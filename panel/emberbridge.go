// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// Matches resident/approval.py: persistent lock inode, withdrawn tombstone,
// executing claim before sudo, and token-matched completion. Never replay claims.
func withBridgeLock(bridge, id string, fn func() error) error {
	if !validBridgeID(id) {
		return fmt.Errorf("invalid bridge request id")
	}
	parent := filepath.Join(bridge, "req")
	path := filepath.Join(parent, id+".lock")
	flags := os.O_RDWR | syscall.O_NOFOLLOW | syscall.O_NONBLOCK
	f, err := os.OpenFile(path, flags|os.O_CREATE|os.O_EXCL, 0o660)
	if os.IsExist(err) {
		f, err = os.OpenFile(path, flags, 0)
	} else if err == nil {
		info, e := os.Stat(parent)
		if e == nil {
			e = f.Chmod(0o660)
		}
		if e == nil {
			if stat, ok := info.Sys().(*syscall.Stat_t); ok {
				e = f.Chown(-1, int(stat.Gid))
			}
		}
		if e != nil {
			f.Close()
			return e
		}
	}
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("bridge lock must be a regular file")
	}
	// The panel writes this on the keypress that runs sudo, inside the update loop: wait long enough for
	// another writer's short critical section, never long enough to freeze the interface.
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			break
		}
		if err != syscall.EAGAIN && err != syscall.EWOULDBLOCK {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("bridge request is busy; press again")
		}
		time.Sleep(2 * time.Millisecond)
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return fn()
}

// handoffText is a hand-off's fields as text, the body last and whole: what the owner's account will post.
func handoffText(h map[string]any) (head []string, body string) {
	s := func(k string) string { v, _ := h[k].(string); return v }
	add := func(label, v string) {
		if v != "" {
			head = append(head, label+": "+v)
		}
	}
	add("action", s("action"))
	add("repo", s("repo"))
	if s("head") != "" {
		add("branches", s("head")+" -> "+s("base"))
	}
	if n, ok := h["number"].(float64); ok {
		add("number", fmt.Sprintf("#%d", int(n)))
	}
	if n, ok := h["run_id"].(float64); ok {
		add("run", fmt.Sprintf("%d", int64(n)))
	}
	add("title", s("title"))
	return head, s("body")
}

// handoffLines is handoffText for the approvals detail: the fields, then the body in full, wrapped.
func handoffLines(h map[string]any, width int) []string {
	head, body := handoffText(h)
	var out []string
	for _, l := range head {
		out = append(out, lipgloss.NewStyle().Foreground(cText).Render(ansi.Truncate(terminalText(l, false), max(8, width), "…")))
	}
	if strings.TrimSpace(body) != "" {
		out = append(out, emberWarn.Render("body (posted as you):"))
		for _, piece := range strings.Split(ansi.Hardwrap(terminalText(body, false), max(8, width-2), true), "\n") {
			out = append(out, "  "+lipgloss.NewStyle().Foreground(cText).Render(piece))
		}
	}
	return out
}

// approveHandoff runs an approved GitHub hand-off through `caminus ember answer ID approved --digest D`: the CLI
// re-checks the request against the digest the owner reviewed, builds the gh command from its checked fields and runs
// it as the owner (no sudo, no shell). The outcome lands in the status line and reaches the agent as the answer.
// handoffDoneMsg: a hand-off's run came back; answered is false when it never reached the bridge (no CLI, the CLI
// failed), so the card comes back instead of vanishing while the request is still pending.
type handoffDoneMsg struct {
	id       string
	answered bool
	text     string
}

func approveHandoff(m emberModel, r emberReq) tea.Cmd {
	return func() tea.Msg {
		if len(m.cli) == 0 {
			return handoffDoneMsg{r.ID, false, "GitHub hand-offs need the caminus CLI (--cli)"}
		}
		argv := append(append([]string{}, m.cli[1:]...), "ember", "answer", r.ID, "approved", "--digest", r.Digest, "--json")
		out, err := exec.Command(m.cli[0], argv...).CombinedOutput()
		text := strings.TrimSpace(string(out))
		if i := strings.LastIndex(text, "\n"); i >= 0 {
			text = text[i+1:]
		}
		var res struct {
			RC  int    `json:"rc"`
			Out string `json:"out"`
		}
		if err != nil || json.Unmarshal([]byte(text), &res) != nil {
			return handoffDoneMsg{r.ID, false, "GitHub hand-off failed: " + terminalText(cliError(out), false)}
		}
		if res.RC != 0 {
			return handoffDoneMsg{r.ID, true, "GitHub refused the hand-off (the agent hears why)"}
		}
		return handoffDoneMsg{r.ID, true, "done on GitHub: " + terminalText(res.Out, false)}
	}
}

func requestDigest(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

func checkPendingRequest(bridge string, r emberReq) (emberReq, error) {
	var current emberReq
	if _, err := os.Lstat(filepath.Join(bridge, "res", r.ID+".json")); !os.IsNotExist(err) {
		return current, fmt.Errorf("request already answered, withdrawn or executing")
	}
	data, err := os.ReadFile(filepath.Join(bridge, "req", r.ID+".json"))
	if err != nil {
		return current, err
	}
	if err = json.Unmarshal(data, &current); err != nil {
		return current, err
	}
	if current.ID != r.ID {
		return current, fmt.Errorf("bridge request id mismatch")
	}
	if r.Digest != "" && requestDigest(data) != r.Digest {
		return current, fmt.Errorf("request changed; review it again")
	}
	return current, nil
}

func claimRoot(bridge, actor string, r emberReq, decision, cmd string) (string, error) {
	return claimRootGrant(bridge, actor, r, decision, cmd, false)
}

// claimRootGrant is claimRoot that also grants the approved command for its goal: the answer carries grant, the
// resident records it, and the same exact command in that goal is approved without the two-press confirmation.
func claimRootGrant(bridge, actor string, r emberReq, decision, cmd string, grant bool) (string, error) {
	token := ""
	err := withBridgeLock(bridge, r.ID, func() error {
		current, err := checkPendingRequest(bridge, r)
		if err != nil {
			return err
		}
		if current.kind() != "root" || (decision != "approved" && decision != "edited") || cmd == "" {
			return fmt.Errorf("invalid root decision")
		}
		if decision == "approved" && current.Cmd != cmd {
			return fmt.Errorf("approved command changed")
		}
		nonce := make([]byte, 16)
		if _, err = rand.Read(nonce); err != nil {
			return err
		}
		token = fmt.Sprintf("%x", nonce)
		value := map[string]any{"id": r.ID, "decision": "executing", "claim": token, "final_decision": decision,
			"approved_by": actor, "cmd": cmd, "ts": float64(time.Now().UnixMilli()) / 1000}
		if grant && decision == "approved" {
			value["grant"] = true // finishRoot keeps it: the resident reads it from the final answer
		}
		data, _ := json.Marshal(value)
		return writeBridgeJSON(filepath.Join(bridge, "res", r.ID+".json"), data)
	})
	return token, err
}

func finishRoot(bridge, id, token string, rc int, out string) error {
	return withBridgeLock(bridge, id, func() error {
		path := filepath.Join(bridge, "res", id+".json")
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var value map[string]any
		if err = json.Unmarshal(data, &value); err != nil {
			return err
		}
		if token == "" || value["decision"] != "executing" || value["claim"] != token {
			return fmt.Errorf("root execution claim changed; inspect the audit log")
		}
		value["decision"] = value["final_decision"]
		delete(value, "final_decision")
		value["rc"], value["out"], value["err"], value["ts"] = rc, out, "", float64(time.Now().UnixMilli())/1000
		data, _ = json.Marshal(value)
		return writeBridgeJSON(path, data)
	})
}

// grantSeconds is a grant's limit on the owner's side (the Python CLI's GRANT_SECONDS): whatever the agent's goal does,
// a grant is not honoured after this.
const grantSeconds = 12 * 3600

// grantRecorded: the owner's own audit holds a granted, approved run of exactly this command in exactly this goal,
// less than grantSeconds ago. The agent's granted flag and goal id are not trusted on their own.
func grantRecorded(auditPath string, r emberReq, now time.Time) bool {
	if !r.Granted || r.Goal == "" || r.Cmd == "" {
		return false
	}
	data, err := os.ReadFile(auditPath)
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		var e struct {
			Event, Decision, Cmd, Goal string
			Granted                    bool
			Ts                         float64
		}
		if json.Unmarshal([]byte(line), &e) != nil {
			continue
		}
		if e.Event == "root_result" && e.Decision == "approved" && e.Granted && e.Cmd == r.Cmd && e.Goal == r.Goal &&
			float64(now.Unix())-e.Ts <= grantSeconds {
			return true
		}
	}
	return false
}

// presence is the human's last key or mouse event in this chat, stamped in the bridge (presence.json) so the resident
// knows whether they are at the screen: a work question waits question_timeout while they are, and while they are
// away it waits until question_timeout after they come back (resident approval.wait_left; questionLeft mirrors it).
type presence struct {
	Seen    float64 `json:"seen"` // the last input (Unix seconds)
	Back    float64 `json:"back"` // the first input after a gap longer than the away threshold
	written float64
}

// presenceAwayAfter is the away threshold when a request does not carry one (resident approval.AWAY_AFTER).
const presenceAwayAfter = 300.0

// presenceEvery is how often the stamp is rewritten while the human keeps typing (the threshold is minutes).
const presenceEvery = 30.0

func (p *presence) input(now time.Time) {
	t := float64(now.UnixMilli()) / 1000
	if p.Seen == 0 || t-p.Seen > presenceAwayAfter {
		p.Back = t
	}
	p.Seen = t
}

// sync takes a newer stamp from the bridge (another panel's) and writes this one's when it changed: at once when the
// human came back, else at most every presenceEvery seconds. A write that fails is not retried until the next input
// (each tries an fsync, and the bridge's sticky folder refuses a stamp another account made).
func (p *presence) sync(bridge string) {
	if bridge == "" {
		return
	}
	path := filepath.Join(bridge, "presence.json")
	var file presence
	if data, err := os.ReadFile(path); err == nil && json.Unmarshal(data, &file) == nil && file.Seen > p.Seen {
		p.Seen, p.Back, p.written = file.Seen, file.Back, file.Seen
	}
	if p.Seen <= p.written || (p.Back <= p.written && p.Seen-p.written < presenceEvery) {
		return
	}
	data, _ := json.Marshal(p)
	_ = writeBridgeJSON(path, data)
	p.written = p.Seen
}
