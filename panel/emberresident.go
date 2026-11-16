// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

type residentGoal struct {
	ID       string `json:"id"`
	Prompt   string `json:"prompt"`
	State    string `json:"state"`
	Summary  string `json:"summary"`
	Error    string `json:"error"`
	Steps    int    `json:"steps"`
	Plan     int    `json:"plan"`      // 0 a plain goal, 1 in plan mode, 2 its plan approved
	PlanText string `json:"plan_text"` // the plan it proposed (state waiting_plan)
}
type residentStatus struct {
	Name             string             `json:"name"`
	Model            string             `json:"model"`
	Context          int                `json:"context"`
	ContextUsed      int                `json:"context_used"` // measured at the last turn boundary; 0 means never measured
	TokPerS          float64            `json:"tok_per_s"`    // measured decode speed of the last streamed reply; 0 means none reported
	CompactThreshold float64            `json:"compact_threshold"`
	CompactTarget    float64            `json:"compact_target_ratio"`
	SpendToday       float64            `json:"spend_today"` // every priced turn today, across goals
	SpendGoals       map[string]float64 `json:"spend_goals"` // per goal
	Compactions      map[string]int     `json:"compactions"` // per goal: how many times its history was compacted
	OnlineS          float64            `json:"online_s"`    // how long the daemon has been running (0: unknown)
	CacheHit         *float64           `json:"cache_hit"`   // today's input served from the prompt cache, 0-1 (nil: no turns yet)
	DaemonAlive      bool               `json:"daemon_alive"`
	Goals            []residentGoal     `json:"goals"`
	LastEvent        int                `json:"last_event"`
}
type residentEvent struct {
	Seq  int            `json:"seq"`
	Ts   float64        `json:"ts"`
	Goal string         `json:"goal"`
	Kind string         `json:"kind"`
	Data map[string]any `json:"data"`
}
type residentMsg struct {
	agent  string
	epoch  uint64
	status residentStatus
	events []residentEvent
	err    string
}
type residentActionMsg struct{ text string }

// residentLiveMsg is one NDJSON line from `agent stream`: the reply the runtime is generating right now.
// reset carries the whole text of a new turn; later lines carry only the suffix.
type residentLiveMsg struct {
	agent, turn, text, reasoning string
	tools                        []string
	reset, done                  bool
	err                          string
	epoch                        uint64
	sc                           *bufio.Scanner // the reader it came from: a stale reader's end never stops the current one
}

// residentStreamStartMsg hands the started reader process to the model.
type residentStreamStartMsg struct {
	agent string
	epoch uint64
	state *streamState
	err   string
}

// streamState is the one live reader for this conversation page: one long-lived process instead of a poll
// per keystroke, killed when the page, the session or the UI goes away.
type streamState struct {
	cancel context.CancelFunc
	sc     *bufio.Scanner
}

// cliError is a failed agent command's output as one useful line: for a Python traceback the exception (its last
// line), not "Traceback (most recent call last):", which was all the status line showed of a crash.
func cliError(out []byte) string {
	text := strings.TrimSpace(string(out))
	if !strings.Contains(text, "Traceback (most recent call last)") {
		return text
	}
	lines := strings.Split(text, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if l := strings.TrimSpace(lines[i]); l != "" {
			return "agent runtime error: " + l
		}
	}
	return text
}

func residentCommand(m emberModel, args []string, input string) *exec.Cmd {
	if len(m.cli) == 0 {
		return nil // the panel can run without --cli; callers report the missing CLI instead of panicking
	}
	// Bounded: a stalled CLI or SSH session must not leave the composer on "Sending…" forever.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	_ = cancel // the deadline fires on its own; the process is killed when it passes
	argv := append(append([]string{}, m.cli[1:]...), "ember", "agent")
	argv = append(argv, args...)
	c := exec.CommandContext(ctx, m.cli[0], argv...)
	c.Stdin = strings.NewReader(input)
	return c
}

// startStream opens `agent stream NAME`: the runtime rewrites a snapshot of the reply it is generating
// (at most ten times a second), and each line arrives here as one message.
func startStream(m emberModel) tea.Cmd {
	if len(m.cli) == 0 || m.agentName == "" {
		return nil
	}
	name, epoch := m.agentName, m.residentEpoch
	cli := append([]string{}, m.cli...)
	return func() tea.Msg {
		ctx, cancel := context.WithCancel(context.Background())
		argv := append(append([]string{}, cli[1:]...), "ember", "agent", "stream", name, "--interval", "0.1")
		cmd := exec.CommandContext(ctx, cli[0], argv...)
		out, err := cmd.StdoutPipe()
		if err == nil {
			err = cmd.Start()
		}
		if err != nil {
			cancel()
			return residentStreamStartMsg{agent: name, epoch: epoch, err: err.Error()}
		}
		go func() { _ = cmd.Wait() }() // reap when the page closes and the context kills it
		sc := bufio.NewScanner(out)
		sc.Buffer(make([]byte, 0, 64<<10), 1<<20) // a turn's whole text can arrive in one reset line
		return residentStreamStartMsg{agent: name, epoch: epoch, state: &streamState{cancel: cancel, sc: sc}}
	}
}

// streamLine reads one NDJSON line; the update loop asks for the next one after handling each.
func streamLine(agent string, epoch uint64, sc *bufio.Scanner) tea.Cmd {
	return func() tea.Msg {
		if sc == nil || !sc.Scan() {
			return residentLiveMsg{agent: agent, epoch: epoch, sc: sc, done: true}
		}
		var line struct {
			Turn      string   `json:"turn"`
			Content   string   `json:"content"`
			Reasoning string   `json:"reasoning"`
			Tools     []string `json:"tools"`
			Reset     bool     `json:"reset"`
			Done      bool     `json:"done"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			return residentLiveMsg{agent: agent, epoch: epoch, sc: sc, err: err.Error(), done: true}
		}
		return residentLiveMsg{agent: agent, epoch: epoch, sc: sc, turn: line.Turn, text: line.Content,
			reasoning: line.Reasoning, tools: line.Tools, reset: line.Reset, done: line.Done}
	}
}

func residentRefresh(m emberModel) tea.Cmd {
	name, after := m.agentName, m.chat.seq
	epoch := m.residentEpoch
	if name == "" {
		name = "resident"
	}
	return func() tea.Msg {
		if len(m.cli) == 0 {
			return residentMsg{agent: name, epoch: epoch, err: "resident controls need the caminus CLI (--cli)"}
		}
		out, err := residentCommand(m, []string{"status", name, "--json"}, "").CombinedOutput()
		if err != nil {
			return residentMsg{agent: name, epoch: epoch, err: cliError(out)}
		}
		var status residentStatus
		if err = json.Unmarshal(out, &status); err != nil {
			return residentMsg{agent: name, epoch: epoch, err: err.Error()}
		}
		var events []residentEvent
		// Catch up from the durable cursor. Bound each refresh; another tick continues a large backlog without
		// replacing older entries or skipping a page while the agent works.
		pages := 4
		if after == 0 && status.LastEvent > maxResidentEvents/2 {
			// A fresh open starts at the newest events the panel would keep anyway, in one refresh: it read a long
			// resident's whole history from the first event, 800 a tick, drawing each batch (the chat appeared to
			// replay the entire session on every open) and then dropped all but the newest.
			after = status.LastEvent - maxResidentEvents/2
			pages = 8
		}
		for page := 0; page < pages && after < status.LastEvent; page++ {
			out, err = residentCommand(m, []string{"events", name, "--after", fmt.Sprint(after), "--json"}, "").CombinedOutput()
			if err != nil {
				return residentMsg{agent: name, epoch: epoch, err: cliError(out)}
			}
			var batch []residentEvent
			if err = json.Unmarshal(out, &batch); err != nil {
				return residentMsg{agent: name, epoch: epoch, err: err.Error()}
			}
			if len(batch) == 0 {
				break
			}
			events = append(events, batch...)
			after = batch[len(batch)-1].Seq
		}
		return residentMsg{agent: name, epoch: epoch, status: status, events: events}
	}
}

// auditedOps are the panel's resident controls the owner audit log records.
var auditedOps = map[string]bool{"heartbeat": true, "start": true, "stop": true, "cancel": true, "pause": true, "resume": true,
	"update": true, "rewind": true, "interrupt": true, "model": true}

func residentAction(m emberModel, args []string, input string) tea.Cmd {
	return func() tea.Msg {
		if len(m.cli) == 0 {
			return residentActionMsg{"resident controls need the caminus CLI (--cli)"}
		}
		// The owner's controls go in the audit log: a resident stopped from the panel left no trace of who stopped
		// it. Only the operation and its target, never message text.
		if len(args) > 0 && auditedOps[args[0]] && m.auditPath != "" {
			ev := map[string]any{"event": "resident_" + args[0], "agent": m.agentName}
			if len(args) > 2 && !strings.HasPrefix(args[2], "-") {
				ev["target"] = args[2]
			}
			_ = appendAudit(m.auditPath, ev)
		}
		out, err := residentCommand(m, args, input).CombinedOutput()
		text := strings.TrimSpace(string(out))
		if err != nil {
			text = err.Error() + ": " + cliError(out)
		}
		return residentActionMsg{text}
	}
}

func residentText(s string) string {
	return terminalText(s, true)
}
