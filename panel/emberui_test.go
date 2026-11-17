// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	tea "charm.land/bubbletea/v2"
	"encoding/json"
	"github.com/charmbracelet/x/ansi"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func writeReq(t *testing.T, bridge, id, cmd string, ts float64) {
	t.Helper()
	for _, d := range []string{"req", "res"} {
		if err := os.MkdirAll(filepath.Join(bridge, d), 0o770); err != nil {
			t.Fatal(err)
		}
	}
	data, _ := json.Marshal(emberReq{ID: id, Pid: 42, User: "agent", CWD: "/home/agent", Cmd: cmd, Ts: ts})
	if err := os.WriteFile(filepath.Join(bridge, "req", id+".json"), data, 0o660); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRequestsSkipsAnsweredAndSorts(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "b", "second", 200)
	writeReq(t, bridge, "a", "first", 100)
	writeReq(t, bridge, "c", "answered", 300)
	if err := os.WriteFile(filepath.Join(bridge, "res", "c.json"), []byte(`{"id":"c"}`), 0o660); err != nil {
		t.Fatal(err)
	}
	reqs, err := loadRequests(bridge)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 || reqs[0].ID != "a" || reqs[1].ID != "b" {
		t.Fatalf("want [a b] oldest first, got %+v", reqs)
	}
	if reqs[0].Cmd != "first" || reqs[0].User != "agent" {
		t.Fatalf("fields not parsed: %+v", reqs[0])
	}
}

func TestWriteAnswerAtomicAndSchema(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "x", "id -u", 1)
	rc := 0
	if err := writeAnswer(bridge, "human", emberReq{ID: "x"}, "approved", "id -u", &rc, "0\n", "", ""); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(bridge, "res", "x.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m["decision"] != "approved" || m["approved_by"] != "human" || m["rc"].(float64) != 0 || m["out"] != "0\n" {
		t.Fatalf("bad answer: %v", m)
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "x.json.tmp")); !os.IsNotExist(err) {
		t.Fatal("tmp file left behind")
	}
	// and loadRequests now hides it
	reqs, _ := loadRequests(bridge)
	if len(reqs) != 0 {
		t.Fatalf("answered request still queued: %+v", reqs)
	}
}

func TestAppendAuditMatchesPythonSchema(t *testing.T) {
	p := filepath.Join(t.TempDir(), "state", "env-audit.jsonl")
	if err := appendAudit(p, map[string]any{"event": "root_decision", "id": "x", "decision": "denied"}); err != nil {
		t.Fatal(err)
	}
	lines := tailLines(p, 10)
	if len(lines) != 1 {
		t.Fatalf("want 1 line, got %d", len(lines))
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &m); err != nil {
		t.Fatal(err)
	}
	if m["event"] != "root_decision" || m["decision"] != "denied" {
		t.Fatalf("bad event: %v", m)
	}
	if _, ok := m["ts"].(float64); !ok {
		t.Fatal("ts missing or not a number (python audit.log writes float ts)")
	}
	if got := tailLines(p, 0); len(got) != 0 {
		t.Fatal("tail 0 should be empty")
	}
	if got := tailLines(filepath.Join(t.TempDir(), "missing.jsonl"), 5); len(got) != 0 {
		t.Fatal("missing file should read empty")
	}
}

func TestShQuote(t *testing.T) {
	if got := shQuote("a'b"); got != `'a'\''b'` {
		t.Fatalf("shQuote: %s", got)
	}
}

func TestViewShowsQueueAndAudit(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "r1", "systemctl restart foo", 1)
	audit := filepath.Join(t.TempDir(), "a.jsonl")
	if err := appendAudit(audit, map[string]any{"event": "run", "cmd": "id"}); err != nil {
		t.Fatal(err)
	}
	// The chat is the only screen: a waiting request asks in the composer's place, the context page is an overlay.
	m := chatFixture(t)
	m.user, m.bridge, m.auditPath, m.key, m.seen, m.width, m.height = "agent", bridge, audit, "k", map[string]bool{}, 120, 40
	m.refreshLocal()
	v := ansi.Strip(m.render())
	if strings.Contains(v, "CAMINUS ENV") || !strings.Contains(v, "systemctl restart foo") || strings.Contains(v, "pending ^A") {
		t.Fatalf("the chat is the screen and the waiting request asks where you type:\n%s", v)
	}
	var next tea.Model
	// the audit log, the servers and the jobs are CLI only now (caminus ember audit / ctx / agent jobs)
	for _, gone := range []string{"/audit", "/servers", "/jobs", "/identity", "/update", "/export", "/freetime"} {
		if next, _ = m.chatSlash(gone); next.(emberModel).chat.page >= 0 || !strings.Contains(next.(emberModel).msg, "Unknown command") {
			t.Fatalf("%s left the TUI", gone)
		}
	}
	m.resident.Context, m.resident.ContextUsed, m.resident.CompactThreshold = 1000000, 250000, .38
	next, _ = m.chatSlash("/context")
	if v := ansi.Strip(next.(emberModel).render()); !strings.Contains(v, "context · ") || !strings.Contains(v, "25.0 %") || !strings.Contains(v, "38.0 %") {
		t.Fatalf("context overlay shows the agent's own window:\n%s", v)
	}

}

func TestKeyDenyAndSkip(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "r1", "boom", 1)
	audit := filepath.Join(t.TempDir(), "a.jsonl")
	m := emberModel{user: "agent", bridge: bridge, auditPath: audit, key: "k",
		seen: map[string]bool{}, mode: "normal", width: 120, height: 40}
	m.refreshLocal()
	// deny
	mm, _ := m.onKey("d")
	m = mm.(emberModel)
	if len(m.reqs) != 0 {
		t.Fatalf("denied request still visible: %+v", m.reqs)
	}
	data, _ := os.ReadFile(filepath.Join(bridge, "res", "r1.json"))
	var ans map[string]any
	if err := json.Unmarshal(data, &ans); err != nil || ans["decision"] != "denied" {
		t.Fatalf("deny did not write the answer: %s err=%v", data, err)
	}
	// skip keeps it on disk but hides it this session
	writeReq(t, bridge, "r2", "later", 2)
	m.refreshLocal()
	if len(m.reqs) != 1 || m.reqs[0].ID != "r2" {
		t.Fatalf("want r2 only, got %+v", m.reqs)
	}
	mm, _ = m.onKey("s")
	m = mm.(emberModel)
	if len(m.reqs) != 0 {
		t.Fatal("skip did not hide the request")
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "r2.json")); !os.IsNotExist(err) {
		t.Fatal("skip must not write an answer")
	}
	// edit mode: type, backspace, enter would exec (not run here); esc cancels
	writeReq(t, bridge, "r3", "cmd3", 3)
	m = emberModel{user: "agent", bridge: bridge, auditPath: audit, key: "k",
		seen: map[string]bool{"r2": true}, mode: "normal", width: 120, height: 40}
	m.refreshLocal()
	if len(m.reqs) != 1 || m.reqs[0].ID != "r3" {
		t.Fatalf("want r3 selected, got %+v", m.reqs)
	}
	m.sel = 0
	mm, _ = m.onKey("e")
	m = mm.(emberModel)
	if m.mode != "edit" || m.edit != "cmd3" {
		t.Fatalf("edit mode: %+v", m)
	}
	mm, _ = m.onKey("x")
	m = mm.(emberModel)
	mm, _ = m.onKey("backspace")
	m = mm.(emberModel)
	if m.edit != "cmd3" {
		t.Fatalf("typing+backspace: %q", m.edit)
	}
	mm, _ = m.onKey("esc")
	m = mm.(emberModel)
	if m.mode != "normal" {
		t.Fatal("esc did not leave edit mode")
	}
}

func TestParseJobsServers(t *testing.T) {
	js := parseJobs("build|123|alive|~/env-logs/build.log\nold|99|dead|~/env-logs/old.log\n")
	if len(js) != 2 || js[0].Name != "build" || js[0].State != "alive" || js[1].State != "dead" {
		t.Fatalf("parseJobs: %+v", js)
	}
	ss := parseServers("PORT 8092\n" + `{"model_alias":"coder","n_ctx":262144}` + "\nPORT 8093\n" + `{"model":"x"}` + "\n")
	if len(ss) != 2 || ss[0].Port != 8092 || ss[0].Model != "coder" || ss[0].Nctx != "262144" || ss[1].Nctx != "<nil>" {
		t.Fatalf("parseServers: %+v", ss)
	}
}

func mustRead(t *testing.T, p string) string {
	t.Helper()
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestTabsAndJobKeys(t *testing.T) {
	// The old page switch is gone: Tab and the number keys open nothing.
	m := emberModel{seen: map[string]bool{}, width: 120, height: 40, tab: 6}
	for _, k := range []string{"tab", "shift+tab", "1", "3", "5"} {
		if mm, _ := m.onKey(k); mm.(emberModel).tab != 6 {
			t.Fatalf("%s still switches to an old page", k)
		}
	}
	// jobs: K on a selected job builds a kill script (via returned cmd; just check no panic and selection moves)
	m.tab = 1
	m.jobs = []job{{Name: "build", Pid: "123", State: "alive", Log: "~/env-logs/build.log"}}
	mm, _ := m.onKey("down")
	if mm.(emberModel).jobSel != 0 {
		t.Fatal("jobSel should stay 0 with one job")
	}
	mm, cmd := m.onKey("K")
	if cmd == nil {
		t.Fatal("K on a job should return the kill command")
	}
	_ = mm
}

// TestOverlaysPassNoControlSequencesFromTheAgent: the agent is not trusted, and its text reaches the overlays (memory,
// jobs, processes, logs, servers, the audit, diffs, /btw). Only colour codes may reach the terminal: nothing that moves
// the cursor, clears the screen, sets the title or the clipboard, or recolours the agent's text with codes of its own.
func TestOverlaysPassNoControlSequencesFromTheAgent(t *testing.T) {
	evil := "EVIL\x1b[2J\x1b[H\x1b]0;pwned\x07\x1b]52;c;Zm9v\x1b\\\x1b[31mRED\x1bPdcs\x1b\\\u009b2J\u009d0;t\x07\x07\x08END"
	bad := []string{"\x1b[2J", "\x1b[H", "\x1b]", "\x1bP", "\x1b[31mRED", "\u009b", "\u009d", "\x07", "\x08"}
	audit := filepath.Join(t.TempDir(), "a.jsonl")
	if err := appendAudit(audit, map[string]any{"event": "run" + evil, "cmd": evil}); err != nil {
		t.Fatal(err)
	}
	m := chatFixture(t)
	m.width, m.height, m.auditPath = 120, 40, audit
	m.refreshLocal()
	m.sessionMemory = "# memory\n" + evil
	m.jobs = []job{{Name: evil, Pid: evil, State: evil, Log: evil}}
	m.procs = []string{evil}
	m.logName, m.logView = evil, evil
	m.servers = []server{{Port: 8092, Model: evil}}
	m.resident.Model = evil
	check := func(what, out string, skip ...string) {
		t.Helper()
		for _, b := range bad {
			if slices.Contains(skip, b) {
				continue
			}
			if strings.Contains(out, b) {
				t.Errorf("%s: %q reached the terminal", what, b)
			}
		}
	}
	for name := range pageOverlays {
		next, _ := m.chatSlash(name)
		check(name, next.(emberModel).render())
	}
	m.chat.view = textView{open: true, title: "diff", lines: []string{residentText(evil), evil}, colourDiff: true}
	// a producer that forgot to sanitise: the overlay's own filter still stops everything but colour (at worst the
	// agent recolours its own text there)
	check("a text view (a producer that forgot to sanitise)", m.render(), "\x1b[31mRED")
	m.chat.view = textView{}
	m.chat.btw = btwState{open: true, question: evil}
	next, _ := m.Update(btwMsg{question: evil, answer: evil})
	check("/btw", next.(emberModel).render())
}
