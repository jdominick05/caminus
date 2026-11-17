// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"encoding/xml"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

// TestEmberShot writes ember.svg, the README's picture of Ember's chat (`caminus ember ui`): its real render() in the
// same GNOME terminal window as panel.svg, redrawn with it at every release by tools/release/panel_shot.py. The goal
// is made up (a small stock-list project, a bug, its tests) and walks the loop the README describes: read, state what
// done looks like, change, run the check, verify, and the judge's verdict on the finish. Nothing about a real machine
// goes in: no paths, no sessions, no live numbers.
//
//	python -I tools/release/panel_shot.py
func TestEmberShot(t *testing.T) {
	dir := os.Getenv("CAMINUS_SHOTS")
	if dir == "" {
		t.Skip("run tools/release/panel_shot.py to redraw the README's pictures")
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir()) // the chat remembers its last session there; never the human's
	m := emberModel{agentName: "resident", tab: 6, width: 112, height: 50, mode: "normal", sshOK: true}
	m.initChat()

	const goal = "g1"
	prompt := "The stock total in inventory.py is too low. Find out why, fix it, and run the tests."
	call := func(id, name string, args map[string]any) map[string]any {
		raw, _ := json.Marshal(args)
		return map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": string(raw)}}
	}
	turn := func(text string, calls ...map[string]any) map[string]any {
		d := map[string]any{"role": "assistant", "content": text}
		if len(calls) > 0 {
			list := make([]any, len(calls))
			for i, c := range calls {
				list[i] = c
			}
			d["tool_calls"] = list
		}
		return d
	}
	result := func(id, name string, res map[string]any) map[string]any {
		return map[string]any{"name": name, "call_id": id, "result": res}
	}
	source := "     1\t\"\"\"A small stock list: read it, total its value, find what runs low.\"\"\"\n" +
		"    14\tdef total_value(items):\n    15\t    total = 0.0\n    16\t    for name, qty, price in items[1:]:\n" +
		"    17\t        total += qty * price\n    18\t    return round(total, 2)\n"
	diff := "@@ -16,1 +16,1 @@\n-    for name, qty, price in items[1:]:\n+    for name, qty, price in items:"
	events := []map[string]any{
		{"kind": "goal_queued", "data": map[string]any{"prompt": prompt}},
		{"kind": "assistant", "data": turn("", call("c1", "read", map[string]any{"path": "inventory.py"}))},
		{"kind": "tool_result", "data": result("c1", "read", map[string]any{"path": "inventory.py", "lines": 25.0, "shown": "1-25", "text": source})},
		{"kind": "assistant", "data": turn("`total_value` starts at `items[1:]`, so the first row never counts.",
			call("c2", "define_done", map[string]any{"check": "total_value counts every row and the tests pass", "gate": "python3 -m unittest -q"}))},
		{"kind": "tool_result", "data": result("c2", "define_done", map[string]any{"done_check": "total_value counts every row and the tests pass", "gate": "python3 -m unittest -q"})},
		{"kind": "assistant", "data": turn("", call("c3", "edit", map[string]any{"path": "inventory.py", "old": "items[1:]:", "new": "items:"}))},
		{"kind": "tool_result", "data": result("c3", "edit", map[string]any{"path": "inventory.py", "replaced": 1.0, "added": 1.0, "removed": 1.0, "diff": diff})},
		{"kind": "assistant", "data": turn("", call("c4", "terminal", map[string]any{"command": "python3 -m unittest -q"}))},
		{"kind": "tool_result", "data": result("c4", "terminal", map[string]any{"session": "s1", "output": "....\n----------------------------------------------------------------------\nRan 4 tests in 0.002s\n\nOK\n", "exit_code": 0.0})},
		{"kind": "assistant", "data": turn("", call("c5", "verify", map[string]any{"evidence": "unittest ran 4 tests: OK; the total is now 93.11"}))},
		{"kind": "tool_result", "data": result("c5", "verify", map[string]any{"verified": "unittest ran 4 tests: OK; the total is now 93.11"})},
		{"kind": "assistant", "data": turn("Fixed: the loop skipped the first row. The total is now 93.11 and all four tests pass.",
			call("c6", "finish", map[string]any{"summary": "total_value skipped the first row; it now counts every row (93.11). Tests: 4 OK."}))},
		{"kind": "gate", "data": map[string]any{"command": "python3 -m unittest -q", "passed": true, "output": "OK"}},
		{"kind": "judged", "data": map[string]any{"verdict": "met", "reason": "the gate passed and the edit's diff shows the slice removed"}},
		{"kind": "finished", "data": map[string]any{"summary": "total_value skipped the first row; it now counts every row (93.11). Tests: 4 OK."}},
	}
	var evs []residentEvent
	for i, e := range events {
		evs = append(evs, residentEvent{Seq: i + 1, Ts: 1.7e9 + float64(i*4), Goal: goal, Kind: e["kind"].(string), Data: e["data"].(map[string]any)})
	}
	used := 9400
	st := residentStatus{Name: "resident", Model: "qwen3-30b-a3b", Context: 40960, ContextUsed: used, TokPerS: 0, DaemonAlive: true,
		LastEvent: len(evs), Goals: []residentGoal{{ID: goal, Prompt: prompt, State: "completed", Steps: 6,
			Summary: "total_value skipped the first row; it now counts every row (93.11). Tests: 4 OK."}}}
	m.receiveResident(residentMsg{agent: "resident", status: st, events: evs})
	m.chat.transcript.GotoBottom()

	frame := m.render()
	plain := ansi.Strip(frame)
	for _, want := range []string{"inventory.py", "unittest", "met"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("the picture lost %q:\n%s", want, plain)
		}
	}
	svg := svgFrames([]string{frame}, "caminus ember ui", 1)
	for d := xml.NewDecoder(strings.NewReader(svg)); ; {
		if _, err := d.Token(); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("ember.svg is not well-formed XML: %v", err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "ember.svg"), []byte(svg), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("ember.svg: %d KB", len(svg)/1024)
}
