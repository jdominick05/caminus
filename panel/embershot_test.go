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
	"time"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// shotWidth is the README pictures' terminal width: wide enough for Ember's side pane (from 160 columns), as a real
// terminal on a desktop is.
const shotWidth, shotHeight = 170, 40

// TestEmberShot writes ember-chat.svg, the README's picture of Ember's chat (`caminus ember ui`), from the program's
// own drawing: its real render() painted with the default theme's background as its View does, at shotWidth, so the
// side pane (tasks, now), the live reply under the transcript, the activity line and the status line show. The goal is made up (a stock-list bug:
// read, a done check, a failing test, a wrong path, the fix, the tests running again); nothing from a real machine
// goes in. tools/release/panel_shot.py redraws it with panel.svg at every release.
//
//	python -I tools/release/panel_shot.py
func TestEmberShot(t *testing.T) {
	dir := os.Getenv("CAMINUS_SHOTS")
	if dir == "" {
		t.Skip("run tools/release/panel_shot.py to redraw the README's pictures")
	}
	t.Setenv("XDG_STATE_HOME", t.TempDir()) // the chat remembers its last session there; never the human's
	// the default theme, never the saved one of whoever runs the redraw (the package loads that at start)
	savedTheme := themeCurrent
	if err := themeLoad("caminus"); err != nil {
		t.Fatal(err)
	}
	defer func() { themeCurrent = savedTheme; applyThemeStyles() }()
	m := shotEmber(t, shotWidth, shotHeight)
	frame := m.render()
	protect := max(0, m.height-lipgloss.Height(m.chatBottom())-lipgloss.Height(m.chatFoot(shotWidth)))
	frame = paintBackground(frame, protect, shotWidth, shotHeight, true)
	plain := ansi.Strip(frame)
	for _, want := range []string{"inventory.py", "tasks", "now", "swift-1.5", "tok/s"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("the picture lost %q (the side pane and the status line must show):\n%s", want, plain)
		}
	}
	// the tasks head the right column (no thoughts box above them); the reply being written runs under the transcript
	top := lipgloss.Height(m.chatHead(shotWidth))
	if side := ansi.Cut(strings.Split(plain, "\n")[top], m.transcriptCols(), shotWidth); !strings.Contains(side, "tasks") || strings.Contains(plain, "thoughts") {
		t.Fatalf("the picture's top right is the tasks, with no thoughts box (got %q):\n%s", side, plain)
	}
	svg := svgFrames([]string{frame}, "caminus ember ui", 1)
	for d := xml.NewDecoder(strings.NewReader(svg)); ; {
		if _, err := d.Token(); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("ember-chat.svg is not well-formed XML: %v", err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "ember-chat.svg"), []byte(svg), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("ember-chat.svg: %d KB", len(svg)/1024)
}

// shotEmber: a goal in progress on Swift 1.5: read, a done check, a todo list, an edit (a diff), a failing test run (an
// error line), the fix, and the tests running now; the side pane, the activity line and the status pills.
func shotEmber(t *testing.T, w, h int) emberModel {
	m := emberModel{agentName: "resident", tab: 6, width: w, height: h, mode: "normal", sshOK: true}
	m.initChat()
	const goal = "g1"
	prompt := "The stock total in inventory.py is too low. Find out why, fix it, and run the tests."
	call := func(id, name string, args map[string]any) map[string]any {
		raw, _ := json.Marshal(args)
		return map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": string(raw)}}
	}
	turn := func(text, reasoning string, calls ...map[string]any) map[string]any {
		d := map[string]any{"role": "assistant", "content": text, "reasoning": reasoning}
		if len(calls) > 0 {
			l := make([]any, len(calls))
			for i, c := range calls {
				l[i] = c
			}
			d["tool_calls"] = l
		}
		return d
	}
	res := func(id, name string, r map[string]any) map[string]any {
		return map[string]any{"name": name, "call_id": id, "result": r}
	}
	todo := func(states ...string) map[string]any {
		texts := []string{"read inventory.py", "state the done check", "fix total_value", "run the tests"}
		items := make([]any, len(texts))
		for i, s := range texts {
			items[i] = map[string]any{"text": s, "state": states[i]}
		}
		return map[string]any{"items": items}
	}
	src := "    14\tdef total_value(items):\n    15\t    total = 0.0\n    16\t    for name, qty, price in items[1:]:\n" +
		"    17\t        total += qty * price\n    18\t    return round(total, 2)\n"
	events := []map[string]any{
		{"kind": "goal_queued", "data": map[string]any{"prompt": prompt}},
		{"kind": "todo", "data": todo("in_progress", "pending", "pending", "pending")},
		{"kind": "assistant", "data": turn("", "Start where the total is computed.", call("c1", "read", map[string]any{"path": "inventory.py"}))},
		{"kind": "tool_result", "data": res("c1", "read", map[string]any{"path": "inventory.py", "lines": 25.0, "shown": "14-18", "text": src})},
		{"kind": "assistant", "data": turn("`total_value` starts at `items[1:]`, so the first row never counts.", "",
			call("c2", "define_done", map[string]any{"check": "total_value counts every row and the tests pass", "gate": "python3 -m unittest -q"}))},
		{"kind": "tool_result", "data": res("c2", "define_done", map[string]any{"done_check": "total_value counts every row and the tests pass"})},
		{"kind": "todo", "data": todo("done", "done", "in_progress", "pending")},
		{"kind": "assistant", "data": turn("", "", call("c3", "terminal", map[string]any{"command": "python3 -m unittest -q"}))},
		{"kind": "tool_result", "data": res("c3", "terminal", map[string]any{"session": "s1", "exit_code": 1.0,
			"output": "F...\nFAIL: test_total (tests.test_inventory.T.test_total)\nAssertionError: 71.51 != 93.11\n\nRan 4 tests in 0.002s\n\nFAILED (failures=1)\n"})},
		{"kind": "assistant", "data": turn("", "", call("c3b", "read", map[string]any{"path": "tests/test_stock.py"}))},
		{"kind": "tool_result", "data": res("c3b", "read", map[string]any{"error": "tests/test_stock.py: no such file (the tests are in tests/test_inventory.py)"})},
		{"kind": "assistant", "data": turn("", "", call("c4", "edit", map[string]any{"path": "inventory.py", "old": "items[1:]:", "new": "items:"}))},
		{"kind": "tool_result", "data": res("c4", "edit", map[string]any{"path": "inventory.py", "replaced": 1.0, "added": 1.0, "removed": 1.0,
			"diff": "@@ -16,1 +16,1 @@\n-    for name, qty, price in items[1:]:\n+    for name, qty, price in items:"})},
		{"kind": "todo", "data": todo("done", "done", "done", "in_progress")},
		{"kind": "assistant", "data": turn("", "", call("c5", "terminal", map[string]any{"command": "python3 -m unittest -q"}))},
	}
	var evs []residentEvent
	for i, e := range events {
		evs = append(evs, residentEvent{Seq: i + 1, Ts: float64(time.Now().Unix()-240) + float64(i*15), Goal: goal, Kind: e["kind"].(string), Data: e["data"].(map[string]any)})
	}
	hit := 0.91
	st := residentStatus{Name: "resident", Model: "swift-1.5", Context: 32768, ContextUsed: 9400, TokPerS: 42.3, DaemonAlive: true,
		OnlineS: 5400, CacheHit: &hit, Compactions: map[string]int{goal: 0}, LastEvent: len(evs),
		Goals: []residentGoal{{ID: goal, Prompt: prompt, State: "working", Steps: 5}}}
	m.receiveResident(residentMsg{agent: "resident", status: st, events: evs})
	m.chat.live.active, m.chat.live.tools = true, []string{"terminal"}
	m.chat.live.reasoning = "The slice dropped the first row; with it back the total should be 93.11. Running the tests again to check."
	m.chat.transcript.GotoBottom()
	return m
}
