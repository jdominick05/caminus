// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func TestToolCardsSayWhatCameBackInPlainWords(t *testing.T) {
	home, cwd := "/srv/agent/homes/resident", "/srv/agent/homes/resident/work"
	card := func(name string, args, result map[string]any) string {
		return ansi.Strip(actionCard(name, args, result, 160, home, cwd))
	}
	cases := []struct {
		name         string
		args, result map[string]any
		want         string
	}{
		{"terminal", map[string]any{"command": "cd " + cwd + " && python3 -m unittest"}, map[string]any{"exit_code": 0.0, "output": "\x1b[32mOK\x1b[0m\n"}, "$ python3 -m unittest"},
		{"read", map[string]any{"path": cwd + "/inventory.py"}, map[string]any{"shown": "1-20"}, "read  inventory.py"},
		{"read", map[string]any{"path": home + "/notes/a.md"}, map[string]any{"shown": "1-2"}, "~/notes/a.md"},
		{"search", map[string]any{"pattern": "def total", "path": cwd}, map[string]any{"matches": 2.0, "files": 1.0,
			"text": "work/inventory.py\n    14-  \n    15:  def total_value(items):\n    16-      \"\"\"The value\"\"\"\n    --\n    40:  def total(x):"},
			"search  “def total” under ~/work  ·  2 matches in 1 file\n  ⎿ work/inventory.py:15  def total_value(items):\n    work/inventory.py:40  def total(x):"},
		{"search", map[string]any{"glob": "*.py"}, map[string]any{"files": 2.0, "text": "inventory.py\ntools/report.py"}, "search  *.py  ·  2 files\n  ⎿ inventory.py\n    tools/report.py"},
		{"search", map[string]any{"pattern": "nope", "glob": "*.go"}, map[string]any{"matches": 0.0, "files": 0.0, "searched": 12.0, "text": ""}, "“nope” in *.go  ·  no matches in 12 files"},
		{"search", map[string]any{"pattern": "("}, map[string]any{"error": "pattern is not a valid regex (missing ), unterminated subpattern at position 0)"}, "“(”  ·  pattern is not a valid regex"},
		{"search", map[string]any{"pattern": "x", "path": "gone"}, map[string]any{"error": "[Errno 2] No such file or directory: '" + home + "/gone'"}, "“x” under gone  ·  [Errno 2] No such file or directory: '~/gone'"},
		{"peers", nil, map[string]any{"peers": []any{map[string]any{"name": "lab"}}}, "1 peer: lab"},
		{"lease", map[string]any{"action": "list"}, map[string]any{"leases": []any{}}, "no leases"},
		{"discover", nil, map[string]any{"account": map[string]any{"user": "agent"}, "programs": map[string]any{"build": map[string]any{"make": "/usr/bin/make", "gcc": "/usr/bin/gcc"}}}, "account agent · 2 programs found"},
		{"budget", nil, map[string]any{"goal": map[string]any{"steps": 17.0}, "context": map[string]any{"tokens": 13000.0, "window": 65536.0}}, "17 steps · 13K / 65K context"},
		{"todo", nil, map[string]any{"items": 2.0, "left": 2.0}, "2 tasks · 2 left"},
		{"wait", map[string]any{"until": "time"}, map[string]any{"happened": "timeout", "waited_s": 30.0}, "timeout · waited 30s"},
		{"rollback", map[string]any{"id": "cp1"}, map[string]any{"restored": []any{"a.py"}}, "restored 1 file"},
		{"machine", nil, map[string]any{"gpus": []any{map[string]any{"vram_used_mib": 2048.0, "vram_total_mib": 24576.0, "busy_percent": 3.0}}, "memory": map[string]any{"available_mib": 40960.0}, "note": "GPUs are shared with your human"}, "1 GPU · 2/24 GB 3 % · RAM 40 GB free"},
		{"schedule", nil, map[string]any{"sleeping_until": "23:13", "note": "end your turn now"}, "parked until 23:13"},
		{"send_message", map[string]any{"to": "ghost", "text": "hi"}, map[string]any{"error": "unknown peer; initialize the recipient on the same mail bus first"}, "no session called ghost gets mail here"},
	}
	for _, c := range cases {
		v := card(c.name, c.args, c.result)
		if !strings.Contains(v, c.want) {
			t.Fatalf("%s: want %q in %q", c.name, c.want, v)
		}
		for _, bad := range []string{"\\x1b", "[32m", home, "use lease before", "end your turn", "mail bus"} {
			if strings.Contains(v, bad) {
				t.Fatalf("%s: %q must not show: %q", c.name, bad, v)
			}
		}
	}
}

func TestAWriteSaysWriteAndAReportHasNoSecondCard(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 140, 40
	m.resident.Home = "/srv/agent/homes/resident"
	m.chat.cards, m.chat.expanded = false, false // the compact view
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "write", "call_id": "w", "result": map[string]any{
			"path": "/srv/agent/homes/resident/answers/notes.txt", "created": true, "note": "created a new file", "added": 2.0, "diff": "@@ -0,0 +1,2 @@\n+a\n+b"}}},
		{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "report", "call_id": "r", "result": map[string]any{"delivered": "fyi", "note": "in the digest"}}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	if !strings.Contains(v, "write") || !strings.Contains(v, "answers/notes.txt") || strings.Contains(v, "/srv/agent") || !strings.Contains(v, "created") || strings.Contains(v, " edit ") || strings.Contains(v, "$ write") {
		t.Fatalf("a write reads write, created, with a short path:\n%s", v)
	}
	if strings.Contains(v, "in the digest") {
		t.Fatalf("a report's card repeated its line:\n%s", v)
	}
}

func TestAFinishedCommandLeavesNoRunningLineAbove(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 140, 40
	call := map[string]any{"id": "c1", "type": "function", "function": map[string]any{"name": "terminal", "arguments": `{"command":"python3 -m unittest"}`}}
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "", "tool_calls": []any{call}}},
		{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "terminal", "call_id": "c1", "result": map[string]any{"exit_code": 0.0, "output": "OK\n"}}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	if v := ansi.Strip(m.chat.content); strings.Contains(v, "running:") {
		t.Fatalf("the finished command's running line stayed:\n%s", v)
	}
}

func TestTheExpandedViewIsKeyValueLinesAndKeepsYourPlace(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 140, 30
	var evs []residentEvent
	for i := 0; i < 40; i++ {
		evs = append(evs, residentEvent{Seq: i + 1, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "discover", "call_id": fmt.Sprint(i),
			"result": map[string]any{"account": map[string]any{"user": "agent", "uid": 1001.0}}}})
	}
	m.residentEvents = evs
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	m.chat.transcript.GotoBottom()
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'o', Mod: tea.ModCtrl})
	m = next.(emberModel)
	v := ansi.Strip(m.chat.content)
	if strings.Contains(v, `"account"`) || !strings.Contains(v, "user: agent") || !strings.Contains(v, "uid: 1001") {
		t.Fatalf("key: value lines, not JSON:\n%s", v)
	}
	if !m.chat.transcript.AtBottom() {
		t.Fatal("Ctrl+O at the bottom stays at the bottom (it jumped to the top)")
	}
	if !strings.Contains(m.msg, "Ctrl+O") || strings.Contains(m.msg, "Ctrl+E") {
		t.Fatalf("the toast names Ctrl+O: %q", m.msg)
	}
	m.msgOf, m.msgAt = m.msg, time.Now().Add(-7*time.Second)
	if strings.Contains(ansi.Strip(m.chatComposer()), "Ctrl+O back") {
		t.Fatal("an expired toast leaves the frame even before the tick clears it")
	}
}

func TestAnErrorCardDoesNotRepeatItsFirstLine(t *testing.T) {
	m := chatFixture(t)
	e := residentEvent{Seq: 1, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "verify", "call_id": "f",
		"result": map[string]any{"error": "the done check is not met\nthe tests still fail"}}}
	m.chat.expanded = true
	v := ansi.Strip(m.chat.eventBlock(e))
	if strings.Count(v, "the done check is not met") != 1 || !strings.Contains(v, "the tests still fail") {
		t.Fatalf("the reason once, then the rest:\n%s", v)
	}
}
