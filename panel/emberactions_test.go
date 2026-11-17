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

func TestASandboxCardIsMarkedAndSaysItsNetworkAndTheLimitHit(t *testing.T) {
	conns := []any{map[string]any{"target": "example.com:443", "verdict": "allowed"},
		map[string]any{"target": "www.iana.org:443", "verdict": "refused: not in hosts"}}
	args := map[string]any{"command": "python3 build.py", "hosts": []any{"example.com"}}
	result := map[string]any{"exit_code": -9.0, "output": "building\n", "limit_hit": "memory", "hosts": []any{"example.com"},
		"connections": conns, "box": "/srv/agent/homes/r/boxes/g1",
		"limits": map[string]any{"time_s": 600.0, "memory_mb": 4096.0, "tasks": 512.0, "cpu_percent": 800.0, "enforced": "systemd"}}
	v := ansi.Strip(actionCard("sandbox", args, result, 200, "", ""))
	for _, want := range []string{"sandbox", "$ python3 build.py", "memory limit hit: box killed", "net: example.com", "1 refused"} {
		if !strings.Contains(v, want) {
			t.Fatalf("want %q in the compact card:\n%s", want, v)
		}
	}
	if v := ansi.Strip(actionCard("sandbox", map[string]any{"command": "make"}, map[string]any{"exit_code": 0.0, "output": "ok\n", "hosts": []any{}}, 200, "", "")); !strings.Contains(v, "no network") || strings.Contains(v, "limit hit") {
		t.Fatalf("a box without hosts says no network, and no limit when none hit:\n%s", v)
	}

	// expanded: the call card is marked sandbox (not "terminal"), the result lists limits and every connection
	m := chatFixture(t)
	m.chat.expanded = true
	call := residentEvent{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"id": "s1",
		"function": map[string]any{"name": "sandbox", "arguments": `{"command":"python3 build.py","hosts":["example.com"]}`}}}}}
	if v := ansi.Strip(m.chat.eventBlock(call)); !strings.Contains(v, "sandbox") || strings.Contains(v, "terminal") || !strings.Contains(v, "net: example.com") {
		t.Fatalf("the call card reads sandbox with its hosts:\n%s", v)
	}
	res := residentEvent{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "sandbox", "call_id": "s1", "result": result}}
	v = ansi.Strip(m.chat.eventBlock(res))
	for _, want := range []string{"memory limit hit", "limits: 600s · 4096 MB · 512 processes · CPU 800 % (systemd)",
		"connection: example.com:443  allowed", "connection: www.iana.org:443  refused: not in hosts", "building"} {
		if !strings.Contains(v, want) {
			t.Fatalf("want %q in the expanded result:\n%s", want, v)
		}
	}
	ev := residentEvent{Seq: 3, Goal: "original", Kind: "sandbox_connections", Data: map[string]any{"session": "x", "connections": conns}}
	if v := ansi.Strip(m.chat.eventBlock(ev)); !strings.Contains(v, "sandbox network") || !strings.Contains(v, "www.iana.org:443  refused") {
		t.Fatalf("the connections event is its own line:\n%s", v)
	}
}

func TestABrowserCardSaysWhatItDidThePageAndItsNetwork(t *testing.T) {
	snap := "- heading \"Thanks\" (level 1)\n- text \"sent ?who=Ada\"\n- link \"Home\" [e6] -> /\n- button \"Again\" [e7]"
	result := map[string]any{"did": "clicked button \"Send\"", "title": "Done", "url": "http://127.0.0.1:8000/done.html?who=Ada",
		"snapshot": snap, "network": map[string]any{"allowed": 3.0, "refused": []any{"10.0.0.5:80 (refused: LAN)"}}}
	v := ansi.Strip(actionCard("browser", map[string]any{"action": "click", "ref": "e5"}, result, 200, "", ""))
	for _, want := range []string{"browser", "clicked button \"Send\"", "Done · 127.0.0.1:8000/done.html?who=Ada", "1 refused",
		"heading \"Thanks\"", "link \"Home\" [e6]"} {
		if !strings.Contains(v, want) {
			t.Fatalf("want %q in the compact card:\n%s", want, v)
		}
	}
	if strings.Contains(v, "button \"Again\"") {
		t.Fatalf("the compact card shows a short excerpt of the snapshot:\n%s", v)
	}
	shot := ansi.Strip(actionCard("browser", map[string]any{"action": "screenshot"},
		map[string]any{"screenshot": "/h/w/boxes/g/screenshots/shot-1.png", "title": "Done", "url": "http://x/"}, 200, "/h", "/h/w"))
	if !strings.Contains(shot, "screenshot") || !strings.Contains(shot, "→ boxes/g/screenshots/shot-1.png") {
		t.Fatalf("a screenshot card names its file:\n%s", shot)
	}
	refused := ansi.Strip(actionCard("browser", map[string]any{"action": "type", "ref": "e4", "text": "x"},
		map[string]any{"error": "that is a password field: the browser never types passwords"}, 200, "", ""))
	if !strings.Contains(refused, "password field") {
		t.Fatalf("a refusal shows its reason:\n%s", refused)
	}

	// expanded: the whole snapshot and every connection (the event's panel field, not the model's result)
	m := chatFixture(t)
	m.chat.expanded = true
	conns := []any{map[string]any{"target": "127.0.0.1:8000", "verdict": "allowed (named)"},
		map[string]any{"target": "10.0.0.5:80", "verdict": "refused: LAN"}}
	res := residentEvent{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "browser", "call_id": "b1",
		"result": result, "panel": map[string]any{"connections": conns}}}
	v = ansi.Strip(m.chat.eventBlock(res))
	for _, want := range []string{"browser", "clicked button \"Send\"", "button \"Again\" [e7]",
		"connection: 127.0.0.1:8000  allowed (named)", "connection: 10.0.0.5:80  refused: LAN"} {
		if !strings.Contains(v, want) {
			t.Fatalf("want %q in the expanded card:\n%s", want, v)
		}
	}
}
