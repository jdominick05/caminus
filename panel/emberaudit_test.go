// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the 2026-11-16 audit's first group: what blocked use or changed what the human typed.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func longText(n int, word string) string {
	var l []string
	for i := 0; i < n; i++ {
		l = append(l, word+" line")
	}
	return strings.Join(l, "\n")
}

func TestAWaitingPlanAndALongHandoffLeaveTheComposerAndOptionsOnScreen(t *testing.T) {
	for _, size := range [][2]int{{160, 48}, {100, 30}, {80, 24}} {
		m := chatFixture(t)
		m.width, m.height = size[0], size[1]
		m.resident.Goals = []residentGoal{{ID: "g1abcdef", State: "waiting_plan", Prompt: "p", PlanText: longText(30, "plan")}}
		m.chat.target = "g1abcdef"
		m.syncChat()
		v := ansi.Strip(m.render())
		if h := lipgloss.Height(m.render()); h > size[1] {
			t.Fatalf("%v: the frame is %d rows, taller than the screen", size, h)
		}
		if !strings.Contains(v, "/plan approve") || !strings.Contains(v, "⇧Tab sessions") || !strings.Contains(v, "/plan shows the whole plan") {
			t.Fatalf("%v: the plan's hint, the status line and a pointer to the rest stay on screen:\n%s", size, v)
		}

		m2, bridge := promptFixture(t)
		m2.width, m2.height = size[0], size[1]
		writeHandoff(t, bridge, "h1", longText(40, "body"))
		m2.refreshLocal()
		m2.syncQuestion()
		m2.syncChat()
		v2 := ansi.Strip(m2.render())
		if h := lipgloss.Height(m2.render()); h > size[1] {
			t.Fatalf("%v: the hand-off frame is %d rows, taller than the screen", size, h)
		}
		for _, want := range []string{"1. Yes", "› 2. No", "3. Tell it something else", "Esc later", "more lines"} {
			if !strings.Contains(v2, want) {
				t.Fatalf("%v: %q must stay on screen (No preselected for a hand-off):\n%s", size, want, v2)
			}
		}
	}
}

func writeHandoff(t *testing.T, bridge, id, body string) {
	t.Helper()
	for _, d := range []string{"req", "res"} {
		_ = os.MkdirAll(filepath.Join(bridge, d), 0o770)
	}
	r := emberReq{ID: id, Kind: "github", Agent: "resident", Summary: "open a PR on a/b", Reason: "from my fork", Ts: float64(time.Now().Unix()),
		Github: map[string]any{"action": "open_pr", "repo": "a/b", "title": "t", "body": body}}
	data := mustJSON(r)
	if err := os.WriteFile(filepath.Join(bridge, "req", id+".json"), data, 0o660); err != nil {
		t.Fatal(err)
	}
}

func TestAnAtWordThatIsNotAPathKeepsItsAt(t *testing.T) {
	m := pathFixture(t)
	if got := expandAttachments(m, "ask @alice about @work/inventory.py and @notes.md"); got != "ask @alice about work/inventory.py and notes.md" {
		t.Fatalf("only a path loses its @: %q", got)
	}
}

func TestHistoryGivesBackWhatYouTyped(t *testing.T) {
	m := chatFixture(t)
	for _, typed := range []string{"//th", "summarize +work/inventory.py please", "and look at @work/inventory.py too"} {
		saveTyped(m.agentName, typed)
	}
	// the events hold what was sent: one slash dropped, the file inlined, the @ gone
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g", Kind: "goal_queued", Data: map[string]any{"prompt": "/th"}},
		{Seq: 2, Goal: "g", Kind: "user_message", Data: map[string]any{"text": "summarize work/inventory.py (attached) please\n\nAttached work/inventory.py:\n```\nimport x\n```"}},
		{Seq: 3, Goal: "g", Kind: "user_message", Data: map[string]any{"text": "and look at work/inventory.py too"}},
		{Seq: 4, Goal: "g", Kind: "user_message", Data: map[string]any{"text": "/etc is full, from an older panel"}},
	}
	got := m.pastPrompts()[:4] // (then the fixture's own goal)
	want := []string{"and look at @work/inventory.py too", "summarize +work/inventory.py please", "//th", "//etc is full, from an older panel"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("history as typed, newest first, the sent forms not repeated:\n got %q\nwant %q", got, want)
	}
}

func TestAShellCommandsOutputGoesToTheChat(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 24
	m.chat.cards = false      // the compact view (the default)
	for i := 2; i < 30; i++ { // a history taller than its window, read from the top
		m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: i, Goal: "original",
			Kind: "assistant", Data: map[string]any{"content": fmt.Sprintf("Observation %d.", i)}}}})
	}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 30, Goal: "original",
		Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"id": "c1", "function": map[string]any{
			"name": "terminal", "arguments": `{"command":"make test"}`}}}}}}}) // the agent's own command, still running
	m.chat.transcript.GotoTop()
	next, _ := m.chatKey(tea.KeyPressMsg{Code: '!', Text: "!"})
	m = next.(emberModel)
	m.chat.compose.SetValue("echo hi")
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	if m = next.(emberModel); cmd == nil || m.chat.shell || m.chat.compose.Value() != "" {
		t.Fatalf("Enter runs the command, leaves shell mode and empties the composer (%q)", m.chat.compose.Value())
	}
	m.chat.compose.SetValue("words typed while it ran")
	m = m.bangDone(bangDoneMsg{cmd: "echo hi", output: "hi"}) // (what runBang returns for it)
	if v := m.chat.compose.Value(); v != "words typed while it ran" {
		t.Fatalf("the output stays out of the composer; a draft typed meanwhile is kept: %q", v)
	}
	view := ansi.Strip(m.chat.transcript.View())
	if !strings.Contains(view, "! shell  $ echo hi") || !strings.Contains(view, "│ hi") || strings.Contains(view, "my shell") {
		t.Fatalf("the command and its output are a card at the bottom of the chat, scrolled into view:\n%s", view)
	}
	if m.chat.group.running != "make test" || !strings.Contains(view, "running: make test") {
		t.Fatalf("your card leaves the agent's running command under it (it is still running):\n%s", view)
	}
	t.Logf("after ! echo hi:\n%s", ansi.Strip(m.render()))
	m.chat.content, m.chat.rendered = "", 0 // a redraw from the start (a resize, a theme)
	m.syncChat()
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'o', Mod: tea.ModCtrl})
	if m = next.(emberModel); !strings.Contains(ansi.Strip(m.chat.content), "$ echo hi") {
		t.Fatal("the card stays through a redraw and in the everything view")
	}
	m.chat.compose.SetValue("!echo again")
	next, cmd = m.submitCompose()
	if m = next.(emberModel); cmd == nil || m.chat.compose.Value() != "" {
		t.Fatalf("a recalled !cmd sent from the composer runs and empties it too: %q", m.chat.compose.Value())
	}
	if m.chat.seq != 30 || m.residentEvents[len(m.residentEvents)-1].Seq >= 0 {
		t.Fatal("the card is the panel's own event: it never moves the resident's event cursor")
	}
}

func TestCtrlCAndCtrlNKeepTheDraftAndEscGoesBack(t *testing.T) {
	m := chatFixture(t)
	m.chat.target, m.chat.newGoal = "original", false
	m.chat.compose.SetValue("an unsent draft I care about")
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	m = next.(emberModel)
	if cmd != nil || m.chat.compose.Value() != "" || loadTyped(m.agentName)[0] != "an unsent draft I care about" {
		t.Fatal("the first Ctrl+C clears the draft into the history; it does not quit")
	}
	if _, cmd = m.chatKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl}); cmd == nil {
		t.Fatal("the second Ctrl+C quits")
	}
	m.chat.compose.SetValue("keep me")
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'n', Mod: tea.ModCtrl})
	m = next.(emberModel)
	if m.chat.compose.Value() != "keep me" || !m.chat.newGoal {
		t.Fatal("Ctrl+N keeps the draft for the new goal")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.newGoal || m.chat.target != "original" || m.chat.compose.Value() != "keep me" {
		t.Fatal("Esc goes back to the goal you were talking to, with the draft")
	}
}

func TestGoalCommandsRefuseAGoalThatIsOver(t *testing.T) {
	m := chatFixture(t)
	m.resident.Goals = []residentGoal{{ID: "00f32512aa", State: "cancelled", Prompt: "p"}}
	m.chat.target = "00f32512aa"
	for _, c := range []string{"/cancel", "/pause", "/compact"} {
		next, cmd := m.chatSlash(c)
		if cmd != nil || !strings.Contains(next.(emberModel).msg, "already cancelled") {
			t.Fatalf("%s on a cancelled goal does nothing and says so: %q", c, next.(emberModel).msg)
		}
	}
}

func TestATimedOutRootRequestReadsTimedOutNotNo(t *testing.T) {
	m := chatFixture(t)
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g", Kind: "approval_requested", Data: map[string]any{"id": "r1", "cmd": "touch /etc/hello"}},
		{Seq: 2, Goal: "g", Kind: "request_answered", Data: map[string]any{"request": "r1", "decision": "denied", "by": "timeout"}},
	}
	if got := m.requestOutcomes()["r1"]; got != "timed out (no)" {
		t.Fatalf("a timeout is not a human's no: %q", got)
	}
}

func TestAPlusPathThatCannotBeAttachedIsNotSent(t *testing.T) {
	m := pathFixture(t)
	dir := t.TempDir()
	sent := filepath.Join(dir, "sent")
	script := filepath.Join(dir, "fake-caminus")
	body := "#!/bin/sh\nif [ \"$2\" = run ]; then echo \"head: cannot open '/srv/agent/homes/r/work/nope.py' for reading: No such file or directory\"; exit 1; fi\ncat > " + sent + "\necho '{\"id\":\"g\"}'\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	m.cli = []string{script}
	m.chat.compose.SetValue("check +work/nope.py and +../../etc/hostname")
	_, cmd := m.sendCompose("check +work/nope.py and +../../etc/hostname")
	res := cmd().(residentComposeResult)
	if _, err := os.Stat(sent); err == nil {
		t.Fatal("nothing is sent when a file cannot be attached")
	}
	if !strings.Contains(res.err, "work/nope.py: no such file") || !strings.Contains(res.err, "../../etc/hostname: a path with .. is not attached") ||
		strings.Contains(res.err, "/srv/agent") {
		t.Fatalf("plain words, short paths: %q", res.err)
	}
}

func TestProseWithDotsIsNotAPathAndATimeoutStaysNo(t *testing.T) {
	m := pathFixture(t)
	if out, problems := expandAttachmentsChecked(m, "I'd say +1... but let's wait"); len(problems) > 0 || out != "I'd say +1... but let's wait" {
		t.Fatalf("+1... is a word: %q %v", out, problems)
	}
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g", Kind: "question", Data: map[string]any{"id": "q1", "text": "May I delete the cache?", "approval": true}},
		{Seq: 2, Goal: "g", Kind: "question_unanswered", Data: map[string]any{"id": "q1", "approval": true, "outcome": "no"}},
		{Seq: 3, Goal: "g", Kind: "request_answered", Data: map[string]any{"request": "q1", "decision": "answered", "by": "timeout"}},
	}
	if got := m.requestOutcomes()["q1"]; got != "timed out (no)" {
		t.Fatalf("an unanswered permission stays a no: %q", got)
	}
}
