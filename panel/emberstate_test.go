// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the 2026-11-16 audit's second group: what showed the wrong state.

import (
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func TestAParkedGoalAndAStoppedLoopShowNoSpinner(t *testing.T) {
	m := chatFixture(t)
	m.statusAt = time.Now()
	m.resident.DaemonAlive = true
	m.resident.Goals = []residentGoal{{ID: "g1abcdef", State: "working", Prompt: "p"}}
	m.chat.target = "g1abcdef"
	wake := float64(time.Now().Add(10 * time.Minute).Unix())
	m.residentEvents = []residentEvent{{Seq: 1, Goal: "g1abcdef", Kind: "scheduled", Data: map[string]any{"wake_at": wake, "note": "check CI"}}}
	if m.working() != nil || !strings.Contains(ansi.Strip(m.activityLine(120)), "parked until") {
		t.Fatalf("a parked goal is not working: %q", ansi.Strip(m.activityLine(120)))
	}
	if now, _ := m.nowState(&m.resident.Goals[0]); !strings.HasPrefix(now, "parked until") {
		t.Fatalf("the now line says parked: %q", now)
	}
	m.residentEvents = nil
	m.resident.DaemonAlive = false
	if m.working() != nil || m.activityLine(120) != "" {
		t.Fatal("no spinner under a stopped header")
	}
	if now, _ := m.nowState(&m.resident.Goals[0]); now != "stopped · /start" {
		t.Fatalf("the now line says stopped, not between steps: %q", now)
	}
}

func TestARunningLineNeverStaysInTheHistory(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 140, 40
	call := map[string]any{"id": "c1", "type": "function", "function": map[string]any{"name": "terminal", "arguments": `{"command":"ls work"}`}}
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "", "tool_calls": []any{call}}},
		{Seq: 2, Goal: "original", Kind: "paused", Data: map[string]any{}},
		{Seq: 3, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "Resuming."}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	if v := ansi.Strip(m.chat.transcript.View()); strings.Contains(v, "running: ls work") {
		t.Fatalf("a call that never ran left a running line:\n%s", v)
	}
}

func TestCompactCardsKeepTheOutcomeAndWhatWasLeftOpen(t *testing.T) {
	long := `python3 -c "import sys; print(\"checking stock levels for every item in the inventory list now\"); sys.exit(3)" && echo done-with-everything-here`
	v := ansi.Strip(actionCard("terminal", map[string]any{"command": long}, map[string]any{"exit_code": 3.0, "output": "checking\n"}, 100, "", ""))
	if !strings.Contains(strings.Split(v, "\n")[0], "exit 3") {
		t.Fatalf("a long command is cut, never its exit code: %q", v)
	}
	c := &residentChat{}
	e := residentEvent{Seq: 1, Goal: "g", Kind: "tool_result", Data: map[string]any{"name": "finish", "call_id": "f",
		"result": map[string]any{"judge": "met", "open": "data/stock.csv was cut to two rows and was not restored."}}}
	if v := ansi.Strip(c.actionBlock(e, 120)); !strings.Contains(v, "left open: data/stock.csv") {
		t.Fatalf("what was left open shows in compact: %q", v)
	}
}

func TestTodoBlockersShow(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.resident.Goals = []residentGoal{{ID: "g1abcdef", State: "working", Prompt: "p"}}
	m.chat.target = "g1abcdef"
	m.residentEvents = []residentEvent{{Seq: 1, Goal: "g1abcdef", Kind: "todo", Data: map[string]any{
		"items": []any{map[string]any{"text": "run the tests", "state": "in_progress"}}, "blockers": []any{"pytest output is long"}}}}
	if v := ansi.Strip(m.render()); !strings.Contains(v, "pytest output is long") {
		t.Fatalf("the blocker shows:\n%s", v)
	}
}

func TestGoalsOverlayMovesInRunOrderAndSaysWhatItDid(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 30
	// newest first, as status lists them; the queue says B runs before C
	m.resident.Goals = []residentGoal{{ID: "cccccccc", State: "queued", Prompt: "C", Queue: 2}, {ID: "bbbbbbbb", State: "queued", Prompt: "B", Queue: 1},
		{ID: "aaaaaaaa", State: "working", Prompt: "A"}}
	order := m.goalOrder()
	if m.resident.Goals[order[1]].ID != "bbbbbbbb" || m.resident.Goals[order[2]].ID != "cccccccc" {
		t.Fatalf("queued goals in run order: %v", order)
	}
	m.chat.goals, m.chat.goalsSel = true, 1 // B
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: tea.KeyUp, Mod: tea.ModShift})
	if cmd != nil || next.(emberModel).msg != "it runs next already" {
		t.Fatalf("B runs next: nothing to move, and it says so: %q", next.(emberModel).msg)
	}
	if v := ansi.Strip(next.(emberModel).render()); !strings.Contains(v, "it runs next already") {
		t.Fatalf("the message shows in the overlay, not under it:\n%s", v)
	}
	m.chat.goalsSel = 2 // C
	if _, cmd = m.chatKey(tea.KeyPressMsg{Code: tea.KeyUp, Mod: tea.ModShift}); cmd == nil {
		t.Fatal("C moves up")
	}
}

func TestRewindWithNothingToUndoAndTheRecordsLeft(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 30
	m.chat.rewind = rewindState{open: true, goal: "original"}
	next, _ := m.Update(rewindListMsg{goal: "original"})
	if v := ansi.Strip(next.(emberModel).render()); !strings.Contains(v, "nothing to rewind yet") {
		t.Fatalf("an empty list says so; it is not still loading:\n%s", v)
	}
	m.chat.rewind = rewindState{}
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "plan_proposed", Data: map[string]any{"plan": "1. add a test\n2. fix it"}},
		{Seq: 2, Goal: "original", Kind: "plan_rejected", Data: map[string]any{"note": "name it farewell.txt instead"}},
		{Seq: 3, Goal: "original", Kind: "plan_approved", Data: map[string]any{}},
		{Seq: 4, Goal: "original", Kind: "rewound", Data: map[string]any{"to": 4.0, "restored": []any{"work/hello.txt"}, "removed": []any{"work/farewell.txt"}}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	for _, want := range []string{"plan proposed: 1. add a test", "plan sent back (you): name it farewell.txt instead", "plan approved (you)",
		"rewound to step 4: restored work/hello.txt · removed work/farewell.txt"} {
		if !strings.Contains(v, want) {
			t.Fatalf("a record for each: %q missing:\n%s", want, v)
		}
	}
}
