// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the second recheck of the audit fixes.

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func TestUpBringsBackAClearedDraftBeforeAnySentMessage(t *testing.T) {
	m := chatFixture(t)
	m.chat.target, m.chat.newGoal = "original", false
	m.chat.compose.SetValue("a draft I keep")
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	m = next.(emberModel)
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: tea.KeyUp})
	m = next.(emberModel)
	if m.chat.compose.Value() != "a draft I keep" || cmd != nil {
		t.Fatalf("the draft comes back first, and nothing is pulled back from the agent: %q cmd=%v", m.chat.compose.Value(), cmd != nil)
	}
}

func TestAPulledBackMessageComesBackAsTyped(t *testing.T) {
	m := chatFixture(t)
	saveTyped(m.agentName, "//th then look at @work/x.py")
	if got := m.typedFormOf("/th then look at work/x.py"); got != "//th then look at @work/x.py" {
		t.Fatalf("the typed form: %q", got)
	}
	if got := m.typedFormOf("/etc is full"); got != "//etc is full" {
		t.Fatalf("an unknown one keeps its leading / as a message: %q", got)
	}
}

func TestAGoalPickedByHandStaysPickedWhenOver(t *testing.T) {
	m := chatFixture(t)
	m.resident.Goals = []residentGoal{{ID: "bbbbbbbb", State: "cancelled", Prompt: "B"}, {ID: "aaaaaaaa", State: "completed", Prompt: "A"}}
	m.goalSel, m.chat.pickedGoal = 0, "bbbbbbbb"
	m.receiveResident(residentMsg{agent: m.agentName, epoch: m.residentEpoch, status: m.resident})
	if g := m.selectedResidentGoal(); g == nil || g.ID != "bbbbbbbb" {
		t.Fatal("a refresh does not move off a goal you picked")
	}
}

func TestTheRewindRecordWrapsAndTheConfirmLeadsWithTheKey(t *testing.T) {
	c := &residentChat{}
	e := residentEvent{Seq: 1, Goal: "g", Kind: "rewound", Data: map[string]any{"to": 4.0, "restored": []any{"work/hello.txt"}, "removed": []any{"work/farewell.txt"}}}
	v := ansi.Strip(c.eventBlock(e))
	if !strings.Contains(strings.ReplaceAll(v, "\n   ", " "), "the goal is paused (/resume)") {
		t.Fatalf("the record's end can be read:\n%s", v)
	}
}

func TestCopyFixes(t *testing.T) {
	if v := ansi.Strip(actionCard("machine", nil, map[string]any{"gpus": []any{map[string]any{"vram_used_mib": 100.0, "vram_total_mib": 512.0, "busy_percent": 0.0}}}, 160, "", "")); !strings.Contains(v, "100/512 MiB") {
		t.Fatalf("a small iGPU in MiB: %q", v)
	}
	if v := ansi.Strip(actionCard("read", map[string]any{"path": "x"}, map[string]any{"error": "[Errno 2] No such file: '/srv/agent/homes/r/work/x'"}, 160, "/srv/agent/homes/r", "")); strings.Contains(v, "/srv/agent") {
		t.Fatalf("errors read short paths: %q", v)
	}
	m := chatFixture(t)
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g", Kind: "question", Data: map[string]any{"id": "q1", "text": "Which?", "recommended": "3.14"}},
		{Seq: 2, Goal: "g", Kind: "question_unanswered", Data: map[string]any{"id": "q1", "recommended": "3.14"}},
	}
	if got := m.requestOutcomes()["q1"]; got != "3.14 (recommended, timed out)" {
		t.Fatalf("a timeout names the option: %q", got)
	}
}
