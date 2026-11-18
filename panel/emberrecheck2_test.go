// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the second recheck of the audit fixes.

import (
	"strings"
	"testing"
	"time"

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
	if v := ansi.Strip(actionCard("machine", nil, map[string]any{"gpus": []any{map[string]any{"vram_used_mib": 100.0, "vram_total_mib": 512.0, "busy_percent": 0.0}}}, 160, "", "", false, "")); !strings.Contains(v, "100/512 MiB") {
		t.Fatalf("a small iGPU in MiB: %q", v)
	}
	if v := ansi.Strip(actionCard("read", map[string]any{"path": "x"}, map[string]any{"error": "[Errno 2] No such file: '/srv/agent/homes/r/work/x'"}, 160, "/srv/agent/homes/r", "", false, "")); strings.Contains(v, "/srv/agent") {
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

func TestAClearedDraftInHistoryReadsYourDraft(t *testing.T) {
	m := chatFixture(t)
	m.chat.target, m.chat.newGoal = "original", false
	m.chat.compose.SetValue("never sent")
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})
	m = next.(emberModel)
	m.chat.clearedDraft = "" // (already brought back once)
	m, _ = m.recallStep("up", true)
	if m.chat.compose.Value() != "never sent" || !strings.HasPrefix(m.msg, "your draft 1 of") {
		t.Fatalf("a never-sent draft is labelled as one: %q / %q", m.chat.compose.Value(), m.msg)
	}
}

func TestThirdRecheck(t *testing.T) {
	g := residentGoal{ID: "g1", State: "working", Parked: float64(time.Now().Add(time.Hour).Unix())}
	if g.shownState() != "parked" {
		t.Fatal("a parked goal reads parked")
	}
	c := &residentChat{renderWidth: 120}
	v := ansi.Strip(c.eventBlock(residentEvent{Seq: 1, Goal: "g1", Kind: "turn_retried", Data: map[string]any{"step": 4.0}}))
	if !strings.Contains(v, "agent restarted · step 4 retried") {
		t.Fatalf("a restart's retry shows: %q", v)
	}
	m := chatFixture(t)
	m.chat.clearedDraft = "old draft"
	m.chat.target, m.chat.newGoal = "original", false
	m.chat.compose.SetValue("a new message")
	m.submitCompose()
	if next, _ := m.submitCompose(); next.(emberModel).chat.clearedDraft != "" {
		t.Fatal("a send drops the cleared draft")
	}
}

func TestTheGoalsOverlayStopsFollowingAMovedGoalOnAnyOtherKey(t *testing.T) {
	m := chatFixture(t)
	m.chat.goals, m.chat.goalsFollow = true, "g1"
	next, _ := m.goalsKey(tea.KeyPressMsg{Code: tea.KeyDown})
	if next.(emberModel).chat.goalsFollow != "" {
		t.Fatal("navigating stops the follow: a refresh pulled the ▸ back, and x or p hit the moved goal")
	}
}
