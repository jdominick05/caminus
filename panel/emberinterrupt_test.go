// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Esc interrupts the model call in flight: from its first moment (not only once text streams), and afterwards the
// activity line says interrupted instead of counting on as if the agent still worked.

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func interruptFixture(t *testing.T) emberModel {
	m := chatFixture(t)
	m.width, m.height = 140, 40
	m.statusAt = time.Now()
	m.resident.DaemonAlive = true
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: "a goal"}}
	m.chat.target = "original"
	return m
}

var clockRe = regexp.MustCompile(`\(\d+s\)|\(\d+m \d+s\)`)

func TestEscInterruptsACallBeforeItsFirstToken(t *testing.T) {
	m := interruptFixture(t)
	m.chat.live = residentLive{} // nothing streamed yet: a long prompt read, a paid model thinking
	mm, cmd := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = mm.(emberModel)
	if cmd == nil || m.msg != "interrupting the turn…" {
		t.Fatalf("Esc while the goal works sends the interrupt even before the first token: %q", m.msg)
	}
	// "new goal" mode, a call out: Esc goes back to the goal you were talking to (a second Esc interrupts)
	m = interruptFixture(t)
	m.chat.forceNew, m.chat.prevTarget = true, "original"
	m.chat.target = ""
	m.chat.live = residentLive{turn: "t1", active: true}
	mm, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = mm.(emberModel)
	if m.chat.target != "original" || m.msg == "interrupting the turn…" {
		t.Fatalf("Esc leaves new-goal mode first: target %q msg %q", m.chat.target, m.msg)
	}
}

func TestAnInterruptedTurnStopsTheClockAndSaysSo(t *testing.T) {
	m := interruptFixture(t)
	now := float64(time.Now().Unix())
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "user_message", Ts: now - 30, Data: map[string]any{"text": "a goal"}},
		{Seq: 2, Goal: "original", Kind: "turn_interrupted", Ts: now - 2, Data: map[string]any{"hold_s": 120.0, "turn": "t1"}},
	}
	m.resident.TokPerS, m.resident.ContextUsed = 55, 20000
	if foot := ansi.Strip(m.chatFoot(140)); strings.Contains(foot, "tok/s") {
		t.Fatalf("the last reply's speed is not shown as current after an Esc: %q", foot)
	}
	line := ansi.Strip(m.activityLine(140))
	if m.working() != nil || !strings.Contains(line, "turn interrupted") || clockRe.MatchString(line) ||
		strings.Contains(line, "esc interrupts") {
		t.Fatalf("an interrupted goal shows no spinner and no clock: %q", line)
	}
	if !strings.Contains(line, "carries on by itself at "+time.Unix(int64(now-2+120), 0).Format("15:04")) {
		t.Fatalf("it says when the agent carries on by itself: %q", line)
	}
	if now, _ := m.nowState(&m.resident.Goals[0]); !strings.HasPrefix(now, "interrupted") {
		t.Fatalf("the now line says interrupted, not thinking: %q", now)
	}
	if m.interruptibleGoal() != "" {
		t.Fatal("nothing to interrupt any more: Esc sends nothing")
	}
	// your message lifts it: the agent works again
	m.residentEvents = append(m.residentEvents, residentEvent{Seq: 3, Goal: "original", Kind: "user_message", Ts: now, Data: map[string]any{"text": "shorter"}})
	if m.working() == nil || strings.Contains(ansi.Strip(m.activityLine(140)), "turn interrupted") {
		t.Fatal("after your message the goal works again")
	}
	if foot := ansi.Strip(m.chatFoot(140)); !strings.Contains(foot, "55 tok/s") {
		t.Fatalf("working again: the speed shows again: %q", foot)
	}
	// so does a new call going out after the hold (its live turn is not the dropped one)
	m.residentEvents = m.residentEvents[:2]
	m.chat.live = residentLive{turn: "t2", active: true}
	if m.working() == nil {
		t.Fatal("a new call is out: the goal works again")
	}
	// and a hold that ran out (the runtime carried on by itself)
	m.chat.live = residentLive{}
	m.residentEvents[1].Ts = now - 200
	if m.working() == nil {
		t.Fatal("past the hold the goal works again")
	}
}

func TestTheInterruptEventDropsTheLiveTailAndItsLateLines(t *testing.T) {
	m := interruptFixture(t)
	m.chat.live = residentLive{turn: "t1", text: "DROPPED-REPLY", active: true, wrapText: &wrapCache{}, wrapThink: &wrapCache{}}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{
		{Seq: 2, Goal: "original", Kind: "turn_interrupted", Ts: float64(time.Now().Unix()), Data: map[string]any{"hold_s": 120.0, "turn": "t1"}}}})
	if m.chat.live.active || strings.Contains(m.chat.live.text, "DROPPED") {
		t.Fatal("the dropped turn's tail goes")
	}
	next, _ := m.Update(residentLiveMsg{agent: m.agentName, epoch: m.residentEpoch, turn: "t1", text: " late piece"})
	m = next.(emberModel)
	if m.chat.live.active {
		t.Fatal("a late line of the dropped turn does not bring its tail back")
	}
	next, _ = m.Update(residentLiveMsg{agent: m.agentName, epoch: m.residentEpoch, turn: "t2", reset: true})
	m = next.(emberModel)
	if !m.chat.live.active || m.chat.live.turn != "t2" {
		t.Fatal("the next turn streams as usual")
	}
}

func TestTheInterruptSaysWhenThereIsNothingToInterrupt(t *testing.T) {
	saved := interruptSettle
	interruptSettle = 0
	defer func() { interruptSettle = saved }()
	m := interruptFixture(t)
	dir := t.TempDir()
	for _, c := range []struct{ out, want string }{
		{`{"goal":"original","queued":false,"generating":false}`, "nothing to interrupt"},
		{`{"goal":"original","queued":true,"generating":true}`, "turn interrupted"},
	} {
		script := filepath.Join(dir, "fake-caminus")
		if err := os.WriteFile(script, []byte("#!/bin/sh\necho '"+c.out+"'\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		m.cli = []string{script}
		got, _ := interruptTurn(m, "original")().(residentActionMsg)
		if !strings.Contains(got.text, c.want) {
			t.Fatalf("%s: %q", c.out, got.text)
		}
	}
}
