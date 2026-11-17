// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the 2026-11-16 audit's third group: layout at the release widths, and the copy.

import (
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func TestTheThemeScreenFitsBelow160Columns(t *testing.T) {
	for _, sz := range [][2]int{{160, 48}, {100, 30}, {80, 24}, {60, 20}} {
		m := chatFixture(t)
		m.width, m.height = sz[0], sz[1]
		next, _ := m.chatSlash("/theme")
		m = next.(emberModel)
		m.syncChat()
		v := m.render()
		if h := lipgloss.Height(v); h > sz[1] {
			t.Fatalf("%v: %d rows", sz, h)
		}
		for _, l := range strings.Split(ansi.Strip(v), "\n") {
			if ansi.StringWidth(l) > sz[0] {
				t.Fatalf("%v: a line wider than the screen: %q", sz, l)
			}
		}
		if p := ansi.Strip(v); !strings.Contains(p, "↑↓ preview") || !strings.Contains(p, "⇧Tab sessions") {
			t.Fatalf("%v: the hint and the status line stay:\n%s", sz, p)
		}
	}
}

func TestRecordsAndCardsMarkACutAndKeepTheState(t *testing.T) {
	if got := ansi.Strip(emberClip("Refactor the configuration loader so that every option", 30)); !strings.HasSuffix(got, "…") {
		t.Fatalf("a cut ends in …: %q", got)
	}
	for _, w := range []int{80, 100, 160} {
		line := ansi.Strip(recordState("⇡", "github: open a pull request on example/inventory: labfork:fix-docstrings-and-the-tests -> main",
			"no: open it as a draft PR", "", w))
		if !strings.HasSuffix(line, "· no: open it as a draft PR") || ansi.StringWidth(line) > w {
			t.Fatalf("%d: the state stays whole and the line fits: %q", w, line)
		}
	}
}

func TestALongNoticeIsReadWholeAboveTheComposer(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 80, 24
	m.msg = "Unknown command. /help lists commands; //text sends a message; your draft is retained"
	if v := ansi.Strip(m.chatBottom()); !strings.Contains(v, "your draft is") || !strings.Contains(v, "retained") {
		t.Fatalf("the whole notice shows:\n%s", v)
	}
	m.msgOf, m.msgAt = m.msg, time.Now().Add(-20*time.Second)
	if strings.Contains(ansi.Strip(m.chatBottom()), "retained") {
		t.Fatal("it goes when it expires")
	}
	m2, bridge := promptFixture(t)
	writeReq(t, bridge, "57024b55b7d04e6a", "rm -rf /tmp/x", float64(time.Now().Unix()))
	m2.refreshLocal()
	m2.sel = 0
	next, _ := m2.reqKeys("d")
	if msg := next.(emberModel).msg; msg != "root request: no" {
		t.Fatalf("a flash in words, not a request id: %q", msg)
	}
}

func TestHintsWrapAndTheMenuUsesTheRoomItHas(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 80, 24
	m.chat.goals = true
	if v := ansi.Strip(m.render()); !strings.Contains(v, "n new goal") || !strings.Contains(v, "v the whole plan") {
		t.Fatalf("the goals hint wraps, every key kept:\n%s", v)
	}
	m.chat.goals = false
	m.chat.compose.SetValue("/")
	if rows := strings.Count(ansi.Strip(m.slashMenu(80)), "│ /"); rows < 4 {
		t.Fatalf("the / menu shows more than 3 rows at 80x24:\n%s", ansi.Strip(m.slashMenu(80)))
	}
	m.chat.compose.SetValue("")
	m.resident.Goals = []residentGoal{{ID: "g1abcdef", State: "working", Prompt: "Check /srv/projects/very/deeply/nested/application/configuration/loader/ConfigurationLoaderImplementationFactoryBean.java now"}}
	m.chat.target = "g1abcdef"
	if v := strings.ReplaceAll(ansi.Strip(m.chatHead(80)), "\n", ""); !strings.Contains(strings.ReplaceAll(v, " ", ""), "FactoryBean.java") {
		t.Fatalf("a long path wraps in the header; the file name shows:\n%s", ansi.Strip(m.chatHead(80)))
	}
}

func TestEscClosesTheMenuShiftTabClosesTheSwitcherHomeGoesToTheTop(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.chat.compose.SetValue("/th")
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.slashMenuOpen() || m.chat.compose.Value() != "/th" {
		t.Fatal("Esc closes the / menu and the text stays")
	}
	m.chat.compose.SetValue("")
	m.chat.picker = true
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift})
	m = next.(emberModel)
	if m.chat.picker {
		t.Fatal("Shift+Tab closes the switcher it opened")
	}
	var evs []residentEvent
	for i := 0; i < 60; i++ {
		evs = append(evs, residentEvent{Seq: i + 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "line"}})
	}
	m.residentEvents = evs
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	m.chat.transcript.GotoBottom()
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyHome})
	if top := next.(emberModel); !top.chat.transcript.AtTop() {
		t.Fatal("Home with an empty draft goes to the top of the history")
	}
}

func TestANudgeIsNotLabelledSteerAndOneStepIsOneStep(t *testing.T) {
	c := &residentChat{}
	nudge := ansi.Strip(c.eventBlock(residentEvent{Seq: 1, Goal: "original", Kind: "user_message", Data: map[string]any{"text": "line one\nline two", "kind": "nudge"}}))
	steer := ansi.Strip(c.eventBlock(residentEvent{Seq: 2, Goal: "original", Kind: "user_message", Data: map[string]any{"text": "focus", "kind": "steer"}}))
	if strings.Contains(nudge, "steer") || !strings.Contains(steer, "steer") {
		t.Fatalf("only a steer reads steer:\n%s\n%s", nudge, steer)
	}
	if !strings.Contains(nudge, "line one  \n  line two") {
		t.Fatalf("your line breaks stay:\n%s", nudge)
	}
	if got := plural(1, "step", "steps"); got != "1 step" {
		t.Fatal(got)
	}
}
