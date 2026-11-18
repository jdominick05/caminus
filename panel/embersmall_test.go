// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the 0.3.0 release test's small fixes.

import (
	"fmt"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func TestArrowsAndEnterMoveThroughThePathSuggestions(t *testing.T) {
	m := pathFixture(t)
	m.width, m.height = 100, 30
	m.chat.compose.SetValue("look at @")
	m.chat.compose.Focus()
	if m.pathMenu(98) == "" {
		t.Fatal("the suggestions show for @")
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	if m.chat.compose.Value() != "look at @" {
		t.Fatalf("Down moves the highlight, the draft stays: %q", m.chat.compose.Value())
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if m.chat.compose.Value() != "look at @notes.md" {
		t.Fatalf("Enter takes the highlighted path (the second: notes.md): %q", m.chat.compose.Value())
	}
}

func TestShellCommandsGoIntoTheHistory(t *testing.T) {
	m := chatFixture(t)
	m.setShell(true)
	m.chat.compose.SetValue("ls work")
	m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	if got := loadTyped(m.agentName); len(got) == 0 || got[0] != "!ls work" {
		t.Fatalf("a ! command is in the history as typed: %q", got)
	}
}

func TestAClickOnAFoldedCardOpensIt(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.chat.cards, m.chat.expanded = false, false
	m.chat.transcript.SetContent("a reply\n  " + g.collapsed + " terminal make test …\nmore")
	if !m.foldedAt(1) || m.foldedAt(0) {
		t.Fatal("the folded line is found, a plain one is not")
	}
	m.chat.sel = chatSelection{dragging: true, a: [2]int{1, 3}, b: [2]int{1, 3}}
	next, _, _ := m.selectMouse(tea.MouseReleaseMsg{Button: tea.MouseLeft})
	if !next.chat.expanded {
		t.Fatal("a click on it opens everything, as Ctrl+O does")
	}
}

// The picker previews the hovered theme without touching the history; Enter keeps the hovered theme and Esc puts the
// old one back, and either close draws the history once, its markdown in the theme now in force. The markdown renderer
// survives an arrow: dropped there, every frame drawn while the picker was open redrew the whole history (117 ms a
// keypress at 2000 events, BenchmarkThemePicker).
func TestThePickerRestylesWithoutRedrawingTheHistory(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	defer func() { _ = themeLoad("caminus") }()
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 2, Goal: "original",
		Kind: "assistant", Data: map[string]any{"content": "A reply in **markdown**."}}}})
	textSGR := func() string { // the text role as the markdown draws it
		r, g, b, _ := hexRGB(themeRole("text"))
		return fmt.Sprintf("38;2;%d;%d;%d", r, g, b)
	}
	hover := func(m emberModel) emberModel { // down to the next theme with another text colour
		was := themeRole("text")
		for range m.chat.themes.names {
			next, _ := m.Update(tea.KeyPressMsg{Code: tea.KeyDown})
			m = next.(emberModel)
			if themeRole("text") != was {
				return m
			}
		}
		t.Fatal("no theme with another text colour")
		return m
	}
	for _, key := range []rune{tea.KeyEnter, tea.KeyEscape} {
		before := themeRole("text")
		next, _ := m.chatSlash("/theme")
		m = next.(emberModel)
		md := m.chat.markdown
		if md == nil {
			t.Fatal("the fixture's history has a markdown renderer")
		}
		m.chat.content = "SENTINEL"
		m = hover(m)
		hovered := m.chat.themes.names[m.chat.themes.sel]
		frame := strings.ToLower(ansi.Strip(m.View().Content)) // (the box writes its title in lower case)
		if !strings.Contains(m.chat.content, "SENTINEL") || m.chat.markdown != md || !m.chat.themes.dirty ||
			!strings.Contains(frame, "themes  "+hovered) {
			t.Fatalf("an arrow previews %s and restyles only; the history and its renderer wait for the close", hovered)
		}
		want := themeRole("text") // Enter keeps the hovered theme
		if key == tea.KeyEscape {
			want = before // Esc puts back what was showing
		}
		next, _ = m.Update(tea.KeyPressMsg{Code: key})
		m = next.(emberModel)
		switch { // (the saved name and Esc's exact colours: TestThemeScreenPreviewsKeepsAndRestores)
		case m.chat.themes.open || themeRole("text") != want:
			t.Fatalf("key %d: the picker closes with text %s in force, got %s", key, want, themeRole("text"))
		case strings.Contains(m.chat.content, "SENTINEL") || m.chat.markdown == md || !strings.Contains(m.chat.content, textSGR()):
			t.Fatalf("key %d: closing draws the history again, its markdown in the theme in force", key)
		}
	}
}

func TestARecalledHostCommandRunsInYourShell(t *testing.T) {
	m := chatFixture(t)
	m.chat.compose.SetValue("!!echo hi")
	next, cmd := m.submitCompose()
	if cmd == nil || next.(emberModel).msg != "running in your shell…" {
		t.Fatalf("!!cmd from the history runs in your shell, not the agent's: %q", next.(emberModel).msg)
	}
}
