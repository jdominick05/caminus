// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// Tests for the 0.3.0 release test's small fixes.

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
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

func TestThePickerRestylesWithoutRedrawingTheHistory(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	next, _ := m.chatSlash("/theme")
	m = next.(emberModel)
	m.chat.content = "SENTINEL"
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	if !strings.Contains(m.chat.content, "SENTINEL") || !m.chat.themes.dirty {
		t.Fatal("an arrow in the picker restyles only; the history waits for the close")
	}
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	if strings.Contains(next.(emberModel).chat.content, "SENTINEL") {
		t.Fatal("closing the picker redraws the history")
	}
}
