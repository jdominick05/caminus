package main

import (
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// No route from the chat leads to the old tabbed screen: Tab, Shift+Tab and the number keys, typing or browsing.
func TestNoKeyLeavesTheChat(t *testing.T) {
	for _, browsing := range []bool{false, true} {
		m := chatFixture(t)
		if browsing {
			m.chat.compose.Blur()
		}
		for _, k := range []tea.KeyPressMsg{{Code: tea.KeyTab}, {Code: tea.KeyTab, Mod: tea.ModShift}, {Code: '1', Text: "1"}, {Code: '5', Text: "5"}} {
			next, _ := m.Update(k)
			m = next.(emberModel)
			if v := ansi.Strip(m.render()); m.tab != 6 || strings.Contains(v, "CAMINUS ENV") {
				t.Fatalf("browsing=%v %v left the chat:\n%s", browsing, k, v)
			}
		}
	}
}

// Editing a root command is a typed prompt in the composer's slot, not the old page.
func TestEditPromptStaysInTheChat(t *testing.T) {
	m := chatFixture(t)
	bridge := t.TempDir()
	writeReq(t, bridge, "r1", "apt-get install -y foo", 1)
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "a"), map[string]bool{}
	m.refreshLocal()
	m.chat.approvals = true // (the approvals overlay, Ctrl+A, decides and edits requests)
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'e', Text: "e"})
	m = next.(emberModel)
	v := ansi.Strip(m.render())
	if m.tab != 6 || m.mode != "edit" || !strings.Contains(v, "edit the root command") || !strings.Contains(v, "apt-get install -y foo") {
		t.Fatalf("the edit prompt is not in the chat (tab %d, mode %q):\n%s", m.tab, m.mode, v)
	}
	next, _ = m.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
	if m = next.(emberModel); m.mode != "normal" || m.tab != 6 {
		t.Fatal("Esc must cancel the prompt and stay in the chat")
	}
}
