package main

import (
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
