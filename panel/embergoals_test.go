package main

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func goalsFixture(t *testing.T) emberModel {
	m := chatFixture(t)
	m.width, m.height = 100, 24
	m.resident.Goals = []residentGoal{
		{ID: "aaa111", State: "completed", Prompt: "an old goal", Summary: "did it", Steps: 9},
		{ID: "bbb222", State: "working", Prompt: "make the parser tests pass", Steps: 14},
	}
	return m
}

func key(m emberModel, k tea.KeyPressMsg) (emberModel, tea.Cmd) {
	next, cmd := m.chatKey(k)
	return next.(emberModel), cmd
}

// Ctrl+G and /goals open the overlay newest first; q does not quit inside; Esc closes; Enter targets the goal.
func TestGoalsOverlayOpensListsAndTargets(t *testing.T) {
	m := goalsFixture(t)
	m, _ = key(m, tea.KeyPressMsg{Code: 'g', Mod: tea.ModCtrl})
	frame := ansi.Strip(m.render())
	if !m.chat.goals || !strings.Contains(frame, "goals  1 open of 2") {
		t.Fatalf("Ctrl+G did not open the overlay:\n%s", frame)
	}
	if strings.Index(frame, "bbb222") > strings.Index(frame, "aaa111") {
		t.Fatal("the newest goal must come first")
	}
	for _, row := range strings.Split(frame, "\n") {
		if len([]rune(row)) > m.width {
			t.Fatalf("a row is wider than the terminal: %q", row)
		}
	}
	if _, cmd := key(m, tea.KeyPressMsg{Code: 'q', Text: "q"}); cmd != nil {
		t.Fatal("q must do nothing inside an overlay")
	}
	m, _ = key(m, tea.KeyPressMsg{Code: 'j', Text: "j"})
	m, _ = key(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if m.chat.goals || m.chat.newGoal || m.chat.target != "aaa111" {
		t.Fatalf("Enter must close and target the selected goal: open=%v target=%q", m.chat.goals, m.chat.target)
	}
	next, _ := m.chatSlash("/goals")
	m = next.(emberModel)
	if !m.chat.goals {
		t.Fatal("/goals must open the overlay")
	}
	m, _ = key(m, tea.KeyPressMsg{Code: tea.KeyEscape})
	if m.chat.goals {
		t.Fatal("Esc must close it")
	}
}

// One stray x must not cancel the agent's work: the first press arms and names the goal, another key disarms.
func TestGoalsCancelNeedsASecondPress(t *testing.T) {
	m := goalsFixture(t)
	m, _ = key(m, tea.KeyPressMsg{Code: 'g', Mod: tea.ModCtrl})
	m, cmd := key(m, tea.KeyPressMsg{Code: 'x', Text: "x"})
	if cmd != nil || m.chat.armed != "cancel:bbb222" || !strings.Contains(m.msg, "press x again") {
		t.Fatalf("the first x must only arm: armed=%q msg=%q", m.chat.armed, m.msg)
	}
	m, _ = key(m, tea.KeyPressMsg{Code: 'k', Text: "k"})
	if m.chat.armed != "" {
		t.Fatal("another key must disarm")
	}
	m, _ = key(m, tea.KeyPressMsg{Code: 'x', Text: "x"})
	if _, cmd = key(m, tea.KeyPressMsg{Code: 'x', Text: "x"}); cmd == nil {
		t.Fatal("a second x must send the cancel")
	}
}
