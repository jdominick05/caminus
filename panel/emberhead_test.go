// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// #316: the goal at the top of the chat expands and collapses — by Alt+G and by a click on the header, at 80, 100
// and 160 columns, with the "… (N more lines, Alt+G expands)" hint only while it is cut, the whole goal drawn as
// Markdown (lists, code), and select-to-copy taking the whole goal rather than the lines the header had room for.
// The later rounds added the expansion as a view of the goal it was opened on (Ctrl+N or a switch to
// another goal hides it, and the next Esc is not eaten), the hint cut to the terminal's width, the click that clears
// a copy not also toggling the goal under it, the scrolled row clamped to what the goal on the screen has, headMD
// bounded, an expansion opened on a goal whose prompt is blank still keyed to that goal, and the expansion's own
// renderer kept per width instead of rebuilt on every frame.

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// longGoal is about the 2,000 characters the issue saw, so it is cut even at 160 columns, with Markdown the expanded
// form has to draw (a list and a fenced block) and a last line only the expansion shows.
const longGoal = "Review this PR properly.\n\n" +
	"- the header must expand and collapse\n" +
	"- the collapsed form must say it is cut\n" +
	"- the full goal must render as Markdown\n" +
	"- select-to-copy must copy the whole goal\n\n" +
	"```go\nfunc chatHead(w int) string {\n\treturn m.headExpanded(w)\n}\n```\n\n" +
	"Keep the three-line summary readable while the header is closed, and draw the open goal with the same renderer " +
	"the replies use, so a list, a table and the fenced block above come out as they would in a reply rather than as " +
	"the plain source a screen cut in three lines shows. The transcript under it keeps its own scrolling, and the " +
	"composer keeps typing while the goal is open. A goal this long is the normal case here: PR reviews and long " +
	"design briefs are sent as one message, and none of that text should be unreachable from the chat.\n\n" +
	"The last line of the goal: Esc collapses it again.\n"

func longGoalChat(t *testing.T, w, h int) emberModel {
	t.Helper()
	m := chatFixture(t)
	m.width, m.height = w, h
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: longGoal}}
	m.goalSel = 0
	m.chat.target = "original"
	return m
}

func rows(s string) int { return strings.Count(s, "\n") + 1 }

// TestTheLongGoalExpandsByKeyAndClickAtEveryWidth covers the toggle by Alt+G and by a click on the header, the hint
// while cut, and Esc collapsing it again, at the three widths the issue names.
func TestTheLongGoalExpandsByKeyAndClickAtEveryWidth(t *testing.T) {
	for _, w := range []int{80, 100, 160} {
		t.Run(fmt.Sprint(w), func(t *testing.T) {
			m := longGoalChat(t, w, 30)
			collapsed := ansi.Strip(m.chatHead(w))
			if !m.headUnderTitle(w) {
				t.Fatalf("fixture: the goal is cut at %d columns:\n%s", w, collapsed)
			}
			if !strings.Contains(collapsed, "more line") || !strings.Contains(collapsed, "Alt+G expands") {
				t.Fatalf("the collapsed header says it is cut and names the key:\n%s", collapsed)
			}
			if strings.Contains(collapsed, "The last line of the goal") {
				t.Fatalf("fixture: the whole goal is in the collapsed header:\n%s", collapsed)
			}
			// Alt+G opens it: the whole Markdown, its list and its code drawn as such.
			next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
			m = next.(emberModel)
			if !m.chat.headOpen {
				t.Fatal("Alt+G did not open the goal")
			}
			open := ansi.Strip(m.chatHead(w))
			for _, want := range []string{"the header must expand", "select-to-copy must copy the whole goal",
				"func chatHead(w int) string", "Review this PR properly"} {
				if !strings.Contains(open, want) {
					t.Fatalf("the open goal lost %q at %d columns:\n%s", want, w, open)
				}
			}
			if strings.Contains(open, "Alt+G expands") {
				t.Fatalf("the hint goes while the goal is open:\n%s", open)
			}
			if !strings.Contains(open, "PgUp/PgDn scrolls") {
				t.Fatalf("a goal taller than half the screen says how to scroll:\n%s", open)
			}
			// PgDn reaches the rest: the last line is only in the scrolled part.
			for i := 0; i < 20; i++ {
				next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
				m = next.(emberModel)
			}
			if end := ansi.Strip(m.chatHead(w)); !strings.Contains(end, "The last line of the goal") {
				t.Fatalf("PgDn did not reach the last line of the goal at %d columns:\n%s", w, end)
			}
			next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgUp})
			m = next.(emberModel)
			if got := rows(m.chatHead(w)); got > max(12, m.height)/2+2 {
				t.Fatalf("the open goal takes %d rows: more than half the screen at %d", got, w)
			}
			// Esc collapses it, and the header is exactly what it was.
			next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
			m = next.(emberModel)
			if m.chat.headOpen {
				t.Fatal("Esc did not collapse the goal")
			}
			if back := ansi.Strip(m.chatHead(w)); back != collapsed {
				t.Fatalf("collapsed again, the header differs:\n%s\n--- was:\n%s", back, collapsed)
			}
			// A click on the header toggles it, at every width. The row is the header's own first line, not a
			// brand spelled out here: the session draws EMBER since #350, and this test is about the click.
			title := strings.TrimRight(strings.Split(collapsed, "\n")[0], " ")
			y := -1
			for i, row := range strings.Split(ansi.Strip(m.render()), "\n") {
				if strings.HasPrefix(strings.TrimRight(row, " "), title) {
					y = i
					break
				}
			}
			if y < 0 {
				t.Fatal("no header on the screen")
			}
			m, _ = click(m, 2, y)
			if !m.chat.headOpen {
				t.Fatal("a click on the header did not open the goal")
			}
			m, _ = click(m, 2, y)
			if m.chat.headOpen {
				t.Fatal("a second click did not close it")
			}
		})
	}
}

// TestTheCutLineCountsWhatTheHeaderHides: the number in the hint is the rows the expansion draws under the title
// that the collapsed header does not — the expansion's own unit, so the hint and its footer agree (#370 review P1:
// counting the flattened wrap said "no more lines" for a goal whose drawn rows were many).
func TestTheCutLineCountsWhatTheHeaderHides(t *testing.T) {
	m := longGoalChat(t, 100, 30)
	_, _, one := m.headParts()
	shown := len(headGoalRowsUnderTitle(one, 100))
	head := ansi.Strip(m.chatHead(100))
	cut := m.headExpansion(100).lines - shown
	if cut < 1 {
		t.Fatalf("fixture: the long goal's expansion draws no hidden rows (%d lines, %d shown)", m.headExpansion(100).lines, shown)
	}
	want := "… (" + plural(cut, "more line", "more lines") + ", Alt+G expands)"
	if !strings.Contains(head, want) {
		t.Fatalf("the hint counts the hidden drawn rows (%q):\n%s", want, head)
	}
}

// TestAShortGoalHasNoHintAndDoesNotExpand: a goal that fits is not offered an expansion that would do nothing.
func TestAShortGoalHasNoHintAndDoesNotExpand(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	if m.headUnderTitle(100) {
		t.Fatal("a goal of one short line is not cut")
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
	m = next.(emberModel)
	if m.chat.headOpen {
		t.Fatal("Alt+G opened a goal that already fits")
	}
	if strings.Contains(ansi.Strip(m.chatHead(100)), "Alt+G expands") {
		t.Fatal("a goal that fits does not say it is cut")
	}
}

// listGoal is twelve short list items: one flattened line, three wrapped rows at 100 columns, but the expansion
// draws a row per item (#370 review P1).
const listGoal = "- item 01 is short\n- item 02 is short\n- item 03 is short\n- item 04 is short\n" +
	"- item 05 is short\n- item 06 is short\n- item 07 is short\n- item 08 is short\n" +
	"- item 09 is short\n- item 10 is short\n- item 11 is short\n- item 12 is short\n"

func listGoalChat(t *testing.T, w, h int) emberModel {
	t.Helper()
	m := chatFixture(t)
	m.width, m.height = w, h
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: listGoal}}
	m.goalSel = 0
	m.chat.target = "original"
	return m
}

// TestAShortLineMarkdownGoalExpands: a goal whose flattened wrap fits the header's three rows but whose Markdown
// draws more is expandable — the hint says so, Alt+G opens it, and the hint counts the rows the expansion draws
// (#370 review P1: the gate asked the flattened text and refused the key for a whole class of goals).
func TestAShortLineMarkdownGoalExpands(t *testing.T) {
	for _, w := range []int{80, 100, 160} {
		t.Run(fmt.Sprint(w), func(t *testing.T) {
			m := listGoalChat(t, w, 30)
			if !m.headExpandable(w) {
				t.Fatalf("fixture: the list goal's expansion draws more rows than the header shows at %d columns", w)
			}
			collapsed := ansi.Strip(m.chatHead(w))
			if !strings.Contains(collapsed, "Alt+G expands") {
				t.Fatalf("a goal whose expansion has more rows says so:\n%s", collapsed)
			}
			shown := len(headGoalRowsUnderTitle(strings.Join(strings.Fields(m.headGoalText()), " "), w))
			cut := m.headExpansion(w).lines - shown
			if want := plural(cut, "more line", "more lines"); !strings.Contains(collapsed, want) {
				t.Fatalf("the hint counts the drawn rows the header hides (%q):\n%s", want, collapsed)
			}
			next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
			m = next.(emberModel)
			if !m.chat.headOpen {
				t.Fatal("Alt+G did not open a goal whose expansion draws more rows")
			}
			open := ansi.Strip(m.chatHead(w))
			if !strings.Contains(open, "item 01 is short") {
				t.Fatalf("the open list goal lost its first item at %d columns:\n%s", w, open)
			}
			if strings.Contains(open, "Alt+G expands") {
				t.Fatalf("the hint goes while the goal is open:\n%s", open)
			}
			// the expansion is capped at half the screen: the rows past it are reachable with PgDn
			if !strings.Contains(open, "item 12 is short") {
				if !strings.Contains(open, "PgUp/PgDn scrolls") {
					t.Fatalf("the open list goal shows neither its last item nor the scroll footer:\n%s", open)
				}
				for i := 0; i < 5 && !strings.Contains(ansi.Strip(m.chatHead(w)), "item 12 is short"); i++ {
					next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
					m = next.(emberModel)
				}
				if end := ansi.Strip(m.chatHead(w)); !strings.Contains(end, "item 12 is short") {
					t.Fatalf("PgDn did not reach the list goal's last item at %d columns:\n%s", w, end)
				}
			}
		})
	}
}

// TestCopyingAFewRowsOfTheOpenGoalIsNotTheWholeGoal: the collapsed header's rows are a lossy cut, so a copy that
// touches them takes the goal whole; the open header draws the goal's own lines, so a selection inside it that
// stops short copies what it spans — a one-row clip is one row, not the whole goal (#370 review P2).
func TestCopyingAFewRowsOfTheOpenGoalIsNotTheWholeGoal(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	_ = m.render()
	m.chat.sel = m.selectionAt(2, 1)
	m.chat.sel.b = [2]int{1, max(20, m.width) - 1} // one row, all its columns
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	got := m.selectedText()
	if strings.Contains(got, "The last line of the goal") || strings.Contains(got, "func chatHead") {
		t.Fatalf("a one-row clip of the open goal copied rows the selection does not span:\n%s", got)
	}
	if !strings.Contains(got, "Review this PR properly") {
		t.Fatalf("the one-row clip lost the row it spans:\n%s", got)
	}
	// the whole block still copies the goal whole, as the #316 affordance promises
	w := max(20, m.width)
	m.chat.sel = m.selectionAt(2, 1)
	m.chat.sel.b = [2]int{len(m.headExpansion(w).rows), w - 1}
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	if got := m.selectedText(); !strings.Contains(got, "The last line of the goal") {
		t.Fatalf("a selection over the whole open goal lost the goal's own last line:\n%s", got)
	}
}

// TestPgUpAndPgDnPageTheTranscriptUnderAGoalThatFits: an expanded goal with nowhere to scroll does not take the
// paging keys from the history under it.
func TestPgUpAndPgDnPageTheTranscriptUnderAGoalThatFits(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	// long enough to be cut in the collapsed header (4+ wrapped lines), short enough to render in fewer rows than
	// the expansion shows at once.
	goal := strings.Repeat("A short brief. ", 20)
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: goal}}
	m.goalSel, m.chat.target = 0, "original"
	if !m.headUnderTitle(100) {
		t.Fatalf("fixture: the goal is cut in the collapsed header")
	}
	// a history long enough to page through
	for i := 0; i < 40; i++ {
		m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{
			Seq: 100 + i, Goal: "original", Kind: "assistant",
			Data: map[string]any{"content": fmt.Sprintf("Observation %d stays in the transcript.", i)}}}})
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
	m = next.(emberModel)
	if !m.chat.headOpen {
		t.Fatal("Alt+G did not open the goal")
	}
	lines := lipgloss.Height(m.headExpandedBody(100))
	if lines > m.headRows() {
		t.Fatalf("fixture: the open goal fits (%d rows of %d)", lines, m.headRows())
	}
	if m.chat.headTop != 0 {
		t.Fatalf("fixture: the open goal starts at row %d", m.chat.headTop)
	}
	// PgUp over an open goal that cannot scroll pages the history instead of doing nothing.
	if !m.chat.transcript.AtBottom() {
		t.Fatalf("fixture: the transcript starts at the bottom")
	}
	beforeView := ansi.Strip(m.chat.transcript.View())
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgUp})
	m = next.(emberModel)
	if ansi.Strip(m.chat.transcript.View()) == beforeView {
		t.Fatalf("PgUp under a goal that fits left the transcript where it was (offset %d)", m.chat.transcript.YOffset())
	}
	if m.chat.headTop != 0 {
		t.Fatalf("a goal that fits scrolled to row %d", m.chat.headTop)
	}
	// PgDn walks back down.
	up := m.chat.transcript.YOffset()
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
	m = next.(emberModel)
	if m.chat.transcript.YOffset() <= up {
		t.Fatalf("PgDn under a goal that fits did not page the transcript down (offset %d, was %d)",
			m.chat.transcript.YOffset(), up)
	}
}

// TestCopyingTheHeaderTakesTheWholeGoal: a selection over the header copies the goal as written, not the three lines
// the screen shows.
func TestCopyingTheHeaderTakesTheWholeGoal(t *testing.T) {
	m := longGoalChat(t, 100, 30)
	_ = m.render() // (the frame the selection maps onto)
	m.chat.sel = m.selectionAt(2, 1)
	m.chat.sel.b = [2]int{1, 20}
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	got := m.selectedText()
	for _, want := range []string{"the header must expand and collapse", "select-to-copy must copy the whole goal",
		"func chatHead(w int) string", "The last line of the goal"} {
		if !strings.Contains(got, want) {
			t.Fatalf("the copy of the header lost %q:\n%s", want, got)
		}
	}
	if !strings.Contains(got, "\n") {
		t.Fatalf("the whole goal kept its lines:\n%q", got)
	}
}

// ---- #370 rounds: the expansion as a view of its goal, the click, the scroll row, the caches ----

// openHead expands the goal under the title through the key the chat really reads.
func openHead(t *testing.T, m emberModel) emberModel {
	t.Helper()
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
	m = next.(emberModel)
	if !m.headShown() {
		t.Fatalf("fixture: Alt+G did not open the goal (headOpen %v, for %q)", m.chat.headOpen, m.chat.headFor)
	}
	return m
}

// TestTheExpansionDoesNotEatEscInNewGoalMode: Ctrl+N hides the goal under the title, so the Esc after it goes back to
// the goal you were talking to instead of closing an expansion that is not on the screen; back on its goal the
// expansion is drawn again, and the Esc after that collapses it.
func TestTheExpansionDoesNotEatEscInNewGoalMode(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'n', Mod: tea.ModCtrl})
	m = next.(emberModel)
	if m.chat.target != "" || !m.chat.forceNew {
		t.Fatalf("fixture: Ctrl+N did not start a new goal (target %q)", m.chat.target)
	}
	if m.headShown() {
		t.Fatal("the goal is still drawn under the title over a new-goal composer")
	}
	if head := ansi.Strip(m.chatHead(100)); strings.Contains(head, "the header must expand") {
		t.Fatalf("the header draws the goal it has blanked:\n%s", head)
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.target != "original" {
		t.Fatalf("Esc did not go back to the goal (target %q, message %q)", m.chat.target, m.msg)
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.headShown() {
		t.Fatal("the Esc back on the goal did not collapse the expansion")
	}
}

// TestSwitchingGoalsDoesNotExpandTheNewGoal: the header draws the selected goal, so an expansion opened on another
// goal is hidden rather than showing the wrong goal's text expanded, and the paging keys do not scroll it.
func TestSwitchingGoalsDoesNotExpandTheNewGoal(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	m.resident.Goals = append(m.resident.Goals, residentGoal{ID: "second", State: "queued", Prompt: "A short second goal."})
	m.goalSel = 1
	if m.headShown() {
		t.Fatal("the expansion stayed open after a switch to another goal")
	}
	head := ansi.Strip(m.chatHead(100))
	if !strings.Contains(head, "A short second goal.") {
		t.Fatalf("the header does not show the selected goal:\n%s", head)
	}
	if strings.Contains(head, "the header must expand and collapse") {
		t.Fatalf("the header draws the goal the expansion was opened on:\n%s", head)
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
	m = next.(emberModel)
	if m.chat.headTop != 0 {
		t.Fatalf("PgDn scrolled an expansion that is not on the screen (top %d)", m.chat.headTop)
	}
}

// TestTheExpandedHeadFitsNarrowTerminals: the scroll hint is cut to the frame's width as the footer's help lines are.
// Left long it wrapped at 40 or 60 columns, and the wrapped row pushed the title off the top of the screen.
func TestTheExpandedHeadFitsNarrowTerminals(t *testing.T) {
	for _, w := range []int{40, 60} {
		t.Run(fmt.Sprint(w), func(t *testing.T) {
			m := longGoalChat(t, w, 24)
			m.syncChat() // (a resize rebuilds the renderer at the width the frame is drawn at)
			m = openHead(t, m)
			head := ansi.Strip(m.chatHead(w))
			if !strings.Contains(head, "… lines") {
				t.Fatalf("the open goal lost the hint that says how to scroll:\n%s", head)
			}
			for _, row := range strings.Split(head, "\n") {
				if got := ansi.StringWidth(row); got > w {
					t.Fatalf("a header row is %d columns wide at %d: %q", got, w, row)
				}
			}
			for _, row := range strings.Split(ansi.Strip(m.render()), "\n") {
				if got := ansi.StringWidth(row); got > w {
					t.Fatalf("the frame is %d columns wide at %d: %q", got, w, row)
				}
			}
		})
	}
}

// TestTheClickThatClearsACopyDoesNotToggleTheGoal: a drag over the header copies the whole goal, and the click that
// clears that copy must only clear it. The press is over the header's rows, so clickAt's toggle used to run too.
func TestTheClickThatClearsACopyDoesNotToggleTheGoal(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	_ = m.render() // (the frame the selection maps onto)
	// what a drag over the header leaves: the copy shown on the rows it took
	m.chat.sel = chatSelection{shown: true, x0: 0, x1: 20, y0: 1, y1: 3, a: [2]int{2, 1}, b: [2]int{2, 3}}
	next, _ := m.chatUpdate(tea.MouseClickMsg{X: 2, Y: 1, Button: tea.MouseLeft})
	m = next.(emberModel)
	if !m.chat.sel.cleared {
		t.Fatal("the press over a shown copy did not mark the click as clearing it")
	}
	next, _ = m.chatUpdate(tea.MouseReleaseMsg{X: 2, Y: 1, Button: tea.MouseLeft})
	m = next.(emberModel)
	if m.chat.sel.shown || m.chat.sel.dragging {
		t.Fatalf("the copy was not cleared (shown %v, dragging %v)", m.chat.sel.shown, m.chat.sel.dragging)
	}
	if !m.headShown() {
		t.Fatal("the click that cleared the copy also closed the goal")
	}
	// With nothing shown, a click on the header is the toggle again.
	next, _ = m.chatUpdate(tea.MouseClickMsg{X: 2, Y: 1, Button: tea.MouseLeft})
	m = next.(emberModel)
	if m.chat.sel.cleared {
		t.Fatal("a press with nothing shown is marked as clearing a copy")
	}
	next, _ = m.chatUpdate(tea.MouseReleaseMsg{X: 2, Y: 1, Button: tea.MouseLeft})
	m = next.(emberModel)
	if m.headShown() {
		t.Fatal("a plain click on the header no longer closes the goal")
	}
}

// shortGoalChat: the long goal's fixture, scrolled down, then re-aimed to a short text — the "↪ " case, where the
// expansion stays open on the same goal id while its text shrinks under the row the scroll left behind.
func shortGoalChat(t *testing.T) emberModel {
	t.Helper()
	m := openHead(t, longGoalChat(t, 100, 30))
	for i := 0; i < 3 && m.chat.headTop == 0; i++ {
		next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
		m = next.(emberModel)
	}
	if m.chat.headTop == 0 {
		t.Fatal("fixture: PgDn did not scroll the long goal (it fits, so there is nothing to leave stale)")
	}
	for i := 2; i < 60; i++ {
		m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: i, Goal: "original", Kind: "assistant", Data: map[string]any{"content": fmt.Sprintf("Observation %d stays in the transcript.", i)}}}})
	}
	// the re-aim a message followed by a new done check makes: the header now shows a short text
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 60, Goal: "original", Kind: "done_check", Data: map[string]any{"check": "something"}},
		residentEvent{Seq: 61, Goal: "original", Kind: "user_message", Data: map[string]any{"text": "short"}},
		residentEvent{Seq: 62, Goal: "original", Kind: "done_check", Data: map[string]any{"check": "short is done"}})
	m.render()
	return m
}

// TestTheScrollRowFollowsTheGoalOnTheScreen: the goal shrinks while the expansion stays open, so the row the state
// holds is past the new goal's last row. headExpanded clamps it for drawing; the press must move from that drawn row
// too, or it is spent on rows the screen never showed and the transcript does not page.
func TestTheScrollRowFollowsTheGoalOnTheScreen(t *testing.T) {
	m := shortGoalChat(t)
	if !strings.Contains(ansi.Strip(m.chatHead(100)), "↪ short") {
		t.Fatalf("fixture: the re-aim is not what the header shows:\n%s", ansi.Strip(m.chatHead(100)))
	}
	m.chat.transcript.GotoTop()
	before := m.chat.transcript.YOffset()
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
	m = next.(emberModel)
	if m.chat.headTop != 0 {
		t.Fatalf("the press left the stale row in the state (headTop %d, the goal has a row or two)", m.chat.headTop)
	}
	if m.chat.transcript.YOffset() == before {
		t.Fatal("PgDn did not page the transcript: it was spent on a row the expanded goal does not have")
	}
}

// TestTheExpansionDrawnIsTheExpansionScrolled: the drawing (headExpanded), the copy (emberselect) and PgUp/PgDn
// each used to work out the rows, the start row and the goal's length on their own, and a stale headTop against a
// shorter body could put the scroll footer under rows the copy did not call the goal. headExpansion is one
// computation now, and this walks a row past both ends and far past the goal to hold it to what it says (#370
// review).
func TestTheExpansionDrawnIsTheExpansionScrolled(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	w := max(20, m.width)
	body := trimBlankEdges(strings.Split(m.headExpandedBody(w), "\n")) // (the rows the expansion counts)
	for _, headTop := range []int{-5, 0, 3, 1000} {
		m.chat.headTop = headTop
		e := m.headExpansion(w)
		if e.lines != len(body) {
			t.Fatalf("headTop %d: the expansion says %d rows, the body has %d", headTop, e.lines, len(body))
		}
		if e.top < 0 || e.top > e.maxTop() {
			t.Fatalf("headTop %d: the goal starts at row %d (last row %d)", headTop, e.top, e.maxTop())
		}
		if len(e.rows) > m.headRows() {
			t.Fatalf("headTop %d: %d rows drawn, the header has %d", headTop, len(e.rows), m.headRows())
		}
		if e.top+len(e.rows) > e.lines {
			t.Fatalf("headTop %d: rows %d–%d of a %d-row goal", headTop, e.top+1, e.top+len(e.rows), e.lines)
		}
		if e.top < e.maxTop() && len(e.rows) != m.headRows() {
			t.Fatalf("headTop %d: %d rows before the last row, want a screenful of %d", headTop, len(e.rows), m.headRows())
		}
		for i, ln := range e.rows { // the rows are the body's, from the row the expansion says
			if ln != body[e.top+i] {
				t.Fatalf("headTop %d: row %d is not body row %d\ngot  %q\nwant %q", headTop, i, e.top+i, ln, body[e.top+i])
			}
		}
		// The footer names exactly the rows drawn and the goal's length, and is there exactly when the goal does
		// not fit whole.
		foot := fmt.Sprintf("lines %d–%d of %d", e.top+1, e.top+len(e.rows), e.lines)
		head := ansi.Strip(m.chatHead(w))
		overflow := len(e.rows) > 0 && (e.top > 0 || e.lines > e.top+len(e.rows))
		if got := strings.Contains(head, foot); got != overflow {
			t.Fatalf("headTop %d: footer %q present=%v, want %v:\n%s", headTop, foot, got, overflow, head)
		}
	}
}

// TestTheExpandedGoalKeepsThreeRowsOnATinyScreen: headRows() had a max(3, …) around max(12, height)/2-3 that could
// never be reached, because the 12-row floor already keeps the result at three or more. The floor is the thing that
// matters at a small height, so it is written down here rather than left implicit in an unreachable bound (#370
// review).
func TestTheExpandedGoalKeepsThreeRowsOnATinyScreen(t *testing.T) {
	m := chatFixture(t)
	m.width = 100
	for _, h := range []int{6, 12, 23, 24, 30, 60, 120} {
		m.height = h
		rows := m.headRows()
		if rows < 3 {
			t.Errorf("height %d: headRows %d gives the goal fewer than three rows", h, rows)
		}
		if rows > max(12, h)/2 || (h > 24 && rows > h/2) {
			t.Errorf("height %d: headRows %d is more than half the screen", h, rows)
		}
		if h >= 30 && rows != h/2-3 {
			t.Errorf("height %d: headRows %d is not half less the title and rule (%d)", h, rows, h/2-3)
		}
	}
}

// TestTheGoalDrawingsStayBounded: headMD is keyed by width and goal text and syncChat clears it only on a resize, so
// a long run that shows many goals would keep every drawing. Past the bound one other drawing is evicted per insert:
// the cache stays bounded and the goal on the screen keeps the drawing the next frame asks for.
func TestTheGoalDrawingsStayBounded(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	m.syncChat()
	for i := 0; i < headMDMax+4; i++ {
		m.residentEvents = append(m.residentEvents,
			residentEvent{Seq: 10 + 2*i, Goal: "original", Kind: "user_message", Data: map[string]any{"text": fmt.Sprintf("goal %d", i)}},
			residentEvent{Seq: 11 + 2*i, Goal: "original", Kind: "done_check", Data: map[string]any{"check": fmt.Sprintf("goal %d is done", i)}})
		_ = m.chatHead(100)
	}
	if n := len(m.chat.headMD); n > headMDMax {
		t.Fatalf("headMD holds %d drawings of goals no longer on the screen (bound %d)", n, headMDMax)
	}
	// the goal on the screen is drawn once and then taken from the cache: asking for it again adds nothing
	_ = m.chatHead(100)
	n := len(m.chat.headMD)
	_ = m.chatHead(100)
	if got := len(m.chat.headMD); got != n {
		t.Fatalf("drawing the same goal twice added an entry (%d then %d)", n, got)
	}
	// The goal on the screen is the one the cache keeps: clearing the whole map with every insert dropped its
	// drawing too, and the very next frame (chatHead runs on every frame) rendered it again.
	if _, ok := m.chat.headMD[headKey{width: 100, text: m.headGoalText()}]; !ok {
		t.Fatalf("the goal on the screen is not in headMD after %d goals (%d entries)", headMDMax+4, len(m.chat.headMD))
	}
}

// TestAnExpansionOnABlankPromptStillBelongsToItsGoal: headGoalID keyed off headGoalText, which is "" for a goal
// whose prompt is blank, so it answered "" for a goal that is on the screen. The id comes from the cases headGoalText
// blanks for *status* (no status yet, no selected goal, a new goal being written), not from the text the goal happens
// to have. The key cannot reach the old bug today - Alt+G only opens a goal that is cut, and a blank
// prompt is never cut - but the id it stores is what headShown compares against, and any other caller would have hit
// it.
func TestAnExpansionOnABlankPromptStillBelongsToItsGoal(t *testing.T) {
	m := chatFixture(t)
	m.resident.Goals = []residentGoal{{ID: "blank", State: "working", Prompt: "   "}}
	m.goalSel, m.chat.target = 0, "blank"
	if got := m.headGoalID(); got != "blank" {
		t.Fatalf("a goal with a blank prompt has no id under its title (%q)", got)
	}
	// (and the key still refuses to expand a goal with nothing to show)
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
	if next.(emberModel).chat.headOpen {
		t.Fatal("Alt+G opened a goal with no text")
	}
	// the cases that really are "no goal" still have no id
	newGoal := m
	newGoal.chat.newGoal, newGoal.chat.forceNew = true, true
	if newGoal.headGoalID() != "" {
		t.Fatal("a new goal being written still has an id")
	}
	noStatus := m
	noStatus.statusAt = time.Time{}
	if noStatus.headGoalID() != "" {
		t.Fatal("a session with no status yet has an id")
	}
	noSel := m
	noSel.goalSel = -1
	if noSel.headGoalID() != "" {
		t.Fatal("no selected goal still has an id")
	}
}

// TestCopyingTheHeaderPastTheGoalTakesTheHintAsItself: the goal under the title occupies rows 1..rows, and the row
// after them is the cut hint (collapsed) or the scroll footer (open). A selection that reaches that row copies it as
// what it is: the header's goal substitution stops at the goal's own rows instead of running to the rule (#316).
func TestCopyingTheHeaderPastTheGoalTakesTheHintAsItself(t *testing.T) {
	t.Run("collapsed", func(t *testing.T) {
		m := longGoalChat(t, 100, 30)
		_ = m.render() // (the frame the selection maps onto)
		w := max(20, m.width)
		rows := len(headGoalRowsUnderTitle(strings.Join(strings.Fields(m.headGoalText()), " "), w))
		if rows != headLines {
			t.Fatalf("fixture: the cut goal fills %d lines, want %d", rows, headLines)
		}
		if !strings.Contains(ansi.Strip(m.chatHead(w)), "Alt+G expands") {
			t.Fatalf("fixture: the collapsed header has no cut hint:\n%s", ansi.Strip(m.chatHead(w)))
		}
		m.chat.sel = m.selectionAt(2, 1)
		m.chat.sel.b = [2]int{1 + rows, w} // the goal's last row on through the hint
		m.chat.sel.moved, m.chat.sel.dragging = true, false
		got := m.selectedText()
		if !strings.Contains(got, "Alt+G expands") {
			t.Fatalf("a selection through the cut hint copied the goal over it:\n%s", got)
		}
	})
	t.Run("expanded", func(t *testing.T) {
		m := openHead(t, longGoalChat(t, 100, 30))
		_ = m.render()
		w := max(20, m.width)
		rows := len(m.headExpansion(w).rows)
		if rows != m.headRows() {
			t.Fatalf("fixture: the open goal fills %d of %d rows", rows, m.headRows())
		}
		if !strings.Contains(ansi.Strip(m.chatHead(w)), "PgUp/PgDn scrolls") {
			t.Fatalf("fixture: the open goal has no scroll footer:\n%s", ansi.Strip(m.chatHead(w)))
		}
		m.chat.sel = m.selectionAt(2, 1)
		m.chat.sel.b = [2]int{1 + rows, w} // the goal's last shown row on through the footer
		m.chat.sel.moved, m.chat.sel.dragging = true, false
		got := m.selectedText()
		if !strings.Contains(got, "PgUp/PgDn scrolls") {
			t.Fatalf("a selection through the scroll footer copied the goal over it:\n%s", got)
		}
	})
}

// TestCopyingAnOpenScrolledGoalAndTheRowsUnderIt: the review's last combination — expanded, scrolled, a selection
// from the title row to the header's last row — copies the goal once, then the scroll footer as itself; and a
// selection pressed in the transcript takes transcript text only, even dragged up over the header's rows: the goal
// substitution is for screen selections, and a press under the header makes a transcript selection (selectionAt).
func TestCopyingAnOpenScrolledGoalAndTheRowsUnderIt(t *testing.T) {
	m := openHead(t, longGoalChat(t, 100, 30))
	for i := 0; i < 3 && m.chat.headTop == 0; i++ {
		next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
		m = next.(emberModel)
	}
	if m.chat.headTop == 0 {
		t.Fatal("fixture: PgDn did not scroll the open goal")
	}
	for i := 2; i < 40; i++ {
		m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{
			Seq: i, Goal: "original", Kind: "assistant",
			Data: map[string]any{"content": fmt.Sprintf("Observation %d stays in the transcript.", i)}}}})
	}
	_ = m.render()
	w := max(20, m.width)
	head := lipgloss.Height(m.chatHead(w))
	if foot := ansi.Strip(m.chatHead(w)); !strings.Contains(foot, "PgUp/PgDn scrolls") {
		t.Fatalf("fixture: the scrolled header has no footer:\n%s", foot)
	}
	// press on the title row, release on the header's last row (the rule under the scroll footer)
	m.chat.sel = m.selectionAt(2, 0)
	if !m.chat.sel.screen {
		t.Fatal("fixture: a press on the title row is not a screen selection")
	}
	m.chat.sel.b = [2]int{head - 1, w - 1}
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	got := m.selectedText()
	if n := strings.Count(got, "The last line of the goal"); n != 1 {
		t.Fatalf("the goal was copied %d times over rows 1..%d, want once:\n%s", n, head-1, got)
	}
	if !strings.Contains(got, "PgUp/PgDn scrolls") {
		t.Fatalf("the scroll footer is not copied as itself:\n%s", got)
	}
	if strings.Contains(got, "Observation") {
		t.Fatalf("a selection that starts on the title row reached the transcript:\n%s", got)
	}
	// pressed in the transcript, dragged up over the header's rows: a transcript selection, no goal injected
	m.chat.transcript.GotoTop()
	m.chat.sel = m.selectionAt(2, head)
	if m.chat.sel.screen {
		t.Fatal("fixture: a press under the header is a screen selection")
	}
	m.chat.sel.b = [2]int{m.chat.transcript.YOffset(), 20}
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	if got := m.selectedText(); strings.Contains(got, "The last line of the goal") {
		t.Fatalf("a transcript selection took the header's goal:\n%s", got)
	}
}

// TestAGoalWithItsOwnEllipsisIsNotCut: the header's own report is what says the goal was cut, not a "…" anywhere in
// the drawn title line. A short goal that writes "…" itself is whole on the title row, and the header says so.
func TestAGoalWithItsOwnEllipsisIsNotCut(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	const goal = "Fix the parser… then the printer"
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: goal}}
	m.goalSel, m.chat.target = 0, "original"
	if m.headUnderTitle(100) {
		t.Fatalf("a goal whose own text holds a … was taken for a cut one:\n%s", ansi.Strip(m.chatHead(100)))
	}
	head := ansi.Strip(m.chatHead(100))
	title, _, _ := strings.Cut(head, "\n")
	if !strings.Contains(title, goal) {
		t.Fatalf("the goal is not whole on the title line:\n%s", head)
	}
	if got := rows(head); got != 2 { // the title and the rule, with no goal rows under the title
		t.Fatalf("the header drew %d rows: it moved the goal under the title:\n%s", got, head)
	}
}

// TestAGoalGlamourCannotRenderIsWrappedToTheTerminal: a goal with one pasted line indented past mdMaxDepth is left as
// it is by renderWith, and the expansion drew it at its own width — 2,003 columns at 100, which the terminal clips or
// wraps, either way unreadable again (#316). The raw fallback wraps to the width the footer and headExpansion count
// (#370 review P1).
func TestAGoalGlamourCannotRenderIsWrappedToTheTerminal(t *testing.T) {
	const w, h = 100, 30
	goal := strings.Repeat("Keep the goal readable while the header is closed, and draw it whole when it opens. ", 18) +
		"\n" + strings.Repeat(" ", 40) + "pasted line that nests past mdMaxDepth\n" +
		"The last words are reachable."
	if renderable(goal) {
		t.Fatal("fixture: the goal is renderable, so the raw fallback is not what is drawn")
	}
	m := chatFixture(t)
	m.width, m.height = w, h
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: goal}}
	m.goalSel, m.chat.target = 0, "original"
	if collapsed := ansi.Strip(m.chatHead(w)); !strings.Contains(collapsed, "Alt+G expands") {
		t.Fatalf("the collapsed header does not say the goal is cut:\n%s", collapsed)
	}
	m = openHead(t, m)
	for _, c := range []struct{ what, frame string }{{"the open header", m.chatHead(w)}, {"the frame", m.render()}} {
		for _, row := range strings.Split(ansi.Strip(c.frame), "\n") {
			if got := ansi.StringWidth(row); got > w {
				t.Fatalf("%s has a %d-column row at %d: %q", c.what, got, w, row)
			}
		}
	}
	// the goal's own lines kept their breaks, and PgDn reaches the words past the first screenful of rows
	for i := 0; i < 10 && !strings.Contains(ansi.Strip(m.chatHead(w)), "The last words are reachable."); i++ {
		next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
		m = next.(emberModel)
	}
	if end := ansi.Strip(m.chatHead(w)); !strings.Contains(end, "The last words are reachable.") {
		t.Fatalf("PgDn did not reach the goal's last words:\n%s", end)
	}
}

// TestCopyingAShortGoalsHeaderKeepsTheRule: a goal whole on the title row has no rows under it, so the copy took the
// rule row as goal text and put the goal where the rule is (#370 review P2).
func TestCopyingAShortGoalsHeaderKeepsTheRule(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	const goal = "fix the parser"
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: goal}}
	m.goalSel, m.chat.target = 0, "original"
	if m.headUnderTitle(100) {
		t.Fatalf("fixture: the goal went under the title:\n%s", ansi.Strip(m.chatHead(100)))
	}
	_ = m.render()
	m.chat.sel = m.selectionAt(2, 0)
	m.chat.sel.b = [2]int{2, max(20, m.width) - 1} // rows 0-2: the title, the rule, the transcript's first row
	m.chat.sel.moved, m.chat.sel.dragging = true, false
	copied := m.selectedText()
	if got := strings.Count(copied, goal); got != 1 {
		t.Fatalf("the goal is in the copy %d times (the title row draws it once):\n%s", got, copied)
	}
	got := strings.Split(copied, "\n")
	if len(got) < 2 || strings.Trim(strings.TrimSpace(got[1]), "─") != "" {
		t.Fatalf("row 2 of the copy is not the rule:\n%s", copied)
	}
}

// TestNoHintWhenTheGoalFitsTheRowsUnderTheTitle: the words all fit the rows the collapsed header draws, so opening the
// goal only reflows them; the hint promised a hidden line that was never hidden (#370 review P2).
func TestNoHintWhenTheGoalFitsTheRowsUnderTheTitle(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 160, 30
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: strings.Repeat("word ", 30)}}
	m.goalSel, m.chat.target = 0, "original"
	head := ansi.Strip(m.chatHead(160))
	if !m.headUnderTitle(160) {
		t.Fatalf("fixture: the goal did not go under the title:\n%s", head)
	}
	if strings.Contains(head, "Alt+G expands") {
		t.Fatalf("a goal whole on the rows under the title says it is cut:\n%s", head)
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'g', Mod: tea.ModAlt})
	if m = next.(emberModel); m.chat.headOpen {
		t.Fatal("Alt+G opened a goal with nothing hidden")
	}
}

// TestWrapRawKeepsTheLineBreaksAndBreaksWhatIsWider: wrapRaw is the header's fallback (#370 P1). It keeps each line
// the human wrote, keeps a blank line as blank, and breaks a line with no space in it (a pasted URL or path) so no row
// it returns is wider than the terminal.
func TestWrapRawKeepsTheLineBreaksAndBreaksWhatIsWider(t *testing.T) {
	long := strings.Repeat("x", 60)
	rows := strings.Split(wrapRaw("one two three\n\n"+long, 20), "\n")
	if rows[0] != "one two three" {
		t.Fatalf("row 0 lost the goal's own line: %q", rows)
	}
	if rows[1] != "" {
		t.Fatalf("row 1 is not the blank line the goal had: %q", rows)
	}
	for i, ln := range rows {
		if w := ansi.StringWidth(ln); w > 20 {
			t.Fatalf("row %d is %d columns wide, past the terminal's 20: %q", i, w, ln)
		}
	}
	if got := strings.Join(rows[2:], ""); got != long {
		t.Fatalf("the broken line does not hold the %d characters it was given: %d %q", len(long), len(got), got)
	}
	if got := wrapRaw("a b c", 0); got != "a b c" {
		t.Fatalf("a width under 1 changed the text instead of leaving it: %q", got)
	}
}
