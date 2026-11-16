// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
	"io/fs"
)

// chatFixture isolates the panel's user-state writes (last-session file) in a
// temporary state dir; a test or benchmark must never touch the human's state.
func chatFixture(tb testing.TB) emberModel {
	tb.Helper()
	tb.Setenv("XDG_STATE_HOME", tb.TempDir())
	m := emberModel{agentName: "resident", tab: 6, width: 90, height: 28, mode: "normal"}
	m.initChat()
	m.chat.cards = true // the card tests predate the compact view; TestCompactTranscript covers it
	m.receiveResident(residentMsg{agent: "resident", status: residentStatus{Name: "resident", Model: "local-model", Context: 1048574, DaemonAlive: true, LastEvent: 1, Goals: []residentGoal{{ID: "original", State: "working", Prompt: "a goal"}}}, events: []residentEvent{{Seq: 1, Goal: "original", Kind: "goal_queued", Data: map[string]any{"prompt": "a goal"}}}})
	return m
}

func TestChatRetainsTranscriptAndScrollAnchorOnRefresh(t *testing.T) {
	m := chatFixture(t)
	for i := 2; i < 32; i++ {
		m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: i, Goal: "original", Kind: "assistant", Data: map[string]any{"content": fmt.Sprintf("Observation %d stays in the transcript.", i)}}}})
	}
	m.chat.transcript.GotoTop()
	before := ansi.Strip(m.chat.transcript.View())
	m.chat.compose.SetValue("a draft survives")
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 32, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "new activity"}}}})
	if m.chat.transcript.YOffset() != 0 || ansi.Strip(m.chat.transcript.View()) != before {
		t.Fatal("new activity moved the reader")
	}
	plain := ansi.Strip(m.chat.content)
	if !strings.Contains(plain, "Observation 2 stays") || !strings.Contains(plain, "new activity") {
		t.Fatal("old or new transcript disappeared")
	}
	if m.chat.compose.Value() != "a draft survives" {
		t.Fatal("poll replaced input")
	}
	m.chat.transcript.GotoBottom()
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 33, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "latest activity"}}}})
	if !m.chat.transcript.AtBottom() || !strings.Contains(ansi.Strip(m.chat.transcript.View()), "latest activity") {
		t.Fatal("following the live tail failed")
	}
}

func TestChatComposerCapturesTargetAndKeepsFailedDraft(t *testing.T) {
	m := chatFixture(t)
	m.chat.compose.SetValue("Use existing tools.\nKeep café and 漢字 intact.")
	m.receiveResident(residentMsg{agent: "resident", status: residentStatus{LastEvent: 2, Goals: []residentGoal{{ID: "other", State: "working"}, {ID: "original", State: "working"}}}})
	if m.chat.target != "original" {
		t.Fatal("poll redirected steering")
	}
	next, _ := m.finishCompose(residentComposeResult{agent: "resident", err: "connection failed"})
	m = next.(emberModel)
	if !strings.Contains(m.chat.compose.Value(), "café") {
		t.Fatal("failed delivery lost Unicode draft")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.compose.Focused() || m.chat.compose.Value() == "" {
		t.Fatal("Escape should browse without deleting the draft")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if !m.chat.compose.Focused() {
		t.Fatal("Enter should resume editing")
	}
}

func TestChatSlashCommandsAndNewGoalRemainExplicit(t *testing.T) {
	m := chatFixture(t)
	m.chat.compose.SetValue("/new")
	next, cmd := m.submitCompose()
	m = next.(emberModel)
	if cmd != nil || !m.chat.newGoal {
		t.Fatal("/new should prepare input, not submit")
	}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident})
	if !m.chat.newGoal || m.chat.target != "" {
		t.Fatal("poll changed an explicit new goal into steering")
	}
	m.chat.compose.SetValue("/requests")
	next, _ = m.submitCompose()
	m = next.(emberModel)
	if m.tab != 6 || !m.chat.approvals {
		t.Fatal("/requests did not open the approvals overlay in the chat")
	}
	m.chat.approvals = false
	m.chat.compose.SetValue("/unknown keep this")
	next, cmd = m.submitCompose()
	m = next.(emberModel)
	if cmd != nil || m.chat.compose.Value() != "/unknown keep this" {
		t.Fatal("unknown slash command was sent or lost")
	}
}

func TestChatPasteAndFrameStayWithinTerminal(t *testing.T) {
	for _, size := range [][2]int{{40, 18}, {80, 24}, {140, 40}} {
		m := chatFixture(t)
		m.width, m.height = size[0], size[1]
		text := "a Unicode café draft\nsecond line\nthird line\nfourth line"
		next, _ := m.Update(tea.PasteMsg{Content: text})
		m = next.(emberModel)
		if m.chat.compose.Value() != text {
			t.Fatalf("paste changed: %q", m.chat.compose.Value())
		}
		frame := m.render()
		rows := strings.Split(frame, "\n")
		if len(rows) > size[1] {
			t.Fatalf("%v: %d lines overflow terminal", size, len(rows))
		}
		for _, line := range rows {
			if ansi.StringWidth(line) > size[0] {
				t.Fatalf("%v: line too wide: %q", size, ansi.Strip(line))
			}
		}
		if !strings.Contains(ansi.Strip(frame), "to goal original") {
			t.Fatal("bottom composer missing its target")
		}
	}
}

// The composer is the shell's: the target rides on its top border (the old label line, moved), one prompt mark sits on
// the first line only, the hint says what Enter does now, and the frame still fits the terminal at both extremes.
func TestComposerCarriesItsTargetOnTheBorder(t *testing.T) {
	for _, size := range [][2]int{{40, 16}, {100, 30}} {
		m := chatFixture(t) // the selected goal "original" is working
		m.width, m.height = size[0], size[1]
		m.syncChat()
		frame := ansi.Strip(m.render())
		if !strings.Contains(frame, "─ to goal original ─") {
			t.Fatalf("%v: the composer's border does not name its goal:\n%s", size, frame)
		}
		box := ansi.Strip(m.chatComposer())
		if n := strings.Count(box, g.user); n != 1 {
			t.Fatalf("%v: the composer drew %d prompt marks, want one:\n%s", size, n, box)
		}
		if !strings.Contains(box, "message the agent") || !strings.Contains(box, "Enter nudges") || !strings.Contains(box, "/steer") {
			t.Fatalf("%v: the hint does not say what Enter does:\n%s", size, box)
		}
		m.chat.compose.SetValue("first line\nsecond line")
		box = ansi.Strip(m.chatComposer())
		if n := strings.Count(box, g.user); n != 1 {
			t.Fatalf("%v: a multi-line draft drew %d prompt marks:\n%s", size, n, box)
		}
		if !strings.Contains(box, "  second line") {
			t.Fatalf("%v: the continuation line lost its indent:\n%s", size, box)
		}
		m.chat.newGoal, m.chat.target = true, ""
		m.chat.compose.Reset()
		m.syncChat()
		box = ansi.Strip(m.chatComposer())
		if !strings.Contains(box, "─ new goal ─") || !strings.Contains(box, "type a goal") || !strings.Contains(box, "Enter starts it") {
			t.Fatalf("%v: a new goal does not say what Enter does:\n%s", size, box)
		}
		frame = ansi.Strip(m.render())
		rows := strings.Split(frame, "\n")
		if len(rows) > size[1] {
			t.Fatalf("%v: %d rows overflow the terminal", size, len(rows))
		}
		for _, line := range rows {
			if ansi.StringWidth(line) > size[0] {
				t.Fatalf("%v: line too wide: %q", size, line)
			}
		}
	}
}

func TestChatDisplaysCommandArgumentsAndEscapesControls(t *testing.T) {
	m := chatFixture(t)
	e := residentEvent{Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"function": map[string]any{"name": "terminal", "arguments": "{\"command\":\"printf hello #\\u001b[2K\\rhidden\"}"}}}}}
	block := m.chat.eventBlock(e)
	plain := ansi.Strip(block)
	for _, want := range []string{g.tool + " terminal", "printf", `\x1b`, "hidden"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("missing %q in %q", want, plain)
		}
	}
	if strings.Contains(plain, "\x1b") || strings.Contains(plain, "\r") {
		t.Fatal("untrusted terminal controls survived rendering")
	}
	if !strings.Contains(block, "\x1b[") {
		t.Fatal("markdown/code rendering produced no highlighting")
	}
}

func TestChatLoadsHistoryAcrossMoreThanOnePageAndRejectsOldAgent(t *testing.T) {
	dir := t.TempDir()
	cli := filepath.Join(dir, "fake-cli")
	script := "#!/usr/bin/env python3\nimport sys,json\nif sys.argv[3]=='status': print(json.dumps(dict(name='resident',last_event=1201,goals=[])))\nelse:\n after=int(sys.argv[6]); print(json.dumps([dict(seq=i,goal='g',kind='assistant',data=dict(content=str(i))) for i in range(after+1,min(1202,after+201))]))\n"
	if err := os.WriteFile(cli, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	m := emberModel{cli: []string{cli}, agentName: "resident", width: 80, height: 24}
	m.initChat()
	// a fresh open: the newest maxResidentEvents/2 in one refresh, not the whole history from event 1
	m.receiveResident(residentRefresh(m)().(residentMsg))
	if m.chat.seq != 1201 || len(m.residentEvents) != maxResidentEvents/2 || m.residentEvents[0].Seq != 1201-maxResidentEvents/2+1 {
		t.Fatalf("fresh open: cursor=%d count=%d first=%d", m.chat.seq, len(m.residentEvents), m.residentEvents[0].Seq)
	}
	// a session behind its resident (it was open, the agent ran on) catches up 800 a refresh, nothing skipped
	b := emberModel{cli: []string{cli}, agentName: "resident", width: 80, height: 24}
	b.initChat()
	b.chat.seq = 1
	b.receiveResident(residentRefresh(b)().(residentMsg))
	if b.chat.seq != 801 {
		t.Fatalf("bounded catch-up cursor=%d", b.chat.seq)
	}
	b.receiveResident(residentRefresh(b)().(residentMsg))
	if b.chat.seq != 1201 || len(b.residentEvents) != 1200 {
		t.Fatalf("history missing: cursor=%d count=%d", b.chat.seq, len(b.residentEvents))
	}
	next, _ := m.Update(residentMsg{agent: "another", status: residentStatus{LastEvent: 9000}})
	if next.(emberModel).resident.LastEvent != 1201 {
		t.Fatal("old session reply contaminated current session")
	}
}

func TestSessionHomeOpensChatAndKeepsFiveMainPages(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	m := emberModel{agentName: "resident", tab: 7, sessionNames: []string{"resident", "research"}, sessionSel: 1, width: 90, height: 28}
	next, _ := m.sessionKeys("enter")
	m = next.(emberModel)
	if m.tab != 6 || m.agentName != "research" || !m.chat.compose.Focused() {
		t.Fatal("session did not open into composer")
	}
	if len(emberNavigation) != 5 || nextEnvTab(7, 1) != 5 || nextEnvTab(6, 1) != 7 {
		t.Fatal("navigation should start at sessions with five pages")
	}
}

func TestChatRejectsResponsesFromAnEarlierVisitToTheSameSession(t *testing.T) {
	m := chatFixture(t)
	m.openResident("other")
	m.openResident("resident")
	next, _ := m.Update(residentMsg{agent: "resident", epoch: 0, status: residentStatus{LastEvent: 900}, events: []residentEvent{{Seq: 900}}})
	m = next.(emberModel)
	if m.chat.seq != 0 || len(m.residentEvents) != 0 {
		t.Fatal("old visit skipped initial history pages")
	}
	m.chat.target, m.chat.selectID, m.chat.newGoal = "new-id", "new-id", false
	m.receiveResident(residentMsg{agent: "resident", epoch: m.residentEpoch, status: residentStatus{Goals: []residentGoal{{ID: "old", State: "completed"}}}})
	if m.chat.target != "new-id" || m.chat.newGoal {
		t.Fatal("stale status redirected a just-submitted goal")
	}
}

func TestMainPagesScrollInsideTheTerminal(t *testing.T) {
	// The old audit page is an overlay over the chat now: it fits the terminal and Page Down scrolls it.
	m := chatFixture(t)
	m.width, m.height = 80, 24
	next, _ := m.chatSlash("/audit") // (opening reads the audit file; the fixture fills the lines after)
	m = next.(emberModel)
	for i := 0; i < 90; i++ {
		m.events = append(m.events, fmt.Sprintf("audit entry %d with a long but readable message", i))
	}
	frame := m.render()
	if m.tab != 6 || m.chat.page != 5 || !strings.Contains(ansi.Strip(frame), "audit · ") {
		t.Fatalf("/audit did not open the overlay:\n%s", ansi.Strip(frame))
	}
	if len(strings.Split(frame, "\n")) > m.height {
		t.Fatal("audit overlay exceeded terminal height")
	}
	for _, line := range strings.Split(frame, "\n") {
		if ansi.StringWidth(line) > m.width {
			t.Fatal("audit overlay exceeded terminal width")
		}
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyPgDown})
	m = next.(emberModel)
	if m.chat.pageTop == 0 || ansi.Strip(m.render()) == ansi.Strip(frame) {
		t.Fatal("Page Down did not scroll the audit overlay")
	}
	if _, cmd := m.chatKey(tea.KeyPressMsg{Code: 'q', Text: "q"}); cmd != nil {
		t.Fatal("q must not quit inside an overlay")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	if next.(emberModel).chat.page != -1 {
		t.Fatal("Esc must close the overlay")
	}
}

func TestHistoryWrapPreservesLongUnicodeOutputAndReflowsOnResize(t *testing.T) {
	m := chatFixture(t)
	text := "start_" + strings.Repeat("A", 160) + strings.Repeat("界🙂", 40) + "_end"
	m.chat.expanded = true // output is folded by default; an open card must keep every character
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"output": text}}}}})
	for _, width := range []int{90, 40, 120} {
		m.width = width
		m.syncChat()
		plain := ansi.Strip(m.chat.content)
		if !utf8.ValidString(plain) {
			t.Fatal("wrapping split a UTF-8 character")
		}
		// The card rule sits at the start of every wrapped line, so drop it before asking whether the
		// unbroken run survived: the invariant is that no character is lost or split, not that the gutter
		// is invisible.
		if !strings.Contains(strings.ReplaceAll(strings.Join(strings.Fields(plain), ""), "│", ""), text) {
			t.Fatal("long output disappeared during wrapping/resize")
		}
		for _, line := range strings.Split(m.chat.content, "\n") {
			if ansi.StringWidth(line) > width {
				t.Fatalf("history line exceeds %d cells", width)
			}
		}
		m.chat.compose.SetValue("draft")
		before := m.chat.content
		next, _ := m.Update(tea.KeyPressMsg{Code: 'x', Text: "x"})
		m = next.(emberModel)
		_ = m.View()
		if m.chat.content != before || m.chat.compose.Value() != "draftx" {
			t.Fatal("typing replaced history or lost input")
		}
	}
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func TestSlashAutocompleteMatchesAndTabCycles(t *testing.T) {
	if got := slashMatches("/se"); fmt.Sprint(sortedCopy(got)) != "[/servers /session /sessions /settings]" {
		t.Fatalf("unexpected matches: %v", got)
	}
	if got := slashMatches("/st"); fmt.Sprint(sortedCopy(got)) != "[/start /steer /stop /studio]" {
		t.Fatalf("unexpected matches: %v", got)
	}
	if got := slashMatches("/s"); fmt.Sprint(sortedCopy(got)) != "[/servers /session /sessions /settings /start /steer /stop /studio]" {
		t.Fatalf("unexpected matches: %v", got)
	}
	for _, draft := range []string{"", "hello", "/remember queue pace", "/new\nsome goal"} {
		if got := slashMatches(draft); got != nil {
			t.Fatalf("unexpected matches for %q: %v", draft, got)
		}
	}
	for _, c := range slashCommands {
		if !strings.Contains("/help /quit /new /requests /approvals /sessions /newsession /remember /goals /goal /agent /session /start /stop /cancel /compact /resume /theme /glyphs /deny /steer /memory /context /audit /settings /model /effort /update /pause /halt /freetime /recap /studio /jobs /servers /agents /btw /diff /export /rewind /plan /", c.name+" ") {
			t.Fatalf("autocomplete offers %q, which chatSlash does not handle", c.name)
		}
	}
	m := chatFixture(t)
	m.chat.compose.SetValue("/s")
	next, done := m.cycleSlash()
	m = next.(emberModel)
	// (A to Z, as the menu lists them)
	if !done || m.chat.compose.Value() != "/servers " {
		t.Fatalf("first Tab gave %q", m.chat.compose.Value())
	}
	next, done = m.cycleSlash()
	m = next.(emberModel)
	if !done || m.chat.compose.Value() != "/session " {
		t.Fatalf("second Tab gave %q", m.chat.compose.Value())
	}
	next, done = m.cycleSlash()
	m = next.(emberModel)
	if !done || m.chat.compose.Value() != "/sessions " {
		t.Fatalf("cycle skipped a longer command: %q", m.chat.compose.Value())
	}
	m.chat.compose.SetValue("/new steer me")
	next, done = m.cycleSlash()
	if done {
		t.Fatal("Tab completed an argument draft")
	}
	if m.chat.compose.Value() != "/new steer me" {
		t.Fatal("cycle changed the draft")
	}
}

func TestQuickSwitcherOpensRecordsAndFiltersSessions(t *testing.T) {
	m := chatFixture(t)
	m.sessionNames = []string{"research", "resident"}
	m.chat.compose.SetValue("an unsent draft")
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'p', Mod: tea.ModCtrl})
	m = next.(emberModel)
	if !m.chat.picker || m.chat.pickerQuery != "" {
		t.Fatal("Ctrl+P did not open the switcher")
	}
	if m.chat.compose.Value() != "an unsent draft" {
		t.Fatal("the switcher destroyed the draft")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'r', Text: "r"})
	m = next.(emberModel)
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'c', Text: "c"})
	m = next.(emberModel)
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'h', Text: "h"})
	m = next.(emberModel)
	if names := m.pickerNames(); len(names) != 1 || names[0] != "research" {
		t.Fatalf("filter failed: %v", names)
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.picker || !m.chat.compose.Focused() || m.chat.compose.Value() != "an unsent draft" {
		t.Fatal("Escape did not restore the composer")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'p', Mod: tea.ModCtrl})
	m = next.(emberModel)
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if m.agentName != "research" || m.chat.picker || m.chat.compose.Value() != "" {
		t.Fatalf("Enter did not open the selected session: %q", m.agentName)
	}
	// A question arriving with the switcher open keeps F2 as the explicit takeover.
	m2 := chatFixture(t)
	m2.sessionNames = []string{"resident"}
	m2.chat.picker = true
	wrong := emberReq{ID: "q1", Kind: "question"}
	m2.chat.question = &wrong
	next, _ = m2.chatKey(tea.KeyPressMsg{Code: tea.KeyF2})
	m2 = next.(emberModel)
	if m2.chat.picker || !m2.chat.questionArmed || m2.chat.compose.Focused() {
		t.Fatal("F2 with the switcher open did not take over the question")
	}
}

func TestLaunchReopensLastSessionAndFallsBackWhenGone(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_STATE_HOME", dir)
	saveLastSession("research")
	if got := loadLastSession(); got != "research" {
		t.Fatalf("round trip lost the session: %q", got)
	}
	saveLastSession("\x01bad\nname")
	if got := loadLastSession(); got != "" {
		t.Fatalf("unsanitary name loaded: %q", got)
	}
	m := emberModel{agentName: "resident", defaultAgent: "resident", tab: 7, width: 90, height: 28}
	m.initChat()
	saveLastSession("research")
	openLastSession(&m)
	if m.agentName != "research" || m.tab != 6 || m.pendingLast != "research" {
		t.Fatal("launch did not reopen the last session")
	}
	next, _ := m.Update(residentSessionsMsg{agent: "research", names: []string{"resident", "research"}})
	m = next.(emberModel)
	if m.tab != 6 || m.pendingLast != "" || m.agentName != "research" {
		t.Fatal("an existing last session was rejected")
	}
	m2 := emberModel{agentName: "resident", defaultAgent: "resident", tab: 7, width: 90, height: 28}
	m2.initChat()
	saveLastSession("ghost")
	openLastSession(&m2)
	next, _ = m2.Update(residentSessionsMsg{agent: "ghost", names: []string{"resident"}})
	m2 = next.(emberModel)
	if m2.tab != 6 || !m2.chat.picker || m2.agentName != "resident" || !m2.chat.ready {
		t.Fatal("a vanished session did not fall back to the chat with the session picker open")
	}
	m3 := emberModel{agentName: "resident", defaultAgent: "resident", tab: 7, width: 90, height: 28}
	m3.initChat()
	openLastSession(&m3)
	if m3.pendingLast == "" {
		t.Fatal("fixture broken: no pending last session")
	}
	next, _ = m3.Update(residentSessionsMsg{agent: "ghost", err: "no such resident"})
	m3 = next.(emberModel)
	if m3.tab != 6 || !m3.chat.picker || m3.agentName != "resident" || m3.pendingLast != "" {
		t.Fatal("an unresolvable last session did not fall back to the chat's session picker")
	}
}

func TestChatFrameStaysSlimWithHintsAndSwitcher(t *testing.T) {
	for _, size := range [][2]int{{40, 16}, {110, 32}} {
		m := chatFixture(t)
		m.width, m.height = size[0], size[1]
		for i := 2; i < 60; i++ {
			m.receiveResident(residentMsg{agent: "resident", status: m.resident,
				events: []residentEvent{{Seq: i, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "text line"}}}})
		}
		m.chat.compose.SetValue("/s")
		m.syncChat() // (typing syncs the frame; the menu above the composer takes its rows from the transcript)
		frame := ansi.Strip(m.render())
		rows := strings.Split(frame, "\n")
		if len(rows) > size[1] {
			t.Fatalf("%v: %d lines overflow the terminal", size, len(rows))
		}
		for _, row := range rows {
			if ansi.StringWidth(row) > size[0] {
				t.Fatalf("%v: line too wide: %q", size, row)
			}
		}
		if !strings.Contains(frame, "/session") || (size[1] >= 32 && !strings.Contains(frame, "open a session NAME")) {
			t.Fatalf("%v: the slash menu (name and what it does) is missing:\n%s", size, frame)
		}
		if len(rows) < 3 {
			t.Fatal("frame lost the header and footer lines")
		}
		if !strings.Contains(frame, "Ctrl+P sessions") && !strings.Contains(frame, "session") {
			t.Fatalf("%v: footer lost the switcher hint", size)
		}
		next, _ := m.chatKey(tea.KeyPressMsg{Code: 'p', Mod: tea.ModCtrl})
		m = next.(emberModel)
		frame = ansi.Strip(m.render())
		if !strings.Contains(frame, "OPEN SESSION") {
			t.Fatalf("%v: switcher panel missing", size)
		}
		for _, row := range strings.Split(frame, "\n") {
			if ansi.StringWidth(row) > size[0] {
				t.Fatalf("%v: switcher line too wide", size)
			}
		}
	}
}

func TestLiveTailShowsThinkingToolsAndTextThenYieldsToTheDurableBlock(t *testing.T) {
	m := chatFixture(t)
	// a turn's first line carries the whole text, later lines carry the suffix
	next, _ := m.Update(residentLiveMsg{agent: "resident", turn: "t1", text: "Half", reasoning: "Weighing options.",
		tools: []string{"terminal"}, reset: true})
	m = next.(emberModel)
	if !m.chat.live.active {
		t.Fatal("the live tail did not start")
	}
	live := ansi.Strip(m.liveBlock(m.width))
	for _, want := range []string{g.agent + " agent", "Weighing options.", g.tool + " terminal", "Half", "live"} {
		if !strings.Contains(live, want) {
			t.Fatalf("live tail missing %q:\n%s", want, live)
		}
	}
	next, _ = m.Update(residentLiveMsg{agent: "resident", turn: "t1", text: " an answer."})
	m = next.(emberModel)
	if !strings.Contains(ansi.Strip(m.liveBlock(m.width)), "Half an answer.") {
		t.Fatal("streamed deltas did not accumulate")
	}
	// the durable reply replaces the tail and shows its reasoning as its own block
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 2, Goal: "original",
		Kind: "assistant", Data: map[string]any{"content": "Half an answer.", "reasoning": "Weighing options."}}}})
	if m.chat.live.active {
		t.Fatal("the durable reply must end the live tail")
	}
	durable := ansi.Strip(m.chat.content)
	if !strings.Contains(durable, g.collapsed+" thinking") || !strings.Contains(durable, g.agent+" agent") || !strings.Contains(durable, "Half an answer.") {
		t.Fatalf("durable blocks missing:\n%s", durable)
	}
	if strings.Contains(ansi.Strip(m.liveBlock(m.width)), "Half an answer.") {
		t.Fatal("the live tail outlived the durable block")
	}
}

func TestLiveTailStaysBoundedAndRejectsAnotherVisit(t *testing.T) {
	long := strings.Repeat("word ", 400)
	if got := strings.Count(emberTail(long, 40, 8), "\n"); got > 8 {
		t.Fatalf("live tail kept %d lines", got)
	}
	m := chatFixture(t)
	m.residentEpoch = 3
	next, _ := m.Update(residentLiveMsg{agent: "resident", epoch: 2, turn: "t", text: "ghost", reset: true})
	if next.(emberModel).chat.live.active {
		t.Fatal("a stream from an earlier visit painted the tail")
	}
}

func TestQuickSwitcherScrollsPastTheFirstPageWithoutPanicking(t *testing.T) {
	m := chatFixture(t)
	m.height = 12 // four visible rows, so a ten-session list needs paging
	names := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	m.sessionNames = names
	m.chat.picker = true
	for sel := range names {
		m.chat.pickerSel = sel
		box := ansi.Strip(m.pickerBox()) // panicked with "slice bounds out of range" before the fix
		if !strings.Contains(box, "› "+names[sel]) {
			t.Fatalf("selection %d (%s) is not shown:\n%s", sel, names[sel], box)
		}
	}
}

func TestResidentCommandsWithoutTheCLIReportInsteadOfPanicking(t *testing.T) {
	m := chatFixture(t)
	m.cli = nil
	if cmd := residentCommand(m, []string{"status", "resident", "--json"}, ""); cmd != nil {
		t.Fatal("a missing --cli must not build a command")
	}
	msg := residentSessions(m)()
	if sessions, ok := msg.(residentSessionsMsg); !ok || sessions.err == "" {
		t.Fatalf("sessions without a CLI must report an error, got %#v", msg)
	}
}

// The context bar shows what the daemon measured, graded against the
// compaction trigger, and never invents a percentage of its own. The accent
// piece (label, bar, share) takes the level's color; the counts stay dim.
func TestContextBarShowsOnlyMeasuredUsage(t *testing.T) {
	accent, rest := contextBar(0, 1048574, .85, 0)
	if accent != "" || rest != "ctx 1.0M window" || strings.Contains(rest, "%") {
		t.Fatalf("unmeasured context showed usage: %q / %q", accent, rest)
	}
	if accent, rest := contextBar(0, 0, .85, 0); accent != "" || rest != "" {
		t.Fatalf("no capacity, no bar: %q / %q", accent, rest)
	}
	accent, rest = contextBar(314573, 1048574, .85, 0) // 30%
	if filled := strings.Count(accent, "█"); filled != 3 || !strings.Contains(accent, "░") {
		t.Fatalf("30%% carved the bar wrong: %q", accent)
	}
	if !strings.Contains(accent, "ctx") || !strings.Contains(accent, "30%") || strings.Count(accent, "%") != 1 {
		t.Fatalf("accent lost its bar and share: %q", accent)
	}
	if !strings.Contains(rest, "of 1.0M · compact 85%") {
		t.Fatalf("dim piece lost counts and trigger: %q", rest)
	}
	if _, rest := contextBar(314573, 1048574, .85, 87.3); !strings.Contains(rest, "87 tok/s") || !strings.Contains(rest, "compact 85%") {
		t.Fatalf("measured tok/s did not annotate the dim piece: %q", rest)
	}
	if _, rest := contextBar(314573, 1048574, .85, 0); strings.Contains(rest, "tok/s") {
		t.Fatalf("no speed reported, none may show: %q", rest)
	}
	if _, rest := contextBar(0, 1048574, .85, 87.3); strings.Contains(rest, "tok/s") {
		t.Fatalf("tok/s without a measured context bar: %q", rest)
	}
	if accent, _ := contextBar(1048575, 1048574, .85, 0); !strings.HasPrefix(accent, "ctx ██████████ 100%") {
		t.Fatalf("usage past the window did not clamp: %q", accent)
	}
	if got := humanTokens(999); got != "999" || humanTokens(482000) != "482k" || humanTokens(1048574) != "1.0M" {
		t.Fatalf("token formatting drifted: %s %s %s", humanTokens(999), humanTokens(482000), humanTokens(1048574))
	}
	tests := []struct {
		used, capacity int
		threshold      float64
		level          int
	}{
		{0, 1048574, .85, 0},
		{314573, 1048574, .85, 0},
		{838861, 1048574, .85, 1}, // 80%: within fifteen points of the trigger
		{891288, 1048574, .85, 2}, // at the trigger
	}
	for _, tc := range tests {
		if got := contextBarLevel(tc.used, tc.capacity, tc.threshold); got != tc.level {
			t.Fatalf("level(%d/%d, %.2f) = %d, want %d", tc.used, tc.capacity, tc.threshold, got, tc.level)
		}
	}
	states := map[string]int{"working": 1, "queued": 1, "waiting_model": 1, "listening": 1,
		"waiting": 2, "waiting_user": 2, "paused": 2, "error": 3, "stopped": 0, "completed": 0}
	for state, want := range states {
		if got := contextStateStyle(state); got.Render("#") == emberDim.Render("#") && want != 0 {
			t.Fatalf("state %q rendered in the resting tone", state)
		}
	}
	if contextStateStyle("stopped").Render("#") != emberDim.Render("#") || contextStateStyle("nonsense").Render("#") != emberDim.Render("#") {
		t.Fatal("unknown states must rest in dim")
	}
}

// The measured bar lands in the live footer, warning color included, without
// pushing the frame past its width.
func TestChatFooterCarriesTheMeasuredContextBar(t *testing.T) {
	m := chatFixture(t)
	m.resident.ContextUsed = 891288 // at the trigger: hot
	m.resident.TokPerS = 91.4       // the daemon's last measured decode speed
	raw := m.render()
	frame := ansi.Strip(raw)
	if !strings.Contains(frame, "891K/1M") || !strings.Contains(frame, "91 tok/s") {
		t.Fatalf("footer lost the measured bar:\n%s", frame)
	}
	if !strings.Contains(raw, emberCtxHot.Render(strings.Repeat(g.bar, 7))) {
		t.Fatal("at the compaction trigger the bar must be hot")
	}
	for _, row := range strings.Split(frame, "\n") {
		if ansi.StringWidth(row) > m.width {
			t.Fatalf("footer line too wide: %q", row)
		}
	}
}

func TestSlashPrefixedMessageSendsVerbatimWithDoubleSlash(t *testing.T) {
	m := chatFixture(t)
	m.chat.compose.SetValue("//tmp is full, the build failed")
	next, _ := m.submitCompose()
	m = next.(emberModel)
	if m.msg != "" || !m.chat.sending {
		t.Fatalf("//text must be sent as a message, got msg=%q sending=%v", m.msg, m.chat.sending)
	}
	m2 := chatFixture(t)
	m2.chat.compose.SetValue("/tmp is full")
	next, _ = m2.submitCompose()
	m2 = next.(emberModel)
	if !strings.Contains(m2.msg, "Unknown command") || m2.chat.sending {
		t.Fatalf("an unknown command must be reported, not sent: msg=%q sending=%v", m2.msg, m2.chat.sending)
	}
}

func TestSessionCommandRejectsAnUnknownNameWithoutClearingTheTranscript(t *testing.T) {
	m := chatFixture(t)
	m.sessionNames = []string{"resident", "research"}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident,
		events: []residentEvent{{Seq: 2, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "keep me"}}}})
	before := len(m.residentEvents)
	m.chat.compose.SetValue("/session rescarch")
	next, _ := m.submitCompose()
	m = next.(emberModel)
	if m.agentName != "resident" || len(m.residentEvents) != before {
		t.Fatalf("a typo switched sessions: agent=%q events=%d/%d", m.agentName, len(m.residentEvents), before)
	}
	if !strings.Contains(ansi.Strip(m.chat.content), "keep me") {
		t.Fatal("a typo cleared the transcript")
	}
	if !strings.Contains(m.msg, "no such session") {
		t.Fatalf("no warning shown: %q", m.msg)
	}
	m.chat.compose.SetValue("/session research")
	next, _ = m.submitCompose()
	m = next.(emberModel)
	if m.agentName != "research" {
		t.Fatalf("a known session did not open: %q", m.agentName)
	}
}

func TestTruncatedReplyShowsWhatTheModelWroteBeforeItStopped(t *testing.T) {
	m := chatFixture(t)
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 2, Goal: "original",
		Kind: "model_truncated", Data: map[string]any{
			"error":   "the reply ended before it was complete; nothing was executed",
			"partial": map[string]any{"content": "Half an answer"}}}}})
	block := ansi.Strip(m.chat.content)
	if !strings.Contains(block, "Half an answer") {
		t.Fatalf("the partial reply is not shown:\n%s", block)
	}
	if !strings.Contains(block, "before it was complete") {
		t.Fatalf("the reason is not shown:\n%s", block)
	}
}

// liveWindow is the visible body of the live block, without its border or label line.
func liveWindow(m emberModel) string {
	body := m.chat.live.body(m.width - 4)
	top := m.liveTop(len(body))
	return strings.Join(body[top:min(top+m.liveHeight(), len(body))], "\n")
}

func TestLiveWindowScrollsAndHoldsWhileTheReplyKeepsArriving(t *testing.T) {
	m := chatFixture(t)
	m.height = 30
	next, _ := m.Update(residentLiveMsg{agent: "resident", turn: "t1",
		reasoning: strings.Repeat("thinking line ", 200), reset: true})
	m = next.(emberModel)
	if !strings.Contains(ansi.Strip(m.liveBlock(m.width)), g.agent+" thinking") || !strings.Contains(ansi.Strip(m.liveBlock(m.width)), "live ·") {
		t.Fatalf("the live window is not following:\n%s", ansi.Strip(m.liveBlock(m.width)))
	}
	m.scrollLive(-8)
	held := ansi.Strip(m.liveBlock(m.width))
	if !strings.Contains(held, "held") {
		t.Fatalf("scrolling did not hold the window:\n%s", held)
	}
	// more of the reply arrives: the held window must keep showing the same lines
	before := liveWindow(m)
	next, _ = m.Update(residentLiveMsg{agent: "resident", turn: "t1", text: "and now an answer"})
	m = next.(emberModel)
	if after := liveWindow(m); after != before {
		t.Fatalf("the held window moved while text arrived:\n%s\n---\n%s", before, after)
	}
	// Ctrl+T resumes following, so the newest text is visible again
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 't', Mod: tea.ModCtrl})
	m = next.(emberModel)
	if m.chat.live.hold {
		t.Fatal("Ctrl+T did not resume following")
	}
	if !strings.Contains(ansi.Strip(m.liveBlock(m.width)), "and now an answer") {
		t.Fatal("the resumed window does not show the newest text")
	}
	// scrolling back to the bottom resumes too
	m.scrollLive(-4)
	m.scrollLive(100)
	if m.chat.live.hold {
		t.Fatal("scrolling to the bottom must resume following")
	}
}

func TestEmbersBoxKeepsOneSizeGeneratingOrNot(t *testing.T) {
	m := chatFixture(t)
	m.height = 40
	m.residentEvents = append(m.residentEvents, residentEvent{Seq: 800, Goal: "original", Kind: "assistant", Data: map[string]any{"reasoning": "an earlier thought"}})
	m.syncChat()
	idle := lipgloss.Height(m.liveBlock(m.width))
	next, _ := m.Update(residentLiveMsg{agent: "resident", turn: "t1", reasoning: "short", reset: true})
	m = next.(emberModel)
	short := lipgloss.Height(m.liveBlock(m.width))
	next, _ = m.Update(residentLiveMsg{agent: "resident", turn: "t1", reasoning: strings.Repeat("thinking about it ", 60)})
	m = next.(emberModel)
	long := lipgloss.Height(m.liveBlock(m.width))
	if idle != short || short != long || long != thoughtRows+1 { // (a heading and the rows: no frame)
		t.Fatalf("one size: idle %d, a short thought %d, a long one %d (want %d)", idle, short, long, thoughtRows+1)
	}
	body, transcript := m.bodyHeight(), m.chat.transcript.Height()
	if transcript < body-long-2 || transcript > body-long+2 {
		t.Fatalf("transcript is %d lines, expected about %d", transcript, body-long)
	}
}

// sgr is the escape a color renders as, so a test can assert a palette role without hardcoding its hex.
func sgr(color color.Color) string {
	out := lipgloss.NewStyle().Foreground(color).Render("x")
	return strings.SplitN(out, "x", 2)[0]
}

func TestLiveTextIsColoredLikeTheTranscript(t *testing.T) {
	m := chatFixture(t)
	m.chat.live = residentLive{active: true, turn: "t1", text: "run `go test ./...` now", tools: []string{"terminal"}}
	block := m.liveBlock(m.width)
	if !strings.Contains(block, sgr(cTeal)) {
		t.Fatalf("inline code is not colored:\n%q", block)
	}
	if !strings.Contains(block, sgr(cOrange)) {
		t.Fatalf("the running tool is not colored:\n%q", block)
	}
}

func TestEachEventIsFramedAsACardWithItsBodyBehindARule(t *testing.T) {
	m := chatFixture(t)
	m.height = 30
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g1", Kind: "goal_queued", Data: map[string]any{"prompt": "fix the build"}},
		{Seq: 2, Goal: "g1", Kind: "assistant", Data: map[string]any{"content": "on it"}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	lines := strings.Split(ansi.Strip(m.chat.content), "\n")
	if !strings.HasPrefix(lines[0], g.user+" you · goal") {
		t.Fatalf("the first line is not the goal's label: %q", lines[0])
	}
	if !strings.HasPrefix(lines[1], "│ fix the build") {
		t.Fatalf("the body is not behind the card rule: %q", lines[1])
	}
	var seen bool
	for _, ln := range lines {
		if strings.HasPrefix(ln, g.agent+" agent") {
			seen = true
		}
	}
	if !seen {
		t.Fatalf("the reply has no label:\n%s", strings.Join(lines, "\n"))
	}
}

func TestMarkdownColorsFencedCodeWithTheNordTheme(t *testing.T) {
	r, err := glamour.NewTermRenderer(glamour.WithStyles(emberMarkdownStyle()), glamour.WithWordWrap(80))
	if err != nil {
		t.Fatal(err)
	}
	out, err := r.Render("```go\nfunc main() { println(\"hi\", 42) }\n```\n")
	if err != nil {
		t.Fatal(err)
	}
	if spans := strings.Count(out, "38;2;"); spans < 3 {
		t.Fatalf("fenced code has %d colored spans, expected syntax coloring:\n%q", spans, out)
	}
}

// themeFixture saves the panel's theme state and restores it after the test;
// the theme is global and no test may leave a switched palette behind.
func themeFixture(tb testing.TB) {
	tb.Helper()
	saved := themeCurrent
	tb.Cleanup(func() {
		themeCurrent = saved
		applyThemeStyles()
	})
}

func TestThemeParsesTheEmbeddedCaminusFile(t *testing.T) {
	data, err := fs.ReadFile(themeFS, "themes/caminus.theme")
	if err != nil {
		t.Fatal(err)
	}
	got, bad := parseTheme(data)
	if len(bad) != 0 {
		t.Fatalf("the built-in caminus theme has bad lines: %v", bad)
	}
	for _, role := range []string{"text", "accent", "ok", "warn", "action", "hot", "user", "chrome", "code_theme"} {
		if got[role] == "" {
			t.Fatalf("the built-in caminus theme does not set %s", role)
		}
	}
}

func TestThemeLoadSwitchesRolesAndStyles(t *testing.T) {
	themeFixture(t)
	before := themeRole("accent")
	if err := themeLoad("nord"); err != nil {
		t.Fatal(err)
	}
	after := themeRole("accent")
	if before == after {
		t.Fatalf("loading nord left accent at %s", after)
	}
	if emberTitle.Render("x") == "" {
		t.Fatal("the rebuilt styles no longer render")
	}
	if themeLoad("no-such-theme") == nil {
		t.Fatal("a nonexistent theme loaded without error")
	}
	if themeRole("accent") != after {
		t.Fatal("a failed load changed the palette")
	}
}

func TestThemeNamesListTheBuiltins(t *testing.T) {
	names := themeNames()
	for _, want := range []string{"caminus", "nord", "catppuccin-mocha", "dracula", "gruvbox-dark", "tokyo-night"} {
		found := false
		for _, n := range names {
			if n == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("%s missing from themeNames: %v", want, names)
		}
	}
}

func TestThemeParseToleratesBtopKeysAndUnknownLines(t *testing.T) {
	text := `# a comment
main_fg = #e5e9f0
title_fg = #88c0d0
cpu_load_start_low = #abcdef
not a mapping at all
code_theme = monokai
`
	got, bad := parseTheme([]byte(text))
	if got["text"] != "#e5e9f0" || got["accent"] != "#88c0d0" || got["code_theme"] != "monokai" {
		t.Fatalf("the aliases or code_theme did not land: %v", got)
	}
	if _, present := got["cpu_load_start_low"]; present {
		t.Fatalf("an unmapped btop key was applied as a role: %v", got)
	}
	if len(bad) != 1 || !strings.Contains(bad[0], "not a mapping") {
		t.Fatalf("only lines that are not key = value should be bad, got %v", bad)
	}
}

func TestThemeSetPersistsAndTheSlashCommandSwitches(t *testing.T) {
	themeFixture(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	m := chatFixture(t)
	tm, _ := m.chatSlash("/theme")
	list := tm.(emberModel)
	if !list.chat.themes.open || !slices.Contains(list.chat.themes.names, "caminus") || !slices.Contains(list.chat.themes.names, "nord") {
		t.Fatalf("/theme opens the theme screen with the themes: %v", list.chat.themes.names)
	}
	tn, _ := m.chatSlash("/theme nope")
	nope := tn.(emberModel)
	if !strings.Contains(nope.msg, "no such theme") {
		t.Fatalf("/theme nope did not report the failure: %q", nope.msg)
	}
	ts, _ := m.chatSlash("/theme nord")
	m = ts.(emberModel)
	// syncChat rebuilds the renderer straight away when markdown was nilled, so
	// the next reply is styled in the new theme, not the old one.
	if m.chat.markdown == nil {
		t.Fatal("/theme nord left the markdown renderer unbuilt")
	}
	if themeRole("accent") != "#88c0d0" {
		t.Fatalf("/theme nord did not switch the palette: %s", themeRole("accent"))
	}
	saved, err := os.ReadFile(filepath.Join(dir, "caminus", "panel-theme"))
	if err != nil || strings.TrimSpace(string(saved)) != "nord" {
		t.Fatalf("the choice was not persisted (err=%v): %q", err, saved)
	}
}

func TestPathTokenDetectsPathsButLeavesCommandsAlone(t *testing.T) {
	for _, want := range []string{"/home/ig", "~/src/", "./build/x", "../notes", "/home"} {
		if _, ok := pathToken(want); !ok {
			t.Fatalf("%q should be a path token", want)
		}
	}
	for _, no := range []string{"", "hello", "/h", "goal one", "see /home later ", "a\nb /home"} {
		if tok, ok := pathToken(no); ok {
			t.Fatalf("%q should not be a path token (got %q)", no, tok)
		}
	}
}

func TestPathMatchesListsDirectoriesFirstAndHidesDotfiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "alpha"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "zeta.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".hidden"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "alpha", "inner.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	d := dir + "/"
	if got := pathMatches(d); len(got) != 2 || got[0] != d+"alpha/" || got[1] != d+"zeta.txt" {
		t.Fatalf("unexpected matches: %v", got)
	}
	if got := pathMatches(d + "a"); len(got) != 1 || got[0] != d+"alpha/" {
		t.Fatalf("prefix filter failed: %v", got)
	}
	if got := pathMatches(d + "."); len(got) != 1 || !strings.HasSuffix(got[0], ".hidden") {
		t.Fatalf("dot entries should surface with a dot prefix: %v", got)
	}
	if got := pathMatches(dir + "/nowhere/"); len(got) != 0 {
		t.Fatalf("an unreadable directory should match nothing: %v", got)
	}
}

func TestTabCompletesAWalksIntoDirectoriesAndCyclesFiles(t *testing.T) {
	themeFixture(t)
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "walk", "alpha"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "beta.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "gamma.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "walk", "alpha", "inner.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := chatFixture(t)
	m.chat.compose.SetValue(dir + "/walk/a")
	m.chat.compose.Focus()
	next, _ := m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyTab})
	m = next.(emberModel)
	if m.chat.compose.Value() != dir+"/walk/alpha/" {
		t.Fatalf("first Tab should take the directory: %q", m.chat.compose.Value())
	}
	next, _ = m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyTab})
	m = next.(emberModel)
	if m.chat.compose.Value() != dir+"/walk/alpha/inner.txt" {
		t.Fatalf("Tab in the directory should descend: %q", m.chat.compose.Value())
	}
	// From a files-only view cycling walks the matches in order; directories
	// would have gone first.
	files := t.TempDir()
	if err := os.WriteFile(filepath.Join(files, "beta.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(files, "gamma.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	reset := chatFixture(t)
	reset.chat.compose.SetValue(files + "/")
	for _, want := range []string{files + "/beta.txt", files + "/gamma.txt"} {
		nx, _ := reset.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyTab})
		reset = nx.(emberModel)
		if got := reset.chat.compose.Value(); got != want {
			t.Fatalf("cycling files: got %q, want %q", got, want)
		}
	}
}

func TestChatComposerHintsAboutPathsThematically(t *testing.T) {
	themeFixture(t)
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	m := chatFixture(t)
	m.chat.compose.SetValue("./")
	view := m.chatComposer()
	if !strings.Contains(view, "Tab completes") || !strings.Contains(view, "sub/") {
		t.Fatalf("a path in progress gets no completion hint:\n%s", view)
	}
	m.chat.compose.SetValue("./zzz")
	if view := m.chatComposer(); !strings.Contains(view, "no match") {
		t.Fatalf("a dead path should announce itself:\n%s", view)
	}
}

// The redesign's transcript: tool calls fold to one line until Ctrl+E opens every card; a failed call stays open
// with its reason; a root request is the inline approval card.
func TestTranscriptCardsFoldOpenAndShowRootInline(t *testing.T) {
	m := chatFixture(t)
	m.height = 40
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g1", Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"function": map[string]any{
			"name": "terminal", "arguments": `{"command":"make test\nmake lint"}`}}}}},
		{Seq: 2, Goal: "g1", Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{
			"output": "41 passed\nall green", "exit_code": float64(0)}}},
		{Seq: 3, Goal: "g1", Kind: "tool_result", Data: map[string]any{"name": "write", "result": map[string]any{
			"error": "permission denied: /etc/hosts"}}},
		{Seq: 4, Goal: "g1", Kind: "approval_requested", Data: map[string]any{"cmd": "apt-get install -y libcmocka-dev", "reason": "the test needs it"}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	folded := ansi.Strip(m.chat.content)
	for _, want := range []string{g.collapsed + " " + g.tool + " terminal  make test …", "exit 0 · 2 lines", "permission denied", "ROOT", "apt-get install -y libcmocka-dev", "a approve", "/deny TEXT"} {
		if !strings.Contains(folded, want) {
			t.Fatalf("folded transcript missing %q:\n%s", want, folded)
		}
	}
	if strings.Contains(folded, "all green") || strings.Contains(folded, "make lint") {
		t.Fatalf("a folded card showed its body:\n%s", folded)
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: 'e', Mod: tea.ModCtrl})
	m = next.(emberModel)
	open := ansi.Strip(m.chat.content)
	if !m.chat.expanded || !strings.Contains(open, "all green") || !strings.Contains(open, "make lint") {
		t.Fatalf("Ctrl+E did not open the cards:\n%s", open)
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'e', Mod: tea.ModCtrl})
	if next.(emberModel).chat.expanded {
		t.Fatal("a second Ctrl+E must fold them again")
	}
}

// A waiting root request is decided in the chat through the Requests page's own handler: d writes the denial to the
// bridge (and the audit), /deny TEXT denies with what to do instead in the same answer (the runtime steers with it).
func TestChatDecidesRootRequestsInline(t *testing.T) {
	for _, viaSlash := range []bool{false, true} {
		m := chatFixture(t)
		bridge := t.TempDir()
		writeReq(t, bridge, "r1", "apt-get install -y libfoo-dev", 1)
		m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
		m.refreshLocal()
		if m.pendingRoot() != 0 {
			t.Fatal("the waiting request is not found")
		}
		if viaSlash {
			var next tea.Model
			next, _ = m.chatSlash("/deny use the vendored copy in third_party/ instead")
			m = next.(emberModel)
		} else {
			m.chat.compose.Blur()
			next, _ := m.chatKey(tea.KeyPressMsg{Code: 'd', Text: "d"})
			m = next.(emberModel)
		}
		data, err := os.ReadFile(filepath.Join(bridge, "res", "r1.json"))
		if err != nil || !strings.Contains(string(data), `"denied"`) {
			t.Fatalf("no denial in the bridge (slash=%v): %s %v", viaSlash, data, err)
		}
		if viaSlash && !strings.Contains(string(data), `"answer":"use the vendored copy in third_party/ instead"`) {
			t.Fatalf("/deny carries its instruction in the denial: %s", data)
		}
		if m.pendingRoot() != -1 {
			t.Fatal("a decided request still shows as waiting")
		}
	}
}

// Approving runs a root command: one stray "a" while reading must not do it. The first press only arms it and names
// the command; a second approves; any other key in between disarms.
func TestChatApprovalNeedsASecondPress(t *testing.T) {
	m := chatFixture(t)
	bridge := t.TempDir()
	writeReq(t, bridge, "r1", "rm -rf /var/cache/foo", 1)
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
	m.refreshLocal()
	m.chat.compose.Blur()
	press := func(r rune) {
		next, _ := m.chatKey(tea.KeyPressMsg{Code: r, Text: string(r)})
		m = next.(emberModel)
	}
	press('a')
	if m.chat.armed != "a:r1" || !strings.Contains(m.msg, "rm -rf /var/cache/foo") || !strings.Contains(m.msg, "press a again") {
		t.Fatalf("the first a must only arm and name the command: %q", m.msg)
	}
	press('j') // scrolling in between
	if m.chat.armed != "" {
		t.Fatal("another key must disarm the approval")
	}
	press('a')
	if m.chat.armed != "a:r1" {
		t.Fatal("a fresh a arms again")
	}
	m.chat.armedAt = m.chat.armedAt.Add(-11 * time.Second)
	press('a')
	if m.chat.armed != "a:r1" {
		t.Fatal("a stale arm (over 10 s) only re-arms")
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "r1.json")); err == nil {
		t.Fatal("nothing may be decided before a confirmed approval")
	}
}

// An edit is a diff card, open while short; a read is one quiet line; their call cards are not repeated.
func TestFileToolsRenderAsDiffAndReadCards(t *testing.T) {
	m := chatFixture(t)
	m.height = 40
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g1", Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"function": map[string]any{
			"name": "edit", "arguments": `{"path":"src/parse.c","old":"n > len","new":"n >= len"}`}}}}},
		{Seq: 2, Goal: "g1", Kind: "tool_result", Data: map[string]any{"name": "edit", "result": map[string]any{
			"path": "src/parse.c", "added": float64(1), "removed": float64(1),
			"diff": "@@ -1,3 +1,3 @@\n int f(void) {\n-    if (n > len) return -1;\n+    if (n >= len) return -1;"}}},
		{Seq: 3, Goal: "g1", Kind: "tool_result", Data: map[string]any{"name": "read", "result": map[string]any{
			"path": "src/parse.c", "shown": "1-40", "text": "secret body"}}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	for _, want := range []string{g.edit + " edit  src/parse.c  ·  +1 −1", "-    if (n > len) return -1;", "+    if (n >= len) return -1;", "read  src/parse.c  ·  lines 1-40"} {
		if !strings.Contains(v, want) {
			t.Fatalf("missing %q:\n%s", want, v)
		}
	}
	if strings.Contains(v, "@@") || strings.Contains(v, "secret body") || strings.Count(v, "src/parse.c") != 2 {
		t.Fatalf("hunk headers, read bodies or a repeated call card leaked:\n%s", v)
	}
}

// The selected goal's newest todo list is pinned above the composer while the goal is open; the frame still fits.
func TestTodoIsPinnedAboveTheComposer(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 24
	todo := func(items ...map[string]any) residentEvent {
		raw := make([]any, len(items))
		for i, it := range items {
			raw[i] = it
		}
		return residentEvent{Seq: 9, Goal: "original", Kind: "todo", Data: map[string]any{"items": raw}}
	}
	m.residentEvents = append(m.residentEvents,
		todo(map[string]any{"text": "stale", "state": "pending"}),
		todo(map[string]any{"text": "find the test", "state": "done"}, map[string]any{"text": "fix the bounds", "state": "in_progress"},
			map[string]any{"text": "run the gate", "state": "pending"}))
	m.syncChat()
	frame := ansi.Strip(m.render())
	if !strings.Contains(frame, "Task 2 of 3 ●◐○  fix the bounds") || strings.Contains(frame, "stale") || !strings.Contains(frame, "Now") {
		t.Fatalf("the current task and what is happening now are pinned:\n%s", frame)
	}
	m.chat.group.running, m.chat.group.runningSince = "make test", float64(time.Now().Add(-14*time.Second).Unix())
	if f := ansi.Strip(m.render()); !strings.Contains(f, "Now   running 1") || !strings.Contains(f, "s  make test") {
		t.Fatalf("a running command is named with how long it has run:\n%s", f)
	}
	m.chat.group.running = ""
	if rows := strings.Split(frame, "\n"); len(rows) > m.height {
		t.Fatalf("the pinned line pushed the frame past the terminal: %d rows", len(rows))
	}
	m.resident.Goals[0].State = "completed"
	if strings.Contains(ansi.Strip(m.render()), "fix the bounds") {
		t.Fatal("a finished goal's list must not stay pinned")
	}
}

// A big output moved to a file says so on its card, with the whole size and the path.
func TestSpilledOutputCardNamesTheFile(t *testing.T) {
	m := chatFixture(t)
	m.width = 120
	m.residentEvents = []residentEvent{{Seq: 1, Goal: "g1", Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{
		"output": "head\n… [90,000 characters omitted] …\ntail", "exit_code": float64(0),
		"saved": []any{map[string]any{"field": "output", "path": "outputs/20261115-terminal-abc-output.txt", "lines": float64(6000)}}}}}}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	if v := ansi.Strip(m.chat.content); !strings.Contains(v, "exit 0 · 6000 lines → outputs/20261115-terminal-abc-output.txt") {
		t.Fatalf("the card does not name the file:\n%s", v)
	}
}

// A need_you report stands out as a turn; fyi, scheduling and waking are quiet notes.
func TestReportsAndScheduleNotes(t *testing.T) {
	m := chatFixture(t)
	m.height = 40
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g1", Kind: "report", Data: map[string]any{"level": "need_you", "text": "the disk is full"}},
		{Seq: 2, Goal: "g1", Kind: "report", Data: map[string]any{"level": "fyi", "text": "halfway there"}},
		{Seq: 3, Goal: "g1", Kind: "scheduled", Data: map[string]any{"wake_at": float64(1794780000), "note": "check the build"}},
		{Seq: 4, Goal: "g1", Kind: "woke", Data: map[string]any{"note": "check the build"}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	for _, want := range []string{"! needs you", "the disk is full", "fyi: halfway there", "parked until", "check the build", "woke up"} {
		if !strings.Contains(v, want) {
			t.Fatalf("missing %q:\n%s", want, v)
		}
	}
}

// The gate and the judge show their verdicts: a pass is a quiet line, a refusal stays open with its reason.
func TestGateAndJudgeVerdicts(t *testing.T) {
	m := chatFixture(t)
	m.height = 40
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "g1", Kind: "gate", Data: map[string]any{"command": "make test", "passed": false, "output": "exit 2\n1 failed"}},
		{Seq: 2, Goal: "g1", Kind: "judged", Data: map[string]any{"verdict": "not_yet", "reason": "no test output shows the fix"}},
		{Seq: 3, Goal: "g1", Kind: "judged", Data: map[string]any{"verdict": "met", "reason": "41 passed"}},
	}
	m.chat.rendered, m.chat.content = 0, ""
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	for _, want := range []string{"gate: make test  ·  failed", "1 failed", "judge: not yet", "no test output shows the fix", "judge: met  ·  41 passed"} {
		if !strings.Contains(v, want) {
			t.Fatalf("missing %q:\n%s", want, v)
		}
	}
}

func TestModelsOverlayListsTheCatalogAndPicks(t *testing.T) {
	m := chatFixture(t)
	next, _ := m.chatSlash("/model")
	m = next.(emberModel)
	if !m.chat.models.open || !m.chat.models.loading {
		t.Fatalf("/model alone opens the overlay and asks for the catalog: %+v", m.chat.models)
	}
	ctx, big, out := 202752, 1000000, 16384
	in, outp := 0.6, 2.2
	m = m.modelsLoaded(modelsLoadedMsg{
		models: []catalogModel{
			{ID: "claude-opus-5-5", Provider: "anthropic", Context: &big, Efforts: []string{"low", "medium", "high", "xhigh", "max"}, Thinking: "adaptive", Vision: true},
			{ID: "zai-org/GLM-5.2", Provider: "deepinfra", Context: &ctx, MaxOutput: &out, Efforts: []string{"none", "low", "medium", "high"}, Prices: &catalogPrices{Input: &in, Output: &outp}},
		},
		current: modelCurrent{Model: "zai-org/GLM-5.2", Provider: "deepinfra"},
	})
	if m.chat.models.sel != 1 {
		t.Fatalf("the selection starts on the current model, got %d", m.chat.models.sel)
	}
	view := ansi.Strip(m.renderChat())
	for _, want := range []string{"models  2", "now zai-org/GLM-5.2", "claude-opus-5-5", "anthropic · 1M · effort low/medium/high/xhigh/max · vision",
		"deepinfra · 203K · effort none/low/medium/high · $0.6/$2.2"} {
		if !strings.Contains(view, want) {
			t.Fatalf("overlay misses %q:\n%s", want, view)
		}
	}
	next, _ = m.modelsKey(tea.KeyPressMsg{Code: tea.KeyUp})
	m = next.(emberModel)
	next, cmd := m.modelsKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if m.chat.models.open || cmd == nil || !strings.Contains(m.msg, "switching to claude-opus-5-5") {
		t.Fatalf("Enter picks the selected model and closes: open=%v msg=%q", m.chat.models.open, m.msg)
	}
	next, cmd = m.chatSlash("/effort high")
	if cmd == nil {
		t.Fatal("/effort LEVEL runs the effort command")
	}
}

func TestRecapCardAndStudioOverlay(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	m := chatFixture(t)
	m = m.receiveRecap(recapMsg{agent: m.agentName, text: "While you were away (1 stretch):\n- 03:10 free time: Sketched a tide pool.", now: 1794790000})
	view := ansi.Strip(m.renderChat())
	if !strings.Contains(view, "While you were away (1 stretch)") || !strings.Contains(view, "Sketched a tide pool.") {
		t.Fatalf("the recap card is missing:\n%s", view)
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.recap != "" || loadSeen(m.agentName) != 1794790000 {
		t.Fatalf("Esc dismisses the card and remembers when you looked: recap=%q seen=%v", m.chat.recap, loadSeen(m.agentName))
	}
	next, _ = m.chatSlash("/studio")
	m = next.(emberModel)
	if !m.chat.studio.open {
		t.Fatal("/studio opens the overlay")
	}
	m.chat.studio.loading = false
	m.chat.studio.list = []studioEntry{{Path: "/h/studio/for-you/sketch.svg", Name: "sketch.svg", Mtime: 1794780000, group: "for you"},
		{Path: "/h/journal/2026-11-15.md", Name: "2026-11-15.md", Head: "Tide pools", Mtime: 1794770000, group: "journal"}}
	view = ansi.Strip(m.renderChat())
	for _, want := range []string{"studio  2", "for you", "sketch.svg", "journal", "Tide pools"} {
		if !strings.Contains(view, want) {
			t.Fatalf("studio misses %q:\n%s", want, view)
		}
	}
	next, cmd := m.studioKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if cmd == nil || m.chat.studio.reading != "/h/studio/for-you/sketch.svg" {
		t.Fatal("Enter reads the selected entry")
	}
}

func TestStatusLineShowsSpendOnAPaidBrain(t *testing.T) {
	m := chatFixture(t)
	if strings.Contains(ansi.Strip(m.renderChat()), "today") {
		t.Fatal("a local brain shows no spend")
	}
	m.resident.SpendToday = 1.25
	m.resident.SpendGoals = map[string]float64{m.chat.target: 0.4}
	view := ansi.Strip(m.renderChat())
	if !strings.Contains(view, "0.40 · 1.25 today") {
		t.Fatalf("the goal's spend and today's are in the status line:\n%s", view)
	}
}

func TestNotificationsOncePerItemAndNotForWhatWasAlreadyThere(t *testing.T) {
	var got []string
	old := notifyRun
	notifyRun = func(u, title, body string) { got = append(got, u+"|"+title+"|"+body) }
	defer func() { notifyRun = old }()
	m := chatFixture(t)
	m.reqs = []emberReq{{ID: "old", Kind: "question", Text: "already pending"}}
	m.notifyRequests()
	if len(got) != 0 {
		t.Fatalf("what is pending at start is not announced: %v", got)
	}
	m.reqs = append(m.reqs, emberReq{ID: "new", Cmd: "apt-get install jq"})
	m.notifyRequests()
	m.notifyRequests()
	if len(got) != 1 || !strings.Contains(got[0], "critical|Caminus: root approval needed|apt-get install jq") {
		t.Fatalf("one notification for the new root request: %v", got)
	}
	m.notifyResident(nil) // first pass: silent
	m.resident.Goals = append(m.resident.Goals, residentGoal{ID: "g2", State: "completed", Summary: "Report written."})
	m.notifyResident([]residentEvent{{Seq: 99, Kind: "report", Data: map[string]any{"level": "need_you", "text": "the disk is full"}}})
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "needs you|the disk is full") || !strings.Contains(joined, "goal done|Report written.") {
		t.Fatalf("need_you and a finished goal are announced: %v", got)
	}
}

func TestComposerExtras(t *testing.T) {
	m := chatFixture(t)
	// Ctrl+R: earlier prompts containing the draft, newest first
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 900, Kind: "goal_queued", Data: map[string]any{"prompt": "fix the parser tests"}},
		residentEvent{Seq: 901, Kind: "user_message", Data: map[string]any{"text": "also run the parser benchmark"}})
	m.chat.compose.SetValue("parser")
	next, _, ok := m.composerKey(tea.KeyPressMsg{Code: 'r', Mod: tea.ModCtrl})
	if !ok || next.chat.compose.Value() != "also run the parser benchmark" {
		t.Fatalf("Ctrl+R gives the newest matching prompt: %q", next.chat.compose.Value())
	}
	next, _, _ = next.composerKey(tea.KeyPressMsg{Code: 'r', Mod: tea.ModCtrl})
	if next.chat.compose.Value() != "fix the parser tests" {
		t.Fatalf("a second Ctrl+R goes further back: %q", next.chat.compose.Value())
	}
	// Ctrl+X Ctrl+E opens the editor (here: true, which returns at once)
	t.Setenv("VISUAL", "true")
	n2, _, _ := m.composerKey(tea.KeyPressMsg{Code: 'x', Mod: tea.ModCtrl})
	if _, cmd, ok := n2.composerKey(tea.KeyPressMsg{Code: 'e', Mod: tea.ModCtrl}); !ok || cmd == nil {
		t.Fatal("Ctrl+X then Ctrl+E opens the editor")
	}
	if _, _, ok := m.composerKey(tea.KeyPressMsg{Code: 'e', Mod: tea.ModCtrl}); ok {
		t.Fatal("Ctrl+E alone still folds the cards (not the editor)")
	}
	// !cmd: the output lands in the draft, fenced
	m = m.bangDone(bangDoneMsg{cmd: "df -h ~", output: "/dev/x 100G 40G"})
	if v := m.chat.compose.Value(); !strings.Contains(v, "```\n$ df -h ~\n/dev/x 100G 40G\n```") {
		t.Fatalf("the command and its output are in the draft: %q", v)
	}
	// Ctrl+O: every line of the open cards
	next, _, _ = m.composerKey(tea.KeyPressMsg{Code: 'o', Mod: tea.ModCtrl})
	if !next.chat.full {
		t.Fatal("Ctrl+O shows every line")
	}
	// custom commands: a markdown file, $ARGUMENTS replaced, offered by autocomplete
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	_ = os.MkdirAll(filepath.Join(dir, "caminus", "commands"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "caminus", "commands", "review.md"), []byte("---\ndescription: review\n---\nReview $ARGUMENTS for bugs."), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "caminus", "commands", "goals.md"), []byte("shadowing a built-in is ignored"), 0o644)
	if got := customNames(); fmt.Sprint(got) != "[/review]" {
		t.Fatalf("custom commands: %v", got)
	}
	if got := slashMatches("/rev"); fmt.Sprint(got) != "[/review]" {
		t.Fatalf("autocomplete offers the custom command: %v", got)
	}
	if body, _ := expandCustom(customCommands()["/review"], "panel/shell.go"); body != "Review panel/shell.go for bugs." {
		t.Fatalf("expanded: %q", body)
	}
}

func TestSteeringViews(t *testing.T) {
	m := chatFixture(t)
	// /diff opens a coloured view
	m.chat.view = textView{open: true, title: "DIFF", colourDiff: true, lines: []string{"### /work/repo", "+added line", "-removed line", "@@ -1 +1 @@"}}
	view := ansi.Strip(m.renderChat())
	for _, want := range []string{"diff", "+added line", "-removed line"} {
		if !strings.Contains(view, want) {
			t.Fatalf("diff view misses %q:\n%s", want, view)
		}
	}
	next, _ := m.textKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.view.open {
		t.Fatal("Esc closes the view")
	}
	// /rewind takes two presses
	m.chat.rewind = rewindState{open: true, goal: "goal1234abcd", steps: []rewindStep{{Step: 2, Label: "edit a.py"}, {Step: 5, Label: "write b.py"}}, sel: 1}
	if v := ansi.Strip(m.renderChat()); !strings.Contains(v, "undo from step 5 on  (write b.py)") || !strings.Contains(v, "side effects are not undone") {
		t.Fatalf("rewind picker:\n%s", v)
	}
	next, cmd := m.rewindKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if cmd != nil || !strings.Contains(m.msg, "press Enter again") {
		t.Fatalf("the first Enter only arms it: %q", m.msg)
	}
	next, cmd = m.rewindKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	if cmd == nil || next.(emberModel).chat.rewind.open {
		t.Fatal("the second Enter rewinds and closes")
	}
	// a waiting plan is a card above the composer
	m = chatFixture(t)
	m.resident.Goals = append(m.resident.Goals, residentGoal{ID: "plan5678abcd", State: "waiting_plan", PlanText: "1. read the parser\n2. add the tests"})
	if v := ansi.Strip(m.renderChat()); !strings.Contains(v, "PLAN · goal") || !strings.Contains(v, "2. add the tests") || !strings.Contains(v, "/plan approve") {
		t.Fatalf("plan card:\n%s", v)
	}
	if _, cmd, ok := m.steerSlash("/plan", "approve"); !ok || cmd == nil {
		t.Fatal("/plan approve decides the waiting plan")
	}
	// ↑ on an empty composer pulls back; Esc while a reply streams interrupts
	m.chat.compose.SetValue("")
	m.chat.newGoal = false
	m.chat.target = "plan5678abcd"
	if _, cmd := m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyUp}); cmd == nil {
		t.Fatal("↑ on an empty composer asks to pull back the last message")
	}
	text := "edited again"
	next, _ = m.Update(unsendMsg{text: &text})
	if next.(emberModel).chat.compose.Value() != "edited again" {
		t.Fatal("the pulled-back message is in the composer")
	}
	m.chat.live.active = true
	if _, cmd := m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyEscape}); cmd == nil {
		t.Fatal("Esc while a reply is written interrupts the turn")
	}
}

func TestSeveralRootRequestsShareOneCard(t *testing.T) {
	m := chatFixture(t)
	bridge := t.TempDir()
	writeReq(t, bridge, "r1", "apt-get install -y jq", 1)
	writeReq(t, bridge, "r2", "systemctl restart foo", 2)
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
	m.refreshLocal()
	v := ansi.Strip(m.renderChat())
	if !strings.Contains(v, "2 ROOT REQUESTS WAITING") || !strings.Contains(v, "1  apt-get install -y jq") || !strings.Contains(v, "2  systemctl restart foo") {
		t.Fatalf("one card lists both:\n%s", v)
	}
	if m.reqs[m.pendingRoot()].ID != "r2" {
		t.Fatal("the keys act on the newest by default")
	}
	m.chat.compose.Blur()
	next, _ := m.chatKey(tea.KeyPressMsg{Code: '[', Text: "["})
	m = next.(emberModel)
	if m.reqs[m.pendingRoot()].ID != "r1" {
		t.Fatal("[ moves to the older request")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: 'd', Text: "d"})
	m = next.(emberModel)
	if data, err := os.ReadFile(filepath.Join(bridge, "res", "r1.json")); err != nil || !strings.Contains(string(data), `"denied"`) {
		t.Fatalf("d denies the highlighted one: %s %v", data, err)
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "r2.json")); err == nil {
		t.Fatal("the other request is untouched")
	}
	if strings.Contains(ansi.Strip(m.renderChat()), "ROOT REQUESTS WAITING") {
		t.Fatal("one request left: no shared card")
	}
}

func TestTheLiveReplyHasTheWholeWidth(t *testing.T) {
	old := g
	defer func() { g = old }()
	g = glyphTiers["unicode"]
	m := chatFixture(t)
	m.chat.live = residentLive{active: true, turn: "t1", text: "writing the report now"}
	a := m.liveBlock(100)
	// Ember is drawn only on a new goal's welcome screen now (the user's call): the live text has the whole width
	if strings.Contains(ansi.Strip(a), "▀") || !strings.Contains(ansi.Strip(a), "writing the report now") {
		t.Fatalf("the live text, no Ember beside it:\n%s", ansi.Strip(a))
	}
	if n := m.liveBlock(40); strings.Contains(ansi.Strip(n), "▀") {
		t.Fatalf("a narrow box leaves Ember out:\n%s", ansi.Strip(n))
	}
}

func TestGrantTakesTwoPressesAndAGrantedRepeatOne(t *testing.T) {
	m := chatFixture(t)
	bridge := t.TempDir()
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
	writeReq(t, bridge, "r1", "systemctl restart foo", 1)
	m.refreshLocal()
	m.chat.compose.Blur()
	press := func(r rune) tea.Cmd {
		next, cmd := m.chatKey(tea.KeyPressMsg{Code: r, Text: string(r)})
		m = next.(emberModel)
		return cmd
	}
	if press('g'); m.chat.armed != "g:r1" || !strings.Contains(m.msg, "grant for this goal") {
		t.Fatalf("the first g only arms the grant: %q", m.msg)
	}
	if press('a'); m.chat.armed != "a:r1" {
		t.Fatal("a after g does not run the grant: it arms a plain approval")
	}
	// a request the agent marks granted: honoured only when your own audit holds the grant, within 12 hours
	r := emberReq{ID: "r2", Goal: "goal1", Cmd: "systemctl restart foo", Granted: true}
	if grantRecorded(m.auditPath, r, time.Now()) {
		t.Fatal("the agent's flag alone is not a grant")
	}
	_ = appendAudit(m.auditPath, map[string]any{"event": "root_result", "decision": "approved", "cmd": r.Cmd,
		"goal": "goal1", "granted": true})
	if !grantRecorded(m.auditPath, r, time.Now()) {
		t.Fatal("your audited grant for this goal and command is honoured")
	}
	if grantRecorded(m.auditPath, emberReq{ID: "r3", Goal: "goal2", Cmd: r.Cmd, Granted: true}, time.Now()) {
		t.Fatal("another goal's grant is not honoured")
	}
	if grantRecorded(m.auditPath, r, time.Now().Add(13*time.Hour)) {
		t.Fatal("a grant ends after 12 hours on the owner's side")
	}
}

func TestTheBoundedHistoryTrimKeepsTheDrawnPosition(t *testing.T) {
	m := chatFixture(t)
	evs := make([]residentEvent, 0, maxResidentEvents)
	for i := 1; i < maxResidentEvents; i++ { // one short of the bound: all of it is drawn
		evs = append(evs, residentEvent{Seq: 10000 + i, Kind: "note", Data: map[string]any{}})
	}
	m.receiveResident(residentMsg{agent: m.agentName, status: m.resident, events: evs})
	if m.chat.rendered != len(m.residentEvents) {
		t.Fatalf("everything drawn: %d of %d", m.chat.rendered, len(m.residentEvents))
	}
	// crossing the bound trims the oldest half; drawing must not slice past the end (it panicked)
	m.receiveResident(residentMsg{agent: m.agentName, status: m.resident, events: []residentEvent{
		{Seq: 10000 + maxResidentEvents, Kind: "note", Data: map[string]any{}},
		{Seq: 10000 + maxResidentEvents + 1, Kind: "note", Data: map[string]any{}}}})
	if m.chat.rendered != len(m.residentEvents) || len(m.residentEvents) > maxResidentEvents {
		t.Fatalf("after the trim: rendered %d, events %d", m.chat.rendered, len(m.residentEvents))
	}
}

func TestCompactTranscriptFoldsToolWorkIntoOneLine(t *testing.T) {
	m := chatFixture(t)
	m.chat.cards, m.chat.content, m.chat.rendered = false, "", 0
	call := func(name, args string) map[string]any {
		return map[string]any{"id": "c", "type": "function", "function": map[string]any{"name": name, "arguments": args}}
	}
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 500, Kind: "assistant", Data: map[string]any{"content": "Checking the CVEs now.", "reasoning": "long thoughts",
			"tool_calls": []any{call("terminal", `{"command":"ls cve/"}`), call("read", `{"path":"a.md"}`)}}},
		residentEvent{Seq: 501, Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"running": false, "output": "x"}}},
		residentEvent{Seq: 502, Kind: "gate", Data: map[string]any{"command": "test -s PRIVILEGED-STAGE1.md && grep -q Step1", "passed": false}},
		residentEvent{Seq: 503, Kind: "assistant", Data: map[string]any{"tool_calls": []any{call("terminal", `{"command":"jq . n.json"}`), call("terminal", `{"session":"abc"}`)}}},
		residentEvent{Seq: 504, Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"error": "boom"}}},
		residentEvent{Seq: 505, Kind: "gate", Data: map[string]any{"command": "test -s PRIVILEGED-STAGE1.md", "passed": false}},
	)
	m.syncChat()
	v := ansi.Strip(m.chat.transcript.View())
	if !strings.Contains(v, "● Checking the CVEs now.") || strings.Contains(v, "agent\n") {
		t.Fatalf("the agent's words stay whole:\n%s", v)
	}
	if !strings.Contains(v, "Ran 2 commands, read 1 file · 1 failed · gate fails") {
		t.Fatalf("the tool work is one line (a poll is not a command):\n%s", v)
	}
	for _, clutter := range []string{"PRIVILEGED-STAGE1", "output  terminal", "thinking ·", "jq . n.json"} {
		if strings.Contains(v, clutter) {
			t.Fatalf("compact hides %q:\n%s", clutter, v)
		}
	}
	m.residentEvents = append(m.residentEvents, residentEvent{Seq: 506, Kind: "assistant", Data: map[string]any{"content": "Two CVEs are patched."}})
	m.syncChat()
	if v = ansi.Strip(m.chat.transcript.View()); !strings.Contains(v, "Two CVEs are patched.") || !strings.Contains(v, "gate fails") {
		t.Fatalf("the next reply fixes the line above it:\n%s", v)
	}
}

func TestTheLatestThoughtsStayUp(t *testing.T) {
	old := g
	defer func() { g = old }()
	g = glyphTiers["unicode"]
	m := chatFixture(t)
	m.height = 44 // below about 22 body rows the idle box gives its rows to the transcript
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 600, Goal: "original", Kind: "assistant", Data: map[string]any{"reasoning": "first I check the CVE list"}},
		residentEvent{Seq: 601, Goal: "original", Kind: "assistant", Data: map[string]any{"reasoning": "then the kernel source for the missing return"}})
	idle := ansi.Strip(m.liveBlock(110))
	if strings.Contains(idle, "▀") || !strings.Contains(idle, "thoughts") || !strings.Contains(idle, "missing return") {
		t.Fatalf("idle, the latest thoughts stay up (no Ember beside them):\n%s", idle)
	}
	if strings.Index(idle, "first I check") > strings.Index(idle, "missing return") {
		t.Fatal("the thoughts run downward, newest last")
	}
	m.chat.live = residentLive{active: true, turn: "t", reasoning: "now drafting the report"}
	live := ansi.Strip(m.liveBlock(110))
	if !strings.Contains(live, "now drafting the report") || strings.Index(live, "missing return") > strings.Index(live, "now drafting") {
		t.Fatalf("live thoughts append below the past ones:\n%s", live)
	}
}

func TestARecoveredCutOffIsANoteNotAnError(t *testing.T) {
	m := chatFixture(t)
	m.chat.cards, m.chat.renderWidth = false, 140
	block := ansi.Strip(m.chat.eventBlock(residentEvent{Seq: 700, Kind: "model_truncated", Data: map[string]any{
		"error": "model output budget exhausted; increase --max-tokens and resume this goal", "recovered": true, "streak": 1.0}}))
	if strings.Contains(block, "error") || strings.Contains(block, "--max-tokens") || !strings.Contains(block, "retries in smaller pieces (1 of 3") {
		t.Fatalf("a handled cut-off is a quiet note:\n%s", block)
	}
}

func TestAWideTerminalPutsEmberTasksAndNowInTheSidePanel(t *testing.T) {
	old := g
	defer func() { g = old }()
	g = glyphTiers["unicode"]
	m := chatFixture(t)
	m.width, m.height = 220, 50
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 900, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "Checking the CVEs.", "reasoning": "first the NVD feed"}},
		residentEvent{Seq: 901, Goal: "original", Kind: "todo", Data: map[string]any{"items": []any{
			map[string]any{"text": "probe the API", "state": "done"}, map[string]any{"text": "write the CLI", "state": "in_progress"},
			map[string]any{"text": "the web UI", "state": "pending"}}}})
	m.residentEvents = append(m.residentEvents, residentEvent{Seq: 902, Goal: "original", Kind: "assistant",
		Data: map[string]any{"reasoning": "a\ttabbed\tthought"}})
	m.syncChat()
	m.chat.group.running, m.chat.group.runningSince = "make test", float64(time.Now().Unix())
	frame := ansi.Strip(m.render())
	rows := strings.Split(frame, "\n")
	if len(rows) > m.height {
		t.Fatalf("%d rows overflow %d", len(rows), m.height)
	}
	for _, r := range rows {
		if ansi.StringWidth(r) > m.width {
			t.Fatalf("row too wide (%d): %q", ansi.StringWidth(r), r)
		}
	}
	for _, want := range []string{"first the NVD feed", "tasks  1 of 3 done", "◐ write the CLI", "now", "make test", "Checking the CVEs."} {
		if !strings.Contains(frame, want) {
			t.Fatalf("missing %q:\n%s", want, frame)
		}
	}
	if strings.Contains(frame, "Task 2 of 3") || strings.Contains(frame, "\t") {
		t.Fatal("the line above the composer moves into the side panel; tabs are expanded")
	}
	m.resident.Goals[0].State = "completed"
	if f := ansi.Strip(m.render()); strings.Contains(f, "make test") {
		t.Fatal("a command of a goal that ended is not running now")
	}
	m.resident.Goals[0].State = "working"
	// the thoughts sit beside the transcript, on the same rows
	for _, r := range rows {
		if strings.Contains(r, "Checking the CVEs.") && !strings.Contains(r, "│") {
			continue
		}
	}
	m.width = 120
	if f := ansi.Strip(m.render()); strings.Contains(f, "Tasks  1 of 3") || !strings.Contains(f, "Task 2 of 3") {
		t.Fatalf("a narrow terminal keeps the line above the composer:\n%s", f)
	}
}

func TestEmbersEmptyNoteWrapsBesideEmber(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 200, 50
	box := ansi.Strip(m.emberBox(70, 8))
	for _, row := range strings.Split(box, "\n") {
		if ansi.StringWidth(row) > 70 {
			t.Fatalf("row wider than the box: %q", row)
		}
	}
	if !strings.Contains(box, "no thoughts yet") {
		t.Fatalf("the empty note:\n%s", box)
	}
}

func TestDragOverTheHistorySelectsAndCopiesItsText(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.chat.cards = false
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 950, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "First reply about the kernel."}},
		residentEvent{Seq: 951, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "Second reply with a path /etc/hosts."}})
	m.syncChat()
	m.chat.transcript.GotoTop()
	lines := strings.Split(ansi.Strip(m.chat.transcript.GetContent()), "\n")
	first, second := -1, -1
	for i, ln := range lines {
		if strings.Contains(ln, "First reply") {
			first = i
		}
		if strings.Contains(ln, "Second reply") {
			second = i
		}
	}
	if first < 0 || second < 0 {
		t.Fatalf("fixture: replies not in the transcript:\n%s", strings.Join(lines, "\n"))
	}
	top := lipgloss.Height(m.chatHead(m.width))
	y := func(line int) int { return top + line - m.chat.transcript.YOffset() }
	cell := func(line, word string) int { return ansi.StringWidth(line[:strings.Index(line, word)]) } // cells, not bytes
	x1 := cell(lines[first], "First")
	x2 := cell(lines[second], "hosts") + len("hosts") - 1
	next, _ := m.chatUpdate(tea.MouseClickMsg{X: x1, Y: y(first), Button: tea.MouseLeft})
	m = next.(emberModel)
	next, _ = m.chatUpdate(tea.MouseMotionMsg{X: x2, Y: y(second), Button: tea.MouseLeft})
	m = next.(emberModel)
	if !strings.Contains(m.highlightSelection(m.chat.transcript.View()), "\x1b[7m") {
		t.Fatal("the drag is highlighted while it goes on")
	}
	next, cmd := m.chatUpdate(tea.MouseReleaseMsg{X: x2, Y: y(second), Button: tea.MouseLeft})
	m = next.(emberModel)
	got := m.selectedText()
	if !strings.HasPrefix(got, "First reply about the kernel.") || !strings.HasSuffix(got, "Second reply with a path /etc/hosts") {
		t.Fatalf("selected text:\n%q", got)
	}
	if cmd == nil || !strings.Contains(m.msg, "copied") {
		t.Fatalf("the release copies (msg %q)", m.msg)
	}
	next, _ = m.chatUpdate(tea.MouseClickMsg{X: x1, Y: y(first), Button: tea.MouseLeft})
	m = next.(emberModel)
	next, _ = m.chatUpdate(tea.MouseReleaseMsg{X: x1, Y: y(first), Button: tea.MouseLeft})
	m = next.(emberModel)
	if m.chat.sel.shown || m.chat.sel.dragging {
		t.Fatal("a click without a drag clears the selection")
	}
}

func TestEmbersBoxShowsTheThoughtChainWithoutReasoning(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 200, 50
	m.residentEvents = append(m.residentEvents,
		residentEvent{Seq: 990, Goal: "original", Kind: "assistant", Data: map[string]any{
			"content": "Let me check the BL density first.",
			"tool_calls": []any{map[string]any{"id": "c1", "type": "function", "function": map[string]any{
				"name": "terminal", "arguments": `{"command":"python3 density.py xbl.img"}`}}}}},
		residentEvent{Seq: 991, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"output": "0.0256"}}})
	box := ansi.Strip(m.emberBox(90, 8))
	if !strings.Contains(box, "Let me check the BL density first.") || !strings.Contains(box, "→ terminal  python3 density.py xbl.img") {
		t.Fatalf("without reasoning, the words and the tool calls are the chain:\n%s", box)
	}
	if strings.Contains(box, "no thoughts yet") {
		t.Fatal("the note shows only before any reply")
	}
	// a goal with no replies yet keeps the latest chain from any goal: the box never empties
	m.resident.Goals = append(m.resident.Goals, residentGoal{ID: "fresh", State: "working", Prompt: "new"})
	m.resident.Goals[0].State = "completed"
	if box := ansi.Strip(m.emberBox(90, 8)); !strings.Contains(box, "BL density") {
		t.Fatalf("a fresh goal keeps the latest chain:\n%s", box)
	}
}

func TestSlashMenuMovesAndTakesTheHighlightedCommand(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 40
	m.chat.compose.Focus()
	m.chat.compose.SetValue("/st")
	m.syncChat()
	if !m.slashMenuOpen() || !strings.Contains(ansi.Strip(m.render()), "run this session's loop") {
		t.Fatal("/st opens the menu with each command's description")
	}
	first, matches, _ := m.slashPick()
	next, _ := m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	if pick, _, sel := m.slashPick(); sel != 1 || pick != matches[1] || pick == first {
		t.Fatalf("↓ moves the highlight: %q %d", pick, sel)
	}
	next, _ = m.chatDraftKey(tea.KeyPressMsg{Code: tea.KeyTab})
	m = next.(emberModel)
	if m.chat.compose.Value() != matches[1]+" " || m.slashMenuOpen() {
		t.Fatalf("Tab takes the highlighted command: %q", m.chat.compose.Value())
	}
	// Enter on a complete command runs it rather than completing it again
	m.chat.compose.SetValue("/stop")
	if _, done := m.slashMenuKey("enter"); done {
		t.Fatal("Enter on an exact command is left to submit")
	}
}

func TestUpDownRecallSentMessagesLikeAShell(t *testing.T) {
	m := chatFixture(t)
	m.chat.compose.Focus()
	for i, txt := range []string{"first message", "second message"} {
		m.receiveResident(residentMsg{agent: "resident", status: m.resident,
			events: []residentEvent{{Seq: 10 + i, Goal: "original", Kind: "user_message", Data: map[string]any{"text": txt}}}})
	}
	m.chat.newGoal, m.chat.target = true, "" // (no goal to pull a message back from: ↑ recalls at once)
	up := tea.KeyPressMsg{Code: tea.KeyUp}
	down := tea.KeyPressMsg{Code: tea.KeyDown}
	next, _ := m.chatDraftKey(up)
	m = next.(emberModel)
	if m.chat.compose.Value() != "second message" {
		t.Fatalf("↑ brings back the last sent message: %q", m.chat.compose.Value())
	}
	next, _ = m.chatDraftKey(up)
	m = next.(emberModel)
	if m.chat.compose.Value() != "first message" {
		t.Fatalf("↑ again goes further back: %q", m.chat.compose.Value())
	}
	next, _ = m.chatDraftKey(down)
	m = next.(emberModel)
	if m.chat.compose.Value() != "second message" {
		t.Fatalf("↓ comes forward: %q", m.chat.compose.Value())
	}
	next, _ = m.chatDraftKey(down)
	m = next.(emberModel)
	if m.chat.compose.Value() != "" {
		t.Fatalf("↓ past the newest leaves the draft empty: %q", m.chat.compose.Value())
	}
	// with a goal, ↑ asks the resident to pull back an unread message; "nothing waiting" recalls instead
	m.chat.newGoal, m.chat.target = false, "original"
	next, _ = m.Update(unsendMsg{})
	if next.(emberModel).chat.compose.Value() != "second message" {
		t.Fatalf("nothing to pull back: ↑ recalls: %q", next.(emberModel).chat.compose.Value())
	}
}

func TestBtwAnswerIsASectionAboveTheComposerNotAnOverlay(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 40
	m.receiveResident(residentMsg{agent: "resident", status: m.resident,
		events: []residentEvent{{Seq: 2, Goal: "original", Kind: "assistant", Data: map[string]any{"content": "the transcript stays in view"}}}})
	m.chat.btw = btwState{open: true, loading: true, question: "what is left?"}
	m.syncChat()
	if v := ansi.Strip(m.render()); !strings.Contains(v, "thinking…") || !strings.Contains(v, "the transcript stays in view") {
		t.Fatalf("while it thinks: the section and the transcript both show:\n%s", v)
	}
	next, _ := m.Update(btwMsg{question: "what is left?", answer: "**Two** steps remain."})
	m = next.(emberModel)
	v := m.render()
	plain := ansi.Strip(v)
	if !strings.Contains(plain, "Two steps remain") || strings.Contains(plain, "**") || !strings.Contains(plain, "by the way") ||
		!strings.Contains(plain, "the transcript stays in view") || strings.Contains(v, `\x1b`) {
		t.Fatalf("the answer renders as Markdown in its own section, the transcript still visible:\n%s", plain)
	}
	if lines := strings.Split(plain, "\n"); len(lines) > m.height {
		t.Fatalf("%d lines overflow a %d-line terminal", len(lines), m.height)
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	if next.(emberModel).chat.btw.open {
		t.Fatal("Esc on an empty draft closes the section")
	}
}

func TestThemeScreenPreviewsKeepsAndRestores(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	m := chatFixture(t)
	m.width, m.height = 140, 44
	before := themeRole("accent")
	next, _ := m.chatSlash("/theme")
	m = next.(emberModel)
	if !m.chat.themes.open || len(m.chat.themes.names) < 20 {
		t.Fatalf("the screen opens with every theme (%d)", len(m.chat.themes.names))
	}
	frame := m.render()
	plain := ansi.Strip(frame)
	for _, want := range []string{"themes  ", "add tests for the parser", "the gate fails", "a framed box", "esc browse", "↑↓ preview"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("the screen misses %q:\n%s", want, plain)
		}
	}
	if len(strings.Split(plain, "\n")) > m.height || strings.Contains(frame, `\x1b`) {
		t.Fatal("the screen overflows the terminal or prints escape codes as text")
	}
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	moved := m.chat.themes.names[m.chat.themes.sel]
	if _, err := os.Stat(themeSavePath()); err == nil {
		t.Fatal("previewing must not save")
	}
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if m.chat.themes.open || themeRole("accent") != before {
		t.Fatal("Esc closes the screen and puts the old theme back")
	}
	next, _ = m.chatSlash("/theme")
	m = next.(emberModel)
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	next, _ = m.themesKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	if b, _ := os.ReadFile(themeSavePath()); strings.TrimSpace(string(b)) != moved {
		t.Fatalf("Enter keeps the previewed theme: saved %q, want %q", b, moved)
	}
	_ = themeLoad("caminus")
}

func TestEveryBundledThemeSetsEveryRole(t *testing.T) {
	for _, n := range themeNames() {
		data, err := themeFS.ReadFile("themes/" + n + ".theme")
		if err != nil {
			continue // a user's own file
		}
		for role := range defaultTheme() {
			if !strings.Contains(string(data), "\n"+role+" = ") {
				t.Errorf("%s does not set %s", n, role)
			}
		}
	}
}

func TestCompactingShowsUntilItEndsAndTheStatusLineCountsIt(t *testing.T) {
	old := g
	defer func() { g = old }()
	g = glyphTiers["unicode"]
	m := chatFixture(t)
	m.width, m.height = 160, 40
	m.chat.target = "original"
	m.resident.Compactions = map[string]int{"original": 2}
	m.resident.OnlineS, m.resident.DaemonAlive = 3*3600+12*60, true
	hit := 0.874
	m.resident.CacheHit = &hit
	m.receiveResident(residentMsg{agent: "resident", status: m.resident,
		events: []residentEvent{{Seq: 2, Goal: "original", Kind: "compacting", Data: map[string]any{"part": 3.0, "parts": 5.0, "before_tokens": 398000.0}}}})
	m.syncChat()
	v := ansi.Strip(m.render())
	for _, want := range []string{"⇊ compacting", "part 3 of 5", "398K", "⇊ 2 compactions", "◉ online 3h12m", "↺ 87% cached"} {
		if !strings.Contains(v, want) {
			t.Fatalf("missing %q:\n%s", want, v)
		}
	}
	if !strings.Contains(v, "▓▓░░░") {
		t.Fatalf("the parts gauge (2 of 5 done before part 3):\n%s", v)
	}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident,
		events: []residentEvent{{Seq: 3, Goal: "original", Kind: "compacted", Data: map[string]any{"before": 300.0, "after": 11.0}}}})
	if v := ansi.Strip(m.render()); strings.Contains(v, "compacting ") {
		t.Fatalf("the indicator ends with the compaction:\n%s", v)
	}
	// a narrow terminal: the new bubbles go before the keys
	m.width = 80
	m.syncChat()
	if foot := ansi.Strip(m.chatFoot(80)); !strings.Contains(foot, "/help") {
		t.Fatalf("the keys stay on a narrow status line: %q", foot)
	}
}

func TestAWrittenThemeKeepsOnlyWhatValidates(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	ok := "Here you go!\n```\nname = Nord!!\n"
	for _, r := range []string{"text", "muted", "think", "dim", "accent", "ok", "warn", "action", "hot", "purple", "user", "teal", "chrome", "code_bg", "codeblock_bg"} {
		ok += r + " = #A1B2C3   (why)\n"
	}
	ok += "code_theme = rm -rf\nevil = \x1b[2J\nmain_fg = #ffffff\n```\n"
	name, file, err := themeFromAnswer(ok, "a cool nordic night")
	if err != nil {
		t.Fatal(err)
	}
	if name == "nord" || !strings.HasPrefix(name, "nord") || strings.ContainsAny(name, "! ") {
		t.Fatalf("the name is safe and never a built-in's: %q", name)
	}
	if !strings.Contains(file, "accent = #a1b2c3") || !strings.Contains(file, "code_theme = dracula") ||
		strings.Contains(file, "evil") || strings.Contains(file, "\x1b") || strings.Contains(file, "rm -rf") || strings.Contains(file, "main_fg") {
		t.Fatalf("only known roles, #rrggbb colours and a known code style survive:\n%s", file)
	}
	if roles, _ := parseTheme([]byte(file)); roles["accent"] != "#a1b2c3" {
		t.Fatalf("the file loads as a theme: %v", roles)
	}
	if _, _, err := themeFromAnswer("text = #ffffff\naccent = #00ff00", "half a theme"); err == nil || !strings.Contains(err.Error(), "missed") {
		t.Fatalf("an answer missing roles is refused: %v", err)
	}
}

func TestAMessageSitsOnTheComposerFrameAndNothingUnderTheStatusLine(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.msg = "Delivered to goal 579e81bc"
	top := strings.Split(ansi.Strip(m.chatComposer()), "\n")[0]
	if !strings.Contains(top, "Delivered to goal 579e81bc") {
		t.Fatalf("the message rides on the composer's top border: %q", top)
	}
	if foot := m.chatFoot(100); strings.Contains(foot, "\n") || strings.Contains(ansi.Strip(foot), "Delivered") {
		t.Fatalf("no line under the status line: %q", ansi.Strip(foot))
	}
	if w := lipgloss.Width(top); w != lipgloss.Width(strings.Split(ansi.Strip(m.chatComposer()), "\n")[1]) {
		t.Fatalf("the border keeps its width: %d", w)
	}
}

func TestTheModelPillCarriesItsLabsBadge(t *testing.T) {
	for model, want := range map[string]string{"claude-haiku-5-5": "✳", "zai-glm-5-3-flash": "Z", "mistral-large-latest": "▞",
		"codestral-latest": "▞", "deepseek-ai/DeepSeek-V4.1-Flash-1017": "◆", "k3-coder": "◈", "some-other-model": ansi.Strip(g.model)} {
		if got := ansi.Strip(labBadge(model)); got != want {
			t.Fatalf("%s: badge %q, want %q", model, got, want)
		}
	}
}

func TestAStoppedDaemonShowsStoppedNotItsGoalsLastState(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.resident.DaemonAlive = false
	if h := ansi.Strip(m.chatHead(100)); !strings.Contains(h, "stopped · /start") || strings.Contains(h, "working") {
		t.Fatalf("the header says stopped: %q", h)
	}
	if strings.Contains(m.windowTitle(), "working") {
		t.Fatalf("the tab title says stopped too: %q", m.windowTitle())
	}
}

func TestTheBadgeKeepsThePillWhole(t *testing.T) {
	saved := g
	defer func() { g = saved }()
	g = glyphTiers["nerd"]
	bar := footBar([]string{strings.TrimSpace(labBadge("claude-haiku-5-5") + " claude-haiku-5-5")}, nil, 80)
	bg := strings.TrimSuffix(strings.TrimPrefix(strings.SplitN(lipgloss.NewStyle().Background(cGray).Render("\x00"), "\x00", 2)[0], "\x1b["), "m")
	i := strings.Index(bar, "claude-haiku")
	before := bar[:i]
	last := max(strings.LastIndex(before, "\x1b[m"), strings.LastIndex(before, "\x1b[0m"))
	if last >= 0 && !strings.Contains(before[last:], bg) {
		t.Fatalf("after the badge's reset the pill's colours come back before the name: %q", before)
	}
}

func TestShellModeLikeClaudeCode(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	press := func(k string) {
		var code rune
		if len(k) == 1 {
			code = rune(k[0])
		}
		msg := tea.KeyPressMsg{Code: code, Text: k}
		switch k {
		case "backspace":
			msg = tea.KeyPressMsg{Code: tea.KeyBackspace}
		case "enter":
			msg = tea.KeyPressMsg{Code: tea.KeyEnter}
		}
		next, _ := m.chatKey(msg)
		m = next.(emberModel)
	}
	press("!")
	if !m.chat.shell || m.chat.compose.Value() != "" {
		t.Fatalf("! on an empty draft enters shell mode, not the text: shell=%v %q", m.chat.shell, m.chat.compose.Value())
	}
	if box := ansi.Strip(m.chatComposer()); !strings.Contains(box, "! ") || !strings.Contains(box, "shell · runs as the agent") {
		t.Fatalf("the prompt and the border say shell mode: %q", box)
	}
	press("backspace")
	if m.chat.shell {
		t.Fatal("Backspace on an empty line leaves shell mode")
	}
	press("a")
	press("!")
	if m.chat.shell || !strings.HasSuffix(m.chat.compose.Value(), "!") {
		t.Fatalf("a ! inside a draft is text: %q", m.chat.compose.Value())
	}
}

func TestAGithubHandoffIsACardAndTakesTwoPresses(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 100, 30
	if card := ansi.Strip(approvalCard("github", "open a pull request on a/b: me:x -> testing: t", "token is read-only", 90)); !strings.Contains(card, "GITHUB") ||
		!strings.Contains(card, "run it as you") || strings.Contains(card, "grant") {
		t.Fatalf("a hand-off card offers run and deny, never a grant: %q", card)
	}
	m.reqs = []emberReq{{ID: "abcdef0123456789", Kind: "github", Agent: m.agentName, Summary: "open a pull request on a/b"}}
	m.chat.compose.Blur() // (the card's keys work while browsing: "esc, then a a")
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: 'a', Text: "a"})
	m = next.(emberModel)
	if cmd != nil || !strings.Contains(m.msg, "run on GitHub as you") {
		t.Fatalf("the first a only arms it: %q", m.msg)
	}
	if next, _ = m.chatKey(tea.KeyPressMsg{Code: 'g', Text: "g"}); strings.Contains(next.(emberModel).msg, "grant") {
		t.Fatal("g never grants a hand-off")
	}
}
