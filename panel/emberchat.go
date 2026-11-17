// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"slices"
	"strings"
	"time"

	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// The chat owns editing, durable-history rendering and scroll position. Polls
// append events; they never replace the transcript or steal an unfinished draft.
type residentChat struct {
	ready, sending, newGoal, help, forceNew bool
	shell                                   bool      // shell mode: "!" on an empty draft; Enter runs the line as the agent
	expanded                                bool      // Ctrl+E: every card open (tool calls, output, thinking)
	armed                                   string    // the root request a first "a" armed; a second approves
	armedAt                                 time.Time // when it was armed (10 s to confirm)
	compose                                 textarea.Model
	transcript                              viewport.Model
	target, selectID                        string
	seq, rendered, renderWidth              int
	content                                 string
	markdown                                *glamour.TermRenderer
	calls                                   map[string]callInfo // each tool call's arguments by id: its result's card names what it acted on
	question                                *emberReq
	questionChoice                          int
	questionCustom                          bool
	questionArmed                           bool
	questionInput                           textarea.Model
	questionDeferred, questionError         string
	picker                                  bool
	pickerQuery                             string
	pickerSel                               int
	approvals                               bool // the Approvals overlay is over the conversation
	approvalSel                             int  // the row it has selected
	approvalDetail                          bool // Enter opened the selected request in full
	goals, goalsDetail                      bool // the Goals overlay, and its selected goal opened in full
	goalsSel                                int
	models                                  modelsState   // the Models overlay (/model)
	identity                                identityState // the Identity screen (/identity): the agent's GitHub account
	advisor                                 advisorState  // the Advisor screen (/advisor)
	studio                                  studioState   // the Studio overlay (/studio)
	recap                                   string        // the "while you were away" card, until Esc
	chordAt                                 time.Time     // Ctrl+X pressed: Ctrl+E within 2 s opens $EDITOR
	histQuery                               string        // Ctrl+R: the draft it searches for
	histIdx                                 int           // Ctrl+R: how far back it is
	full                                    bool          // Ctrl+O: cards show every line
	rootSel                                 string        // the root request the decision keys act on (shared card)
	group                                   toolGroup     // compact view: the tool work since the last drawn block
	cards                                   bool          // Ctrl+E: every tool card instead of the compact line
	sel                                     chatSelection // select to copy (envselect.go)
	view                                    textView      // /btw's answer, /diff
	rewind                                  rewindState   // the /rewind step picker
	recapNow                                float64       // when the recap was taken: saved as seen on dismiss
	recapAsked                              int           // recapNotAsked, recapAtOpen or recapOnDemand
	page, pageTop                           int           // an old page shown as an overlay (-1: none), and its scroll
	slashIdx                                int
	slashSel                                int         // the slash menu's highlighted row
	btw                                     btwState    // the /btw side answer, in its own section above the composer
	themes                                  themeScreen // /theme: the theme screen
	recallIdx                               int         // Up/Down history: 1 = your newest sent message; 0 = not recalling
	recallText                              string      // what the recall put in the draft (an edit ends the recall)
	slashFor                                string      // the draft slashSel belongs to (a new draft starts at the top)
	slashToken, slashText                   string
	pathIdx                                 int
	pathPrefix, pathPick                    string
	live                                    residentLive
	stream                                  *streamState
	streamStarting                          bool      // a reader is being started: the tick starts no second one
	streamRetry                             time.Time // a reader that ended is restarted no sooner than this
}

// residentLive is the reply being generated: shown under the durable transcript and replaced by it when the
// turn lands. Nothing here is durable; the transcript stays the record. hold keeps the window where the reader
// put it while more text arrives, so a long reply can be read at its own pace.
type residentLive struct {
	turn, text, reasoning string
	tools                 []string
	active, hold          bool
	top                   int
	wrapText, wrapThink   *wrapCache // finished lines wrapped once per turn (body)
}

// wrapCache keeps the wrapped, styled lines of a growing text's finished part (everything up to its last newline),
// so a snapshot wraps only what arrived since the last one and the partial line at the end. A streamed reply only
// grows within a turn; anything else (a new width, a shorter text) starts over.
type wrapCache struct {
	width, done int
	lines       []string
}

func (c *wrapCache) wrap(s string, width int, style func(string) string) []string {
	cut := strings.LastIndexByte(s, '\n') + 1
	if c.width != width || cut < c.done {
		c.width, c.done, c.lines = width, 0, nil
	}
	if cut > c.done {
		for _, ln := range strings.Split(ansi.Hardwrap(strings.TrimSuffix(s[c.done:cut], "\n"), width, true), "\n") {
			c.lines = append(c.lines, style(ln))
		}
		c.done = cut
	}
	out := slices.Clip(c.lines)
	for _, ln := range strings.Split(ansi.Hardwrap(s[cut:], width, true), "\n") {
		out = append(out, style(ln))
	}
	return out
}

// highlightLive colors the inline code spans of a streaming line, so live text reads like the rendered
// transcript: `code` in the frost accent, everything else as written. Plain text otherwise - this runs on
// every snapshot, so it stays a single pass with no parsing.
func highlightLive(line string) string {
	var b strings.Builder
	for {
		open := strings.IndexByte(line, '`')
		if open < 0 {
			b.WriteString(line)
			return b.String()
		}
		shut := strings.IndexByte(line[open+1:], '`')
		if shut < 0 {
			b.WriteString(line)
			return b.String()
		}
		b.WriteString(line[:open])
		b.WriteString(emberCode.Render(line[open+1 : open+1+shut]))
		line = line[open+1+shut+1:]
	}
}

// appendText adds a streamed delta to the reply (the text only ever grows within a turn).
func (l residentLive) appendText(delta string) residentLive { l.text += delta; return l }

// body is the live reply as wrapped lines: its reasoning, the tools it started, then the text.
func (l residentLive) body(width int) []string {
	var out []string
	if l.reasoning != "" {
		if l.wrapThink == nil {
			l.wrapThink = &wrapCache{} // (a fixture without one; the live handler gives each turn its own)
		}
		out = append(out, l.wrapThink.wrap(l.reasoning, width, func(s string) string { return emberThink.Render(s) })...)
	}
	for _, name := range l.tools {
		out = append(out, emberDim.Render(g.collapsed+" ")+emberTool.Render(g.tool+" "+residentText(name))+emberDim.Render(" …"))
	}
	if l.text != "" {
		if l.wrapText == nil {
			l.wrapText = &wrapCache{}
		}
		out = append(out, l.wrapText.wrap(l.text, width, highlightLive)...)
	}
	if len(out) == 0 {
		out = []string{emberDim.Render("generating…")}
	}
	return out
}

type residentComposeResult struct {
	agent, id, err string
	note           string // the resident's remark on delivery (a stopped agent: it starts the goal at /start)
	epoch          uint64
}

func (m *emberModel) initChat() {
	if m.chat.ready {
		return
	}
	c := residentChat{ready: true, newGoal: true, page: -1}
	c.compose = textarea.New()
	// One prompt mark on the first line, an equal indent under it: the editor asks per display line, so a wrapped
	// draft keeps the column its text starts in.
	c.compose.SetPromptFunc(2, promptMark)
	c.compose.ShowLineNumbers = false
	c.compose.CharLimit = 0
	c.compose.MaxHeight = 0 // long pasted goals scroll inside the visible editor
	c.compose.SetHeight(3)
	c.compose.Focus()
	c.transcript = viewport.New()
	// Wrap once when events or terminal width change. Viewport soft wrapping
	// recomputes ANSI widths across the entire history even for AtBottom/View,
	// turning every editor keystroke and cursor blink into a full transcript scan.
	c.transcript.SoftWrap = false
	m.chat = c
	m.chatStyles()
}

// promptMark is the composer's prompt: the glyph on the first line, nothing on the lines under it (the editor pads
// them to the same width, so the text stays in one column).
func promptMark(info textarea.PromptInfo) string {
	if info.LineNumber == 0 {
		return g.user + " "
	}
	return ""
}

// shellMark is the prompt in shell mode.
func shellMark(info textarea.PromptInfo) string {
	if info.LineNumber == 0 {
		return "! "
	}
	return ""
}

// setShell switches shell mode: the prompt, its colour and the placeholder say which mode the draft is in.
func (m *emberModel) setShell(on bool) {
	m.chat.shell = on
	if on {
		m.chat.compose.SetPromptFunc(2, shellMark)
	} else {
		m.chat.compose.SetPromptFunc(2, promptMark)
	}
	m.chatStyles()
	m.chat.compose.Placeholder = m.composerHint()
}

// chatStyles gives the composer the palette's prompt and placeholder styles. A style keeps the colour it was built
// with, so a theme switch calls this again.
func (m *emberModel) chatStyles() {
	s := m.chat.compose.Styles()
	s.Focused.Prompt, s.Blurred.Prompt = emberBusy, emberBusy
	if m.chat.shell {
		s.Focused.Prompt, s.Blurred.Prompt = emberShell, emberShell
	}
	s.Focused.Placeholder, s.Blurred.Placeholder = emberDim, emberDim
	// the editor's default cursor line has a grey background: the placeholder sat under a highlight bar
	s.Focused.CursorLine, s.Blurred.CursorLine = lipgloss.NewStyle(), lipgloss.NewStyle()
	m.chat.compose.SetStyles(s)
}

func (m emberModel) selectedResidentGoal() *residentGoal {
	if m.goalSel < 0 || m.goalSel >= len(m.resident.Goals) {
		return nil
	}
	g := m.resident.Goals[m.goalSel]
	return &g
}

// maxResidentEvents bounds the events the panel keeps in memory for one conversation.
const maxResidentEvents = 2000

func (m *emberModel) receiveResident(msg residentMsg) {
	m.initChat()
	selected := ""
	if g := m.selectedResidentGoal(); g != nil {
		selected = g.ID
	}
	// A slower poll must not roll newer goal state backwards.
	if msg.status.LastEvent >= m.resident.LastEvent {
		m.resident = msg.status
		m.statusAt = time.Now() // (the online clock ticks on from here between refreshes)
	}
	defer m.notifyResident(msg.events)
	for _, e := range msg.events {
		if e.Seq > m.chat.seq {
			if len(m.residentEvents) >= maxResidentEvents {
				// Bounded: a resident that runs for days must not grow the panel without limit. The oldest
				// events leave memory; the journal on the agent side keeps them.
				drop := len(m.residentEvents) - maxResidentEvents/2
				m.residentEvents = append(m.residentEvents[:0:0], m.residentEvents[drop:]...)
				// rendered counts drawn events by position: shift it with them (left as it was, the next draw sliced
				// past the end and crashed the panel once a busy session crossed the bound). The drawn text stays.
				m.chat.rendered = max(0, m.chat.rendered-drop)
			}
			m.residentEvents = append(m.residentEvents, e)
			m.chat.seq = e.Seq
			if e.Kind == "assistant" {
				// The durable block replaces the live tail; keeping both would show the reply twice.
				m.chat.live = residentLive{}
			}
		}
	}
	if m.chat.selectID != "" {
		selected = m.chat.selectID
	}
	m.goalSel = min(m.goalSel, max(0, len(m.resident.Goals)-1))
	for i, g := range m.resident.Goals {
		if g.ID == selected {
			m.goalSel = i
			m.chat.selectID = ""
			break
		}
	}
	if m.chat.compose.Value() == "" && !m.chat.sending && !m.chat.forceNew && m.chat.selectID == "" {
		m.targetSelectedGoal()
	}
	m.syncQuestion()
	m.syncChat()
}

func (m *emberModel) targetSelectedGoal() {
	g := m.selectedResidentGoal()
	m.chat.newGoal = g == nil || g.State == "cancelled"
	m.chat.target = ""
	if !m.chat.newGoal {
		m.chat.target = g.ID
	}
}

func (m *emberModel) syncChat() {
	m.initChat()
	w, h := max(20, m.width), max(12, m.height)
	// The composer's box is w-2 wide: four of those cells are its border and padding, so the editor gets w-6 and a
	// full line never eats the frame.
	m.chat.compose.SetWidth(w - 6)
	m.chat.compose.Placeholder = m.composerHint()
	m.chat.compose.SetHeight(min(3, max(1, h-10)))
	if m.chat.question != nil {
		m.chat.questionInput.SetWidth(w - 4)
		m.chat.questionInput.SetHeight(min(3, max(1, h-12)))
	}
	oldBottom := m.chat.transcript.AtBottom()
	// A viewport re-wraps its whole content on a size change, and this runs on every live snapshot, so a
	// 1 MB history would be re-read ten times a second while a reply streams. Only resize when it moved.
	tw := w
	if sw := m.sideWidth(); sw > 0 {
		tw = w - sw - 1 // the side panel takes the right column
	}
	if m.chat.transcript.Width() != tw {
		m.chat.transcript.SetWidth(tw)
	}
	w = tw // the transcript's content (cards, wrapping, the group line) fits its column
	// One header line and one footer line surround the transcript; the live tail sits between it and the
	// composer, so a generating reply never rewrites the durable content.
	// Half the body is the live window while a reply is written; the transcript keeps the other half.
	tail := 0
	if live := m.liveBlock(w); live != "" {
		tail = lipgloss.Height(live)
	}
	height := max(1, m.bodyHeight()-tail)
	if m.chat.help {
		height = max(1, height-3)
	}
	if m.chat.transcript.Height() != height {
		m.chat.transcript.SetHeight(height)
	}
	if m.chat.renderWidth != w || m.chat.markdown == nil {
		// Prose wraps at a reading width, not the terminal's: a 250-column line is hard to read. Tool lines, the
		// header and the status line still use the full width.
		m.chat.markdown, _ = glamour.NewTermRenderer(glamour.WithStyles(emberMarkdownStyle()), glamour.WithWordWrap(max(10, min(w-4, readingWidth))))
		m.chat.content, m.chat.rendered, m.chat.renderWidth = "", 0, w
	}
	changed := m.chat.rendered != len(m.residentEvents)
	if m.chat.rendered == 0 {
		m.chat.group = toolGroup{}   // drawing from the start (a resize, Ctrl+E): the group starts over too
		m.chat.sel = chatSelection{} // and the lines a selection pointed at are gone
	}
	for _, e := range m.residentEvents[m.chat.rendered:] {
		// Compact (the default): a card per action between the agent's words; Ctrl+O draws everything.
		if !m.chat.cards && !m.chat.expanded {
			if e.Kind == "assistant" {
				// its words first (they close the line above them), then its calls start the next line
				if block := m.chat.eventBlock(compactAssistant(e)); block != "" {
					m.chat.flushGroup(w)
					m.chat.content += ansi.Hardwrap(block, w, true) + "\n\n"
				}
				m.chat.fold(e)
				continue
			}
			if e.Kind == "tool_result" {
				// every action is its own card (60 calls, 60 cards: emberactions.go)
				if block := m.chat.actionBlock(e, w); block != "" {
					m.chat.flushGroup(w)
					m.chat.content += block + "\n\n"
					m.chat.fold(e) // (the running line still clears)
					continue
				}
			}
			if m.chat.fold(e) {
				continue
			}
		}
		if block := m.chat.eventBlock(e); block != "" {
			m.chat.flushGroup(w)
			m.chat.content += ansi.Hardwrap(block, w, true) + "\n\n"
		}
	}
	m.chat.rendered = len(m.residentEvents)
	if changed || m.chat.transcript.TotalLineCount() == 0 {
		body := strings.TrimRight(m.chat.content, "\n")
		if !m.chat.group.empty() {
			body += "\n\n" + m.chat.group.line(w) // the work in progress, fixed when the next block arrives
		}
		if body != "" && m.height >= 24 && m.sideWidth() > 0 {
			body += "\n" // the gap above the composer, in the transcript's column only (chatBottom)
		}
		if body == "" {
			body = mascotBeside(0, "", emberTitle.Render("CAMINUS")+emberDim.Render(" env"),
				"Your agent's conversation and commands will stay here.",
				emberDim.Render("Type a goal below. /help shows the controls."))
		}
		m.chat.transcript.SetContent(body)
	}
	if oldBottom {
		m.chat.transcript.GotoBottom()
	}
}

// bodyHeight is the space between the header and the footer, without the composer.
func (m emberModel) bodyHeight() int {
	// the header is two lines (title, rule) or more when the goal wraps under it; the footer one or two (chatFoot)
	w := max(20, m.width)
	return max(4, max(12, m.height)-lipgloss.Height(m.chatHead(w))-lipgloss.Height(m.chatBottom())-lipgloss.Height(m.chatFoot(w)))
}

// chatHead is the header with the whole goal: beside the agent and its state when it fits, else on its own lines
// under them (three at most). The goal was cut at 47 characters before.
// goalAim is what a goal works toward now: its prompt, or, once a message from its human was followed by a new done
// check (the goal re-aimed: a user started a block-breaker game and then asked for a design review in the same goal),
// that message. redirected says which.
func (m emberModel) goalAim(g *residentGoal) (text string, redirected bool) {
	text, msg, checked := g.Prompt, "", false
	for _, e := range m.residentEvents {
		if e.Goal != g.ID {
			continue
		}
		switch e.Kind {
		case "user_message":
			if t, _ := e.Data["text"].(string); strings.TrimSpace(t) != "" && checked {
				msg = t // (a message before the first done check refines the prompt; it does not redirect it)
			}
		case "done_check":
			if msg != "" {
				text, redirected, msg = msg, true, ""
			}
			checked = true
		}
	}
	return text, redirected
}

func (m emberModel) chatHead(w int) string {
	state := "stopped"
	if m.resident.DaemonAlive {
		state = "listening"
	}
	steps, goalText := 0, ""
	if sel := m.selectedResidentGoal(); sel != nil {
		state, steps = sel.State, sel.Steps
		aim, redirected := m.goalAim(sel)
		goalText = strings.Join(strings.Fields(residentText(aim)), " ")
		if redirected {
			goalText = "↪ " + goalText
		}
	}
	if !m.resident.DaemonAlive {
		// a stopped daemon runs nothing, whatever its goals last recorded: the header said "working" for an hour
		// after /stop while a newly queued goal waited
		state = "stopped · /start"
	}
	agent := residentText(m.agentName)
	head := shellHeader(agent, state, goalText, steps, w)
	if first := strings.SplitN(head, "\n", 2)[0]; !strings.Contains(ansi.Strip(first), "…") || goalText == "" {
		return head
	}
	head = shellHeader(agent, state, "", steps, w)
	title, rule, _ := strings.Cut(head, "\n")
	wrapped := strings.Split(ansi.Wordwrap(goalText, max(10, w-4), " "), "\n")
	if len(wrapped) > 3 {
		wrapped = append(wrapped[:2], ansi.Truncate(wrapped[2], max(10, w-5), "")+"…")
	}
	goalSt := lipgloss.NewStyle().Foreground(cBlue)
	for i, ln := range wrapped {
		wrapped[i] = "  " + goalSt.Render(ln)
	}
	return title + "\n" + strings.Join(wrapped, "\n") + "\n" + rule
}

// windowTitle is the terminal tab's title: the agent's state glyph, its name and what it is doing now, so a glance
// at the tab bar says it (Claude Code's own tab shows a glyph and the session name the same way).
func (m emberModel) windowTitle() string {
	state := "stopped"
	if m.resident.DaemonAlive {
		state = "idle"
	}
	for _, gl := range m.resident.Goals {
		if !m.resident.DaemonAlive {
			break // (stopped: its goals' last states run nothing)
		}
		switch gl.State {
		case "working", "waiting_model", "waiting_user", "waiting_root", "waiting_plan":
			state = gl.State
		}
		if state != "idle" && state != "stopped" {
			break
		}
	}
	icon, _ := stateStyle(state)
	now := strings.ReplaceAll(state, "_", " ")
	l := m.chat.live
	switch {
	case l.active && l.text != "":
		now = "writing"
	case l.active:
		now = "thinking"
	case m.chat.group.running != "" && state == "working" && m.activeGoal() != nil &&
		(m.chat.group.runningGoal == "" || m.chat.group.runningGoal == m.activeGoal().ID):
		now = "running"
		if since := m.chat.group.runningSince; since > 0 {
			now += " " + (time.Duration(time.Since(time.Unix(int64(since), 0)).Seconds()) * time.Second).String()
		}
	case state == "waiting_user" || state == "waiting_root":
		now = "needs you"
	}
	return strings.TrimSpace(icon + " " + residentText(m.agentName) + " · " + now + " — caminus ember")
}

// liveHeight is the live window's content lines (its label included): half the body less the border, so the
// transcript keeps the other half while a reply is being written.
func (m emberModel) liveHeight() int { return max(3, m.bodyHeight()/2-3) }

// liveWindow is the number of body lines the live box shows: its content height less the label line.
func (m emberModel) liveWindow() int {
	if sw := m.sideWidth(); sw > 0 {
		return m.sideEmberRows()
	}
	return max(1, min(m.liveHeight(), thoughtRows+1)-1)
}

// sideWidth is the side panel's width on a wide terminal (0: none). The transcript's prose wraps at readingWidth, so
// past about 160 columns the right part of the screen was empty; the panel takes it.
func (m emberModel) sideWidth() int {
	if m.width < sideMinTerminal {
		return 0
	}
	return min(sideMax, m.width-readingWidth-8)
}

const sideMinTerminal, sideMax = 160, 90

// sideEmberRows is the thoughts' window in the side panel: about 60 % of the body, the rest for tasks and now.
// goalCompactions is how many times the goal in view was compacted (the resident counts from its whole history);
// -1 with no goal in view hides the bubble.
func (m emberModel) goalCompactions() int {
	if m.chat.target == "" {
		return -1
	}
	return m.resident.Compactions[m.chat.target]
}

// cacheHit is today's share of input tokens served from the prompt cache; -1 before any turn today.
func (m emberModel) cacheHit() float64 {
	if m.resident.CacheHit == nil {
		return -1
	}
	return *m.resident.CacheHit
}

// onlineSeconds is the daemon's running time as the last status reported it, advanced by the time since (it ticks
// between refreshes); 0 when the daemon is down or the time is unknown.
func (m emberModel) onlineSeconds() float64 {
	if !m.resident.DaemonAlive || m.resident.OnlineS <= 0 {
		return 0
	}
	if m.statusAt.IsZero() {
		return m.resident.OnlineS
	}
	return m.resident.OnlineS + time.Since(m.statusAt).Seconds()
}

// sideEmberRows: the thoughts keep one size (they do not jump as text arrives) but a modest one, about a third of the
// panel, so tasks start right under them; the spare rows go to the bottom of the panel, not between the two (at 60 %
// there was a wide gap above tasks).
func (m emberModel) sideEmberRows() int { return max(6, min(14, m.bodyHeight()*35/100)) }

// sidePanel is the right column: Ember's box on top (one size), then the goal's whole todo list and what runs now.
func (m emberModel) sidePanel(sw, height int) string {
	box := m.emberBox(sw, m.sideEmberRows())
	rows := []string{}
	if box != "" {
		rows = strings.Split(box, "\n")
	}
	goal := m.activeGoal()
	if goal == nil {
		if sel := m.selectedResidentGoal(); sel != nil && (sel.State == "queued" || sel.State == "paused") {
			goal = sel
		}
	}
	textSt := lipgloss.NewStyle().Foreground(cText)
	wrap := func(text string, indent int, st lipgloss.Style, limit int) []string {
		lines := strings.Split(ansi.Wordwrap(text, max(8, sw-indent-1), " "), "\n")
		if len(lines) > limit {
			lines = append(lines[:limit-1], ansi.Truncate(lines[limit-1], max(8, sw-indent-2), "")+"…")
		}
		for i, ln := range lines {
			lines[i] = strings.Repeat(" ", indent) + st.Render(ln)
		}
		return lines
	}
	if goal != nil {
		if items := m.todoItems(goal.ID); len(items) > 0 {
			done := 0
			for _, it := range items {
				if it.state == "done" {
					done++
				}
			}
			rows = append(rows, "", sectionHead("tasks", cAccent, fmt.Sprintf("%d of %d done", done, len(items)), sw))
			for _, it := range items {
				mark, st := emberDim.Render("○"), emberDim
				switch it.state {
				case "done":
					mark, st = emberCtxOK.Render("●"), emberDim
				case "now":
					mark, st = emberBusy.Render("◐"), textSt.Bold(true)
				default:
					st = textSt
				}
				// one line each, cut with … (wrapped tasks made a ragged block)
				rows = append(rows, " "+mark+" "+st.Render(ansi.Truncate(strings.Join(strings.Fields(residentText(it.text)), " "), max(8, sw-4), "…")))
			}
		}
	}
	now, command := m.nowState(goal)
	rows = append(rows, "", sectionHead("now", cBlue, "", sw), " "+emberBusy.Render(now))
	if command != "" {
		rows = append(rows, wrap(command, 3, textSt, 8)...)
	}
	used := len(rows)
	for len(rows) < height {
		rows = append(rows, "")
	}
	rows = rows[:height]
	_ = used // (no Ember over the composer: the model's lab badge in the status line says who is answering)
	for i, r := range rows {
		rows[i] = ansi.Truncate(r, sw, "…")
	}
	return strings.Join(rows, "\n")
}

// todoItems is a goal's newest todo list (state done, now or next).
func (m emberModel) todoItems(gid string) []todoItem {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != gid || e.Kind != "todo" {
			continue
		}
		raw, _ := e.Data["items"].([]any)
		var items []todoItem
		for _, r := range raw {
			it, _ := r.(map[string]any)
			text, _ := it["text"].(string)
			state := map[string]string{"done": "done", "in_progress": "now"}[fmt.Sprint(it["state"])]
			if state == "" {
				state = "next"
			}
			if text != "" {
				items = append(items, todoItem{text, state})
			}
		}
		return items
	}
	return nil
}

// liveTop is the first visible line: the tail while following, the held position otherwise.
func (m emberModel) liveTop(body int) int {
	window := m.liveWindow()
	if !m.chat.live.hold {
		return max(0, body-window)
	}
	return max(0, min(m.chat.live.top, max(0, body-window)))
}

// scrollLive moves the live window and holds it, so text arriving mid-read cannot pull the view away.
// Scrolling back to the bottom resumes following.
func (m *emberModel) scrollLive(delta int) {
	l := &m.chat.live
	if !l.active {
		return
	}
	// the same lines the box shows (the thoughts at its width), so the first scroll does not jump: it measured the
	// live reply alone at the full terminal width
	body := len(m.thoughts(m.thoughtsWidth()))
	window := m.liveWindow()
	top := max(0, min(m.liveTop(body)+delta, max(0, body-window)))
	l.top = top
	l.hold = top < max(0, body-window)
}

// liveBlock renders the reply being generated in a fixed-height window: its reasoning, the tools it has
// started and the text so far. Plain text, not glamour: this is redrawn on every snapshot, and the durable
// block is styled once when the turn lands.
// liveEmberMin is the box width from which Ember sits beside the live reply; narrower, the text keeps every column.
const liveEmberMin = 50

// liveTextWidth is the live reply's text column. Ember is drawn only on the welcome screen of a new goal now (the
// user's call): the thoughts take the whole width.
func liveTextWidth(inner int) int { return inner }

// thoughtsWidth is the text width emberBox draws the thoughts at: the side panel's on a wide terminal, the chat's
// otherwise.
func (m emberModel) thoughtsWidth() int {
	w := max(20, m.width)
	if sw := m.sideWidth(); sw > 0 {
		w = sw
	}
	return min(liveTextWidth(max(10, w-4)), readingWidth)
}

func (m emberModel) liveBlock(w int) string {
	if m.sideWidth() > 0 {
		return "" // a wide terminal shows Ember in the side panel
	}
	return m.emberBox(w, min(m.liveHeight(), thoughtRows+1)-1)
}

// emberBox is Ember and the thoughts, window lines of them, w cells wide: under the transcript on a narrow terminal,
// at the top of the side panel on a wide one.
func (m emberModel) emberBox(w, window int) string {
	l := m.chat.live
	inner := max(10, w-4)
	text := min(liveTextWidth(inner), readingWidth)
	if !l.active && m.sideWidth() == 0 && m.bodyHeight() < 2*(thoughtRows+3) {
		return "" // a small terminal keeps its rows for the transcript between replies
	}
	body := m.thoughts(text)
	if len(body) == 0 {
		if m.activeGoal() == nil && !l.active {
			return "" // between goals with nothing to show: the transcript takes the rows
		}
		msg := "no thoughts yet: they appear with the agent's first reply (its reasoning when the model shows it, else its words and the tools it calls)"
		for _, ln := range strings.Split(ansi.Wordwrap(msg, max(8, text), " "), "\n") {
			body = append(body, emberDim.Render(ln)) // wrapped like the thoughts: unwrapped it ran under Ember
		}
	}
	// One fixed size, generating or not (a box that grew and shrank was hard to read): only the text moves. The
	// thoughts run downward, newest last.
	window = max(1, window)
	top := max(0, len(body)-window)
	if l.active && l.hold {
		top = m.liveTop(len(body))
	}
	var head string
	if l.active {
		spin := g.spinner[m.tickN%len(g.spinner)]
		who := g.agent + " agent"
		if l.reasoning != "" && l.text == "" {
			who = g.agent + " thinking"
		}
		hint := "  live · Ctrl+T hold · Alt+↑/↓ scroll"
		if l.hold {
			hint = fmt.Sprintf("  held · %d below · Ctrl+T follow", max(0, len(body)-(top+window)))
		}
		head = bold(cGreen).Render(who) + " " + emberBusy.Render(spin) + emberDim.Render(hint)
	} else {
		head = sectionHeadRight(g.agent+" thoughts", cGreen, "newest ↓", max(18, w-1))
	}
	shown := append([]string{}, body[top:min(top+window, len(body))]...)
	for len(shown) < window {
		shown = append(shown, "") // filled to its size from the first line, so it never grows as text arrives
	}
	// a reading surface: a heading, no frame (sectionHead)
	if !strings.Contains(head, "─") {
		head += " " + lipgloss.NewStyle().Foreground(cGray).Render(strings.Repeat("─", max(2, w-1-lipgloss.Width(head)-1)))
	}
	for i := range shown {
		shown[i] = " " + shown[i]
	}
	return strings.Join(append([]string{head}, shown...), "\n")
}

// nowState is what the agent is doing this moment, and the command when one runs (the time first in now).
func (m emberModel) nowState(goal *residentGoal) (string, string) {
	now, command := "idle", ""
	l := m.chat.live
	switch {
	case goal != nil && m.compactingNote(goal.ID) != "":
		now = m.compactingNote(goal.ID)
	case l.active && l.text != "":
		now = "writing its reply"
	case l.active && len(l.tools) > 0:
		now = "calling " + residentText(l.tools[len(l.tools)-1])
	case l.active:
		now = "thinking"
	case m.chat.group.running != "" && goal != nil && goal.State == "working" &&
		(m.chat.group.runningGoal == "" || m.chat.group.runningGoal == goal.ID):
		// the time first; the whole command follows, wrapped under it (three lines at most)
		now = "running"
		if since := m.chat.group.runningSince; since > 0 {
			now += " " + (time.Duration(time.Since(time.Unix(int64(since), 0)).Seconds()) * time.Second).String()
		}
		command = m.chat.group.running
	case goal == nil:
		now = "idle (free time runs on its own schedule)"
	default:
		now = map[string]string{
			"waiting_user": "waiting for your answer · Ctrl+A", "waiting_root": "waiting for a root approval · Ctrl+A",
			"waiting_plan": "waiting for your OK on its plan · /plan approve", "waiting_model": "waiting for the model",
			"paused": "paused · /resume", "queued": "queued, starts next", "working": "between steps",
		}[goal.State]
	}
	return now, command
}

// compactingNote: while a goal's history is being summarised (a compacting event with no compacted, error or
// assistant reply after it), "compacting · part i of n · 398K tokens"; else "". The summary takes a model call per
// part and nothing else moves meanwhile: without this the chat looked frozen.
func (m emberModel) compactingNote(gid string) string {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != gid {
			continue
		}
		switch e.Kind {
		case "compacting":
			part, _ := e.Data["part"].(float64)
			parts, _ := e.Data["parts"].(float64)
			before, _ := e.Data["before_tokens"].(float64)
			note := strings.TrimSpace(g.compact + " compacting")
			if parts > 1 {
				// a small gauge of the parts summarised so far, then the count
				n := min(10, int(parts))
				done := int(float64(n) * (part - 1) / parts)
				note += " " + strings.Repeat(g.bar, done) + strings.Repeat(g.barEmpty, n-done) +
					fmt.Sprintf(" part %d of %d", int(part), int(parts))
			}
			if before > 0 {
				note += " · " + short(int(before))
			}
			return note
		case "compacted", "runtime_error", "assistant", "watchdog", "finished":
			return ""
		}
	}
	return ""
}

// activeGoal is the goal the agent is on now (working or waiting on something), nil between goals.
func (m emberModel) activeGoal() *residentGoal {
	for i := range m.resident.Goals {
		switch m.resident.Goals[i].State {
		case "working", "waiting_model", "waiting_user", "waiting_root", "waiting_plan":
			return &m.resident.Goals[i]
		}
	}
	return nil
}

// thoughtRows is the number of lines of thoughts Ember's box shows, generating or not.
const thoughtRows = 8

// thoughts is the stream Ember's box shows: the last few replies' thought chains, oldest first, then the reply being
// written. A chain is the reply's reasoning when the model shows it, else its own words and the tools it called (a
// model that answers without reasoning, as DeepSeek on DeepInfra does, left the box saying "no thinking"). It keeps
// the current goal's chains; with none yet it keeps the latest from any goal, so the box never empties.
func (m emberModel) thoughts(width int) []string {
	gid := ""
	if g := m.activeGoal(); g != nil {
		gid = g.ID
	}
	chains := m.thoughtChains(gid)
	if len(chains) == 0 && gid != "" {
		chains = m.thoughtChains("")
	}
	var out []string
	wrap := func(text string, st lipgloss.Style) {
		text = strings.ReplaceAll(text, "\t", "    ") // a tab is one cell to the wrapper and up to eight on screen
		for _, ln := range strings.Split(ansi.Wordwrap(residentText(strings.TrimSpace(text)), max(8, width), " "), "\n") {
			out = append(out, st.Render(ln))
		}
	}
	for _, c := range chains {
		if len(out) > 0 {
			out = append(out, "")
		}
		if c.text != "" {
			wrap(c.text, emberThink)
		}
		for _, t := range c.tools {
			out = append(out, ansi.Truncate(emberDim.Render("→ ")+emberTool.Render(t), max(8, width), "…"))
		}
	}
	if m.chat.live.active {
		if len(out) > 0 {
			out = append(out, "")
		}
		out = append(out, m.chat.live.body(width)...)
	}
	return out
}

type thoughtChain struct {
	text  string
	tools []string
}

// thoughtChains is the last six replies of a goal ("" = any) that carry a thought: their reasoning, else their words,
// with the tools each called. A long one keeps its tail.
func (m emberModel) thoughtChains(gid string) []thoughtChain {
	var chains []thoughtChain
	for i := len(m.residentEvents) - 1; i >= 0 && len(chains) < 6; i-- {
		e := m.residentEvents[i]
		if e.Kind != "assistant" || (gid != "" && e.Goal != gid) {
			continue
		}
		text, _ := e.Data["reasoning"].(string)
		if strings.TrimSpace(text) == "" {
			text, _ = e.Data["content"].(string)
		}
		if len(text) > 3000 {
			text = "…" + text[len(text)-3000:]
		}
		var tools []string
		calls, _ := e.Data["tool_calls"].([]any)
		for _, raw := range calls {
			call, _ := raw.(map[string]any)
			f, _ := call["function"].(map[string]any)
			name, _ := f["name"].(string)
			var args map[string]any
			if s, _ := f["arguments"].(string); s != "" {
				_ = json.Unmarshal([]byte(s), &args)
			}
			what := name
			for _, k := range []string{"command", "path", "query", "url", "check", "evidence", "summary", "question"} {
				if v, _ := args[k].(string); strings.TrimSpace(v) != "" {
					what += "  " + oneLine(residentText(v))
					break
				}
			}
			tools = append(tools, what)
		}
		if strings.TrimSpace(text) == "" && len(tools) == 0 {
			continue
		}
		chains = append([]thoughtChain{{strings.TrimSpace(text), tools}}, chains...)
	}
	return chains
}

// emberTail keeps the last maxLines wrapped lines of s, so the live tail has a fixed ceiling.
func emberTail(s string, width, maxLines int) string {
	lines := strings.Split(ansi.Hardwrap(s, max(8, width), true), "\n")
	if len(lines) > maxLines {
		lines = append([]string{emberDim.Render("…")}, lines[len(lines)-maxLines:]...)
	}
	return strings.Join(lines, "\n")
}

// stopStream ends the reader process and drops the live tail; the transcript already holds what is durable.
func (c *residentChat) stopStream() {
	if c.stream != nil {
		c.stream.cancel()
		c.stream = nil
	}
	c.live = residentLive{}
	c.streamRetry = time.Now().Add(2 * time.Second) // a CLI that exits at once is not respawned every tick
}

func (c *residentChat) renderMarkdown(text string) string {
	text = residentText(text)
	if c.markdown == nil {
		return text
	}
	// fenced code goes to codePlate (a visible block: plate, language, highlighting); glamour left it as indented
	// text with no background, so a code block could not be told from prose. The prose around it is glamour's.
	var parts []string
	for _, seg := range splitFences(text) {
		if seg.code {
			parts = append(parts, codePlate(seg.lang, seg.text, max(10, min(c.renderWidth-4, readingWidth))))
			continue
		}
		if strings.TrimSpace(seg.text) == "" {
			continue
		}
		if rendered, err := c.markdown.Render(seg.text); err == nil {
			parts = append(parts, strings.Trim(rendered, "\n"))
		} else {
			parts = append(parts, seg.text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// eventCard frames one event: its label, then its body behind a dim left rule, so the conversation reads as a
// stack of blocks instead of one wall of text. Blank lines stay blank, so a card can breathe.
// bulletBlock is a reply in the compact view: a bullet, then the text with a hanging indent; no label line, no
// rule, so the agent's words read as prose (the cards view keeps the labelled card).
// readingWidth is the column prose wraps at in the transcript, whatever the terminal's width.
const readingWidth = 100

func bulletBlock(text string) string {
	lines := strings.Split(strings.Trim(text, "\n"), "\n")
	for len(lines) > 0 && strings.TrimSpace(ansi.Strip(lines[0])) == "" {
		lines = lines[1:]
	}
	for i, ln := range lines {
		ln = strings.TrimPrefix(ln, "  ") // the markdown renderer's own left margin: the bullet's indent replaces it
		if i == 0 {
			lines[i] = bold(cGreen).Render("● ") + ln
		} else {
			lines[i] = "  " + ln
		}
	}
	return strings.Join(lines, "\n")
}

func eventCard(block string) string {
	lines := strings.Split(strings.TrimRight(block, "\n"), "\n")
	if len(lines) < 2 {
		return block
	}
	var b strings.Builder
	b.WriteString(lines[0])
	for _, ln := range lines[1:] {
		if strings.TrimSpace(ansi.Strip(ln)) == "" {
			b.WriteString("\n")
			continue
		}
		b.WriteString("\n" + emberDim.Render("│ ") + ln)
	}
	return b.String()
}

func fenced(text, language string) string {
	fence := "```"
	for strings.Contains(text, fence) {
		fence += "`"
	}
	return fence + language + "\n" + text + "\n" + fence
}

// eventBlock draws one durable event in the shell's style (shell.go): the human's words and the agent's replies as
// labelled turns on a thin rule, every tool call and its result as a one-line card (Ctrl+E expands them all), root
// requests as the inline approval card. The result is final: syncChat caches it and never draws it again.
func (c *residentChat) eventBlock(e residentEvent) string {
	str := func(key string) string { s, _ := e.Data[key].(string); return s }
	w := max(20, c.renderWidth)
	turn := func(mark, label string, col color.Color, body string) string {
		if !c.cards && !c.expanded { // compact: the label, then the text indented, without the rule
			lines := strings.Split(strings.Trim(body, "\n"), "\n")
			for i, ln := range lines {
				if strings.TrimSpace(ansi.Strip(ln)) != "" {
					lines[i] = "  " + strings.TrimPrefix(ln, "  ")
				}
			}
			return bold(col).Render(mark+" "+label) + "\n" + strings.Join(lines, "\n")
		}
		return eventCard(bold(col).Render(mark+" "+label) + "\n" + body)
	}
	firstLine := func(s string) string {
		s = strings.TrimSpace(residentText(s))
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			s = s[:i] + " …"
		}
		return s
	}
	lines := func(s string, limit int) []string {
		all := strings.Split(strings.TrimRight(strings.ReplaceAll(residentText(s), "\r\n", "\n"), "\n"), "\n")
		if c.full {
			limit = max(limit, 5000) // Ctrl+O: the whole output, bounded so one card cannot stall the screen
		}
		if len(all) > limit {
			all = append([]string{fmt.Sprintf("… %d earlier lines (Ctrl+O shows all)", len(all)-limit)}, all[len(all)-limit:]...)
		}
		return all
	}
	switch e.Kind {
	case "goal_queued":
		label := "you · goal " + shortGoal(e.Goal)
		if k, _ := e.Data["kind"].(string); k == "free" || k == "dream" {
			label = "free time · " + k // the resident's own hours: the invitation, not the human's words
			return toolCard(k, "stretch "+shortGoal(e.Goal)+" started", "", true, c.expanded, lines(str("prompt"), 30), w)
		}
		return turn(g.user, label, cBlue, c.renderMarkdown(str("prompt")))
	case "user_message", "peer_message":
		text := str("text")
		if text == "" {
			return ""
		}
		return turn(g.user, "you · steer "+shortGoal(e.Goal), cBlue, c.renderMarkdown(text))
	case "assistant":
		var parts []string
		// The reply's reasoning is shown once, above the answer, from the durable event: it is never
		// sent back to the model, so this block is display only.
		if text := str("reasoning"); text != "" {
			if c.expanded {
				parts = append(parts, eventCard(emberThink.Render(g.expanded+" thinking")+"\n"+emberThink.Render(c.renderMarkdown(text))))
			} else {
				parts = append(parts, emberDim.Render(g.collapsed+" thinking · "+fmt.Sprintf("%d lines", strings.Count(text, "\n")+1)))
			}
		}
		if text := str("content"); text != "" {
			if !c.cards && !c.expanded {
				parts = append(parts, bulletBlock(c.renderMarkdown(text)))
			} else {
				parts = append(parts, turn(g.agent, "agent", cGreen, c.renderMarkdown(text)))
			}
		}
		calls, _ := e.Data["tool_calls"].([]any)
		for _, raw := range calls {
			call, _ := raw.(map[string]any)
			f, _ := call["function"].(map[string]any)
			name, _ := f["name"].(string)
			if name == "ask_user" || name == "read" || name == "edit" || name == "write" {
				continue
			} // the durable question gets a native UI below; a file tool's result card says what it did
			rawArgs, _ := f["arguments"].(string)
			var args map[string]any
			_ = json.Unmarshal([]byte(rawArgs), &args)
			if command, ok := args["command"].(string); ok {
				parts = append(parts, toolCard("terminal", firstLine(command), "", true, c.expanded, lines(command, 40), w))
				continue
			}
			summary := ""
			for _, key := range []string{"path", "question", "summary", "check", "evidence", "label", "id", "topic", "to", "until"} {
				if v, ok := args[key].(string); ok && v != "" {
					summary = firstLine(v)
					break
				}
			}
			var pretty any
			text := rawArgs
			if json.Unmarshal([]byte(rawArgs), &pretty) == nil {
				if b, err := json.MarshalIndent(pretty, "", "  "); err == nil {
					text = string(b)
				}
			}
			parts = append(parts, toolCard(residentText(name), summary, "", true, c.expanded, lines(text, 40), w))
		}
		return strings.Join(parts, "\n")
	case "tool_result":
		result, _ := e.Data["result"].(map[string]any)
		if str("name") == "ask_user" {
			answer, _ := result["answer"].(string)
			if answer == "" {
				answer, _ = result["decision"].(string)
			}
			return turn(g.user, "you · answer", cBlue, c.renderMarkdown(answer))
		}
		if name := str("name"); (name == "edit" || name == "write") && result["error"] == nil {
			// an edit is a diff card: open when it is short, folded (Ctrl+E) when it is long
			path, _ := result["path"].(string)
			added, _ := result["added"].(float64)
			removed, _ := result["removed"].(float64)
			meta := fmt.Sprintf("+%d −%d", int(added), int(removed))
			if created, _ := result["created"].(bool); created {
				note, _ := result["note"].(string)
				meta = note
			}
			diff, _ := result["diff"].(string)
			changed := 0
			for _, ln := range strings.Split(diff, "\n") {
				if ln != "" && !strings.HasPrefix(ln, "@@") {
					changed++
				}
			}
			// line numbers, tinted added / removed plates, the code highlighted for its language (emberdiff.go)
			return editCard(residentText(path), meta, c.expanded || c.full || changed <= 12, diff, w)
		}
		if str("name") == "read" && result["error"] == nil {
			path, _ := result["path"].(string)
			shown, _ := result["shown"].(string)
			meta := "lines " + shown
			if dir, _ := result["directory"].(bool); dir {
				meta = "directory"
			}
			return toolCard("read", residentText(path), meta, true, false, nil, w)
		}
		if err, ok := result["error"].(string); ok {
			// a failed call stays open: the reason is what the reader needs
			return toolCard("error", residentText(str("name")), firstLine(err), false, true, lines(err, 12), w)
		}
		if output, ok := result["output"].(string); ok {
			meta := fmt.Sprintf("%d lines", strings.Count(strings.TrimRight(output, "\n"), "\n")+1)
			if code, ok := result["exit_code"].(float64); ok {
				meta = fmt.Sprintf("exit %d · ", int(code)) + meta
			} else if running, _ := result["running"].(bool); running {
				meta = "running · " + meta
			}
			if strings.TrimSpace(output) == "" {
				meta = strings.TrimSuffix(meta, " · 1 lines") + " · no output"
			}
			if saved, _ := result["saved"].([]any); len(saved) > 0 {
				// a big output was moved to a file: the card says where, with the whole size
				if first, _ := saved[0].(map[string]any); first != nil {
					path, _ := first["path"].(string)
					lines, _ := first["lines"].(float64)
					meta = strings.Split(meta, " · ")[0] + fmt.Sprintf(" · %d lines → %s", int(lines), residentText(path))
				}
			}
			code, _ := result["exit_code"].(float64)
			return toolCard("output", residentText(str("name")), meta, code == 0, c.expanded, lines(output, 30), w)
		}
		if str("name") == "finish" {
			// the "finished" block under it carries the summary: here one line, the judge's verdict, and what was left
			// open when expanded (its raw JSON repeated the summary right above the block)
			verdict := residentText(firstLine(fmt.Sprint(result["judge"])))
			if result["judge"] == nil {
				verdict = "no judge"
			}
			var more []string
			if open, _ := result["open"].(string); strings.TrimSpace(open) != "" {
				more = append(more, "left open: "+residentText(open))
			}
			return toolCard("finish", "judge: "+verdict, "", true, c.expanded, more, w)
		}
		if b, err := json.MarshalIndent(result, "", "  "); err == nil {
			return toolCard("result", residentText(str("name")), "", true, c.expanded, lines(string(b), 30), w)
		}
	case "finished":
		return turn(g.done, "finished", cGreen, c.renderMarkdown(str("summary")))
	case "approval_requested":
		if str("kind") == "github" {
			// the card shows what will be posted under your name: the summary, then the body's first lines (all of it
			// in the approvals screen, Ctrl+A then Enter)
			text := residentText(str("summary"))
			if h, ok := e.Data["github"].(map[string]any); ok {
				if _, body := handoffText(h); strings.TrimSpace(body) != "" {
					lines := strings.Split(strings.TrimRight(residentText(body), "\n"), "\n")
					shown := lines[:min(len(lines), 10)]
					text += "\n\n" + strings.Join(shown, "\n")
					if len(lines) > len(shown) {
						text += fmt.Sprintf("\n… %d more lines · Ctrl+A, Enter shows all of it", len(lines)-len(shown))
					}
				}
			}
			return approvalCard("github", text, residentText(str("reason")), w)
		}
		return approvalCard("root", residentText(str("cmd")), residentText(str("reason")), w)
	case "question":
		return turn("?", "question", cYellow, c.renderMarkdown(str("text")))
	case "compacted":
		return toolCard("note", "context compacted", "the whole conversation stays in this transcript", true, false, nil, w)
	case "cancelled":
		return toolCard("note", "goal "+shortGoal(e.Goal)+" cancelled", "", true, false, nil, w)
	case "model_back":
		down, _ := e.Data["down_s"].(float64)
		tries, _ := e.Data["attempts"].(float64)
		return toolCard("note", "the model is back",
			fmt.Sprintf("after %s down (%d tries)", (time.Duration(down)*time.Second).String(), int(tries)), true, false, nil, w)
	case "model_unavailable", "model_truncated", "context_refused", "model_refused", "model_gave_up":
		if boost, _ := e.Data["boost"].(float64); boost > 0 && e.Kind == "model_truncated" {
			limit, _ := e.Data["limit"].(float64)
			return toolCard("note", "a reply hit the output limit",
				fmt.Sprintf("asked again with room for %s tokens (was %s); nothing in it had run", short(int(boost)), short(int(limit))),
				true, false, nil, w)
		}
		if rec, _ := e.Data["recovered"].(bool); rec && e.Kind == "model_truncated" {
			// handled: the agent was told and retries in smaller pieces; only a third cut-off in a row pauses the goal
			streak, _ := e.Data["streak"].(float64)
			return toolCard("note", "a reply hit the output limit and was dropped",
				fmt.Sprintf("the agent retries in smaller pieces (%d of 3 in a row)", int(streak)), true, false, nil, w)
		}
		text := residentText(str("error"))
		body := lines(text, 12)
		if partial, ok := e.Data["partial"].(map[string]any); ok {
			if content, _ := partial["content"].(string); content != "" {
				// what the model produced before it stopped, then why it stopped
				body = append(lines(content, 20), "", text)
			}
		}
		return toolCard("error", strings.ReplaceAll(e.Kind, "_", " "), firstLine(text), false, true, body, w)
	case "judged":
		v := str("verdict")
		return toolCard("gate", "judge: "+strings.ReplaceAll(v, "_", " "), firstLine(str("reason")), v == "met" || v == "unread", v != "met", lines(str("reason"), 6), w)
	case "gate":
		ok, _ := e.Data["passed"].(bool)
		return toolCard("gate", "gate: "+firstLine(str("command")), map[bool]string{true: "passed", false: "failed"}[ok], ok, !ok, lines(str("output"), 12), w)
	case "report":
		if str("level") == "need_you" {
			return turn("!", "needs you", cRed, c.renderMarkdown(str("text")))
		}
		return toolCard("note", "fyi: "+firstLine(str("text")), "", true, c.expanded, lines(str("text"), 20), w)
	case "scheduled":
		at, _ := e.Data["wake_at"].(float64)
		return toolCard("note", "parked until "+time.Unix(int64(at), 0).Format("15:04"), firstLine(str("note")), true, false, nil, w)
	case "advice":
		return toolCard("note", "asked its advisor", residentText(str("model"))+" · "+firstLine(str("question")), true, false, nil, w)
	case "advisor_review":
		return toolCard("note", "advisor review, asked for by you", residentText(str("model"))+" · the agent reads it before its next step", true, false, nil, w)
	case "watchdog_advised":
		return toolCard("note", "the advisor stepped in", "the goal was about to pause ("+residentText(str("reason"))+"); its advice went to the agent", true, false, nil, w)
	case "woke":
		return toolCard("note", "woke up", firstLine(str("note")), true, false, nil, w)
	case "brain":
		return toolCard("note", "now answering: "+residentText(str("model")), str("brain"), true, false, nil, w)
	}
	return ""
}

func shortGoal(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

// chatComposer is where the human types: the shell's composer box (shell.go), its top border saying which goal the
// draft goes to, the editor's one prompt mark on the first line, and a dim hint that says what Enter does now. The
// Tab completion hint rides above the draft, inside the same box.
func (m emberModel) chatComposer() string {
	if m.chat.question != nil && m.chat.questionArmed {
		return m.questionBox()
	}
	input := m.chat.compose.View()
	if (m.mode == "" || m.mode == "normal") && !m.chat.sending {
		value := strings.TrimRight(m.chat.compose.Value(), " ")
		if m.slashMenuOpen() {
			// the slash menu is its own box above the composer (slashMenu); a terminal too short for it gets the
			// names on one line inside the composer instead
			if m.slashMenu(max(18, m.width-2)) == "" {
				lines := slashMatches(value)
				if len(lines) > 4 {
					lines = lines[:4]
				}
				input = emberDim.Render(emberClip("Tab completes · "+strings.Join(lines, "  "), max(18, m.width-6))) + "\n" + input
			}
		} else if token, ok := pathToken(value); ok {
			// The path so far is rendered in the inline-code role, then what Tab
			// can offer: directories in the navigation accent, files resting, and
			// the absence of any match in the failure color so a typo shows itself.
			input = emberDim.Render("Tab completes · ") + emberCode.Render(token)
			if matches := pathMatches(token); len(matches) > 0 {
				lines := matches
				if len(lines) > 4 {
					lines = lines[:4]
				}
				for i, l := range lines {
					style := emberDim
					if strings.HasSuffix(l, "/") {
						style = emberTitle
					}
					if l == m.chat.pathPick && m.chat.pathIdx > 1 {
						style = emberSel
					}
					lines[i] = style.Render(l)
				}
				input += emberDim.Render(" · " + strings.Join(lines, "  "))
			} else {
				input += emberCtxHot.Render(" no match")
			}
			input = emberClip(input, max(18, m.width-6)) + "\n" + input
		}
	}
	box := composerBox(input, m.composerTitle(), max(18, m.width-2))
	if m.chat.shell {
		box = borderTitle(emberBorder.BorderForeground(cPurple).Padding(0, 1).Width(max(18, m.width-2)).Render(input),
			m.composerTitle(), emberShell)
	}
	// a message (delivered, queued, an error) sits on the frame for a few seconds instead of a line of its own under
	// the status line, which appeared and went with every message
	return borderNote(box, terminalText(m.msg, false), emberDim)
}

// composerTitle is what the composer's top border says: which goal a message goes to, or that it starts a new one.
// The old label's other facts (sending, a waiting question, browsing) keep their place there, shortened to a title.
func (m emberModel) composerTitle() string {
	title := "new goal"
	switch {
	case m.chat.shell:
		title = "shell · runs as the agent"
	case m.chat.sending:
		title = "sending…"
	case m.chat.question != nil:
		title = "question waiting · F2 answers"
	case !m.chat.newGoal:
		title = "to goal " + shortGoal(m.chat.target)
		if g := m.selectedResidentGoal(); g != nil && g.ID == m.chat.target && g.State == "completed" {
			title = "continue goal " + shortGoal(m.chat.target)
		}
	}
	if !m.chat.compose.Focused() && !m.chat.sending {
		title += " · browsing"
	}
	return title
}

// composerHint is the composer's placeholder: what Enter does with the draft now.
func (m emberModel) composerHint() string {
	if m.chat.shell {
		return "a command, run in the agent's account · Enter runs it · Backspace leaves"
	}
	if !m.chat.compose.Focused() {
		return "browsing history · Enter returns to typing"
	}
	if m.chat.newGoal {
		return "type a goal · Enter starts it · ! for shell mode · /help"
	}
	return "message the agent · Enter nudges · /steer redirects · ! for shell mode"
}

// chatBottom renders whatever occupies the composer slot: the steering draft,
// the inline question or the Ctrl+P session switcher. syncChat sizes the
// transcript against its measured height.
func (m emberModel) chatBottom() string {
	if box := m.promptBox(max(18, m.width-2)); box != "" {
		return box
	}
	if m.chat.picker {
		return m.pickerBox()
	}
	out := m.chatComposer()
	if line := m.todoPinned(); line != "" {
		out = line + "\n" + out
	}
	if act := m.activityLine(max(18, m.width-2)); act != "" && m.height >= 20 { // (a small terminal keeps its rows)
		out = act + "\n" + out // above the todo list, where the work is described
	}
	if side := m.btwSection(max(18, m.width-2)); side != "" {
		out = side + "\n" + out
	}
	if menu := m.slashMenu(max(18, m.width-2)); menu != "" {
		out = menu + "\n" + out
	}
	// a blank line between the conversation and what sits under it, as Claude Code leaves: the composer read as part
	// of the last block when it touched it
	// (a small terminal keeps every row for the transcript; with the side panel the gap is the transcript's own last
	// line instead, so the panel, and Ember sitting at its foot, reach down to the composer)
	if m.height >= 24 && m.sideWidth() == 0 {
		out = "\n" + out
	}
	return out
}

// recall is the shell's Up / Down history over what you sent (pastPrompts, newest first): Up on an empty draft brings
// back your last message, Up again goes further back, Down comes forward and past the newest leaves the draft empty.
// An edited draft is yours: the arrows then move the cursor. The first Up on an empty draft still pulls back a newest
// message the agent has not read yet (unsendLast); with none waiting it recalls.
func (m emberModel) recall(k string) (emberModel, bool) { return m.recallStep(k, false) }

// recallStep is recall; asked says the resident already answered that nothing is waiting to pull back.
func (m emberModel) recallStep(k string, asked bool) (emberModel, bool) {
	value := m.chat.compose.Value()
	recalling := m.chat.recallIdx > 0 && value == m.chat.recallText
	if !recalling {
		m.chat.recallIdx = 0
		if k == "down" || strings.TrimSpace(value) != "" {
			return m, false
		}
		if k == "up" && !asked && !m.chat.newGoal && m.chat.target != "" {
			// the resident knows whether your newest message is still unread: ask it to pull that back first; with
			// nothing waiting its answer recalls instead (unsendMsg)
			return m, false
		}
	}
	past := m.pastPrompts()
	i := m.chat.recallIdx
	if k == "up" {
		i++
	} else {
		i--
	}
	switch {
	case i < 1:
		m.chat.compose.SetValue("")
		m.chat.recallIdx, m.chat.recallText = 0, ""
		m.msg = ""
	case i > len(past):
		if len(past) == 0 {
			m.msg = "nothing sent yet"
		}
		return m, true
	default:
		m.chat.compose.SetValue(past[i-1])
		m.chat.compose.CursorEnd()
		m.chat.recallIdx, m.chat.recallText = i, past[i-1]
		m.msg = fmt.Sprintf("sent message %d of %d · ↑ older · ↓ newer · edit or Enter to send", i, len(past))
	}
	m.syncChat()
	return m, true
}

// btwRows is the most answer lines the /btw section shows at once: a third of the terminal.
func (m emberModel) btwRows() int { return max(3, min(14, m.height/3)) }

// btwSection is the /btw side answer in its own box above the composer: a small Ember on the left (still while it
// thinks, it blinks), the answer on the right, scrolled with Shift+Up / Shift+Down; Esc on an empty draft closes it.
// The goal keeps nothing of it.
func (m emberModel) btwSection(w int) string {
	b := m.chat.btw
	if !b.open {
		return ""
	}
	var art []string // (no Ember beside the answer; the section heading says what it is)
	artW := 0
	textW := max(10, min(readingWidth, w-4-artW)) // (the frame and its inner margin: a body line is 1 + textW, the box textW + 2)
	q := lipgloss.NewStyle().Foreground(cText).Render(emberClip(residentText(b.question), textW-4))
	body := []string{emberBusy.Render("you ") + q}
	rows := m.btwRows()
	keys := "esc closes"
	switch {
	case b.loading:
		body = append(body, emberDim.Render(spin[m.tickN%len(spin)]+" thinking… (no tools; nothing enters the goal)"))
	case b.err != "":
		body = append(body, emberCtxHot.Render(emberClip("could not ask: "+b.err, textW)))
	default:
		var lines []string
		for _, l := range b.lines {
			lines = append(lines, strings.Split(ansi.Hardwrap(l, textW, true), "\n")...)
		}
		top := max(0, min(b.top, len(lines)-rows))
		end := min(len(lines), top+rows)
		body = append(body, lines[top:end]...)
		if len(lines) > rows {
			keys = fmt.Sprintf("shift+↑↓ scroll (%d-%d of %d) · esc closes", top+1, end, len(lines))
		}
	}
	// a popup, not more transcript: its own tinted surface in a frame, the title and keys in the top border (a user
	// read the unframed answer as history)
	surface, border := overlayColors()
	seq, edge := bgSeq(surface), lipgloss.NewStyle().Foreground(lipgloss.Color(border))
	inner := min(w-2, textW+2)
	title := " " + lipgloss.NewStyle().Foreground(cAccent).Bold(true).Render("by the way") + " " +
		emberDim.Render(emberClip("not added to the goal · "+keys, max(4, inner-16))) + " "
	fill := max(0, inner-1-ansi.StringWidth(title))
	out := []string{edge.Render("╭─") + title + edge.Render(strings.Repeat("─", fill)+"╮")}
	for i, l := range body {
		a := strings.Repeat(" ", artW)
		if i < len(art) {
			a = art[i] + "  "
		}
		cell := " " + a + l
		pad := max(0, inner-ansi.StringWidth(cell))
		out = append(out, edge.Render("│")+seq+replate(cell, seq)+strings.Repeat(" ", pad)+"\x1b[m"+edge.Render("│"))
	}
	out = append(out, edge.Render("╰"+strings.Repeat("─", inner)+"╯"))
	return strings.Join(out, "\n")
}

// helpLines: every command with what it does (the slash menu's two columns), then the keys.
func helpLines() []string {
	names := make([]string, 0, len(slashCommands))
	for _, c := range slashCommands {
		names = append(names, c.name)
	}
	slices.Sort(names)
	row := func(k, v string, w int) string {
		return lipgloss.NewStyle().Foreground(cText).Bold(true).Render(k) + strings.Repeat(" ", max(1, w-ansi.StringWidth(k))) + emberDim.Render(v)
	}
	out := []string{emberTitle.Render("commands"), ""}
	for _, n := range names {
		out = append(out, row(n, slashHint(n), 16))
	}
	out = append(out, "", emberTitle.Render("keys"), "")
	for _, kv := range [][2]string{
		{"Enter", "send · Alt+Enter adds a line"},
		{"↑ ↓", "your sent messages (an unread one is pulled back first)"},
		{"/", "commands: ↑ ↓ choose, Tab or Enter take"},
		{"Esc", "browse the history; while a reply is written, interrupt that turn"},
		{"PgUp PgDn", "scroll · Ctrl+L follow new activity"},
		{"Ctrl+N", "a new goal"},
		{"Ctrl+P", "sessions"},
		{"Ctrl+G", "goals"},
		{"Ctrl+A", "approvals"},
		{"Ctrl+R", "search what you sent"},
		{"Ctrl+O", "layout: compact (a card per action) or everything"},
		{"Ctrl+T", "hold the live view"},
		{"Ctrl+X Ctrl+E", "edit the draft in $EDITOR"},
	} {
		out = append(out, row(kv[0], kv[1], 16))
	}
	return out
}

// slashMenuRows is how many commands the slash menu shows at once.
const slashMenuRows = 10

// slashMenuOpen: the draft is a command still being typed (a "/" token with more than one way to go on).
func (m emberModel) slashMenuOpen() bool {
	if !(m.mode == "" || m.mode == "normal") || m.chat.sending || !m.chat.compose.Focused() || m.chat.picker {
		return false
	}
	value := strings.TrimRight(m.chat.compose.Value(), " ")
	matches := slashMatches(value)
	return len(matches) > 0 && !(len(matches) == 1 && matches[0] == value)
}

// slashPick is the highlighted command for the current draft.
func (m emberModel) slashPick() (string, []string, int) {
	value := strings.TrimRight(m.chat.compose.Value(), " ")
	matches := slashMatches(value)
	if len(matches) == 0 {
		return "", nil, 0
	}
	sel := 0
	if m.chat.slashFor == value {
		sel = ((m.chat.slashSel % len(matches)) + len(matches)) % len(matches)
	}
	return matches[sel], matches, sel
}

// slashHint is a command's one-line description.
func slashHint(name string) string {
	for _, c := range slashCommands {
		if c.name == name {
			return c.hint
		}
	}
	return "custom command"
}

// slashMenu: the commands the draft can become, as a framed list above the composer: the name bold in a fixed
// column, what it does dim beside it, the highlighted row in the accent, a scrollbar when the list is longer than
// the box. Up / Down move, Tab or Enter takes the highlighted one.
func (m emberModel) slashMenu(w int) string {
	if !m.slashMenuOpen() {
		return ""
	}
	_, matches, sel := m.slashPick()
	// the box takes at most a third of the terminal; on a terminal too small for two rows it stays shut (Tab still
	// cycles the names)
	rows := min(slashMenuRows, len(matches), max(0, (m.height-18)/2))
	if rows < 2 {
		return ""
	}
	top := 0
	if sel >= rows {
		top = sel - rows + 1
	}
	nameW := 0
	for _, n := range matches {
		nameW = max(nameW, ansi.StringWidth(n))
	}
	nameW = min(nameW+4, 26)
	inner := w - 4 // the border, and one space each side of the text
	scroll := len(matches) > rows
	textW := inner
	if scroll {
		textW = inner - 2
	}
	thumbAt, thumbLen := 0, rows
	if scroll {
		thumbLen = max(1, rows*rows/len(matches))
		thumbAt = (rows - thumbLen) * top / max(1, len(matches)-rows)
	}
	var lines []string
	for i := 0; i < rows; i++ {
		n := matches[top+i]
		name := lipgloss.NewStyle().Foreground(cText).Bold(true)
		desc := emberDim
		if top+i == sel {
			name = lipgloss.NewStyle().Foreground(cAccent).Bold(true)
			desc = lipgloss.NewStyle().Foreground(cText)
		}
		line := name.Render(n) + strings.Repeat(" ", max(1, nameW-ansi.StringWidth(n))) +
			desc.Render(emberClip(slashHint(n), max(1, textW-nameW)))
		line = emberClip(line, textW)
		line += strings.Repeat(" ", max(0, textW-ansi.StringWidth(line)))
		if scroll {
			bar := emberDim.Render("│")
			if i >= thumbAt && i < thumbAt+thumbLen {
				bar = lipgloss.NewStyle().Foreground(cAccent).Render("┃")
			}
			line += " " + bar
		}
		lines = append(lines, " "+line+" ")
	}
	return emberBorder.Render(strings.Join(lines, "\n"))
}

// slashMenuKey moves the menu's highlight or takes the highlighted command; false when the key is not the menu's.
func (m emberModel) slashMenuKey(k string) (emberModel, bool) {
	if !m.slashMenuOpen() {
		return m, false
	}
	value := strings.TrimRight(m.chat.compose.Value(), " ")
	pick, matches, sel := m.slashPick()
	switch k {
	case "up", "down":
		d := 1
		if k == "up" {
			d = -1
		}
		m.chat.slashFor, m.chat.slashSel = value, (sel+d+len(matches))%len(matches)
		return m, true
	case "tab", "enter":
		if k == "enter" && slices.Contains(matches, value) && pick == value {
			return m, false // a complete command runs
		}
		m.chat.compose.SetValue(pick + " ")
		m.chat.compose.CursorEnd()
		m.chat.slashFor, m.chat.slashSel = "", 0
		return m, true
	}
	return m, false
}

// todoPinned is the "now" line above the composer: the task the agent is on (its todo list's current item, as
// "Task 2 of 5" with a dot per item) and what it is doing this moment (thinking, writing, running a command and for
// how long, calling a tool, waiting for you, idle). It follows the goal the agent is actually on, not the selection.
func (m emberModel) todoPinned() string {
	w := max(20, m.width)
	if m.chat.question != nil || m.sideWidth() > 0 {
		return "" // the question's own box says the agent waits for you; a wide terminal shows it in the side panel
	}
	goal := m.activeGoal()
	if goal == nil {
		if sel := m.selectedResidentGoal(); sel != nil && (sel.State == "queued" || sel.State == "paused") {
			goal = sel
		}
	}
	// the task: the current todo item, else the goal itself
	task, dots := "", ""
	if goal != nil {
		aim, _ := m.goalAim(goal)
		task = oneLine(residentText(aim))
		for i := len(m.residentEvents) - 1; i >= 0; i-- {
			e := m.residentEvents[i]
			if e.Goal != goal.ID || e.Kind != "todo" {
				continue
			}
			raw, _ := e.Data["items"].([]any)
			cur, n := 0, 0
			for _, r := range raw {
				it, _ := r.(map[string]any)
				text, _ := it["text"].(string)
				if text == "" {
					continue
				}
				n++
				switch fmt.Sprint(it["state"]) {
				case "done":
					dots += emberCtxOK.Render("●")
				case "in_progress":
					dots += emberBusy.Render("◐")
					if cur == 0 {
						cur, task = n, oneLine(residentText(text))
					}
				default:
					dots += emberDim.Render("○")
				}
			}
			if n > 0 {
				label := fmt.Sprintf("Task %d of %d", max(cur, 1), n)
				if cur == 0 {
					label = fmt.Sprintf("Tasks (%d)", n)
				}
				dots = bold(cAccent).Render(label) + " " + dots
			}
			break
		}
	}
	// now: what is happening this moment
	now, command := m.nowState(goal)
	l := m.chat.live
	if goal == nil && !l.active && command == "" {
		return "" // nothing running and no goal: no line
	}
	// two lines, so neither crowds the other: the task, then what is happening now (a command wrapped under it)
	var rows []string
	if goal != nil && dots != "" { // without a todo list the header already shows the whole goal
		rows = append(rows, ansi.Truncate(" "+dots+"  "+lipgloss.NewStyle().Foreground(cText).Render(task), w, "…"))
	}
	nowRow := " " + bold(cBlue).Render("Now") + "   " + emberBusy.Render(now)
	if command != "" {
		indent := 2 + lipgloss.Width(now) + 5
		if indent > w/2 {
			indent = 6
		}
		wrapped := strings.Split(ansi.Wordwrap(command, max(10, w-indent-1), " "), "\n")
		if len(wrapped) > 3 {
			wrapped = append(wrapped[:2], ansi.Truncate(wrapped[2], max(10, w-indent-2), "")+"…")
		}
		cmdSt := lipgloss.NewStyle().Foreground(cText)
		nowRow += "  " + cmdSt.Render(wrapped[0])
		for _, ln := range wrapped[1:] {
			nowRow += "\n" + strings.Repeat(" ", indent) + cmdSt.Render(ln)
		}
	}
	rows = append(rows, nowRow)
	return strings.Join(rows, "\n")
}

// contextBarSlots is the fixed width of the usage bar: readable at a glance,
// small enough to keep the footer one line.
const contextBarSlots = 10

// humanTokens renders a token count at footer density: 482k, 1.0M.
func humanTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%dk", (n+500)/1000)
	default:
		return fmt.Sprint(n)
	}
}

// compactTrigger normalizes the compaction threshold: a status without one
// behaves like the daemon's default, never like "compact 0%".
func compactTrigger(threshold float64) float64 {
	if threshold <= 0 || threshold >= 1 {
		return .85
	}
	return threshold
}

// contextBarLevel grades the measured footprint against the compaction
// trigger: 0 comfortable, 1 approaching, 2 at or past the trigger.
func contextBarLevel(used, capacity int, threshold float64) int {
	if used <= 0 || capacity <= 0 {
		return 0
	}
	switch share := float64(used) / float64(capacity); {
	case share >= compactTrigger(threshold):
		return 2
	case share >= compactTrigger(threshold)-0.15:
		return 1
	default:
		return 0
	}
}

// contextBar renders the measured context footprint for the footer as two
// pieces: an accent (label, bar, share) that takes the level's color, and a
// dim remainder (window size, trigger). A used count of 0 means the daemon
// has not measured anything yet; then only the window size shows — capacity
// is never presented as usage, and no percentage is invented from events.
// tps is the measured decode speed of the agent's last streamed reply; it
// only annotates a measured bar, and a count of 0 shows nothing at all.
func contextBar(used, capacity int, threshold, tps float64) (accent, rest string) {
	if capacity <= 0 {
		return "", ""
	}
	if used <= 0 {
		return "", "ctx " + humanTokens(capacity) + " window"
	}
	share := float64(used) / float64(capacity)
	if share > 1 {
		share = 1
	}
	filled := int(share*contextBarSlots + 0.5)
	bar := strings.Repeat("█", filled) + strings.Repeat("░", contextBarSlots-filled)
	speed := ""
	if tps > 0 {
		speed = fmt.Sprintf(" · %.0f tok/s", tps)
	}
	return fmt.Sprintf("ctx %s %d%%", bar, int(share*100+0.5)),
		fmt.Sprintf(" of %s · compact %.0f%%", humanTokens(capacity), compactTrigger(threshold)*100) + speed
}

// contextStateStyle colors the header's status light by what it means: busy
// work in cyan, healthy listening in green, something-needs-you in amber,
// failure in red, anything else in the dim resting tone.
func contextStateStyle(state string) lipgloss.Style {
	switch state {
	case "working", "queued", "waiting_model":
		return emberBusy
	case "listening":
		return emberHead
	case "waiting", "waiting_user", "waiting_root", "paused":
		return emberWarn
	case "error":
		return emberCtxHot
	default:
		return emberDim
	}
}

func (m emberModel) renderChat() string {
	m.syncChat()
	w := max(20, m.width)
	clip := func(s string) string { return ansi.Truncate(s, w, "") }
	head := m.chatHead(w)
	foot := m.chatFoot(w)
	if m.chat.help {
		foot += "\n" + clip("/new [goal] · /goals · /goal ID · /cancel · /resume · /compact · /approvals") + "\n" + clip("/newsession NAME · /session NAME · /sessions · /approvals · /remember TEXT · /memory · /context · /audit · /settings") + "\n" + clip("/start · /stop · /theme NAME · /help · /quit · Esc browse · Ctrl+N new · Ctrl+L follow · Ctrl+A approvals · Ctrl+G goals")
	}
	if m.chat.approvals {
		// The overlay takes the transcript's and the composer's place; the header and the footer stay. The floor is
		// small so a short terminal (with /help's extra lines) still gets a box inside the frame.
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.approvalOverlay(w, h) + "\n" + foot
	}
	if m.chat.goals {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.goalsOverlay(w, h) + "\n" + foot
	}
	if m.chat.models.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.modelsOverlay(w, h) + "\n" + foot
	}
	if m.chat.identity.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.identityOverlay(w, h) + "\n" + foot
	}
	if m.chat.advisor.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.advisorOverlay(w, h) + "\n" + foot
	}
	if m.chat.themes.open && len(m.chat.themes.names) > 0 {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.themesOverlay(w, h) + "\n" + foot
	}
	if m.chat.studio.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.studioOverlay(w, h) + "\n" + foot
	}
	if m.chat.view.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.textOverlay(w, h) + "\n" + foot
	}
	if m.chat.rewind.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.rewindOverlay(w, h) + "\n" + foot
	}
	if m.chat.page >= 0 {
		// a typed prompt opened from the page (a new note) sits under the overlay
		prompt := m.promptBox(max(18, w-2))
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)-lipgloss.Height(prompt)+1)
		if prompt != "" {
			return head + "\n" + m.pageOverlay(w, h) + "\n" + prompt + "\n" + foot
		}
		return head + "\n" + m.pageOverlay(w, h) + "\n" + foot
	}
	view := m.highlightSelection(m.chat.transcript.View())
	view = m.withJumpPill(view, m.transcriptWidth(w))
	if sw := m.sideWidth(); sw > 0 {
		tw := w - sw - 1
		view = lipgloss.JoinHorizontal(lipgloss.Top, lipgloss.NewStyle().Width(tw).Render(view), " ",
			m.sidePanel(sw, lipgloss.Height(view)))
	}
	body := []string{head, view}
	if live := m.liveBlock(w); live != "" {
		body = append(body, live)
	}
	if card := m.recapCard(w); card != "" {
		body = append(body, card)
	}
	if card := m.planCard(w); card != "" {
		body = append(body, card)
	}
	if card := m.rootsCard(w); card != "" {
		body = append(body, card)
	}
	return strings.Join(append(body, m.chatBottom(), foot), "\n")
}

func (m emberModel) chatUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	m.initChat()
	var cmd tea.Cmd
	if click, ok := msg.(tea.MouseClickMsg); ok && click.Button == tea.MouseLeft && m.onJumpPill(click.X, click.Y) {
		m.chat.transcript.GotoBottom()
		m.chat.live.hold = false
		return m, nil
	}
	if next, c, ok := m.selectMouse(msg); ok {
		return next, c
	} else {
		m = next
	}
	if _, ok := msg.(tea.MouseWheelMsg); ok {
		// the wheel always scrolls the history (the user's choice: the live thinking window took it mid-scroll);
		// PgUp / PgDn scroll the thinking window while a reply streams
		m.chat.transcript, cmd = m.chat.transcript.Update(msg)
	} else if m.chat.question != nil && m.chat.questionArmed {
		if m.chat.questionCustom {
			m.chat.questionInput, cmd = m.chat.questionInput.Update(msg)
		}
	} else if !m.chat.sending && !m.chat.approvals && !m.chat.goals && !m.chat.models.open && !m.chat.identity.open && !m.chat.advisor.open && !m.chat.themes.open && !m.chat.studio.open &&
		!m.chat.view.open && !m.chat.rewind.open {
		// (the approvals overlay owns the keyboard; a paste must not fill the draft behind it)
		m.chat.compose, cmd = m.chat.compose.Update(msg)
	}
	return m, cmd
}

func (m emberModel) chatKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	m.initChat()
	if m.chat.approvals {
		return m.approvalKey(msg)
	}
	if m.chat.goals {
		return m.goalsKey(msg)
	}
	if m.chat.models.open {
		return m.modelsKey(msg)
	}
	if m.chat.identity.open {
		return m.identityKey(msg)
	}
	if m.chat.advisor.open {
		return m.advisorKey(msg)
	}
	if m.chat.themes.open {
		return m.themesKey(msg)
	}
	if m.chat.studio.open {
		return m.studioKey(msg)
	}
	if m.chat.view.open {
		return m.textKey(msg)
	}
	if m.chat.rewind.open {
		return m.rewindKey(msg)
	}
	if msg.String() == "esc" && m.chat.recap != "" && strings.TrimSpace(m.chat.compose.Value()) == "" {
		return m.dismissRecap(), nil
	}
	if m.chat.btw.open {
		switch k := msg.String(); {
		case k == "esc" && strings.TrimSpace(m.chat.compose.Value()) == "":
			m.chat.btw = btwState{}
			m.syncChat()
			return m, nil
		case k == "shift+up":
			m.chat.btw.top = max(0, m.chat.btw.top-1)
			return m, nil
		case k == "shift+down":
			m.chat.btw.top++
			return m, nil
		}
	}
	if m.chat.page >= 0 {
		return m.pageKey(msg)
	}
	if m.chat.picker {
		return m.pickerKey(msg)
	}
	if m.chat.question != nil {
		return m.questionKey(msg)
	}
	return m.chatDraftKey(msg)
}

func (m emberModel) chatDraftKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if next, cmd, ok := m.composerKey(msg); ok {
		return next, cmd
	}
	k := msg.String()
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "ctrl+p":
		m.chat.picker, m.chat.pickerQuery, m.chat.pickerSel = true, "", 0
		m.chat.compose.Blur()
		return m, residentSessions(m)
	case "ctrl+g":
		m.chat.goals, m.chat.goalsSel, m.chat.goalsDetail = true, 0, false
		m.chat.compose.Blur()
		return m, nil
	case "ctrl+a":
		// The Approvals overlay over the conversation: the requests a person has to decide, in the one box every
		// overlay uses. It owns the keys while it is open, so the composer lets go of them.
		m.chat.approvals, m.chat.approvalSel, m.chat.approvalDetail = true, 0, false
		m.chat.armed = ""
		m.chat.compose.Blur()
		return m, nil
	case "tab", "shift+tab":
		if next, done := m.slashMenuKey("tab"); done && k == "tab" {
			next.syncChat()
			return next, nil
		}
		if next, done := m.cycleSlash(); done {
			return next, nil
		}
		if next, done := m.cyclePath(); done {
			return next, nil
		}
		return m, nil // the chat is the only screen: Tab completes and does nothing else
	case "pgup":
		if m.chat.live.active {
			m.scrollLive(-m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		if m.chat.live.active {
			m.scrollLive(m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageDown()
		return m, nil
	case "ctrl+l", "ctrl+end":
		m.chat.transcript.GotoBottom()
		m.chat.live.hold = false // follow the live tail again too
		return m, nil
	case "ctrl+e":
		// (Ctrl+O's layout switch; Ctrl+E kept for the hands that learned it)
		return m.toggleLayout(), nil
	case "ctrl+t":
		// Hold the live window where it is while the reply keeps arriving.
		m.chat.live.hold = !m.chat.live.hold
		if !m.chat.live.hold {
			m.chat.live.top = 0 // following recomputes the tail from the body length
		}
		return m, nil
	case "alt+up":
		m.scrollLive(-3)
		return m, nil
	case "alt+down":
		m.scrollLive(3)
		return m, nil
	case "esc":
		if m.chat.live.active {
			// Esc while a reply is being written interrupts that turn only; the next message steers.
			if goal := m.targetGoal(); goal != "" {
				m.msg = "interrupting the turn…"
				return m, interruptTurn(m, goal)
			}
		}
		m.chat.compose.Blur()
		return m, nil
	case "up", "down":
		if next, done := m.slashMenuKey(k); done {
			return next, nil
		}
		if next, done := m.recall(k); done {
			return next, nil
		}
		if k == "down" {
			break
		}
		if strings.TrimSpace(m.chat.compose.Value()) == "" && !m.chat.newGoal && m.chat.target != "" {
			// ↑ on an empty composer pulls back your newest message the agent has not read yet
			return m, unsendLast(m, m.chat.target)
		}
	case "ctrl+n":
		if !m.chat.sending {
			m.chat.newGoal = true
			m.chat.forceNew = true
			m.chat.target = ""
			m.chat.compose.Reset()
			return m, m.chat.compose.Focus()
		}
		return m, nil
	}
	if !m.chat.compose.Focused() {
		// A pending root request is decided right here, through the Requests page's own handler (the same sudo
		// run, claim, audit entry and answer files): a approves, d denies, e opens it in the Requests editor.
		// Approving runs a root command, so it takes two presses: the first a arms it and shows the exact
		// command, a second a within 10 s approves. Any other key disarms. Denying runs nothing and needs one.
		armed := m.chat.armed
		m.chat.armed = ""
		if i := m.pendingRoot(); i >= 0 && (k == "a" || k == "g" || k == "d" || k == "e") {
			r := m.reqs[i]
			// A command you granted for this goal (checked against your own audit, not the agent's word) runs with
			// one a; anything else, and every grant, takes two presses.
			granted := k == "a" && grantRecorded(m.auditPath, r, time.Now())
			if (k == "a" || k == "g") && !granted && (armed != k+":"+r.ID || time.Since(m.chat.armedAt) > 10*time.Second) {
				m.chat.armed, m.chat.armedAt = k+":"+r.ID, time.Now()
				what := "approve as root: "
				if k == "g" {
					what = "approve and grant for this goal (the same command then needs one a): "
				}
				m.msg = what + terminalText(r.Cmd, false) + " ? press " + k + " again (any other key cancels)"
				if r.kind() == "github" {
					m.chat.armed = ""
					m.msg = "a hand-off runs once, as filed: a a runs it as you, d denies"
					if k == "a" {
						m.chat.armed = k + ":" + r.ID
						m.msg = "run on GitHub as you: " + terminalText(r.Summary, false) + " ? press a again (any other key cancels)"
					}
				}
				return m, nil
			}
			if r.kind() == "github" && (k == "g" || k == "e") {
				return m, nil // a hand-off runs once, as filed
			}
			m.sel = i
			return m.reqKeys(k) // e opens the edit prompt in the composer's slot
		}
		if armed != "" {
			m.msg = "approval cancelled"
		}
		if roots := m.pendingRoots(); len(roots) > 1 && (k == "[" || k == "]") {
			pos := 0
			for n, i := range roots {
				if i == m.pendingRoot() {
					pos = n
				}
			}
			if k == "[" {
				pos = max(0, pos-1)
			} else {
				pos = min(len(roots)-1, pos+1)
			}
			m.chat.rootSel = m.reqs[roots[pos]].ID
			m.msg = fmt.Sprintf("request %d of %d: %s", pos+1, len(roots), terminalText(m.reqs[roots[pos]].Cmd, false))
			return m, nil
		}
		switch k {
		case "enter", "t":
			return m, m.chat.compose.Focus()
		case "n":
			m.chat.newGoal = true
			m.chat.forceNew = true
			m.chat.target = ""
			return m, m.chat.compose.Focus()
		case "q":
			return m, tea.Quit
		case "up", "k":
			m.chat.transcript.ScrollUp(1)
			return m, nil
		case "down", "j":
			m.chat.transcript.ScrollDown(1)
			return m, nil
		case "home":
			m.chat.transcript.GotoTop()
			return m, nil
		case "end":
			m.chat.transcript.GotoBottom()
			return m, nil
		}
		return m, nil
	}
	if m.chat.sending {
		return m, nil
	}
	// Shell mode, as in Claude Code: "!" on an empty draft switches the composer to commands run in the agent's
	// account (its files, its rights: what it sees), shown by a "!" prompt; Backspace on an empty line or Esc leaves.
	if k == "!" && m.chat.compose.Value() == "" && !m.chat.shell {
		m.setShell(true)
		return m, nil
	}
	if m.chat.shell && m.chat.compose.Value() == "" && (k == "backspace" || k == "esc") {
		m.setShell(false)
		return m, nil
	}
	if m.chat.shell && k == "enter" {
		cmd := strings.TrimSpace(m.chat.compose.Value())
		if cmd == "" {
			return m, nil
		}
		m.chat.compose.Reset()
		m.setShell(false)
		m.msg = "running in the agent's account…"
		return m, runBang(m, cmd)
	}
	switch k {
	case "enter":
		if next, done := m.slashMenuKey("enter"); done {
			next.syncChat()
			return next, nil
		}
		return m.submitCompose()
	case "alt+enter", "shift+enter":
		m.chat.compose.InsertString("\n")
		return m, nil
	}
	return m.chatUpdate(msg)
}

func (m emberModel) submitCompose() (tea.Model, tea.Cmd) {
	text := m.chat.compose.Value()
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || m.chat.sending {
		return m, nil
	}
	// A leading // sends the rest as a message, so a goal about /etc or /tmp is not taken for a command.
	if strings.HasPrefix(trimmed, "//") {
		return m.sendCompose(strings.TrimPrefix(trimmed, "/"))
	}
	if strings.HasPrefix(trimmed, "/") {
		return m.chatSlash(trimmed)
	}
	if strings.HasPrefix(trimmed, "!") && len(trimmed) > 1 {
		m.msg = "running in the agent's account…"
		return m, runBang(m, strings.TrimSpace(trimmed[1:]))
	}
	return m.sendCompose(text)
}

func (m emberModel) sendCompose(text string) (tea.Model, tea.Cmd) {
	args := []string{"send", m.agentName, "--stdin", "--json"}
	if !m.chat.newGoal {
		args = append(args, "--goal-id", m.chat.target, "--continue")
	}
	name, epoch := m.agentName, m.residentEpoch
	m.chat.sending = true
	return m, func() tea.Msg {
		if strings.Contains(text, "@") {
			text = expandAttachments(m, text) // @path: the file, read as the agent, goes with the message
		}
		out, err := runResident(residentCommand(m, args, text))
		if err != nil {
			return residentComposeResult{agent: name, epoch: epoch, err: strings.TrimSpace(string(out)) + ": " + err.Error()}
		}
		var reply struct {
			ID   string `json:"id"`
			Note string `json:"note"`
		}
		if err = json.Unmarshal(out, &reply); err != nil || reply.ID == "" {
			return residentComposeResult{agent: name, epoch: epoch, err: "Unrecognized delivery response; draft retained. Inspect /goals before retrying."}
		}
		return residentComposeResult{agent: name, epoch: epoch, id: reply.ID, note: reply.Note}
	}
}

func (m emberModel) finishCompose(msg residentComposeResult) (tea.Model, tea.Cmd) {
	if msg.agent != m.agentName || msg.epoch != m.residentEpoch {
		return m, nil
	}
	m.chat.sending = false
	if msg.err != "" {
		m.msg = msg.err
		return m, nil
	}
	m.chat.compose.Reset()
	m.chat.forceNew = false
	m.chat.target, m.chat.selectID, m.chat.newGoal = msg.id, msg.id, false
	m.msg = "Delivered to goal " + shortGoal(msg.id)
	if msg.note != "" {
		m.msg += " · " + residentText(msg.note)
	}
	m.chat.transcript.GotoBottom()
	return m, residentRefresh(m)
}

func (m emberModel) chatSlash(text string) (tea.Model, tea.Cmd) {
	command, arg, _ := strings.Cut(text, " ")
	arg = strings.TrimSpace(arg)
	clear := func() { m.chat.compose.Reset(); m.msg = "" }
	switch command {
	case "/help":
		clear()
		m.chat.view = textView{open: true, title: "help", lines: helpLines()}
		m.chat.compose.Blur()
	case "/quit":
		return m, tea.Quit
	case "/new":
		m.chat.newGoal, m.chat.target = true, ""
		m.chat.forceNew = true
		clear()
		if arg != "" {
			m.chat.compose.SetValue(arg)
			return m.sendCompose(arg)
		}
	case "/requests":
		clear()
		m.chat.approvals, m.chat.approvalSel, m.chat.approvalDetail = true, 0, false
		m.chat.armed = ""
		m.chat.compose.Blur()
	case "/approvals":
		// The overlay over the conversation; /requests still opens the page that edits and answers.
		clear()
		m.chat.approvals, m.chat.approvalSel, m.chat.approvalDetail = true, 0, false
		m.chat.armed = ""
		m.chat.compose.Blur()
	case "/memory", "/context", "/audit", "/settings", "/jobs", "/servers", "/agents":
		clear()
		return m.openPage(pageOverlays[command].tab)
	case "/sessions":
		clear()
		m.chat.picker, m.chat.pickerQuery, m.chat.pickerSel = true, "", 0
		m.chat.compose.Blur()
		return m, residentSessions(m)
	case "/newsession":
		if arg == "" {
			m.msg = "Use /newsession NAME"
			return m, nil
		}
		clear()
		return m, createResidentSession(m, arg)
	case "/remember":
		if arg == "" {
			m.msg = "Use /remember TEXT"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"memory", m.agentName, "--stdin", "--json"}, arg)
	case "/goals":
		clear()
		m.chat.goals, m.chat.goalsSel, m.chat.goalsDetail = true, 0, false
		m.chat.compose.Blur()
	case "/goal":
		if plan, ok := strings.CutPrefix(arg, "--plan"); ok {
			if strings.TrimSpace(plan) == "" {
				m.msg = "usage: /goal --plan TEXT — it explores and proposes a plan before it changes anything"
				return m, nil
			}
			clear()
			m.chat.newGoal, m.chat.target = true, ""
			return m.sendPlanGoal(strings.TrimSpace(plan))
		}
		var found []int
		for i, g := range m.resident.Goals {
			if arg != "" && strings.HasPrefix(g.ID, arg) {
				found = append(found, i)
			}
		}
		if len(found) != 1 {
			m.msg = "Use /goal with one unambiguous goal ID from /goals"
			return m, nil
		}
		clear()
		m.goalSel = found[0]
		m.chat.forceNew = false
		m.targetSelectedGoal()
	case "/agent", "/session":
		if arg == "" {
			m.msg = "Use /agent NAME"
			return m, nil
		}
		// A typo here used to clear the transcript and leave an empty conversation behind.
		if len(m.sessionNames) > 0 && !slices.Contains(m.sessionNames, arg) {
			m.msg = "no such session " + terminalText(arg, true) + " · Ctrl+P lists them"
			return m, nil
		}
		clear()
		m.openResident(arg)
		return m, tea.Batch(residentRefresh(m), residentSessions(m))
	case "/start", "/stop":
		clear()
		return m, residentAction(m, []string{strings.TrimPrefix(command, "/"), m.agentName, "--json"}, "")
	case "/cancel":
		g := m.selectedResidentGoal()
		if g == nil {
			m.msg = "No selected goal to cancel"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"cancel", m.agentName, g.ID, "--json"}, "")
	case "/model", "/models":
		// alone: the Models overlay over the catalog; with an id (provider/id when two serve it): pick it
		clear()
		if id := strings.TrimSpace(arg); id != "" && command == "/model" {
			m.msg = "switching to " + id + "…"
			return m, pickModel(m, id)
		}
		m.chat.models.open, m.chat.models.loading, m.chat.models.err = true, true, ""
		return m, fetchModels(m, false)
	case "/identity":
		// the agent's accounts: its linked GitHub account, the identity its commits carry; link or unlink here
		clear()
		return m.openIdentity()
	case "/effort":
		clear()
		return m, setEffort(m, strings.TrimSpace(arg))
	case "/freetime":
		// a free-time stretch now: after any running work, within the day's free-time budget
		clear()
		m.msg = "free time asked: it starts at the agent's next pass (after any running work)"
		return m, residentAction(m, []string{"heartbeat", m.agentName, "now", "--json"}, "")
	case "/recap":
		clear()
		m.chat.recapAsked = recapOnDemand
		return m, fetchRecap(m, 0)
	case "/studio":
		clear()
		m.chat.studio = studioState{open: true, loading: true}
		return m, fetchStudio(m)
	case "/pause":
		g := m.selectedResidentGoal()
		if g == nil {
			m.msg = "No selected goal to pause"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"pause", m.agentName, g.ID, "--json"}, "")
	case "/halt":
		clear()
		m.msg = "emergency stop…"
		return m, emergencyStop(m, strings.TrimSpace(arg))
	case "/btw", "/diff", "/export", "/rewind", "/plan", "/advisor":
		clear()
		next, cmd, _ := m.steerSlash(command, arg)
		return next, cmd
	case "/update":
		clear()
		m.msg = "uploading the runtime…"
		return m, updateResident(m, strings.TrimSpace(arg) == "now")
	case "/compact":
		// Compaction needs the model and the single writer, so this asks the runtime to do it on its next
		// pass rather than writing history from here.
		g := m.selectedResidentGoal()
		args := []string{"compact", m.agentName}
		if g != nil {
			args = append(args, g.ID)
		}
		clear()
		return m, residentAction(m, args, "")
	case "/steer":
		// A steer redirects: it supersedes conflicting instructions and voids the plan. A plain message in the
		// composer is a nudge, which keeps the plan, so the heavy one has to be asked for.
		text := strings.TrimSpace(arg)
		if text == "" || m.chat.target == "" {
			m.msg = "usage: /steer TEXT — redirects the goal; a plain message nudges it"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"send", m.agentName, "--goal-id", m.chat.target,
			"--steer", "--stdin", "--json"}, text)
	case "/theme":
		// The panel's colors; the transcript's markdown is rebuilt below to pick
		// up the new roles. Nothing else in the session changes.
		name := strings.TrimSpace(arg)
		if name == "" {
			return m.openThemes(), nil // the theme screen: every theme, previewed on every element
		}
		if err := themeSet(name); err != nil {
			m.msg = "no such theme: " + name + " · /theme lists"
			return m, nil
		}
		m.chat.markdown = nil
		m.chatStyles() // the composer's prompt and hint keep the palette's colours
		m.msg = "theme: " + name
		m.syncChat()
		return m, nil
	case "/deny":
		// "No, do this instead": deny the newest root request and send the text to its goal as a steer.
		i := m.pendingRoot()
		if i < 0 {
			m.msg = "no root request is waiting"
			return m, nil
		}
		text := strings.TrimSpace(arg)
		if text == "" {
			m.msg = "usage: /deny TEXT — denies the root request and tells the agent what to do instead"
			return m, nil
		}
		// The instruction rides in the denial itself; the runtime files it as a steer on the request's goal (one
		// write, so a denial can never arrive without its instruction).
		r := m.reqs[i]
		m.sel = i
		m.mode, m.edit, m.editRequest = "denywhy", text, &r
		clear()
		return m.submitInput()
	case "/glyphs":
		// The symbol set (shell.go): nerd needs a Nerd Font; unicode needs none; ascii is for anything else.
		name := strings.TrimSpace(arg)
		if name == "" {
			name = glyphsNext()
		}
		if err := glyphsSet(name); err != nil {
			m.msg = err.Error()
			return m, nil
		}
		m.msg = "glyphs: " + name + "  " + g.working + " " + g.waiting + " " + g.done + " " + g.root + " " + g.model + " " + g.queue +
			"  (boxes? pick /glyphs unicode)"
		m.syncChat()
		return m, nil
	case "/resume":
		g := m.selectedResidentGoal()
		if g == nil {
			m.msg = "No selected goal"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"resume", m.agentName, g.ID, "--json"}, "")
	default:
		if path, ok := customCommands()[command]; ok {
			body, err := expandCustom(path, arg)
			if err != nil {
				m.msg = "custom command: " + err.Error()
				return m, nil
			}
			clear()
			return m.sendCompose(body)
		}
		m.msg = "Unknown command. /help lists commands; //text sends a message; your draft is retained."
	}
	m.syncChat()
	return m, nil
}

// slashCommands backs the composer autocomplete; every name must be handled by
// chatSlash. The list stays in one place so the two never drift apart.
var slashCommands = []struct{ name, hint string }{
	{"/agent", "open another session NAME"},
	{"/approvals", "open the approvals overlay (Ctrl+A)"},
	{"/cancel", "cancel the selected goal"},
	{"/compact", "summarize this goal's history into memory now"},
	{"/effort", "reasoning depth LEVEL among the model's; alone, its levels"},
	{"/model", "pick the brain: the models overlay; or /model ID"},
	{"/identity", "the agent's GitHub account: link, unlink, what its commits carry"},
	{"/advisor", "the advisor reviews the goal now and the agent reads it before its next step; /advisor setup: model, budget"},
	{"/pause", "pause the selected goal at its next step; /resume continues it"},
	{"/halt", "emergency stop: every resident pauses after its step (/halt clear lifts it)"},
	{"/update", "install this caminus's runtime between goals (now: at the next turn)"},
	{"/goal", "select a goal by ID prefix"},
	{"/goals", "the goals overlay (Ctrl+G)"},
	{"/help", "toggle the help footer"},
	{"/new", "start a new goal"},
	{"/newsession", "create a session NAME"},
	{"/quit", "close the UI; the daemon continues"},
	{"/remember", "append TEXT to this session's memory"},
	{"/requests", "open questions and root approvals"},
	{"/resume", "resume the selected goal"},
	{"/session", "open a session NAME"},
	{"/sessions", "open the Sessions homepage"},
	{"/start", "run this session's loop"},
	{"/steer", "redirect this goal with TEXT (a plain message nudges)"},
	{"/stop", "stop this session's loop"},
	{"/theme", "the theme screen: preview and pick (or /theme NAME)"},
	{"/glyphs", "symbols: nerd, unicode or ascii; alone, the next one"},
	{"/deny", "deny the waiting root request and tell the agent what to do instead"},
	{"/memory", "memory and notes"},
	{"/context", "what the agent's context holds"},
	{"/audit", "the audit log"},
	{"/settings", "this session's settings"},
	{"/jobs", "the agent's jobs and processes"},
	{"/btw", "a side question with no tools; the goal keeps nothing"},
	{"/diff", "what this goal changed in its git repositories"},
	{"/export", "this goal as markdown in the agent's ~/exports"},
	{"/rewind", "back to an earlier step, files included"},
	{"/plan", "approve or revise a waiting plan (/goal --plan TEXT starts one)"},
	{"/servers", "the model servers and their windows"},
	{"/agents", "every resident on this machine (advanced)"},
	{"/freetime", "start a free-time stretch now (after any running work)"},
	{"/recap", "while you were away: free time since your last goal"},
	{"/studio", "what free time made: the shelf, the journal, the studio"},
}

// slashMatches returns the commands whose name the draft token still prefixes.
// An exact command stays listed so repeated Tab can cycle across the longer
// names sharing its prefix; arguments or a multiline draft get nothing.
func slashMatches(draft string) []string {
	if draft == "" || !strings.HasPrefix(draft, "/") {
		return nil
	}
	if strings.ContainsAny(draft, " \n") {
		return nil
	}
	var out []string
	for _, c := range slashCommands {
		if strings.HasPrefix(c.name, draft) {
			out = append(out, c.name)
		}
	}
	for _, n := range customNames() {
		if strings.HasPrefix(n, draft) {
			out = append(out, n)
		}
	}
	slices.Sort(out) // the menu reads A to Z
	return slices.Compact(out)
}

// pathToken reports the trailing token of the draft when it is a path the user
// is still typing: it contains a separator (/, ~/ , ./, ../) and the draft does
// not end in whitespace, so a finished word never matches. A leading / alone is
// left to slashMatches, which sees commands first; a path wins once the token
// goes deeper than a bare command could (/home) or does not start with / at all.
func pathToken(draft string) (string, bool) {
	if draft == "" || strings.Contains(draft, "\n") || strings.HasSuffix(draft, " ") {
		return "", false
	}
	token := draft
	if i := strings.LastIndex(draft, " "); i >= 0 {
		token = draft[i+1:]
	}
	if !strings.Contains(token, "/") {
		return "", false
	}
	if strings.HasPrefix(token, "/") && !strings.HasPrefix(token, "//") {
		if rest := token[1:]; !strings.Contains(rest, "/") {
			// A bare /word still belongs to the command completer until no
			// command could grow into it; /h stays /help, /home is a path.
			if isCommandPrefix(rest) {
				return "", false
			}
		}
	}
	return token, true
}

// isCommandPrefix reports whether the word after the leading slash could still
// complete to any slash command.
func isCommandPrefix(token string) bool {
	for _, c := range slashCommands {
		if strings.HasPrefix(c.name[1:], token) {
			return true
		}
	}
	return false
}

// pathMatches lists what the typed path points at, directories first with
// their trailing separator, so Tab keeps descending into a directory and the
// same token prefix re-lists its children. Hidden entries only surface once
// the typed prefix includes their dot. The list is capped so /usr/lib never
// floods the hint line.
func pathMatches(token string) []string {
	dir, prefix := token, ""
	if !strings.HasSuffix(token, "/") {
		i := strings.LastIndex(token, "/")
		prefix = token[i+1:]
		dir = token[:i+1]
	}
	if dir == "~" {
		dir = "~/"
	}
	fsDir := dir
	if strings.HasPrefix(fsDir, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil
		}
		fsDir = home + fsDir[1:]
	}
	entries, err := os.ReadDir(fsDir)
	if err != nil {
		return nil
	}
	var dirs, files []string
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		if strings.HasPrefix(name, ".") && !strings.HasPrefix(prefix, ".") {
			continue
		}
		if e.IsDir() {
			dirs = append(dirs, dir+name+"/")
		} else {
			files = append(files, dir+name)
		}
	}
	out := append(dirs, files...)
	if len(out) > 64 {
		out = out[:64]
	}
	return out
}

// cyclePath completes the trailing path token on Tab. The first Tab takes the
// best match; later Tabs cycle the rest of that list; taking a directory puts
// its separator on the draft, so the next Tab walks into it. A token that
// already names exactly one thing leaves Tab free. picks state (pathPrefix,
// pathPick, pathIdx) survives only while the token is untouched.
func (m emberModel) cyclePath() (tea.Model, bool) {
	value := m.chat.compose.Value()
	token, ok := pathToken(value)
	if !ok {
		return m, false
	}
	prefix, idx := token, 0
	switch {
	case token == m.chat.pathPick && strings.HasSuffix(token, "/"):
		prefix, idx = token, 0
		m.chat.pathIdx = 0
	case token == m.chat.pathPrefix || token == m.chat.pathPick:
		// pathIdx already points at the next pick of the running cycle.
		prefix, idx = m.chat.pathPrefix, m.chat.pathIdx
	}
	matches := pathMatches(prefix)
	if len(matches) == 0 {
		return m, false
	}
	if len(matches) == 1 && matches[0] == token {
		return m, false
	}
	pick := matches[idx%len(matches)]
	m.chat.pathIdx = idx + 1
	m.chat.pathPrefix = prefix
	m.chat.pathPick = pick
	m.chat.compose.SetValue(value[:len(value)-len(token)] + pick)
	return m, true
}

// cycleSlash completes the draft to the next matching command on Tab without
// altering what the user has typed beyond the command token. Tab cycles across
// every command sharing the originally typed prefix (so /s can reach /start and
// /stop); an exact, unique token leaves Tab free for the page navigation.
func (m emberModel) cycleSlash() (tea.Model, bool) {
	value := m.chat.compose.Value()
	trimmed := strings.TrimRight(value, " ")
	token := trimmed
	if trimmed == m.chat.slashText || trimmed == m.chat.slashToken {
		token = m.chat.slashToken // continue an in-progress cycle over the original prefix
	} else {
		m.chat.slashIdx = 0
	}
	matches := slashMatches(token)
	if len(matches) == 0 {
		return m, false
	}
	pick := matches[m.chat.slashIdx%len(matches)]
	m.chat.slashIdx++
	m.chat.slashToken = token
	m.chat.slashText = pick
	if pick == trimmed {
		return m, false
	}
	m.chat.compose.SetValue(pick + " ")
	return m, true
}

// The Ctrl+P switcher selects another session without leaving the conversation
// page. Enter opens the highlighted session; the section replaces the composer
// slot, so it reuses the composer's measured height.
func (m emberModel) pickerKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "pgup":
		if m.chat.live.active {
			m.scrollLive(-m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		if m.chat.live.active {
			m.scrollLive(m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageDown()
		return m, nil
	case "esc":
		m.chat.picker = false
		return m, m.chat.compose.Focus()
	case "f2":
		m.chat.picker = false
		if m.chat.question != nil {
			m.chat.questionArmed = true
			m.chat.compose.Blur()
			if m.chat.questionCustom {
				return m, m.chat.questionInput.Focus()
			}
			return m, nil
		}
		return m, m.chat.compose.Focus()
	case "up", "k":
		m.chat.pickerSel = max(0, m.chat.pickerSel-1)
	case "down", "j":
		m.chat.pickerSel++
	case "enter":
		names := m.pickerNames()
		if len(names) == 0 {
			return m, nil
		}
		name := names[min(m.chat.pickerSel, len(names)-1)]
		if name == m.agentName {
			m.chat.picker = false
			return m, m.chat.compose.Focus()
		}
		m.openResident(name)
		return m, tea.Batch(residentRefresh(m), residentSessions(m))
	case "backspace":
		if query := []rune(m.chat.pickerQuery); len(query) > 0 {
			m.chat.pickerQuery = string(query[:len(query)-1])
		}
	default:
		if r := msgRunes(k); r != nil {
			m.chat.pickerQuery += string(r)
		}
	}
	m.chat.pickerSel = max(0, min(m.chat.pickerSel, len(m.pickerNames())-1))
	return m, nil
}

func (m emberModel) pickerBox() string {
	names := m.pickerNames()
	w := max(18, m.width-2)
	maxRows := max(1, m.height-8)
	first := 0
	if len(names) > maxRows {
		first = max(0, min(m.chat.pickerSel-maxRows+1, len(names)-maxRows))
	}
	rows := []string{emberWarn.Render("OPEN SESSION"), "› " + residentText(m.chat.pickerQuery) + "█", ""}
	if len(names) == 0 {
		rows = append(rows, emberDim.Render("no matching session"))
	}
	for idx := first; idx < min(first+maxRows, len(names)); idx++ {
		name := names[idx]
		mark := "  "
		if idx == m.chat.pickerSel {
			mark = "› "
		}
		row := mark + residentText(name)
		if idx == m.chat.pickerSel {
			row = lipgloss.NewStyle().Bold(true).Render(row)
		}
		rows = append(rows, emberClip(row, w-4))
	}
	hint := "Enter opens · ↑/↓ choose · Esc cancels"
	return emberBorder.Width(w).Render(strings.Join(rows, "\n") + "\n" + emberDim.Render(emberClip(hint, w-4)))
}

// chatFoot is the status line (shell.go) and, when something is only sometimes true, a dim line under it; bodyHeight
// measures it so the frame never outgrows the terminal.
func (m emberModel) chatFoot(w int) string {
	clip := func(s string) string { return ansi.Truncate(s, w, "") }
	queued := 0
	for _, gl := range m.resident.Goals {
		if gl.State == "queued" {
			queued++
		}
	}
	// The status line: brain, context, speed and queue, then the keys that matter now (shell.go). Everything that is
	// only sometimes true (scroll position, the live hold, history loading, a message) rides on the line under it.
	keys := []string{"esc browse", "^P sessions", "/help"}
	if m.chat.seq < m.resident.LastEvent {
		keys = append([]string{"loading history"}, keys...)
	}
	if !m.chat.transcript.AtBottom() {
		keys = append([]string{fmt.Sprintf("history %.0f%%", m.chat.transcript.ScrollPercent()*100)}, keys...)
	}
	if !m.sshOK {
		keys = append([]string{emberWarn.Render("SSH …")}, keys...)
	}
	if m.chat.live.active && m.chat.live.hold {
		keys = append([]string{"held · ^T follow"}, keys...)
	}
	if len(m.reqs) > 0 {
		keys = append([]string{fmt.Sprintf("%s %d pending ^A", g.root, len(m.reqs))}, keys...)
	}
	// Spend shows once anything has been priced (a paid brain); a local brain's turns cost nothing and hide it.
	usd := -1.0
	if m.resident.SpendToday > 0 || len(m.resident.SpendGoals) > 0 {
		usd = m.resident.SpendGoals[m.chat.target]
	}
	if ag := m.activeGoal(); ag != nil && m.compactingNote(ag.ID) != "" {
		keys = append([]string{emberWarn.Render(g.spinner[m.tickN%len(g.spinner)] + " " + m.compactingNote(ag.ID))}, keys...)
	}
	statusBar := statusLine(status{brain: residentText(m.resident.Model), used: m.resident.ContextUsed,
		window: m.resident.Context, compact: m.resident.CompactThreshold, usd: usd, usdDay: m.resident.SpendToday,
		tps: m.resident.TokPerS, queue: queued, compactions: m.goalCompactions(), online: m.onlineSeconds(), cacheHit: m.cacheHit(), keys: keys}, w)
	_ = clip
	// (no line under the status line: the occasional states ride among its keys and a message sits on the composer's
	// frame, so nothing appears and goes below it)
	return statusBar
}

// pendingRoots are the indexes in m.reqs of this resident's waiting root requests, oldest first.
func (m emberModel) pendingRoots() []int {
	var out []int
	for i, r := range m.reqs {
		if r.kind() != "question" && (r.Agent == "" || r.Agent == m.agentName) { // root requests and GitHub hand-offs
			out = append(out, i)
		}
	}
	return out
}

// pendingRoot is the index in m.reqs of the root request the keys act on: the one highlighted on the shared card
// ([ and ] move it), else the newest; -1 when none waits.
func (m emberModel) pendingRoot() int {
	roots := m.pendingRoots()
	if len(roots) == 0 {
		return -1
	}
	for _, i := range roots {
		if m.reqs[i].ID == m.chat.rootSel {
			return i
		}
	}
	return roots[len(roots)-1]
}

// rootsCard: when several root requests wait, one card above the composer lists them all; the decision keys act on
// the highlighted one. One request needs no extra card (its card is in the transcript).
func (m emberModel) rootsCard(w int) string {
	roots := m.pendingRoots()
	anyGranted := false
	for _, i := range roots {
		anyGranted = anyGranted || grantRecorded(m.auditPath, m.reqs[i], time.Now())
	}
	if len(roots) < 2 && !anyGranted {
		return ""
	}
	cur := m.pendingRoot()
	inner := max(10, min(w, 96)-4)
	var rows []string
	for n, i := range roots {
		r := m.reqs[i]
		line := fmt.Sprintf("%d  %s", n+1, residentText(r.Cmd))
		if grantRecorded(m.auditPath, r, time.Now()) {
			line += "   granted for this goal: one a runs it"
		}
		if i == cur {
			rows = append(rows, emberSel.Render(ansi.Truncate(g.collapsed+" "+line, inner, "…")))
		} else {
			rows = append(rows, emberDim.Render(ansi.Truncate("  "+line, inner, "…")))
		}
	}
	rows = append(rows, emberDim.Render("esc, then [ ] pick  ·  a a approve  ·  g g grant for the goal  ·  e edit  ·  d deny  ·  /deny TEXT what instead"))
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cYellow).Padding(0, 1).
		Width(min(w, 96)).Render(strings.Join(rows, "\n"))
	title := fmt.Sprintf("%s %d ROOT REQUESTS WAITING", g.root, len(roots))
	if len(roots) == 1 {
		title = g.root + " ROOT REQUEST WAITING"
	}
	return borderTitle(box, title, emberWarn.Bold(true))
}

// toggleLayout switches the two layouts: compact (a card per action with short previews, replies in full) and
// everything (every call, result and gate as a card, in full, with the thinking).
func (m emberModel) toggleLayout() emberModel {
	all := !(m.chat.cards && m.chat.expanded)
	m.chat.cards, m.chat.expanded, m.chat.full = all, all, all
	m.msg = map[bool]string{true: "everything: every card in full, the thinking, every gate · Ctrl+O back to compact",
		false: "compact: a card per action, short previews · Ctrl+O shows everything"}[all]
	m.chat.content, m.chat.rendered = "", 0
	m.syncChat()
	return m
}

// The jump pill: scrolled up through the history, its last row carries " Jump to bottom (Ctrl+End) ↓ " in the
// middle; a click on it or Ctrl+End goes back to the newest message.
const jumpPillText = " Jump to bottom (Ctrl+End) ↓ "

func (m emberModel) jumpPillShown() bool {
	t := m.chat.transcript
	return t.TotalLineCount() > t.Height() && !t.AtBottom()
}

// transcriptWidth is the history's width inside a w-wide chat (the side panel takes the rest).
func (m emberModel) transcriptWidth(w int) int {
	if sw := m.sideWidth(); sw > 0 {
		return w - sw - 1
	}
	return w
}

func (m emberModel) jumpPillX(tw int) int {
	return max(0, (tw-ansi.StringWidth(jumpPillText))/2)
}

func (m emberModel) withJumpPill(view string, tw int) string {
	if !m.jumpPillShown() {
		return view
	}
	rows := strings.Split(view, "\n")
	last := rows[len(rows)-1]
	x0, pw := m.jumpPillX(tw), ansi.StringWidth(jumpPillText)
	left := ansi.Truncate(last, x0, "")
	if pad := x0 - ansi.StringWidth(left); pad > 0 {
		left += strings.Repeat(" ", pad)
	}
	rows[len(rows)-1] = left + "\x1b[0m" + emberSel.Render(jumpPillText) + ansi.TruncateLeft(last, x0+pw, "")
	return strings.Join(rows, "\n")
}

// onJumpPill: a click at (x, y) lands on the pill.
func (m emberModel) onJumpPill(x, y int) bool {
	if !m.jumpPillShown() {
		return false
	}
	w := max(20, m.width)
	row := lipgloss.Height(m.chatHead(w)) + m.chat.transcript.Height() - 1
	x0 := m.jumpPillX(m.transcriptWidth(w))
	return y == row && x >= x0 && x < x0+ansi.StringWidth(jumpPillText)
}
