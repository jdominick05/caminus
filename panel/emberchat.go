// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"slices"
	"sort"
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
	expanded                                bool      // Ctrl+O: every card open (tool calls, output, thinking)
	home, cwd                               string    // the agent's home and working folder: cards show paths short
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
	later                                   map[string]bool   // requests put off with Esc: they wait; Tab brings them back
	reqOutcome                              map[string]string // each request's state for its one-line record
	resSeen                                 map[string]string // answers read from the bridge's res/ (final: read once)
	questionAt                              time.Time         // when the prompt opened (a grace period before Enter decides)
	picker                                  bool
	pickerSel                               int
	hostShell                               bool               // shell mode runs in your own shell on this machine (!!), not the agent's
	shellStash                              string             // the draft held aside while a second shell command is typed
	prevTarget                              string             // the goal before Ctrl+N or /new (Esc goes back to it)
	goalsFollow                             string             // the goal the overlay's selection follows after a move
	pickedGoal                              string             // the goal picked by hand in the overlay (no refresh moves off it)
	clearedDraft                            string             // the draft Ctrl+C / Ctrl+U cleared: the next Up brings it back first
	sessionDone                             map[string]float64 // terminal sessions seen to end, with their exit code
	menuClosed                              string             // the draft Esc closed the / or path menu on (it stays shut until it changes)
	quitArmed                               time.Time          // Ctrl+C cleared a draft: a second one within 2 s quits
	pickerNew                               bool               // the switcher is asking for a new session's name
	pickerName                              string             // that name as typed
	pickerDestroy                           string             // the session x armed: a second x destroys it
	goals, goalsDetail                      bool               // the Goals overlay, and its selected goal opened in full
	loadingShown                            bool               // the transcript shows "loading NAME…"
	goalsSel                                int
	models                                  modelsState   // the Models overlay (/model)
	studio                                  studioState   // the Studio overlay (/studio)
	recap                                   string        // the "while you were away" card, until Esc
	chordAt                                 time.Time     // Ctrl+X pressed: Ctrl+E within 2 s opens $EDITOR
	histQuery                               string        // Ctrl+R: the draft it searches for
	histIdx                                 int           // Ctrl+R: how far back it is
	full                                    bool          // Ctrl+O: cards show every line
	rootSel                                 string        // the root request the decision keys act on (shared card)
	group                                   toolGroup     // compact view: the tool work since the last drawn block
	cards                                   bool          // Ctrl+O: every tool card instead of the compact line
	sel                                     chatSelection // select to copy (emberselect.go)
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
	dirs                                    map[string][]string // the agent's folders listed for @ / + completion
	dirsPending                             map[string]bool
	pathPrefix, pathPick                    string
	pathSel                                 int    // the path suggestions' highlighted row (arrows)
	pathSelFor                              string // the token pathSel belongs to (a new token starts at its pick)
	live                                    residentLive
	liveShown                               string // the reply being written as the transcript last drew it (wide terminals)
	stream                                  *streamState
	streamStarting                          bool      // a reader is being started: the tick starts no second one
	streamRetry                             time.Time // a reader that ended is restarted no sooner than this
	deadTurn                                string    // the live turn an Esc dropped: its late lines are ignored
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
	wrapGist              *wrapCache // the reasoning wrapped for the wide transcript's dimmed last lines (liveInline)
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
		for _, ln := range strings.Split(ansi.Wrap(strings.TrimSuffix(s[c.done:cut], "\n"), width, ""), "\n") {
			c.lines = append(c.lines, style(ln))
		}
		c.done = cut
	}
	out := slices.Clip(c.lines)
	for _, ln := range strings.Split(ansi.Wrap(s[cut:], width, ""), "\n") {
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
	out = append(out, l.reply(width)...)
	if len(out) == 0 {
		out = []string{emberDim.Render("generating…")}
	}
	return out
}

// reply is the live reply without its reasoning: the tools it started, then the text.
func (l residentLive) reply(width int) []string {
	var out []string
	for _, name := range l.tools {
		out = append(out, emberDim.Render(g.collapsed+" ")+emberTool.Render(g.tool+" "+residentText(name))+emberDim.Render(" …"))
	}
	if l.text != "" {
		if l.wrapText == nil {
			l.wrapText = &wrapCache{}
		}
		out = append(out, l.wrapText.wrap(l.text, width, highlightLive)...)
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

// hostShellMark is the host shell's prompt: "!!", run as you on this machine.
func hostShellMark(info textarea.PromptInfo) string {
	if info.LineNumber == 0 {
		return "!! "
	}
	return ""
}

// setShell switches shell mode: the prompt, its colour and the placeholder say which mode the draft is in.
func (m *emberModel) setShell(on bool) {
	m.chat.shell = on
	if !on {
		m.chat.hostShell = false
	}
	if on && m.chat.hostShell {
		m.chat.compose.SetPromptFunc(3, hostShellMark)
	} else if on {
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
	if m.chat.hostShell {
		s.Focused.Prompt, s.Blurred.Prompt = hostShellStyle(), hostShellStyle()
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
			if e.Kind == "turn_interrupted" {
				// The dropped turn's tail goes (nothing of it was kept), and a late line of it must not bring it back.
				turn, _ := e.Data["turn"].(string)
				if turn == "" || m.chat.live.turn == turn {
					m.chat.live = residentLive{}
				}
				if turn != "" {
					m.chat.deadTurn = turn
				}
			}
		}
	}
	if m.chat.selectID != "" {
		selected = m.chat.selectID
	}
	if id := m.chat.goalsFollow; id != "" { // the overlay's ▸ on the goal that moved, at its new place
		for row, i := range m.goalOrder() {
			if m.resident.Goals[i].ID == id {
				m.chat.goalsSel = row
			}
		}
	}
	m.goalSel = min(m.goalSel, max(0, len(m.resident.Goals)-1))
	found := false
	for i, g := range m.resident.Goals {
		if g.ID == selected {
			m.goalSel = i
			m.chat.selectID = ""
			found = true
			break
		}
	}
	// nothing chosen yet (a fresh panel), or the chosen goal was cancelled: the goal being worked on, else the newest
	// one not cancelled. The newest goal is not the one in progress when later ones were queued and cancelled: the
	// header, the composer, /compact and /rewind all followed a cancelled goal.
	if g := m.selectedResidentGoal(); !found || (g != nil && g.State == "cancelled" && g.ID != m.chat.pickedGoal) {
		if i := m.liveGoalIndex(); i >= 0 {
			m.goalSel = i
		}
	}
	if m.chat.compose.Value() == "" && !m.chat.sending && !m.chat.forceNew && m.chat.selectID == "" {
		m.targetSelectedGoal()
	}
	m.syncQuestion()
	m.syncChat()
}

// liveGoalIndex is the goal being worked on (or waiting for you), else the newest goal not cancelled; -1 for none.
func (m emberModel) liveGoalIndex() int {
	for i, g := range m.resident.Goals {
		switch g.State {
		case "working", "waiting_model", "waiting_user", "waiting_root", "waiting_plan", "paused":
			return i
		}
	}
	for i, g := range m.resident.Goals {
		if g.State != "cancelled" {
			return i
		}
	}
	return -1
}

// recordLine is a request's one-line record in the transcript.
func recordLine(icon, text string, w int) string {
	return emberClip(" "+emberWarn.Render(icon)+" "+emberDim.Render(text), max(10, w))
}

// recordState is a record whose body is cut, never its state (" · no: open it as a draft PR" stays whole), then
// what follows the state (a waiting question's choices) in the room left.
func recordState(icon, body, state, after string, w int) string {
	tail := ""
	if state != "" {
		tail = " · " + state
	}
	if room := max(10, w) - 4; ansi.StringWidth(tail) > room-12 {
		tail = ansi.Truncate(tail, max(12, room-12), "…")
	}
	room := max(12, max(10, w)-4-ansi.StringWidth(tail))
	text := ansi.Truncate(body, room, "…") + tail
	if after != "" {
		text += "  " + after
	}
	return recordLine(icon, text, w)
}

// humanRefusal says why a request was not filed, for a person: the gate's text addresses the model.
func humanRefusal(err string) string {
	switch {
	case strings.Contains(err, "no done check yet"):
		return "the agent asked before stating what done looks like"
	case strings.Contains(err, "already waiting"), strings.Contains(err, "still waiting"):
		return "another request is already waiting for you"
	}
	return firstLine(err)
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
	oldShare := m.chat.transcript.ScrollPercent() // (a redraw at a new width keeps this share of the history in view)
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
	// One header line and one footer line surround the transcript; on a narrow terminal the live tail sits between it
	// and the composer, so a generating reply never rewrites the durable content.
	// Half the body is the live window while a reply is written; the transcript keeps the other half.
	tail := 0
	if live := m.liveBlock(w); live != "" {
		tail = lipgloss.Height(live)
	}
	// a wide terminal writes the reply into the transcript instead, as its last block (liveInline)
	inline := m.liveInline(w)
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
	// a command first seen running and later seen to end: its card is drawn again with the exit
	if done := m.endedSessions(); fmt.Sprint(done) != fmt.Sprint(m.chat.sessionDone) {
		m.chat.sessionDone = done
		m.chat.content, m.chat.rendered = "", 0
	}
	// a request's one-line record carries its state (waiting, yes, no: …): when one changes, the history is drawn
	// again (rare: a request filed, answered or timed out)
	if out := m.requestOutcomes(); fmt.Sprint(out) != fmt.Sprint(m.chat.reqOutcome) {
		m.chat.reqOutcome = out
		m.chat.content, m.chat.rendered = "", 0
	}
	if m.chat.home != m.resident.Home || m.chat.cwd != m.agentCwdAbs() {
		m.chat.home, m.chat.cwd = m.resident.Home, m.agentCwdAbs()
		m.chat.content, m.chat.rendered = "", 0 // (the paths on the cards are drawn short against them)
	}
	rendering := m.chat.rendered == 0 // drawing the history from the start (a resize, Ctrl+O, a state change)
	changed := m.chat.rendered != len(m.residentEvents)
	if m.chat.rendered == 0 {
		m.chat.group = toolGroup{}   // drawing from the start (a resize, Ctrl+O): the group starts over too
		m.chat.sel = chatSelection{} // and the lines a selection pointed at are gone
	}
	for _, e := range m.residentEvents[m.chat.rendered:] {
		// Compact (the default): a card per action between the agent's words; Ctrl+O draws everything.
		if !m.chat.cards && !m.chat.expanded {
			if e.Kind == "assistant" {
				// its words first (they close the line above them), then its calls start the next line
				if block := m.chat.eventBlock(compactAssistant(e)); block != "" {
					m.chat.flushGroup(w)
					m.chat.content += ansi.Wrap(block, w, "") + "\n\n" // (at words: Hardwrap broke them mid-word)
				}
				m.chat.fold(e)
				continue
			}
			if e.Kind == "tool_result" {
				// every action is its own card (60 calls, 60 cards: emberactions.go)
				if block := m.chat.actionBlock(e, w); block != "" {
					m.chat.fold(e) // first: a finished command's "running:" line must not be fixed above its card
					m.chat.flushGroup(w)
					m.chat.content += block + "\n\n"
					continue
				}
			}
			if m.chat.fold(e) {
				continue
			}
		} else if e.Kind == "assistant" {
			m.chat.noteCalls(e) // (a search result's card names its pattern from the call)
		}
		if block := m.chat.eventBlock(e); block != "" {
			m.chat.flushGroup(w)
			m.chat.content += ansi.Wrap(block, w, "") + "\n\n"
		}
	}
	m.chat.rendered = len(m.residentEvents)
	if changed || m.chat.transcript.TotalLineCount() == 0 || m.chat.loadingShown != m.statusAt.IsZero() || inline != m.chat.liveShown {
		body := strings.TrimRight(m.chat.content, "\n")
		if !m.chat.group.empty() {
			body += "\n\n" + m.chat.group.line(w) // the work in progress, fixed when the next block arrives
		}
		if m.chat.liveShown = inline; inline != "" { // the reply being written, until its durable block replaces it
			if body != "" {
				body += "\n\n"
			}
			body += inline
		}
		if body != "" && m.height >= 24 && m.sideWidth() > 0 {
			body += "\n" // the gap above the composer, in the transcript's column only (chatBottom)
		}
		m.chat.loadingShown = false
		if body == "" && m.statusAt.IsZero() {
			body = emberDim.Render("loading " + residentText(m.agentName) + "…")
			m.chat.loadingShown = true // (replaced by the welcome or the history once the status lands)
		} else if body == "" {
			room := max(10, w-lipgloss.Width(mascot(0)[0])-2) // (a narrow terminal cut the lines with no marker)
			body = mascotBeside(0, "", emberTitle.Render("EMBER")+emberDim.Render(" · Caminus"),
				emberClip("Your agent's conversation and commands will stay here.", room),
				emberDim.Render(emberClip("Type a goal below. /help shows the controls.", room)))
		}
		redrawn := m.chat.transcript.TotalLineCount() > 0 && rendering
		m.chat.transcript.SetContent(body)
		if redrawn && !oldBottom {
			// a resize redrew the history at the new width: the place you were reading stays (80->160->80 lost it)
			m.chat.transcript.SetYOffset(int(oldShare * float64(max(0, m.chat.transcript.TotalLineCount()-m.chat.transcript.Height()))))
		}
	}
	if oldBottom {
		m.chat.transcript.GotoBottom()
	}
}

// bodyHeight is the space between the header and the footer, without the composer.
func (m emberModel) bodyHeight() int {
	// the header is two lines (title, rule) or more when the goal wraps under it; the footer one or two (chatFoot)
	w := max(20, m.width)
	// the cards drawn under the transcript (a recap, a waiting plan) take their rows from it too: they pushed the
	// composer and the status line off the screen
	cards := 0
	for _, c := range []string{m.recapCard(w), m.planCard(w)} {
		if c != "" {
			cards += lipgloss.Height(c)
		}
	}
	return max(1, max(12, m.height)-lipgloss.Height(m.chatHead(w))-lipgloss.Height(m.chatBottom())-lipgloss.Height(m.chatFoot(w))-cards)
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
	// a new goal being written (Ctrl+N, /new, after /cancel) is no goal yet: the header showed the previous one's
	// "completed" over a new-goal composer
	if sel := m.selectedResidentGoal(); sel != nil && !(m.chat.newGoal && m.chat.forceNew) {
		state, steps = sel.State, sel.Steps
		if at := m.parkedUntil(sel.ID); at != "" {
			state = "parked until " + at // (the status still says working)
		}
		aim, redirected := m.goalAim(sel)
		goalText = strings.Join(strings.Fields(residentText(aim)), " ")
		if redirected {
			goalText = "↪ " + goalText
		}
	}
	if sel := m.selectedResidentGoal(); sel != nil && sel.State == "working" && m.waitingForYou(sel.ID) {
		state = "waiting for you"
	}
	if m.statusAt.IsZero() {
		// no status yet (a session just opened or switched to): its blank state is not "stopped", which flashed
		// with the empty-session welcome on every switch
		state, steps, goalText = "loading…", 0, ""
	} else if !m.resident.DaemonAlive {
		// a stopped daemon runs nothing, whatever its goals last recorded: the header said "working" for an hour
		// after a stop while a newly queued goal waited
		state = "stopped · /start"
	}
	agent := residentText(m.agentName)
	head := shellHeader(agent, state, goalText, steps, w)
	if first := strings.SplitN(head, "\n", 2)[0]; !strings.Contains(ansi.Strip(first), "…") || goalText == "" {
		return head
	}
	head = shellHeader(agent, state, "", steps, w)
	title, rule, _ := strings.Cut(head, "\n")
	// Wrap breaks a long path too (Wordwrap left it to be cut at the edge with no marker)
	wrapped := strings.Split(ansi.Wrap(goalText, max(10, w-4), " "), "\n")
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
	if m.statusAt.IsZero() {
		return "◌ " + residentText(m.agentName) + " · loading — caminus ember"
	}
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
func (m emberModel) liveWindow() int { return max(1, min(m.liveHeight(), thoughtRows+1)-1) }

// sideWidth is the side panel's width on a wide terminal (0: none). The transcript's prose wraps at readingWidth, so
// past about 160 columns the right part of the screen was empty; the panel takes it.
func (m emberModel) sideWidth() int {
	if m.width < sideMinTerminal {
		return 0
	}
	return min(sideMax, m.width-readingWidth-8)
}

const sideMinTerminal, sideMax = 160, 90

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

// sidePanel is the right column: the goal's whole todo list at the top, then what runs now. (A thoughts box sat above
// the tasks until the owner asked for the tasks in its place; the reply being written streams into the transcript.)
func (m emberModel) sidePanel(sw, height int) string {
	rows := []string{}
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
			rows = append(rows, sectionHead("tasks", cAccent, fmt.Sprintf("%d of %d done", done, len(items)), sw))
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
			for _, b := range m.todoBlockers(goal.ID) { // what holds it up (the tool tells the model you see these)
				rows = append(rows, " "+emberWarn.Render("!")+" "+emberWarn.Render(ansi.Truncate(oneLine(residentText(b)), max(8, sw-4), "…")))
			}
		}
	}
	now, command := m.nowState(goal)
	if len(rows) > 0 {
		rows = append(rows, "") // (with no todo list, now is the top row)
	}
	rows = append(rows, sectionHead("now", cBlue, "", sw), " "+emberBusy.Render(now))
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
	if !m.liveBoxShown() {
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

// thoughtsWidth is the text width liveBlock draws the thoughts at: the chat's (the box is a narrow terminal's only).
func (m emberModel) thoughtsWidth() int {
	return min(liveTextWidth(max(10, max(20, m.width)-4)), readingWidth)
}

// liveBoxShown: the thoughts box under the transcript holds a reply being written (a narrow terminal), so PgUp / PgDn,
// Ctrl+T and Alt+↑/↓ act on it; a wide terminal writes the reply into the transcript, which they scroll as always.
func (m emberModel) liveBoxShown() bool { return m.chat.live.active && m.sideWidth() == 0 }

// liveInline is the reply being written as the transcript's last block on a wide terminal (no box, so the right
// column keeps its full height): the live heading, the tools it started and the text, with a cursor at the end; ""
// on a narrow terminal or between replies. Its reasoning stays out, as the compact transcript folds it.
func (m emberModel) liveInline(w int) string {
	l := m.chat.live
	if !l.active || m.sideWidth() == 0 {
		return ""
	}
	// no spinner here: a frame that changed this block reset the whole transcript (a render works on a copy, so every
	// tick between live updates redrew it: 5.7 against 3.2 ms a frame at 2000 events); the activity line moves instead
	rows := []string{m.liveWho() + emberDim.Render("  live")}
	width := min(max(10, w-4), readingWidth)
	if l.text != "" {
		for _, ln := range l.reply(width) {
			rows = append(rows, "  "+ln)
		}
	} else {
		// no reply text yet: a model thinking or writing a tool call sent nothing the block showed, so it looked
		// frozen for most of a turn. Its reasoning's last lines show dimmed (the text replaces them; the landed turn
		// folds them as before), else the tool being prepared.
		if strings.TrimSpace(l.reasoning) != "" {
			if l.wrapGist == nil {
				l.wrapGist = &wrapCache{} // (a fixture without one; the live handler gives each turn its own)
			}
			// (a tab is one cell to the wrapper and up to eight on screen; trailing newlines would show empty rows)
			text := strings.TrimRight(strings.ReplaceAll(residentText(l.reasoning), "\t", "    "), "\n")
			lines := l.wrapGist.wrap(text, max(8, width-2), func(s string) string { return s })
			for _, ln := range lines[max(0, len(lines)-liveGistRows):] {
				rows = append(rows, "  "+emberDim.Render("│ "+ln))
			}
		}
		if len(l.tools) > 0 {
			rows = append(rows, "  "+emberDim.Render("preparing ")+emberTool.Render(residentText(strings.Join(l.tools, ", "))))
		}
	}
	rows[len(rows)-1] += emberBusy.Render("▌")
	return strings.Join(rows, "\n")
}

// liveGistRows is how many of the reasoning's last lines a wide transcript shows while no reply text has come.
const liveGistRows = 4

// liveWho is the live reply's heading: who is writing (thinking while only reasoning has come).
func (m emberModel) liveWho() string {
	l := m.chat.live
	who := g.agent + " agent"
	if l.reasoning != "" && l.text == "" {
		who = g.agent + " thinking"
	}
	return bold(cGreen).Render(who)
}

// liveBlock is the thoughts box under the transcript on a narrow terminal, liveWindow lines of them, w cells wide. A
// wide terminal has none: its right column holds the tasks, and the reply being written runs in the transcript.
func (m emberModel) liveBlock(w int) string {
	l := m.chat.live
	inner := max(10, w-4)
	text := min(liveTextWidth(inner), readingWidth)
	if m.sideWidth() > 0 || (!l.active && m.bodyHeight() < 2*(thoughtRows+3)) {
		return "" // (a small terminal keeps its rows for the transcript between replies)
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
	window := m.liveWindow()
	top := max(0, len(body)-window)
	if l.active && l.hold {
		top = m.liveTop(len(body))
	}
	var head string
	if l.active {
		hint := "  live · Ctrl+T hold · Alt+↑/↓ scroll"
		if l.hold {
			hint = fmt.Sprintf("  held · %d below · Ctrl+T follow", max(0, len(body)-(top+window)))
		}
		head = m.liveWho() + " " + emberBusy.Render(g.spinner[m.tickN%len(g.spinner)]) + emberDim.Render(hint)
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
	now, command := "listening", ""
	l := m.chat.live
	switch {
	case m.daemonDown():
		now = "stopped · /start" // (it said "between steps" under a stopped header)
	case goal != nil && m.parkedUntil(goal.ID) != "":
		now = "parked until " + m.parkedUntil(goal.ID)
	case goal != nil && m.compactingNote(goal.ID) != "":
		now = m.compactingNote(goal.ID)
	case goal != nil && goal.State == "working" && func() bool { _, held := m.interruptedUntil(goal.ID); return held }():
		now = "interrupted · your next message steers"
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
		now = "listening (free time runs on its own schedule)" // (the header's word for the same state; it said idle)
	default:
		now = map[string]string{
			"waiting_user": "waiting for your answer · Tab", "waiting_root": "waiting for a root approval · Tab",
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

// thoughtRows is the number of lines of thoughts the box under the transcript (liveBlock) shows, generating or not.
const thoughtRows = 8

// thoughts is the stream liveBlock shows: the last few replies' thought chains, oldest first, then the reply being
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
			for _, k := range []string{"command", "pattern", "glob", "path", "query", "url", "check", "evidence", "summary", "question"} {
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
	lines := strings.Split(ansi.Wrap(s, max(8, width), ""), "\n") // (at words; a long word still breaks)
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
// labelled turns on a thin rule, every tool call and its result as a one-line card (Ctrl+O expands them all), root
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
		// a plain message is a nudge (it keeps the plan); only /steer is labelled steer. Your line breaks stay (as a
		// Markdown hard break: soft breaks joined four lines into one)
		label := "you · " + shortGoal(e.Goal)
		if k, _ := e.Data["kind"].(string); k == "steer" {
			label = "you · steer " + shortGoal(e.Goal)
		}
		if e.Kind == "peer_message" {
			label = "peer · " + shortGoal(e.Goal)
		}
		return turn(g.user, label, cBlue, c.renderMarkdown(strings.ReplaceAll(text, "\n", "  \n")))
	case "assistant":
		var parts []string
		// The reply's reasoning is shown once, above the answer, from the durable event: it is never
		// sent back to the model, so this block is display only.
		if text := str("reasoning"); text != "" {
			if c.expanded {
				parts = append(parts, eventCard(emberThink.Render(g.expanded+" thinking")+"\n"+emberThink.Render(c.renderMarkdown(text))))
			} else {
				parts = append(parts, emberDim.Render(g.collapsed+" thinking · "+plural(strings.Count(text, "\n")+1, "line", "lines")))
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
			if name == "ask_user" || name == "read" || name == "search" || name == "edit" || name == "write" {
				continue
			} // the durable question gets a native UI below; a file tool's result card says what it did
			rawArgs, _ := f["arguments"].(string)
			var args map[string]any
			_ = json.Unmarshal([]byte(rawArgs), &args)
			if command, ok := args["command"].(string); ok {
				label := "terminal" // (any call with a command read "terminal": a root request among them)
				body := lines(command, 40)
				switch name {
				case "ask_root":
					label = "asks for root"
				case "sandbox": // marked as the box it is, with what it may reach
					label = "sandbox"
					body = append(body, sandboxNet(args, nil))
				}
				parts = append(parts, toolCard(label, firstLine(command), "", true, c.expanded, body, w))
				continue
			}
			summary := ""
			for _, key := range []string{"path", "question", "summary", "check", "evidence", "label", "id", "topic", "to", "until"} {
				if v, ok := args[key].(string); ok && v != "" {
					summary = firstLine(v)
					break
				}
			}
			body := lines(rawArgs, 40)
			if len(args) > 0 {
				body = keyLines(args, 40) // key: value lines, not a JSON dump
			}
			parts = append(parts, toolCard(residentText(name), summary, "", true, c.expanded, body, w))
		}
		return strings.Join(parts, "\n")
	case "tool_result":
		result, _ := e.Data["result"].(map[string]any)
		if name := str("name"); name == "ask_user" || name == "ask_root" || name == "github_handoff" {
			// the outcome is on the request's one-line record; only a request never filed shows here
			if errText, _ := result["error"].(string); errText != "" {
				return recordLine(map[string]string{"ask_user": "?", "ask_root": g.root, "github_handoff": "⇡"}[name], "not sent: "+humanRefusal(errText), w)
			}
			return ""
		}
		if name := str("name"); (name == "edit" || name == "write") && result["error"] == nil {
			// an edit is a diff card: open when it is short, folded (Ctrl+O) when it is long
			path, _ := result["path"].(string)
			added, _ := result["added"].(float64)
			removed, _ := result["removed"].(float64)
			meta := fmt.Sprintf("+%d −%d", int(added), int(removed))
			if created, _ := result["created"].(bool); created {
				meta = "created"
			}
			diff, _ := result["diff"].(string)
			changed := 0
			for _, ln := range strings.Split(diff, "\n") {
				if ln != "" && !strings.HasPrefix(ln, "@@") {
					changed++
				}
			}
			// line numbers, tinted added / removed plates, the code highlighted for its language (emberdiff.go)
			return editCard(name, residentText(shortPath(path, c.home, c.cwd)), meta, c.expanded || c.full || changed <= 12, diff, w)
		}
		if str("name") == "read" && result["error"] == nil {
			path, _ := result["path"].(string)
			return toolCard("read", residentText(path), readMeta(result), true, false, nil, w)
		}
		if str("name") == "search" && result["error"] == nil {
			subject, meta, _ := searchSummary(c.calls[str("call_id")].args, result, c.home, c.cwd)
			text, _ := result["text"].(string)
			return toolCard("search", subject, meta, true, c.expanded, lines(text, 40), w)
		}
		if str("name") == "search" && result["error"] == nil {
			subject, meta, _ := searchSummary(c.calls[str("call_id")].args, result, c.home, c.cwd)
			text, _ := result["text"].(string)
			return toolCard("search", subject, meta, true, c.expanded, lines(text, 40), w)
		}
		if str("name") == "browser" && result["error"] == nil {
			panel, _ := e.Data["panel"].(map[string]any)
			return toolCard("browser", browserSubject(nil, result), browserMeta(result, c.home, c.cwd), true, c.expanded,
				browserLines(result, panel), w)
		}
		if err, ok := result["error"].(string); ok {
			// a failed call stays open: the reason is what the reader needs (the first line is the head; the body is the rest)
			var rest []string
			if i := strings.IndexByte(strings.TrimSpace(err), '\n'); i >= 0 {
				rest = lines(strings.TrimSpace(err)[i+1:], 12)
			}
			return toolCard("error", residentText(str("name")), firstLine(err), false, true, rest, w)
		}
		if output, ok := result["output"].(string); ok {
			output = ansi.Strip(output) // (a program's colour codes are not text)
			meta := plural(strings.Count(strings.TrimRight(output, "\n"), "\n")+1, "line", "lines")
			if code, ok := result["exit_code"].(float64); ok {
				meta = fmt.Sprintf("exit %d · ", int(code)) + meta
			} else if running, _ := result["running"].(bool); running {
				meta = "running · " + meta
				if sid, _ := result["session"].(string); sid != "" {
					if code, done := c.sessionDone[sid]; done { // a later check saw it end
						meta = fmt.Sprintf("ended later: exit %d · ", int(code)) + strings.TrimPrefix(meta, "running · ")
					}
				}
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
			body := lines(output, 30)
			if hit := limitHit(str("name"), result); hit != "" {
				meta, code = hit+" · "+meta, 1
			}
			if str("name") == "sandbox" {
				body = append(sandboxLines(result), body...)
			}
			return toolCard("output", residentText(str("name")), meta, code == 0, c.expanded, body, w)
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
		return toolCard("result", residentText(str("name")), "", true, c.expanded, keyLines(result, 30), w)
	case "finished":
		return turn(g.done, "finished", cGreen, c.renderMarkdown(str("summary")))
	case "approval_requested":
		// one line, not a card: you decide in the prompt where you type (the outcome is the result's line)
		// a long command is cut, never its state
		if str("kind") == "github" {
			return recordState("⇡", "github: "+oneLine(residentText(str("summary"))), c.reqOutcome[str("id")], "", w)
		}
		return recordState(g.root, "root: "+oneLine(residentText(str("cmd"))), c.reqOutcome[str("id")], "", w)
	case "question":
		// the state first (a long question's choices must not push it off the line); the choices only while it waits
		// a long question is cut, never its state; a waiting one's choices follow in the room left
		state := c.reqOutcome[str("id")]
		after := ""
		if ch, _ := e.Data["choices"].([]any); len(ch) > 0 && (state == "" || state == "waiting") {
			var opts []string
			for _, x := range ch {
				if t, _ := x.(string); t != "" {
					opts = append(opts, residentText(t))
				}
			}
			after = "(" + strings.Join(opts, " / ") + ")"
		}
		return recordState("?", oneLine(residentText(str("text"))), state, after, w)
	case "plan_proposed":
		// one line each for the plan's steps: once its card closed nothing was left in the history
		return recordLine("▤", "plan proposed: "+oneLine(residentText(str("plan"))), w)
	case "plan_approved":
		return recordLine("▤", "plan approved (you)", w)
	case "plan_rejected":
		line := "plan sent back (you)"
		if n := strings.TrimSpace(str("note")); n != "" {
			line += ": " + oneLine(residentText(n))
		}
		return recordLine("▤", line, w)
	case "rewound":
		// the cards above this line are undone: the files they changed are back (terminal side effects are not)
		to, _ := e.Data["to"].(float64)
		var parts []string
		for _, k := range []string{"restored", "removed"} {
			if l, _ := e.Data[k].([]any); len(l) > 0 {
				var names []string
				for _, x := range l {
					names = append(names, shortPath(fmt.Sprint(x), c.home, c.cwd))
				}
				parts = append(parts, k+" "+strings.Join(names, ", "))
			}
		}
		line := fmt.Sprintf("rewound to step %d", int(to))
		if len(parts) > 0 {
			line += ": " + strings.Join(parts, " · ")
		}
		// wrapped, not cut: its end (the goal is paused) could not be read even in the expanded view
		return wrappedRecord("↶", line+" · the changes above are undone; the goal is paused (/resume)", w)
	case "rewind_failed":
		return recordLine("↶", "rewind failed: "+oneLine(residentText(str("error"))), w)
	case "compacted":
		return toolCard("note", "context compacted", "the whole conversation stays in this transcript", true, false, nil, w)
	case "cancelled":
		return toolCard("note", "goal "+shortGoal(e.Goal)+" cancelled", "", true, false, nil, w)
	case "turn_retried": // a daemon restart cut a step that was generating: it runs again
		step, _ := e.Data["step"].(float64)
		return recordLine("↻", fmt.Sprintf("agent restarted · step %d retried", int(step)), w)
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
		if v == "met" { // the finish line right after it carries the verdict and its reason: one line, not two
			return ""
		}
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
		// what the advisor said, which the agent reads before its next step (a card without it said nothing)
		return toolCard("note", "advisor review, asked for by you", residentText(str("model"))+" · "+firstLine(str("advice")), true, c.expanded, lines(str("advice"), 30), w)
	case "watchdog_advised":
		return toolCard("note", "the advisor stepped in", "the goal was about to pause ("+residentText(str("reason"))+"); its advice went to the agent", true, false, nil, w)
	case "woke":
		return toolCard("note", "woke up", firstLine(str("note")), true, false, nil, w)
	case "brain":
		return toolCard("note", "now answering: "+residentText(str("model")), str("brain"), true, false, nil, w)
	case "sandbox_connections": // every connection a box made or was refused, as it happened
		return wrappedRecord("⇄", "sandbox network: "+strings.Join(sandboxConnections(e.Data["connections"]), " · "), w)
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
		} else if token, ok := pathToken(value); ok && m.pathMenu(max(18, m.width-2)) == "" {
			// The path so far is rendered in the inline-code role, then what Tab
			// can offer: directories in the navigation accent, files resting, and
			// the absence of any match in the failure color so a typo shows itself.
			hint := emberDim.Render("Tab completes · ") + emberCode.Render(token) // (it overwrote the draft, so it drew twice)
			if matches := m.pathMatches(token); len(matches) > 0 {
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
				hint += emberDim.Render(" · " + strings.Join(lines, "  "))
			} else if _, abs, _ := m.pathDir(token); m.chat.dirs[abs] == nil && m.chat.dirsPending[abs] {
				hint += emberDim.Render(" listing the agent's folder…")
			} else {
				hint += emberCtxHot.Render(" no match")
			}
			input = emberClip(hint, max(18, m.width-6)) + "\n" + input
		}
	}
	box := composerBox(input, m.composerTitle(), max(18, m.width-2))
	if m.chat.shell {
		edge, st := cPurple, emberShell
		if m.chat.hostShell { // your own shell, not the agent's: its own colour, so the two never look alike
			edge, st = cOrange, hostShellStyle()
		}
		box = borderTitle(emberBorder.BorderForeground(edge).Padding(0, 1).Width(max(18, m.width-2)).Render(input),
			m.composerTitle(), st)
	}
	// a message (delivered, queued, an error) sits on the frame for a few seconds instead of a line of its own under
	// the status line, which appeared and went with every message
	if m.noticeShown() == "" || m.noticeLong() {
		return box // expired, or too long for the frame: it is drawn whole above the composer (chatBottom)
	}
	return borderNote(box, terminalText(m.msg, false), emberDim)
}

// composerTitle is what the composer's top border says: which goal a message goes to, or that it starts a new one.
// The old label's other facts (sending, a waiting question, browsing) keep their place there, shortened to a title.
func (m emberModel) composerTitle() string {
	title := "new goal"
	switch {
	case m.chat.hostShell:
		title = "host shell · runs as you, on this machine"
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
	return title
}

// composerHint is the composer's placeholder: what Enter does with the draft now.
func (m emberModel) composerHint() string {
	if m.chat.hostShell {
		return "a command, run as you in your own shell · Enter runs it · Backspace leaves"
	}
	if m.chat.shell {
		return "a command, run in the agent's account · ! again: your own shell · Enter runs it · Backspace leaves"
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
	if line := m.waitingLine(); line != "" && m.chat.question == nil {
		out = line + "\n" + out // requests put off with Esc: they still wait, Tab answers
	}
	if m.noticeLong() && m.chat.question == nil {
		// a notice too long for the composer's frame, whole (it was cut where the next step was named)
		w := max(18, m.width-4)
		ls := strings.Split(ansi.Wordwrap(terminalText(m.noticeShown(), false), w, " "), "\n")
		if len(ls) > 3 {
			ls = append(ls[:2], ansi.Truncate(ls[2], w-1, "")+"…")
		}
		out = " " + emberDim.Render(strings.Join(ls, "\n ")) + "\n" + out
	}
	if side := m.btwSection(max(18, m.width-2)); side != "" {
		out = side + "\n" + out
	}
	if menu := m.slashMenu(max(18, m.width-2)); menu != "" {
		out = menu + "\n" + out
	} else if menu := m.pathMenu(max(18, m.width-2)); menu != "" {
		out = menu + "\n" + out
	}
	// a blank line between the conversation and what sits under it, as Claude Code leaves: the composer read as part
	// of the last block when it touched it
	// (a small terminal keeps every row for the transcript; with the side panel the gap is the transcript's own last
	// line instead, so the panel reaches down to the composer)
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
	if !recalling && k == "up" && strings.TrimSpace(value) == "" && m.chat.clearedDraft != "" {
		// a draft Ctrl+C or Ctrl+U just cleared comes back first: Up went to the resident and pulled back (withdrew) a
		// sent message instead, and the draft was gone
		m.chat.compose.SetValue(m.chat.clearedDraft)
		m.chat.clearedDraft = ""
		m.msg = "your draft is back"
		return m, true
	}
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
		what := "sent message"
		if isDraft(m.agentName, past[i-1]) { // a draft Ctrl+C or Ctrl+U cleared was never sent
			what = "your draft"
		}
		m.msg = fmt.Sprintf("%s %d of %d · ↑ older · ↓ newer · edit or Enter to send", what, i, len(past))
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
func helpLines(width int) []string {
	names := make([]string, 0, len(slashCommands))
	for _, c := range slashCommands {
		names = append(names, c.name)
	}
	slices.Sort(names)
	// the description wraps at a word under its own column (an 80-column terminal cut it mid-word: "reads it be"), and
	// a tab never reaches the frame (a custom command's text with one pushed the border out)
	textW := max(20, width-6-16)
	row := func(k, v string, w int) string {
		v = strings.ReplaceAll(v, "\t", " ")
		wrapped := strings.Split(ansi.Wordwrap(v, textW, " "), "\n")
		out := lipgloss.NewStyle().Foreground(cText).Bold(true).Render(k) + strings.Repeat(" ", max(1, w-ansi.StringWidth(k))) + emberDim.Render(wrapped[0])
		for _, more := range wrapped[1:] {
			out += "\n" + strings.Repeat(" ", w) + emberDim.Render(more)
		}
		return out
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
		{"@ path", "names a path (the agent's folders; Tab completes)"},
		{"+ path", "attaches the file (read as the agent, up to 64 KB)"},
		{"Esc", "while the agent waits on a reply, interrupt that turn"},
		{"PgUp PgDn", "scroll · Ctrl+L follow new activity"},
		{"Ctrl+N", "a new goal"},
		{"Shift+Tab", "sessions: a card per session"},
		{"Ctrl+G", "goals"},
		{"Tab", "a waiting request: answer it (it shows where you type)"},
		{"Ctrl+R", "search what you sent"},
		{"Ctrl+O", "layout: compact (a card per action) or everything"},
		{"!  !!", "on an empty draft: a command in the agent's account; again: in your own shell"},
		{"Home  End", "the top or the bottom of the history (with an empty draft)"},
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
	if !(m.mode == "" || m.mode == "normal") || m.chat.sending || !m.chat.compose.Focused() || m.chat.picker || m.menuDismissed() {
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
	return m.completionMenu(matches, slashHint, sel, w)
}

// pathMenu: the @path completions in the same framed list as the commands, folders marked as such; the highlight
// follows Tab's cycle.
func (m emberModel) pathMenu(w int) string {
	if !(m.mode == "" || m.mode == "normal") || m.chat.sending || !m.chat.compose.Focused() || m.chat.picker || m.slashMenuOpen() ||
		m.menuDismissed() {
		return ""
	}
	token, ok := pathToken(strings.TrimRight(m.chat.compose.Value(), " "))
	if !ok {
		return ""
	}
	matches := m.pathMatches(token)
	if len(matches) == 0 {
		return ""
	}
	sel := m.pathSelected(token, matches)
	hint := func(p string) string {
		if strings.HasSuffix(p, "/") {
			return "folder"
		}
		return "file"
	}
	return m.completionMenu(matches, hint, sel, w)
}

// completionMenu draws the list both completions share: one match or many, the same box.
func (m emberModel) completionMenu(matches []string, hintOf func(string) string, sel, w int) string {
	// the box takes at most a third of the terminal; on a terminal too small for two rows it stays shut (Tab still
	// cycles the names, and the composer names them on one line)
	if max(0, (m.height-12)/2) < 2 || len(matches) == 0 {
		return ""
	}
	rows := min(slashMenuRows, len(matches), (m.height-12)/2) // (three rows at 80x24 while five above stayed empty)
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
			desc.Render(emberClip(hintOf(n), max(1, textW-nameW)))
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
		row := " " + dots + "  " + lipgloss.NewStyle().Foreground(cText).Render(task)
		if bl := m.todoBlockers(goal.ID); len(bl) > 0 && m.sideWidth() == 0 { // (the side pane lists them when shown)
			row += emberWarn.Render("  ! " + oneLine(residentText(bl[0])))
		}
		rows = append(rows, ansi.Truncate(row, w, "…"))
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
		foot += "\n" + clip("/new [goal] · /goals · /cancel · /resume · /compact") + "\n" + clip("/sessions · /remember TEXT · /memory · /context") + "\n" + clip("/start · /theme NAME · /help · /quit · Ctrl+N new · Ctrl+L follow · Ctrl+G goals")
	}
	if m.chat.goals {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.goalsOverlay(w, h) + "\n" + foot
	}
	if m.chat.models.open {
		h := max(6, max(12, m.height)-lipgloss.Height(head)-lipgloss.Height(foot)+1)
		return head + "\n" + m.modelsOverlay(w, h) + "\n" + foot
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
	view := m.withJumpPill(m.highlightSelection(m.chat.transcript.View(), false), m.transcriptWidth(w))
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
	return strings.Join(append(body, m.chatBottom(), foot), "\n")
}

func (m emberModel) chatUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	m.initChat()
	var cmd tea.Cmd
	if next, c, ok := m.selectMouse(msg); ok { // (a click on a picker card or the jump pill too: emberselect.go)
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
	} else if !m.chat.sending && !m.chat.goals && !m.chat.models.open && !m.chat.themes.open && !m.chat.studio.open &&
		!m.chat.view.open && !m.chat.rewind.open {
		// (the approvals overlay owns the keyboard; a paste must not fill the draft behind it)
		m.chat.compose, cmd = m.chat.compose.Update(msg)
	}
	return m, cmd
}

func (m emberModel) chatKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	m.initChat()
	if m.chat.goals {
		return m.goalsKey(msg)
	}
	if m.chat.models.open {
		return m.modelsKey(msg)
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
		// an unsent draft is not thrown away with the window: the first Ctrl+C clears it (Up brings it back), the
		// second quits
		if d := m.chat.compose.Value(); strings.TrimSpace(d) != "" && time.Since(m.chat.quitArmed) > 2*time.Second {
			saveTyped(m.agentName, d)
			markDraft(m.agentName, d)
			m.chat.clearedDraft = d
			m.chat.compose.Reset()
			m.chat.quitArmed = time.Now()
			m.msg = "draft cleared (Up brings it back) · Ctrl+C again quits"
			return m, nil
		}
		return m, tea.Quit
	case "ctrl+p":
		m.chat.picker, m.chat.pickerSel = true, 0
		m.chat.compose.Blur()
		return m, residentSessions(m)
	case "ctrl+g":
		m.chat.goals, m.chat.goalsSel, m.chat.goalsDetail = true, 0, false
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
		if k == "tab" { // no completion under way: a waiting request comes back where you type (the draft is kept)
			if next, ok := m.reopenPrompt(); ok {
				return next, nil
			}
		}
		if k == "shift+tab" { // no completion under way: Shift+Tab opens the session switcher (Ctrl+P does too)
			m.chat.picker, m.chat.pickerSel = true, 0
			m.chat.compose.Blur()
			return m, residentSessions(m)
		}
		return m, nil // the chat is the only screen: Tab completes and does nothing else
	case "pgup":
		if m.liveBoxShown() {
			m.scrollLive(-m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		if m.liveBoxShown() {
			m.scrollLive(m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageDown()
		return m, nil
	case "ctrl+u":
		// the whole draft (the editor's Ctrl+U cleared one line of a many-line draft: a command's output stayed, and
		// the next !cmd was sent with it as a message)
		if strings.Contains(m.chat.compose.Value(), "\n") {
			saveTyped(m.agentName, m.chat.compose.Value())
			markDraft(m.agentName, m.chat.compose.Value())
			m.chat.clearedDraft = m.chat.compose.Value()
			m.chat.compose.Reset()
			m.msg = "draft cleared (Up brings it back)"
			return m, nil
		}
	case "home", "end":
		if m.chat.compose.Value() != "" {
			break // (in a draft, Home and End move the cursor)
		}
		if k == "home" {
			m.chat.transcript.GotoTop()
		} else {
			m.chat.transcript.GotoBottom()
			m.chat.live.hold = false
		}
		return m, nil
	case "ctrl+l", "ctrl+end":
		m.chat.transcript.GotoBottom()
		m.chat.live.hold = false // follow the live tail again too
		return m, nil
	case "ctrl+t":
		// Hold the live window where it is while the reply keeps arriving.
		m.chat.live.hold = !m.chat.live.hold && m.liveBoxShown() // (a wide terminal has no box to hold)
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
		if m.chat.shell && m.chat.compose.Value() == "" { // out of shell mode, a draft held aside comes back
			return m.leaveShell(), nil
		}
		if !m.menuDismissed() && (m.slashMenuOpen() || m.pathMenu(m.width) != "") {
			m.chat.menuClosed = m.chat.compose.Value() // the menu closes; the text stays (typing opens it again)
			return m, nil
		}
		// Esc while the agent waits on a reply interrupts that turn only; the next message steers. Not only once text
		// streams: a long prompt read or a paid model's thinking shows nothing for a while, and the call (and the
		// server's work on it) is just as live then. The resident queues it only while a call is out. Leaving
		// "new goal" mode comes first: the call is now live for its whole length, and that Esc means "back".
		if m.chat.forceNew && m.chat.prevTarget != "" { // out of "new goal", back to the goal you were talking to
			m.chat.target, m.chat.newGoal, m.chat.forceNew = m.chat.prevTarget, false, false
			m.chat.prevTarget = ""
			m.msg = "back to goal " + shortGoal(m.chat.target)
			return m, nil
		}
		if goal := m.interruptibleGoal(); goal != "" {
			m.msg = "interrupting the turn…"
			return m, interruptTurn(m, goal)
		}
		return m, nil // (idle, Esc does nothing else: PgUp/PgDn and the wheel scroll)
	case "up", "down":
		if next, done := m.slashMenuKey(k); done {
			return next, nil
		}
		if next, done := m.pathMenuKey(k); done {
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
			// the draft stays (it becomes the new goal's text); Esc goes back to the goal you were talking to
			if m.chat.target != "" {
				m.chat.prevTarget = m.chat.target
			}
			m.chat.newGoal = true
			m.chat.forceNew = true
			m.chat.target = ""
			m.msg = "new goal · Esc goes back"
			return m, m.chat.compose.Focus()
		}
		return m, nil
	}
	if !m.chat.compose.Focused() {
		// (no browse mode any more: a composer left unfocused takes the key back to typing)
		return m, m.chat.compose.Focus()
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
	// "!" on the empty last line of a draft (a command's output waits there): shell mode, the draft held aside; the
	// next command's output joins it, and leaving shell mode gives it back. It was typed into the draft as text,
	// and Enter sent the output and "!!cmd" as a message.
	if k == "!" && !m.chat.shell && strings.HasSuffix(m.chat.compose.Value(), "\n") && m.chat.compose.Line() == m.chat.compose.LineCount()-1 {
		m.chat.shellStash = strings.TrimRight(m.chat.compose.Value(), "\n")
		m.chat.compose.Reset()
		m.setShell(true)
		return m, nil
	}
	// "!" again on the empty agent shell: the host shell, your own account on this machine (!! as in a terminal)
	if k == "!" && m.chat.compose.Value() == "" && m.chat.shell && !m.chat.hostShell {
		m.chat.hostShell = true
		m.setShell(true)
		return m, nil
	}
	if m.chat.shell && m.chat.compose.Value() == "" && (k == "backspace" || k == "esc") {
		return m.leaveShell(), nil
	}
	if m.chat.shell && k == "enter" {
		cmd := strings.TrimSpace(m.chat.compose.Value())
		if cmd == "" {
			return m, nil
		}
		host := m.chat.hostShell
		saveTyped(m.agentName, map[bool]string{true: "!!", false: "!"}[host]+cmd) // (Up brings the command back)
		m.chat.compose.Reset()
		m.setShell(false)
		if host {
			m.msg = "running in your shell…"
			return m, runHostBang(cmd)
		}
		m.msg = "running in the agent's account…"
		return m, runBang(m, cmd)
	}
	switch k {
	case "enter":
		if next, done := m.slashMenuKey("enter"); done {
			next.syncChat()
			return next, nil
		}
		if next, done := m.pathMenuKey("enter"); done {
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
		saveTyped(m.agentName, trimmed) // (history gives back what you typed, // included)
		unmarkDraft(m.agentName, trimmed)
		m.chat.clearedDraft = ""
		return m.sendCompose(strings.TrimPrefix(trimmed, "/"))
	}
	if strings.HasPrefix(trimmed, "/") {
		return m.chatSlash(trimmed)
	}
	if strings.HasPrefix(trimmed, "!!") && len(strings.TrimSpace(trimmed[2:])) > 0 {
		// your own shell, as typed in !! mode (a recalled !!cmd ran in the agent's account with a stray !)
		saveTyped(m.agentName, trimmed)
		m.msg = "running in your shell…"
		return m, runHostBang(strings.TrimSpace(trimmed[2:]))
	}
	if strings.HasPrefix(trimmed, "!") && len(trimmed) > 1 {
		saveTyped(m.agentName, trimmed)
		m.msg = "running in the agent's account…"
		return m, runBang(m, strings.TrimSpace(trimmed[1:]))
	}
	saveTyped(m.agentName, text)
	unmarkDraft(m.agentName, text) // (a cleared draft sent later is a sent message)
	m.chat.clearedDraft = ""       // (Up after a send is the history again, newest first: an old cleared draft came first)
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
		if strings.ContainsAny(text, "@+") { // (it checked only "@": a message with only +path attached nothing)
			var problems []string
			text, problems = expandAttachmentsChecked(m, text) // +path: the file, read as the agent, goes with the message
			if len(problems) > 0 {                             // nothing is sent; the draft stays to fix
				return residentComposeResult{agent: name, epoch: epoch, err: "not sent: " + strings.Join(problems, " · ")}
			}
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
		m.chat.view = textView{open: true, title: "help", lines: strings.Split(strings.Join(helpLines(m.width), "\n"), "\n")}
		m.chat.compose.Blur()
	case "/quit":
		return m, tea.Quit
	case "/new":
		if m.chat.target != "" {
			m.chat.prevTarget = m.chat.target // (Esc goes back to it)
		}
		m.chat.newGoal, m.chat.target = true, ""
		m.chat.forceNew = true
		clear()
		if arg != "" {
			m.chat.compose.SetValue(arg)
			return m.sendCompose(arg)
		}
	case "/memory":
		clear()
		m.msg = "reading the memory…"
		return m, fetchMemoryView(m)
	case "/context":
		clear()
		return m.openPage(pageOverlays[command].tab)
	case "/sessions":
		clear()
		m.chat.picker, m.chat.pickerSel = true, 0
		m.chat.compose.Blur()
		return m, residentSessions(m)
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
	case "/goal": // (not in the menu or help: the Goals overlay targets a goal; kept for a typed ID prefix)
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
	case "/start": // (the header asks for it when the loop is down; Shift+Tab stops and starts any session)
		clear()
		return m, residentAction(m, []string{strings.TrimPrefix(command, "/"), m.agentName, "--json"}, "")
	case "/cancel":
		g := m.selectedResidentGoal()
		if g == nil {
			m.msg = "No selected goal to cancel"
			return m, nil
		}
		if goalEnded(g.State) { // (it said "cancelled" again and recorded another cancel)
			clear() // (left in the composer, the next command was appended to it: "/cancel/pause")
			m.msg = "goal " + shortGoal(g.ID) + " already " + endedWord(g.State)
			return m, nil
		}
		clear()
		// the composer starts a new goal after it (it pointed at the previous, finished goal)
		m.chat.newGoal, m.chat.target, m.chat.forceNew = true, "", true
		return m, residentAction(m, []string{"cancel", m.agentName, g.ID, "--json"}, "")
	case "/model":
		// alone: the Models overlay over the catalog; with an id (provider/id when two serve it): pick it
		clear()
		if id := strings.TrimSpace(arg); id != "" {
			m.msg = "switching to " + id + "…"
			return m, pickModel(m, id)
		}
		m.chat.models.open, m.chat.models.loading, m.chat.models.err = true, true, ""
		return m, fetchModels(m, false)
	case "/effort":
		clear()
		return m, setEffort(m, strings.TrimSpace(arg))
	case "/panel":
		clear()
		return m.openPanel()
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
		if goalEnded(g.State) {
			clear() // (left in the composer, the next command was appended to it: "/cancel/pause")
			m.msg = "goal " + shortGoal(g.ID) + " already " + endedWord(g.State) + ": nothing to pause"
			return m, nil
		}
		clear()
		return m, residentAction(m, []string{"pause", m.agentName, g.ID, "--json"}, "")
	case "/halt":
		clear()
		m.msg = "emergency stop…"
		return m, emergencyStop(m, strings.TrimSpace(arg))
	case "/btw", "/diff", "/rewind", "/plan", "/advisor":
		clear()
		next, cmd, _ := m.steerSlash(command, arg)
		return next, cmd
	case "/compact":
		// Compaction needs the model and the single writer, so this asks the runtime to do it on its next
		// pass rather than writing history from here.
		g := m.selectedResidentGoal()
		if g != nil && goalEnded(g.State) {
			clear() // (left in the composer, the next command was appended to it: "/cancel/pause")
			m.msg = "goal " + shortGoal(g.ID) + " already " + endedWord(g.State) + ": nothing to compact"
			return m, nil
		}
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
		clear() // (it alone kept "/theme" in the composer: the next command was typed onto it)
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
	{"/cancel", "cancel the selected goal"},
	{"/compact", "summarize this goal's history into memory now"},
	{"/effort", "how hard it thinks: none, low, medium, high, xhigh, max (the model's nearest); alone, what this model does"},
	{"/model", "pick the brain: the models overlay; or /model ID"},
	{"/advisor", "the advisor reviews the goal now and the agent reads it before its next step"},
	{"/pause", "pause the selected goal at its next step; /resume continues it"},
	{"/halt", "emergency stop: every resident pauses after its step (/halt clear lifts it)"},
	{"/goals", "the goals overlay (Ctrl+G)"},
	{"/help", "the commands and keys"},
	{"/new", "start a new goal"},
	{"/quit", "close the UI; the daemon continues"},
	{"/remember", "append TEXT to this session's memory"},
	{"/resume", "resume the selected goal"},
	{"/sessions", "every session as a card: switch with Enter or a click"},
	{"/start", "run this session's loop (stop and start any session from Shift+Tab)"},
	{"/steer", "redirect this goal with TEXT (a plain message nudges)"},
	{"/theme", "the theme screen: preview and pick (or /theme NAME)"},
	{"/memory", "this session's memory file"},
	{"/context", "what the agent's context holds"},
	{"/btw", "a side question with no tools; the goal keeps nothing"},
	{"/diff", "what this goal changed in its git repositories"},
	{"/rewind", "back to an earlier step, files included"},
	{"/plan", "alone: the whole plan; TEXT starts a goal that plans first; approve or revise answers a waiting plan"},
	{"/studio", "what free time made: the shelf, the journal, the studio"},
	{"/panel", "the Caminus panel: models, servers and the cards (Esc comes back)"},
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
	matches := m.pathMatches(prefix)
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
	if k != "x" {
		m.chat.pickerDestroy = ""
	}
	if m.chat.pickerNew {
		switch k {
		case "esc":
			m.chat.pickerNew = false
		case "enter":
			name := strings.TrimSpace(m.chat.pickerName)
			m.chat.pickerNew, m.chat.pickerName = false, ""
			if name != "" {
				return m, createResidentSession(m, name)
			}
		case "backspace":
			if r := []rune(m.chat.pickerName); len(r) > 0 {
				m.chat.pickerName = string(r[:len(r)-1])
			}
		default:
			if r := msgRunes(k); r != nil {
				m.chat.pickerName += string(r)
			}
		}
		return m, nil
	}
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "pgup":
		if m.liveBoxShown() {
			m.scrollLive(-m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		if m.liveBoxShown() {
			m.scrollLive(m.liveHeight())
			return m, nil
		}
		m.chat.transcript.PageDown()
		return m, nil
	case "esc", "shift+tab": // (Shift+Tab opened it: it closes it too)
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
	case "x": // destroy the selected session: armed by the first x, done by the second
		cards := m.switcherCards()
		i := m.chat.pickerSel
		if i < 0 || i >= len(cards) {
			return m, nil
		}
		if len(cards) < 2 {
			m.msg = "the last session can't be destroyed"
			return m, nil
		}
		if m.chat.pickerDestroy != cards[i].name {
			m.chat.pickerDestroy = cards[i].name
			return m, nil
		}
		m.chat.pickerDestroy = ""
		c := cards[i]
		if c.name == m.agentName { // never stay in the session being removed: the next card first
			other := cards[(i+1)%len(cards)].name
			m.openResident(other)
			m.chat.picker = true
		}
		return m, destroyResidentSession(m, c.name, c.active)
	case "n": // a new session from this one: its name on a line under the cards
		m.chat.pickerNew, m.chat.pickerName = true, ""
		return m, nil
	case "s": // stop the selected session's loop, or start it when it is stopped
		cards := m.switcherCards()
		if i := m.chat.pickerSel; i >= 0 && i < len(cards) {
			op := "stop"
			if cards[i].status == "stopped" {
				op = "start"
			}
			return m, tea.Sequence(residentAction(m, []string{op, cards[i].name, "--json"}, ""), residentSessions(m))
		}
		return m, nil
	case "left", "h":
		m.chat.pickerSel = max(0, m.chat.pickerSel-1)
	case "right", "l":
		m.chat.pickerSel++
	case "up", "k":
		m.chat.pickerSel = max(0, m.chat.pickerSel-cardCols(max(18, m.width-2)))
	case "down", "j":
		m.chat.pickerSel += cardCols(max(18, m.width-2))
	case "enter":
		if m.chat.pickerSel == len(m.switcherCards()) { // the "+ new session" card
			m.chat.pickerNew, m.chat.pickerName = true, ""
			return m, nil
		}
		return m.switchToCard(m.chat.pickerSel)
	}
	m.chat.pickerSel = max(0, min(m.chat.pickerSel, m.cardCount()-1))
	return m, nil
}

// switchToCard opens card i's session (the one in view: back to the chat).
func (m emberModel) switchToCard(i int) (tea.Model, tea.Cmd) {
	cards := m.switcherCards()
	if i < 0 || i >= len(cards) {
		return m, nil
	}
	if name := cards[i].name; name != m.agentName {
		m.openResident(name)
		return m, tea.Batch(residentRefresh(m), residentSessions(m))
	}
	m.chat.picker = false
	return m, m.chat.compose.Focus()
}

// runtimeOutdated: the daemon runs another runtime than this caminus installs (the /update command's job is now a
// hint: `caminus ember agent install` queues the new one for every running daemon, taken between goals).
func (m emberModel) runtimeOutdated() bool {
	r := m.resident
	return r.DaemonAlive && r.Heartbeat.Runtime != "" && r.RuntimeLatest != "" && r.Heartbeat.Runtime != r.RuntimeLatest
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
	keys := []string{"⇧Tab sessions", "/help"} // (Ctrl+P still opens it; Shift+Tab is the one shown)
	if m.chat.seq < m.resident.LastEvent {
		keys = append([]string{"loading history"}, keys...)
	}
	if !m.chat.transcript.AtBottom() {
		keys = append([]string{fmt.Sprintf("history %.0f%%", m.chat.transcript.ScrollPercent()*100)}, keys...)
	}
	if !m.sshOK {
		keys = append([]string{emberWarn.Render("SSH …")}, keys...)
	}
	if m.liveBoxShown() && m.chat.live.hold {
		keys = append([]string{"held · ^T follow"}, keys...)
	}
	if m.runtimeOutdated() {
		keys = append([]string{emberWarn.Render("older runtime · caminus ember agent install")}, keys...)
	}
	// Spend shows once anything has been priced (a paid brain); a local brain's turns cost nothing and hide it.
	usd := -1.0
	if m.resident.SpendToday > 0 || len(m.resident.SpendGoals) > 0 {
		usd = m.resident.SpendGoals[m.chat.target]
	}
	if ag := m.activeGoal(); ag != nil && m.compactingNote(ag.ID) != "" {
		keys = append([]string{emberWarn.Render(g.spinner[m.tickN%len(g.spinner)] + " " + m.compactingNote(ag.ID))}, keys...)
	}
	// the speed is the last finished reply's: after an Esc it read as if the dropped turn were still generating
	speed := m.resident.speed()
	if ag := m.activeGoal(); ag != nil {
		if _, held := m.interruptedUntil(ag.ID); held {
			speed = replySpeed{}
		}
	}
	statusBar := statusLine(status{brain: residentText(withEffort(m.resident.Model, m.resident.Effort)), cwd: residentText(m.agentCwd()), used: m.resident.ContextUsed,
		window: m.resident.Context, compact: m.resident.CompactThreshold, usd: usd, usdDay: m.resident.SpendToday,
		speed: speed, queue: queued, compactions: m.goalCompactions(), online: m.onlineSeconds(), cacheHit: m.cacheHit(), keys: keys}, w)
	_ = clip
	// (no line under the status line: the occasional states ride among its keys and a message sits on the composer's
	// frame, so nothing appears and goes below it)
	return statusBar
}

// toggleLayout switches the two layouts: compact (a card per action with short previews, replies in full) and
// everything (every call, result and gate as a card, in full, with the thinking).
func (m emberModel) toggleLayout() emberModel {
	all := !(m.chat.cards && m.chat.expanded)
	m.chat.cards, m.chat.expanded, m.chat.full = all, all, all
	m.msg = map[bool]string{true: "everything · Ctrl+O back to compact", false: "compact · Ctrl+O shows everything"}[all]
	// keep the place: at the bottom stays at the bottom; scrolled up keeps the same share of the history
	atBottom := m.chat.transcript.AtBottom()
	frac := m.chat.transcript.ScrollPercent()
	m.chat.content, m.chat.rendered = "", 0
	m.syncChat()
	if atBottom {
		m.chat.transcript.GotoBottom()
	} else {
		m.chat.transcript.SetYOffset(int(frac * float64(max(0, m.chat.transcript.TotalLineCount()-m.chat.transcript.Height()))))
	}
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

// keyLines is a tool's arguments or result as "key: value" lines (nested maps indented, lists joined when short), at
// most limit lines: the expanded view read as a raw JSON dump.
func keyLines(v map[string]any, limit int) []string {
	var out []string
	var walk func(m map[string]any, indent string)
	walk = func(m map[string]any, indent string) {
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			switch x := m[k].(type) {
			case map[string]any:
				out = append(out, indent+k+":")
				walk(x, indent+"  ")
			case []any:
				var parts []string
				simple := true
				for _, e := range x {
					switch e.(type) {
					case map[string]any, []any:
						simple = false
					}
					parts = append(parts, fmt.Sprint(e))
				}
				if simple && len(strings.Join(parts, ", ")) <= 100 {
					out = append(out, indent+k+": "+strings.Join(parts, ", "))
					continue
				}
				out = append(out, indent+k+":")
				for _, e := range x {
					if em, ok := e.(map[string]any); ok {
						out = append(out, indent+"  -")
						walk(em, indent+"    ")
					} else {
						out = append(out, indent+"  - "+fmt.Sprint(e))
					}
				}
			case string:
				ls := strings.Split(strings.TrimRight(x, "\n"), "\n")
				out = append(out, indent+k+": "+ls[0])
				for _, l := range ls[1:] {
					out = append(out, indent+"  "+l)
				}
			case float64:
				if x == float64(int64(x)) {
					out = append(out, fmt.Sprintf("%s%s: %d", indent, k, int64(x)))
				} else {
					out = append(out, fmt.Sprintf("%s%s: %g", indent, k, x))
				}
			default:
				out = append(out, fmt.Sprintf("%s%s: %v", indent, k, x))
			}
		}
	}
	walk(v, "")
	return lines(strings.Join(out, "\n"), limit)
}

// goalEnded: the goal is over (cancelled, completed, failed or an error); goal commands have nothing to act on.
func goalEnded(state string) bool {
	switch state {
	case "cancelled", "completed", "failed", "error":
		return true
	}
	return false
}

func endedWord(state string) string {
	if state == "error" {
		return "failed"
	}
	return state
}

// leaveShell ends shell mode; a draft held aside for a second command comes back.
func (m emberModel) leaveShell() emberModel {
	m.setShell(false)
	if m.chat.shellStash != "" {
		m.chat.compose.SetValue(m.chat.shellStash + "\n")
		m.chat.shellStash = ""
	}
	return m
}

// todoBlockers are the newest todo list's blockers for a goal.
func (m emberModel) todoBlockers(gid string) []string {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != gid || e.Kind != "todo" {
			continue
		}
		var out []string
		raw, _ := e.Data["blockers"].([]any)
		for _, b := range raw {
			if t, _ := b.(string); strings.TrimSpace(t) != "" {
				out = append(out, t)
			}
		}
		return out
	}
	return nil
}

// noticeShown is the notice on the composer's frame now ("" once it expired: the tick clears it, a stalled tick must
// not keep it).
func (m emberModel) noticeShown() string {
	if m.msg == m.msgOf && !m.msgAt.IsZero() && time.Since(m.msgAt) > noticeTTL(m.msg) {
		return ""
	}
	return m.msg
}

// noticeTTL is how long a notice stays: 6 s, and a second more for every 20 characters past the first 60 (15 s at
// most), so a long error can be read before it goes.
func noticeTTL(msg string) time.Duration {
	extra := max(0, ansi.StringWidth(msg)-60) / 20
	return time.Duration(min(15, 6+extra)) * time.Second
}

// noticeLong: the notice does not fit the composer's frame (half its width).
func (m emberModel) noticeLong() bool {
	n := m.noticeShown()
	return n != "" && ansi.StringWidth(n) > max(18, m.width-2)/2-4
}

// menuDismissed: Esc closed the completion menu on this very draft.
func (m emberModel) menuDismissed() bool {
	return m.chat.menuClosed != "" && m.chat.menuClosed == m.chat.compose.Value()
}

// endedSessions are the terminal sessions a result saw end, with their exit code.
func (m emberModel) endedSessions() map[string]float64 {
	out := map[string]float64{}
	for _, e := range m.residentEvents {
		if e.Kind != "tool_result" {
			continue
		}
		if n, _ := e.Data["name"].(string); n != "terminal" && n != "sandbox" {
			continue
		}
		res, _ := e.Data["result"].(map[string]any)
		sid, _ := res["session"].(string)
		if code, ok := res["exit_code"].(float64); ok && sid != "" {
			out[sid] = code
		}
	}
	return out
}

// wrappedRecord is a record line that wraps under itself instead of being cut (a long one whose end matters).
func wrappedRecord(icon, text string, w int) string {
	ls := strings.Split(ansi.Wordwrap(text, max(10, w-4), " "), "\n")
	for i, l := range ls {
		if i == 0 {
			ls[i] = " " + emberWarn.Render(icon) + " " + emberDim.Render(l)
		} else {
			ls[i] = "   " + emberDim.Render(l)
		}
	}
	return strings.Join(ls, "\n")
}

// pathSelected is the path suggestions' highlighted row: the arrows' choice for this token, else Tab's pick.
func (m emberModel) pathSelected(token string, matches []string) int {
	if m.chat.pathSelFor == token && len(matches) > 0 {
		return ((m.chat.pathSel % len(matches)) + len(matches)) % len(matches)
	}
	return max(0, slices.Index(matches, m.chat.pathPick))
}

// pathMenuKey: while the path suggestions show, Up and Down move their highlight (the draft is untouched) and Enter
// takes the highlighted path into the draft (a folder then shows what it holds). Only Tab worked before.
func (m emberModel) pathMenuKey(k string) (emberModel, bool) {
	if m.pathMenu(max(18, m.width-2)) == "" {
		return m, false
	}
	value := strings.TrimRight(m.chat.compose.Value(), " ")
	token, _ := pathToken(value)
	matches := m.pathMatches(token)
	sel := m.pathSelected(token, matches)
	switch k {
	case "up", "down":
		d := 1
		if k == "up" {
			d = -1
		}
		m.chat.pathSelFor, m.chat.pathSel = token, (sel+d+len(matches))%len(matches)
		return m, true
	case "enter":
		pick := matches[sel]
		if pick == token {
			return m, false // already complete: Enter sends
		}
		m.chat.compose.SetValue(value[:len(value)-len(token)] + pick)
		m.chat.compose.CursorEnd()
		m.chat.pathPrefix, m.chat.pathPick, m.chat.pathIdx = pick, pick, 0
		m.chat.pathSelFor = ""
		return m, true
	}
	return m, false
}
