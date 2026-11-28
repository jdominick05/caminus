// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberselect.go: select to copy, anywhere on Ember's screen. The panel takes the mouse (the wheel scrolls the
// transcript, a click opens a card or picks a session), which turns the terminal's own selection off. So the panel
// selects itself: a left-button drag highlights the cells it passes over, and the release copies their plain text
// (no styling; lines break where the screen wraps them) to the clipboard: wl-copy on Wayland, xclip on X11, and
// OSC 52 for terminals that take it (a remote session). Shift+drag still gives the terminal's own selection in
// terminals that offer it (VTE / GNOME Terminal do).
//
// A drag selects inside the area it began in, so text beside it never comes along:
//   - the transcript: its content lines, so dragging past the top or the bottom scrolls and keeps selecting;
//   - the side pane (tasks, now): its columns only;
//   - the band above the transcript (the header), and the band below it (the live window, the activity and task
//     lines, the composer, menus, the session picker, the status line);
//   - the whole frame while an overlay covers the transcript (goals, models, themes, studio, pages, rewind).
//
// Outside the transcript the text comes from the drawn frame, with a box's border (the composer's, a dialog's, the
// seam between the picker's cards: only at the columns the layout draws them in) and a heading's trailing rule left
// out; the highlight covers what is copied.
//
// Click or drag: the press only arms. Moving two or more cells away while the button is down makes it a drag; a
// release without that, or back on the cell it was pressed on, is a click (a shaky hand still clicks), and the click
// acts on release where it was pressed: it opens or folds the one card under it (cardToggles), picks a card in the
// session picker, presses the jump pill, or clears the selection shown.

import (
	"maps"
	"os"
	"os/exec"
	"slices"
	"sort"
	"strconv"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// chatSelection is a range of cells: [line, cell] at both ends, a where the drag began. In the transcript a line is
// a content line; elsewhere (screen) it is a row of the drawn frame.
type chatSelection struct {
	dragging, moved, shown bool   // moved: the pointer went 2 or more cells from the press (less is a shaky click)
	cleared                bool   // this press cleared a copy the screen still showed: the click after it only clears
	edges                  []int  // over the screen: the columns where the area's boxes draw their sides
	screen                 bool   // over the frame's rows, not the transcript's content
	x0, x1, y0, y1         int    // the area the drag began in: cells [x0, x1); for the screen, rows [y0, y1)
	at                     [2]int // where the button went down (x, y): a release without a drag clicks there
	a, b                   [2]int
}

func (s chatSelection) ordered() ([2]int, [2]int) {
	if s.b[0] < s.a[0] || (s.b[0] == s.a[0] && s.b[1] < s.a[1]) {
		return s.b, s.a
	}
	return s.a, s.b
}

// span is the selected cells [start, end) of line i, ok false when the line is outside the selection.
func (s chatSelection) span(i int) (int, int, bool) {
	if !s.moved && !s.shown {
		return 0, 0, false
	}
	a, b := s.ordered()
	if i < a[0] || i > b[0] {
		return 0, 0, false
	}
	start, end := s.x0, s.x1
	if i == a[0] {
		start = a[1]
	}
	if i == b[0] {
		end = b[1] + 1
	}
	return start, max(start, end), true
}

// transcriptShown: the transcript is on the screen (an overlay draws over its place, the session picker does not).
func (m emberModel) transcriptShown() bool {
	c := m.chat
	return !(c.goals || c.models.open || c.themes.open || c.studio.open || c.view.open || c.rewind.open || c.page >= 0)
}

// selectionAt starts a selection in the area under the screen cell (x, y).
func (m emberModel) selectionAt(x, y int) chatSelection {
	w := max(20, m.width)
	top, rows, tw := lipgloss.Height(m.chatHead(w)), m.chat.transcript.Height(), m.transcriptWidth(w)
	s := chatSelection{dragging: true, screen: true, x1: w, y1: max(1, m.height), at: [2]int{x, y}}
	s.edges = []int{0, w - 1} // an overlay's box spans the frame
	if m.transcriptShown() {
		switch {
		case y >= top && y < top+rows && x < tw:
			s.screen, s.x1, s.edges = false, tw, nil
		case y >= top && y < top+rows:
			s.x0, s.y0, s.y1, s.edges = tw+1, top, top+rows, nil // the side pane: no box
		case y < top:
			s.y1, s.edges = top, nil
		default:
			s.y0 = top + rows
			bw := max(18, w-2) // the boxes under the transcript (composer, prompt, menus) are drawn this wide
			s.edges = []int{0, bw - 1}
			if m.chat.picker { // the session picker's cards sit side by side: each card's two sides
				cols, cw := cardCols(bw), cardWidth(bw)
				for c := 0; c < cols; c++ {
					s.edges = append(s.edges, c*(cw+1), c*(cw+1)+cw-1)
				}
			}
		}
	}
	s.a = m.selectionPos(s, x, y)
	s.b = s.a
	return s
}

// selectionPos maps a screen cell to the selection's lines, pulled inside its area (a drag that leaves the area
// keeps selecting along its edge).
func (m emberModel) selectionPos(s chatSelection, x, y int) [2]int {
	x = max(s.x0, min(s.x1-1, x))
	if s.screen {
		return [2]int{max(s.y0, min(s.y1-1, y)), x}
	}
	row := y - lipgloss.Height(m.chatHead(max(20, m.width)))
	row = max(0, min(m.chat.transcript.Height()-1, row))
	return [2]int{m.chat.transcript.YOffset() + row, x}
}

// selectedText is the selection as plain text, each line's trailing spaces trimmed.
func (m emberModel) selectedText() string {
	s := m.chat.sel
	source := m.chat.transcript.GetContent()
	if s.screen {
		source = m.render()
	}
	lines := strings.Split(source, "\n")
	a, b := s.ordered()
	var bodies []mdBody
	if !s.screen {
		bodies = m.markdownBodies(a[0], b[0])
	}
	// The header's goal is drawn cut, or rendered (its Markdown reflowed): a copy that touches it takes the goal
	// whole, as the agent wrote it, the way an open card's body copies its source (markdownBodies) (#316).
	// The rows substituted are the ones the goal occupies under the title — headGoalRowsUnderTitle (cut) and
	// headExpansion (open) are what chatHead draws — so a selection reaching the cut hint or the expansion's
	// scroll footer copies those as what they are, never as goal text.
	goalAt, goalTo, goal := -1, -1, ""
	if s.screen && b[0] >= 1 {
		if g := strings.TrimSpace(m.headGoalText()); g != "" {
			w := max(20, m.width)
			collapsed := !m.headShown()
			// The goal has rows under the title only when the header drew it there (headCutParts). A goal whole on
			// the title row has none: the row under it is the rule, and counting the flattened wrap anyway put the
			// goal where the rule is in a copy of rows 0-2 (#370 review P2).
			rows := 0
			if collapsed && m.headUnderTitle(w) {
				rows = len(headGoalRowsUnderTitle(strings.Join(strings.Fields(g), " "), w))
			}
			if !collapsed {
				rows = len(m.headExpansion(w).rows)
			}
			// The goal block is rows 1..rows. Collapsed, its rows are a lossy cut of the flattened goal, so a copy
			// that touches it takes the goal whole (the #316 affordance, as an open card's body copies its source).
			// Open, the rows are the goal's own lines, and only a selection that spans the block takes the goal
			// whole (its source, which for a scrolled or longer goal is more than the screen shows); a selection
			// inside it that stops short copies the drawn rows as themselves — a one-row clip of a twelve-row
			// expansion is one row, not the whole goal (#370 review P2).
			if rows > 0 && a[0] < 1+rows && (collapsed || (a[0] <= 1 && b[0] >= rows)) {
				goalAt, goalTo, goal = max(1, a[0]), 1+rows, ansi.Strip(g)
			}
		}
	}
	var out []string
	for i := max(0, a[0]); i <= b[0] && i < len(lines); i++ {
		if goalAt >= 0 && i >= goalAt && i < goalTo {
			out = append(out, goal) // the whole goal, not the lines the header had room for
			i = goalTo - 1
			continue
		}
		if k := slices.IndexFunc(bodies, func(md mdBody) bool { return i >= md.from && i < md.to }); k >= 0 {
			out = append(out, bodies[k].source) // a rendered card's body: its source, whole (the copy keeps it)
			i = bodies[k].to - 1
			continue
		}
		_, _, text := s.cells(i, ansi.Strip(lines[i]))
		out = append(out, text)
	}
	return strings.Trim(strings.Join(out, "\n"), "\n")
}

// mdBody is an open card's body drawn as Markdown: content lines [from, to), and the text it was rendered from.
type mdBody struct {
	from, to int
	source   string
}

// markdownBodies are the cards in content lines [a, b] whose body is drawn as Markdown (openCardMarkdown). Each is
// found by drawing its event again with the source shown: the lines that differ are the body. A selection that touches
// one copies its source whole, so a copy is the Markdown as it came (line-exact mapping does not exist: rendering
// reflows it).
func (m emberModel) markdownBodies(a, b int) []mdBody {
	c := m.chat
	if c.mdSource {
		return nil
	}
	var out []mdBody
	line, at := 0, 0
	for _, sp := range c.spans {
		line += strings.Count(c.content[at:sp.from], "\n")
		at = sp.from
		drawn := strings.Split(c.content[sp.from:sp.to], "\n")
		if line+len(drawn) <= a {
			continue
		}
		if line > b {
			break
		}
		i := slices.IndexFunc(m.residentEvents, func(e residentEvent) bool { return e.Seq == sp.seq })
		if i < 0 {
			continue
		}
		e := m.residentEvents[i]
		text, _ := cardSource(e)
		if c.openCardMarkdown(e) == "" { // (drawn as it is: no body to find; the drawing is cached)
			continue
		}
		src := c // (a copy: drawing it changes nothing on the model)
		src.mdSource = true
		raw := strings.Split(src.drawnBlock(e, max(20, c.renderWidth)), "\n")
		h, t := 0, 0
		for h < len(drawn) && h < len(raw) && drawn[h] == raw[h] {
			h++
		}
		for t < len(drawn)-h && t < len(raw)-h && drawn[len(drawn)-1-t] == raw[len(raw)-1-t] {
			t++
		}
		if h < len(drawn)-t {
			out = append(out, mdBody{line + h, line + len(drawn) - t, strings.TrimRight(residentText(text), "\n")})
		}
	}
	return out
}

// cardSource is the text an open card can draw as Markdown, and whether it is a model's words (prose: Markdown, as
// a reply is) rather than a program's output (drawn as Markdown only when looksLikeMarkdown says so): a command's
// output (a ! card's, a terminal result's), an advisor's review, a report's text ("" for any other card).
func cardSource(e residentEvent) (string, bool) {
	str := func(m map[string]any, k string) string { s, _ := m[k].(string); return s }
	output := func(s string) string { return strings.ReplaceAll(ansi.Strip(s), "\r\n", "\n") }
	switch e.Kind {
	case "shell":
		return output(str(e.Data, "output")), false
	case "tool_result":
		result, _ := e.Data["result"].(map[string]any)
		return output(str(result, "output")), false
	case "advisor_review":
		return str(e.Data, "advice"), true
	case "report":
		if str(e.Data, "level") != "need_you" { // (the need-you report is a turn, rendered as a reply is)
			return str(e.Data, "text"), true
		}
	}
	return "", false
}

// cells is what the selection takes of line i (plain): its text and the cells [start, end) that text sits in (empty
// when it takes nothing). The copy and the highlight both read it, so what shows selected is what is copied, but for
// a card drawn as Markdown: its body is copied as its source (markdownBodies).
func (s chatSelection) cells(i int, plain string) (int, int, string) {
	start, end, ok := s.span(i)
	if !ok {
		return 0, 0, ""
	}
	text, lead := strings.TrimRight(ansi.Cut(plain, start, end), " "), 0
	if s.screen {
		text, lead = unframe(plain, start, end, s.edges, s.x1)
	}
	if text == "" {
		return 0, 0, ""
	}
	start += lead
	return start, start + ansi.StringWidth(text), text
}

// unframe is the text of cells [start, end) of a drawn row (plain) without what frames it, judged only at the columns
// the layout puts a frame in, so a │ or ─ inside an overlay's content (a diff, a table, tree art) is copied as is:
//   - at an edge column: a box's side, or a corner with the rule running from it along a top or bottom edge (a title
//     on it stays), copied as blanks so the columns beside it stay put (the seam between the picker's cards);
//   - a rule running to the area's last column (areaEnd - 1): a heading's trailing rule;
//   - a row left with nothing but rule (a heading's underline).
//
// At the start of the text, the dropped cells and the one space of padding inside a side go; lead counts them.
func unframe(plain string, start, end int, edges []int, areaEnd int) (string, int) {
	cell := func(c int) string { return ansi.Cut(plain, c, c+1) }
	blank := map[int]bool{}
	run := func(from, step int) { // the rule from a corner (or the area's end) along its edge
		for i := from; i >= 0 && cell(i) == "─"; i += step {
			blank[i] = true
		}
	}
	for _, c := range edges {
		switch r := cell(c); r {
		case "│":
			blank[c] = true
		case "╭", "╰":
			blank[c] = true
			run(c+1, 1)
		case "╮", "╯":
			blank[c] = true
			run(c-1, -1)
		}
	}
	run(areaEnd-1, -1)
	i := start
	for i < end && blank[i] {
		i++
	}
	if i > start && i < end && cell(i) == " " {
		i++
	}
	var b strings.Builder
	for j := i; j < end; {
		k := j
		for k < end && blank[k] == blank[j] {
			k++
		}
		if blank[j] {
			b.WriteString(strings.Repeat(" ", k-j))
		} else {
			b.WriteString(ansi.Cut(plain, j, k))
		}
		j = k
	}
	text := strings.TrimRight(b.String(), " ")
	if strings.Trim(text, "─ ") == "" {
		return "", 0
	}
	return text, i - start
}

// selectMouse handles a press, a drag and a release anywhere on the chat; handled false leaves the message to the
// rest of the chat.
func (m emberModel) selectMouse(msg tea.Msg) (emberModel, tea.Cmd, bool) {
	switch ev := msg.(type) {
	case tea.MouseClickMsg:
		if ev.Button != tea.MouseLeft {
			return m, nil, false
		}
		was := m.chat.sel.shown // a shown copy: the click that clears it must not also toggle what it covers
		m.syncChat()            // the areas as the screen draws them now (opening the session picker shortens the transcript)
		m.chat.sel = m.selectionAt(ev.X, ev.Y)
		m.chat.sel.cleared = was
		return m, nil, true
	case tea.MouseMotionMsg:
		s := m.chat.sel
		if !s.dragging {
			return m, nil, false
		}
		if !s.screen {
			top := lipgloss.Height(m.chatHead(max(20, m.width)))
			if ev.Y < top {
				m.chat.transcript.ScrollUp(1)
			} else if ev.Y >= top+m.chat.transcript.Height() {
				m.chat.transcript.ScrollDown(1)
			}
		}
		s.b = m.selectionPos(s, ev.X, ev.Y)
		s.moved = s.moved || max(ev.X-s.at[0], s.at[0]-ev.X)+max(ev.Y-s.at[1], s.at[1]-ev.Y) >= 2
		m.chat.sel = s
		return m, nil, true
	case tea.MouseReleaseMsg:
		s := m.chat.sel
		if !s.dragging {
			return m, nil, false
		}
		m.chat.sel = chatSelection{}
		if !s.moved || s.at == [2]int{ev.X, ev.Y} { // a shaky click: never 2 cells away, or back where it began
			next, cmd := m.clickAt(s)
			return next, cmd, true
		}
		s.dragging, s.shown = false, true
		m.chat.sel = s
		text := m.selectedText()
		if text == "" {
			m.chat.sel = chatSelection{}
			return m, nil, true
		}
		n := strings.Count(text, "\n") + 1
		m.msg = map[bool]string{true: "copied 1 line", false: "copied " + strconv.Itoa(n) + " lines"}[n == 1] +
			" to the clipboard · click to clear"
		return m, tea.Batch(tea.SetClipboard(text), copyToClipboard(text)), true
	}
	return m, nil, false
}

// clickAt is a click (a press and a release without a drag) where s was pressed: a card in the session picker
// switches to it, the jump pill goes to the newest message, a folded card opens and an open one folds (that card
// only: Ctrl+O opens them all); anywhere else it only clears.
func (m emberModel) clickAt(s chatSelection) (emberModel, tea.Cmd) {
	x, y := s.at[0], s.at[1]
	// a click on the header opens the whole goal under the title, or closes it again (#316); it is the header's
	// own rows only, so a click in the transcript is untouched. The click that clears a copy the screen still shows
	// only clears: the drag that made it ended where the goal's header is, and opening it there would be a surprise.
	if m.transcriptShown() && !s.cleared && y < lipgloss.Height(m.chatHead(max(20, m.width))) {
		next, _ := m.toggleHead()
		return next.(emberModel), nil
	}
	if m.chat.picker {
		if i := m.cardAt(x, y); i >= 0 {
			if i == len(m.switcherCards()) {
				m.chat.pickerSel, m.chat.pickerNew, m.chat.pickerName = i, true, ""
				return m, nil
			}
			next, cmd := m.switchToCard(i)
			return next.(emberModel), cmd
		}
	}
	if m.onJumpPill(x, y) {
		m.chat.transcript.GotoBottom()
		m.chat.live.hold = false
		return m, nil
	}
	if seq, ok := m.chat.cardAtLine(s.a[0]); ok && !s.screen && m.cardToggles(seq) {
		return m.toggleCard(seq), nil // (it opened every card, as Ctrl+O does)
	}
	return m, nil
}

// highlightSelection draws the selection (reverse video) over the cells it copies. frame says which drawing view is:
// the whole frame (render) for a selection over the screen, the transcript's visible rows (renderChat, under the
// jump pill and only while the transcript shows) for one in the transcript.
func (m emberModel) highlightSelection(view string, frame bool) string {
	s := m.chat.sel
	if s.screen != frame || (!s.moved && !s.shown) {
		return view
	}
	off := 0
	if !frame {
		off = m.chat.transcript.YOffset()
	}
	rows := strings.Split(view, "\n")
	st := lipgloss.NewStyle().Reverse(true)
	for r, row := range rows {
		plain := ansi.Strip(row)
		if start, end, text := s.cells(off+r, plain); text != "" {
			rows[r] = ansi.Cut(row, 0, start) + st.Render(ansi.Cut(plain, start, end)) + ansi.Cut(row, end, ansi.StringWidth(plain))
		}
	}
	return strings.Join(rows, "\n")
}

// copyToClipboard puts text on the desktop clipboard with the session's tool; OSC 52 (tea.SetClipboard) covers a
// terminal that takes it. A failure is reported in the status line.
func copyToClipboard(text string) tea.Cmd {
	return func() tea.Msg {
		var cmd *exec.Cmd
		switch {
		case os.Getenv("WAYLAND_DISPLAY") != "":
			if p, err := exec.LookPath("wl-copy"); err == nil {
				cmd = exec.Command(p)
			}
		case os.Getenv("DISPLAY") != "":
			if p, err := exec.LookPath("xclip"); err == nil {
				cmd = exec.Command(p, "-selection", "clipboard")
			} else if p, err := exec.LookPath("xsel"); err == nil {
				cmd = exec.Command(p, "--clipboard", "--input")
			}
		}
		if cmd == nil {
			return nil // OSC 52 only
		}
		cmd.Stdin = strings.NewReader(text)
		if out, err := cmd.CombinedOutput(); err != nil {
			return residentActionMsg{"copy failed: " + strings.TrimSpace(err.Error()+" "+string(out))}
		}
		return nil
	}
}

// cardSpan is where one event's block sits in the drawn history: bytes [from, to) of content.
type cardSpan struct{ from, to, seq int }

// addBlock appends an event's block to the drawn history and notes where it sits (a click finds its card by that).
func (c *residentChat) addBlock(seq int, block string) {
	c.spans = append(c.spans, cardSpan{len(c.content), len(c.content) + len(block), seq})
	c.content += block + "\n\n"
}

// cardAtLine is the event whose block the transcript's content line i is part of; false for a line between blocks,
// the work line under them or the reply being written.
func (c *residentChat) cardAtLine(i int) (int, bool) {
	at := 0 // line i's first byte
	if i < 0 {
		return 0, false
	}
	for ; i > 0 && at < len(c.content); i-- {
		next := strings.IndexByte(c.content[at:], '\n')
		if next < 0 {
			return 0, false
		}
		at += next + 1
	}
	if at >= len(c.content) {
		return 0, false
	}
	k := sort.Search(len(c.spans), func(k int) bool { return c.spans[k].to > at })
	if k < len(c.spans) && c.spans[k].from <= at {
		return c.spans[k].seq, true
	}
	return 0, false
}

// cardLines is where the event's block sits in the drawn history: its first content line and how many lines it has;
// false when it is not drawn.
func (c *residentChat) cardLines(seq int) (int, int, bool) {
	line, at := 0, 0
	for _, sp := range c.spans {
		line += strings.Count(c.content[at:sp.from], "\n")
		at = sp.from
		if sp.seq == seq {
			return line, strings.Count(c.content[sp.from:sp.to], "\n") + 1, true
		}
	}
	return 0, 0, false
}

// cardToggles: a click on the event's block opens or folds it. Only in the compact view (Ctrl+O shows every card
// in full already), only an open card or one that opening changes (a card with lines left out, a long diff, a
// command of several lines): a click on the agent's words or a short card does nothing.
func (m emberModel) cardToggles(seq int) bool {
	c := &m.chat
	if c.cards || c.expanded {
		return false
	}
	if c.opened[seq] {
		return true
	}
	i := slices.IndexFunc(m.residentEvents, func(e residentEvent) bool { return e.Seq == seq })
	if i < 0 {
		return false
	}
	e, w := m.residentEvents[i], max(20, c.renderWidth)
	folded := c.compactBlock(e, w)
	c.opened = maps.Clone(c.opened) // (a trial: the model's map stays as it was)
	if c.opened == nil {
		c.opened = map[int]bool{}
	}
	c.opened[seq] = true
	return c.compactBlock(e, w) != folded
}

// toggleCard opens or folds one card, and keeps it where it is on the screen: the history above it is unchanged, so
// the same scroll offset puts it under the pointer again.
func (m emberModel) toggleCard(seq int) emberModel {
	opened := maps.Clone(m.chat.opened)
	if opened == nil {
		opened = map[int]bool{}
	}
	if opened[seq] {
		delete(opened, seq)
	} else {
		opened[seq] = true
	}
	m.chat.opened = opened
	top := m.chat.transcript.YOffset()
	m.chat.content, m.chat.rendered = "", 0
	m.syncChat()
	m.chat.transcript.SetYOffset(top)
	return m
}
