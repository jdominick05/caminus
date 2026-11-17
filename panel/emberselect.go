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
// acts on release where it was pressed: it opens a folded card, picks a card in the session picker, presses the jump
// pill, or clears the selection shown.

import (
	"os"
	"os/exec"
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
	var out []string
	for i := max(0, a[0]); i <= b[0] && i < len(lines); i++ {
		_, _, text := s.cells(i, ansi.Strip(lines[i]))
		out = append(out, text)
	}
	return strings.Trim(strings.Join(out, "\n"), "\n")
}

// cells is what the selection takes of line i (plain): its text and the cells [start, end) that text sits in (empty
// when it takes nothing). The copy and the highlight both read it, so what shows selected is what is copied.
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
		m.syncChat() // the areas as the screen draws them now (opening the session picker shortens the transcript)
		m.chat.sel = m.selectionAt(ev.X, ev.Y)
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
// switches to it, the jump pill goes to the newest message, a folded card opens; anywhere else it only clears.
func (m emberModel) clickAt(s chatSelection) (emberModel, tea.Cmd) {
	x, y := s.at[0], s.at[1]
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
	if !s.screen && m.foldedAt(s.a[0]) && !(m.chat.cards && m.chat.expanded) {
		return m.toggleLayout(), nil // a click on a folded card opens it (clicking did nothing; Ctrl+O did)
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

// foldedAt: the transcript line at row shows something folded (a card's fold mark, "… N more lines", "earlier lines").
func (m emberModel) foldedAt(row int) bool {
	lines := strings.Split(m.chat.transcript.GetContent(), "\n")
	if row < 0 || row >= len(lines) {
		return false
	}
	l := strings.TrimSpace(ansi.Strip(lines[row]))
	return (g.collapsed != "" && strings.HasPrefix(l, g.collapsed+" ")) || strings.HasSuffix(l, " …") ||
		strings.Contains(l, "more lines") || strings.Contains(l, "earlier lines")
}
