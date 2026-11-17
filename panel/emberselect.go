// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envselect.go: select to copy in the chat history. The panel takes the mouse (the wheel scrolls the transcript),
// which turned the terminal's own selection off, so nothing in the history could be copied. A left-button drag over
// the transcript now selects (highlighted), and the release copies the plain text to the clipboard: wl-copy on
// Wayland, xclip on X11, and OSC 52 for terminals that take it (a remote session). Dragging past the top or the
// bottom scrolls. A click without a drag clears the selection. Shift+drag still gives the terminal's own selection
// in terminals that offer it (VTE / GNOME Terminal do).

import (
	"os"
	"os/exec"
	"strconv"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// chatSelection is a range in the transcript's content: [line, cell] at both ends, a where the drag began.
type chatSelection struct {
	dragging, shown bool
	a, b            [2]int
}

func (s chatSelection) ordered() ([2]int, [2]int) {
	if s.b[0] < s.a[0] || (s.b[0] == s.a[0] && s.b[1] < s.a[1]) {
		return s.b, s.a
	}
	return s.a, s.b
}

// span is the selected cells [start, end) of content line i, ok false when the line is outside the selection.
func (s chatSelection) span(i, width int) (int, int, bool) {
	if !s.dragging && !s.shown {
		return 0, 0, false
	}
	a, b := s.ordered()
	if i < a[0] || i > b[0] {
		return 0, 0, false
	}
	start, end := 0, width
	if i == a[0] {
		start = a[1]
	}
	if i == b[0] {
		end = b[1] + 1
	}
	return start, max(start, end), true
}

// transcriptCols is the transcript column's width (the side panel takes the rest on a wide terminal).
func (m emberModel) transcriptCols() int {
	w := max(20, m.width)
	if sw := m.sideWidth(); sw > 0 {
		return w - sw - 1
	}
	return w
}

// contentPos maps a screen cell to the transcript's content: ok false outside the transcript. clamp pulls a point
// above or below it to the first or last visible row (a drag that leaves the transcript keeps selecting).
func (m emberModel) contentPos(x, y int, clamp bool) ([2]int, bool) {
	top := lipgloss.Height(m.chatHead(max(20, m.width)))
	rows := m.chat.transcript.Height()
	row := y - top
	if !clamp && (row < 0 || row >= rows || x >= m.transcriptCols()) {
		return [2]int{}, false
	}
	row = max(0, min(rows-1, row))
	x = max(0, min(m.transcriptCols()-1, x))
	return [2]int{m.chat.transcript.YOffset() + row, x}, true
}

// selectedText is the selection as plain text, each line's trailing spaces trimmed.
func (m emberModel) selectedText() string {
	lines := strings.Split(m.chat.transcript.GetContent(), "\n")
	a, b := m.chat.sel.ordered()
	var out []string
	for i := a[0]; i <= b[0] && i < len(lines); i++ {
		if i < 0 {
			continue
		}
		start, end, _ := m.chat.sel.span(i, m.transcriptCols())
		plain := ansi.Strip(lines[i])
		out = append(out, strings.TrimRight(ansi.Cut(plain, start, end), " "))
	}
	return strings.Trim(strings.Join(out, "\n"), "\n")
}

// selectMouse handles a click, a drag and a release over the transcript; handled false leaves the message to the
// rest of the chat.
func (m emberModel) selectMouse(msg tea.Msg) (emberModel, tea.Cmd, bool) {
	switch ev := msg.(type) {
	case tea.MouseClickMsg:
		if ev.Button != tea.MouseLeft {
			return m, nil, false
		}
		pos, ok := m.contentPos(ev.X, ev.Y, false)
		if !ok {
			if m.chat.sel.shown {
				m.chat.sel = chatSelection{}
			}
			return m, nil, false
		}
		m.chat.sel = chatSelection{dragging: true, a: pos, b: pos}
		return m, nil, true
	case tea.MouseMotionMsg:
		if !m.chat.sel.dragging {
			return m, nil, false
		}
		top := lipgloss.Height(m.chatHead(max(20, m.width)))
		if ev.Y < top {
			m.chat.transcript.ScrollUp(1)
		} else if ev.Y >= top+m.chat.transcript.Height() {
			m.chat.transcript.ScrollDown(1)
		}
		m.chat.sel.b, _ = m.contentPos(ev.X, ev.Y, true)
		return m, nil, true
	case tea.MouseReleaseMsg:
		if !m.chat.sel.dragging {
			return m, nil, false
		}
		m.chat.sel.dragging = false
		if m.chat.sel.a == m.chat.sel.b {
			m.chat.sel = chatSelection{} // a click, not a drag
			return m, nil, true
		}
		m.chat.sel.shown = true // before the text: span reads only a selection that is dragging or shown
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

// highlightSelection draws the selection over the transcript's visible rows.
func (m emberModel) highlightSelection(view string) string {
	if !m.chat.sel.dragging && !m.chat.sel.shown {
		return view
	}
	rows := strings.Split(view, "\n")
	off, width := m.chat.transcript.YOffset(), m.transcriptCols()
	st := lipgloss.NewStyle().Reverse(true)
	for r, row := range rows {
		start, end, ok := m.chat.sel.span(off+r, width)
		if !ok {
			continue
		}
		plain := ansi.Strip(row)
		end = min(end, ansi.StringWidth(strings.TrimRight(plain, " ")))
		if end <= start {
			continue
		}
		rows[r] = ansi.Cut(row, 0, start) + st.Render(ansi.Cut(plain, start, end)) + ansi.Cut(row, end, width)
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
