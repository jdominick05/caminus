// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envapprovals.go: the Approvals overlay, the first of the redesign's overlays. It lists the pending root requests and
// questions over the conversation, in the same box and with the same keys as every other overlay (shell.go's
// overlayBox), and every decision goes through the Requests page's own handler (reqKeys): one sudo run, one claim, one
// audit line, one answer file. Nothing here writes a decision itself.

import (
	"fmt"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// approvalOrder lists the pending requests as the overlay shows them: root requests first (they run a command and
// block the agent), then questions, each oldest first. The values are indexes into m.reqs, which is what reqKeys
// takes.
func (m emberModel) approvalOrder() []int {
	var roots, asks []int
	for i, r := range m.reqs {
		if r.kind() != "question" { // root requests and GitHub hand-offs, which block the agent's step
			roots = append(roots, i)
		} else {
			asks = append(asks, i)
		}
	}
	return append(roots, asks...)
}

// clampApproval keeps the selection inside the list a decision may have shortened.
func (m emberModel) clampApproval() int {
	n := len(m.approvalOrder())
	if n == 0 {
		return 0
	}
	return max(0, min(m.chat.approvalSel, n-1))
}

// requestAge renders how long a request has waited, at the density of a list row.
func requestAge(ts float64, now time.Time) string {
	if ts <= 0 {
		return ""
	}
	d := now.Sub(time.Unix(int64(ts), 0))
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d min", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%d h", int(d.Hours()))
	}
	return fmt.Sprintf("%d d", int(d.Hours()/24))
}

// approvalKindWidth keeps the two kind labels in one column in every glyph tier.
func approvalKindWidth() int {
	return max(lipgloss.Width(g.root+" root"), lipgloss.Width("? ask"))
}

func padRight(s string, w int) string {
	if pad := w - lipgloss.Width(s); pad > 0 {
		return s + strings.Repeat(" ", pad)
	}
	return s
}

// approvalRow draws one pending request: its kind, what it wants, and its age. The selected row is one block of
// emberSel, as in every other list.
func (m emberModel) approvalRow(i int, selected bool, width int) string {
	r := m.reqs[i]
	kind, text := g.root+" root", residentText(r.Cmd)
	switch r.kind() {
	case "question":
		kind, text = "? ask", residentText(r.Text)
	case "github":
		kind, text = "⇡ github", residentText(r.Summary)
	}
	text = strings.Join(strings.Fields(text), " ")
	age := requestAge(r.Ts, time.Now())
	kind = padRight(kind, approvalKindWidth())
	if selected {
		row := " " + g.collapsed + " " + kind + "  " + text
		if age != "" {
			row += "  ·  " + age
		}
		return emberSel.Render(ansi.Truncate(row, max(8, width-1), "…"))
	}
	// Styled segment by segment: a pre-styled string inside another Render ends the outer style at its first reset.
	row := "   " + emberWarn.Render(kind) + "  " + lipgloss.NewStyle().Foreground(cText).Render(text)
	if age != "" {
		row += emberDim.Render("  ·  " + age)
	}
	return ansi.Truncate(row, width, "…")
}

// approvalDetail is the selected request in full, under the list once Enter opens it: where it runs, why it was asked,
// and a question's choices.
func (m emberModel) approvalDetail(order []int, sel, width int) []string {
	if !m.chat.approvalDetail || sel < 0 || sel >= len(order) {
		return nil
	}
	r := m.reqs[order[sel]]
	var out []string
	add := func(label, value string) {
		value = strings.TrimSpace(residentText(value))
		if value == "" {
			return
		}
		room := max(8, width-len(label)-2)
		out = append(out, emberDim.Render(label+": ")+lipgloss.NewStyle().Foreground(cText).Render(ansi.Truncate(value, room, "…")))
	}
	if r.kind() == "github" {
		// what goes public under your name, in full: every field, and the body line by line (the summary alone hid it)
		for _, line := range handoffLines(r.Github, width) {
			out = append(out, line)
		}
	} else if r.kind() == "question" {
		add("question", r.Text)
	} else {
		// The row cuts a long command; the detail never does: what you approve must be visible in full.
		for i, piece := range strings.Split(ansi.Hardwrap(residentText(r.Cmd), max(8, width-5), true), "\n") {
			label := "cmd: "
			if i > 0 {
				label = "     "
			}
			out = append(out, emberDim.Render(label)+lipgloss.NewStyle().Foreground(cText).Bold(true).Render(piece))
		}
	}
	add("cwd", r.CWD)
	add("why", r.Reason)
	if len(r.Choices) > 0 {
		parts := make([]string, len(r.Choices))
		for i, c := range r.Choices {
			parts[i] = fmt.Sprintf("%d %s", i+1, c)
		}
		add("choices", strings.Join(parts, "  ·  "))
	}
	return out
}

// approvalOverlay draws the Approvals overlay in the conversation's place. h is the height overlayBox is given: it
// renders h-1 rows, so the caller passes one more than the rows it has.
func (m emberModel) approvalOverlay(w, h int) string {
	order := m.approvalOrder()
	sel := m.clampApproval()
	inner := max(10, w-6)
	detail := m.approvalDetail(order, sel, inner)
	// the title and the rule take two of overlayBox's h-4 lines, the hint one, the detail the rest
	listH := max(1, (h-6)-len(detail)-1)
	first := 0
	if len(order) > listH {
		first = max(0, min(sel-listH+1, len(order)-listH))
	}
	rows := make([]string, 0, listH)
	if len(order) == 0 {
		rows = append(rows, emberDim.Render("nothing is waiting"))
	}
	for i := first; i < min(first+listH, len(order)); i++ {
		rows = append(rows, m.approvalRow(order[i], i == sel, inner))
	}
	hint := "Enter shows the full request  ·  a a approve  ·  e edit  ·  d deny  ·  /deny TEXT in the chat says what instead"
	if m.chat.approvalDetail {
		hint = "Enter folds the full request again  ·  a a approve  ·  e edit  ·  d deny  ·  /deny TEXT says what instead"
	}
	body := strings.Join(rows, "\n")
	if len(detail) > 0 {
		body += "\n\n" + strings.Join(detail, "\n")
	} else {
		body += "\n\n" + emberDim.Render(ansi.Truncate(hint, inner, "…"))
	}
	return overlayBox(fmt.Sprintf("APPROVALS  %d pending", len(order)), body,
		"↑↓ move  ·  Enter open", w, h)
}

// approvalKey decides inside the overlay through reqKeys, the same handler the Requests page and the chat's inline
// card use: a first "a" only arms a root approval (10 s, any other key disarms), a question answers through the same
// answer mode, e opens the command in the editor, d denies. q does nothing here.
func (m emberModel) approvalKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	armed := m.chat.armed
	m.chat.armed = ""
	order := m.approvalOrder()
	sel := m.clampApproval()
	m.chat.approvalSel = sel
	switch k {
	case "esc", "ctrl+a":
		m.chat.approvals, m.chat.approvalDetail = false, false
		return m, m.chat.compose.Focus()
	case "up", "k":
		m.chat.approvalSel = max(0, sel-1)
	case "down", "j":
		m.chat.approvalSel = min(max(0, len(order)-1), sel+1)
	case "enter":
		m.chat.approvalDetail = !m.chat.approvalDetail
	case "a", "d", "e":
		if sel >= len(order) {
			return m, nil
		}
		r := m.reqs[order[sel]]
		// Approving runs a root command, so it takes two presses, exactly as the inline card does.
		if k == "a" && r.kind() != "question" && (armed != r.ID || time.Since(m.chat.armedAt) > 10*time.Second) {
			m.chat.armed, m.chat.armedAt = r.ID, time.Now()
			m.msg = "approve as root: " + terminalText(r.Cmd, false) + " ? press a again (any other key cancels)"
			if r.kind() == "github" {
				m.msg = "run on GitHub as you: " + terminalText(r.Summary, false) + " ? press a again (any other key cancels)"
			}
			return m, nil
		}
		m.sel = order[sel]
		if k == "e" || (k == "a" && r.kind() == "question") {
			// the editor and the answer box are typed prompts in the composer's slot
			m.chat.approvals, m.chat.approvalDetail = false, false
		}
		return m.reqKeys(k)
	}
	return m, nil
}
