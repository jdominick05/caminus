// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envgoals.go: the Goals overlay. The resident's goals, newest first, in the one box every overlay uses. Enter makes
// the selected goal the composer's target; x cancels it and r resumes it. Every action goes through the chat's own
// /goal, /cancel and /resume handlers, so nothing here talks to the resident itself.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// goalGlyph is a goal state's mark and style, from the shell's glyph tier.
func goalGlyph(state string) (string, lipgloss.Style) {
	switch state {
	case "working", "waiting_model":
		return g.working, emberBusy
	case "queued":
		return g.queue, emberDim
	case "completed":
		return g.done, emberCtxOK
	case "paused", "waiting_human", "waiting_root":
		return g.waiting, emberWarn
	case "error", "cancelled":
		return g.fail, emberCtxHot
	}
	return g.idle, emberDim
}

// goalOrder is the overlay's rows: indexes into m.resident.Goals, newest first.
func (m emberModel) goalOrder() []int {
	out := make([]int, len(m.resident.Goals))
	for i := range out {
		out[i] = len(out) - 1 - i
	}
	// the queued goals take their slots in run order (a move showed nowhere: they stayed in creation order)
	var slots, queued []int
	for k, i := range out {
		if m.resident.Goals[i].State == "queued" {
			slots, queued = append(slots, k), append(queued, i)
		}
	}
	sort.SliceStable(queued, func(a, b int) bool { return m.queuePlace(queued[a]) < m.queuePlace(queued[b]) })
	for j, k := range slots {
		out[k] = queued[j]
	}
	return out
}

// queuePlace is a queued goal's place in the run order: the resident's, else its age (the queue runs oldest first).
func (m emberModel) queuePlace(i int) int {
	if q := m.resident.Goals[i].Queue; q > 0 {
		return q
	}
	return 1000 + len(m.resident.Goals) - i
}

func (m emberModel) goalRow(i int, selected bool, width int) string {
	gl := m.resident.Goals[i]
	mark, st := goalGlyph(gl.State)
	text := strings.Join(strings.Fields(residentText(gl.Prompt)), " ")
	meta := gl.State + " · " + plural(gl.Steps, "step", "steps")
	target := ""
	if !m.chat.newGoal && gl.ID == m.chat.target {
		target = "  " + g.user
	}
	if selected {
		row := " " + g.collapsed + " " + mark + " " + shortGoal(gl.ID) + "  " + text + "  ·  " + meta + target
		return emberSel.Render(ansi.Truncate(row, max(8, width-1), "…"))
	}
	row := "   " + st.Render(mark) + " " + emberDim.Render(shortGoal(gl.ID)) + "  " +
		lipgloss.NewStyle().Foreground(cText).Render(text) + emberDim.Render("  ·  "+meta+target)
	return ansi.Truncate(row, width, "…")
}

// goalDetail is the selected goal in full under the list once space opens it (Enter targets it instead).
func (m emberModel) goalDetail(i, width int) []string {
	gl := m.resident.Goals[i]
	var out []string
	add := func(label, value string) {
		value = strings.TrimSpace(residentText(value))
		if value == "" {
			return
		}
		for j, piece := range strings.Split(ansi.Hardwrap(value, max(8, width-10), true), "\n") {
			if j == 6 {
				out = append(out, emberDim.Render("          …"))
				break
			}
			l := label + ": "
			if j > 0 {
				l = strings.Repeat(" ", len(label)+2)
			}
			out = append(out, emberDim.Render(l)+lipgloss.NewStyle().Foreground(cText).Render(piece))
		}
	}
	add("goal", gl.Prompt)
	add("plan", gl.PlanText)
	if strings.TrimSpace(gl.PlanText) != "" {
		out = append(out, emberDim.Render("          v shows the whole plan"))
	}
	add("summary", gl.Summary)
	add("error", gl.Error)
	return out
}

func (m emberModel) goalsOverlay(w, h int) string {
	order := m.goalOrder()
	sel := max(0, min(m.chat.goalsSel, len(order)-1))
	inner := max(10, w-6)
	var detail []string
	if m.chat.goalsDetail && len(order) > 0 {
		detail = m.goalDetail(order[sel], inner)
	}
	listH := max(1, (h-6)-len(detail)-1)
	first := 0
	if len(order) > listH {
		first = max(0, min(sel-listH+1, len(order)-listH))
	}
	var rows []string
	if len(order) == 0 {
		rows = append(rows, emberDim.Render("no goals yet · Esc, then type one"))
	}
	for i := first; i < min(first+listH, len(order)); i++ {
		rows = append(rows, m.goalRow(order[i], i == sel, inner))
	}
	body := strings.Join(rows, "\n")
	if len(detail) > 0 {
		body += "\n\n" + strings.Join(detail, "\n")
	} else {
		// wrapped, not cut (at 80 and 100 columns the last keys were lost)
		body += "\n\n" + emberDim.Render(ansi.Wordwrap("Enter targets it  ·  space shows it  ·  v the whole plan  ·  x x cancel  ·  p pause  ·  r resume  ·  Shift+↑↓ queue order  ·  n new goal", inner, " "))
	}
	active := 0
	for _, gl := range m.resident.Goals {
		if gl.State != "completed" && gl.State != "cancelled" && gl.State != "error" {
			active++
		}
	}
	hint := "↑↓ move  ·  Enter target  ·  v the whole plan"
	if m.msg != "" { // what x, p, r or a move did: the composer's frame, where it went, is under this overlay
		hint = terminalText(m.msg, false)
	}
	return overlayBox(fmt.Sprintf("GOALS  %d open of %d", active, len(order)), body, hint, w, h)
}

// goalsKey: the overlay owns the keyboard while open. Actions reuse the chat's slash handlers.
func (m emberModel) goalsKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	order := m.goalOrder()
	sel := max(0, min(m.chat.goalsSel, len(order)-1))
	m.chat.goalsSel = sel
	// Cancelling ends the agent's work on a goal, so it takes two presses, like a root approval: the first x arms it
	// and names the goal, a second x within 10 s cancels, any other key disarms.
	armed := m.chat.armed
	m.chat.armed = ""
	closeIt := func() { m.chat.goals, m.chat.goalsDetail = false, false }
	switch k {
	case "esc", "ctrl+g":
		closeIt()
		return m, m.chat.compose.Focus()
	case "up", "k":
		m.chat.goalsSel = max(0, sel-1)
	case "down", "j":
		m.chat.goalsSel = min(max(0, len(order)-1), sel+1)
	case "space", " ":
		m.chat.goalsDetail = !m.chat.goalsDetail
	case "shift+up", "shift+down", "K", "J":
		// A queued goal moves one place in the run order (the queue runs oldest first; 1 runs next).
		if len(order) == 0 {
			return m, nil
		}
		gl := m.resident.Goals[order[sel]]
		if gl.State != "queued" {
			m.msg = "only a queued goal can move"
			return m, nil
		}
		// in run order (the list is newest first: Shift+↑ moved the wrong way and said nothing)
		var queued []string
		for _, i := range m.goalOrder() {
			if m.resident.Goals[i].State == "queued" {
				queued = append(queued, m.resident.Goals[i].ID)
			}
		}
		pos := 0
		for i, id := range queued {
			if id == gl.ID {
				pos = i + 1
			}
		}
		to := pos - 1
		if k == "shift+down" || k == "J" {
			to = pos + 1
		}
		if to < 1 || to > len(queued) {
			m.msg = map[bool]string{true: "it runs next already", false: "it is last already"}[to < 1]
			return m, nil
		}
		m.chat.goalsFollow = gl.ID // the ▸ stays on the goal you moved, not on the row (the next move acted on another)
		return m, moveQueued(m, gl.ID, to)
	case "n":
		closeIt()
		m.chat.newGoal, m.chat.forceNew, m.chat.target = true, true, ""
		return m, m.chat.compose.Focus()
	case "v": // the whole plan, in the viewer /diff uses
		if len(order) > 0 && strings.TrimSpace(m.resident.Goals[order[sel]].PlanText) != "" {
			closeIt() // (the viewer draws and takes keys only with the overlay closed)
			m.chat.view = planView(m.resident.Goals[order[sel]], m.width)
		}
		return m, nil
	case "enter", "x", "r", "p":
		if len(order) == 0 {
			return m, nil
		}
		gl := m.resident.Goals[order[sel]]
		id := gl.ID
		if k == "x" && (armed != "cancel:"+id || time.Since(m.chat.armedAt) > 10*time.Second) {
			m.chat.armed, m.chat.armedAt = "cancel:"+id, time.Now()
			m.msg = "cancel goal " + shortGoal(id) + " (" + gl.State + ")? press x again (any other key keeps it)"
			return m, nil
		}
		next, cmd := m.chatSlash("/goal " + id)
		m = next.(emberModel)
		switch k {
		case "enter":
			closeIt()
			return m, tea.Batch(cmd, m.chat.compose.Focus())
		case "x":
			return m.chatSlash("/cancel")
		case "r":
			return m.chatSlash("/resume")
		case "p":
			return m.chatSlash("/pause")
		}
	}
	return m, nil
}
