// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// embersessioncards.go: the session switcher (Ctrl+P, /sessions): one card per session with what the status line and
// the side pane already know about it (its goal, how long it has worked on it, what it cost, its model and speed, its
// GitHub account), arrows to move, Enter or a click to switch, Esc to go back. Nothing else on the screen.

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// sessionCard is one session's card, read from `agent status` and `agent github status` (the same data the chat's
// status line and side pane show for the session in view).
type sessionCard struct {
	name, status, goal, model, github string
	goalID                            string   // the goal the card shows
	active                            []string // every goal still running, queued, paused or waiting: destroy cancels each
	aimed                             bool     // the goal was redirected: the card shows what it works toward now
	workingS, spendGoal, spendToday   float64
	tokPerS                           float64
	ctxUsed, ctx                      int
	lastTS                            float64
}

const cardW = 46 // a card's width with its frame: two fit a 100-column terminal, three a 140

// cardWidth is a card's width at terminal width w: cardW, narrower on a small terminal.
func cardWidth(w int) int { return max(20, min(cardW, w-2)) }

// loadSessionCard reads one session's card; a session that cannot be read still gets a card with its name.
func loadSessionCard(m emberModel, name string) sessionCard {
	c := sessionCard{name: name, status: "unknown"}
	out, err := runResident(residentCommand(m, []string{"status", name, "--json"}, ""))
	var st struct {
		Model      string             `json:"model"`
		Effort     string             `json:"effort"`
		TokPerS    *float64           `json:"tok_per_s"`
		SpendToday float64            `json:"spend_today"`
		SpendGoals map[string]float64 `json:"spend_goals"`
		Aims       map[string]string  `json:"aims"`
		Context    int                `json:"context"`
		Used       int                `json:"context_used"`
		Alive      bool               `json:"daemon_alive"`
		Goals      []residentGoal     `json:"goals"`
	}
	if err != nil || json.Unmarshal(out, &st) != nil {
		return c
	}
	c.model, c.spendToday, c.ctx, c.ctxUsed = withEffort(st.Model, st.Effort), st.SpendToday, st.Context, st.Used
	if st.TokPerS != nil {
		c.tokPerS = *st.TokPerS
	}
	c.status = "listening" // (the word the header uses for the same state)
	if !st.Alive {
		c.status = "stopped"
	}
	for _, x := range st.Goals {
		if x.State == "queued" || x.State == "working" || x.State == "paused" || strings.HasPrefix(x.State, "waiting") {
			c.active = append(c.active, x.ID)
		}
	}
	if len(st.Goals) > 0 {
		g := st.Goals[0] // the newest, unless one is running or waiting
		for _, x := range st.Goals {
			if (x.State == "working" || x.State == "queued" || strings.HasPrefix(x.State, "waiting")) && x.shownState() != "parked" {
				g = x
				break
			}
		}
		c.goal, c.goalID, c.spendGoal, c.lastTS = g.Prompt, g.ID, st.SpendGoals[g.ID], g.Updated
		if aim := st.Aims[g.ID]; aim != "" { // the header's rule: a redirect after a done check
			c.goal, c.aimed = aim, true
		}
		switch {
		case !st.Alive:
		case g.shownState() == "parked":
			c.status = "parked"
		case g.State == "working" || g.State == "queued":
			c.status = "working"
			c.workingS = float64(time.Now().Unix()) - g.Created
		case strings.HasPrefix(g.State, "waiting"):
			c.status = "waiting"
		}
		if c.status != "working" && g.Updated > g.Created {
			c.workingS = g.Updated - g.Created
		}
	}
	if out, err := runResident(residentCommand(m, []string{"github", name, "status", "--json"}, "")); err == nil {
		var gh struct {
			Login string `json:"login"`
		}
		if json.Unmarshal(out, &gh) == nil {
			c.github = gh.Login
		}
	}
	return c
}

func shortDur(s float64) string {
	d := time.Duration(max(0, s)) * time.Second
	switch {
	case d >= time.Hour:
		return fmt.Sprintf("%dh %02dm", int(d.Hours()), int(d.Minutes())%60)
	case d >= time.Minute:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	return fmt.Sprintf("%ds", int(d.Seconds()))
}

func shortTokens(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	}
	if n >= 1000 {
		return fmt.Sprintf("%dK", n/1000)
	}
	return fmt.Sprint(n)
}

// cardCols is how many cards a row holds at width w.
func cardCols(w int) int { return max(1, (w-2)/(cardWidth(w)+1)) }

// sessionCardView draws one card: the session's name and state on the frame, then its goal and numbers.
func (m emberModel) sessionCardView(c sessionCard, selected bool, cw int) []string {
	surface, border := overlayColors()
	seq := bgSeq(surface)
	edge := lipgloss.NewStyle().Foreground(cDim)
	if selected {
		edge = lipgloss.NewStyle().Foreground(lipgloss.Color(border)).Bold(true)
	}
	inner := cw - 2
	dot := map[string]string{"working": emberBusy.Render("● working"), "waiting": emberWarn.Render("● waiting"),
		"listening": emberDim.Render("○ listening"), "parked": emberWarn.Render("◷ parked"),
		"stopped": emberDim.Render("○ stopped")}[c.status]
	if dot == "" {
		dot = emberDim.Render("? " + c.status)
	}
	name := lipgloss.NewStyle().Foreground(cText).Bold(true).Render(emberClip(residentText(c.name), inner-14))
	title := " " + name + " " + dot + " "
	top := edge.Render("╭─") + title + edge.Render(strings.Repeat("─", max(0, inner-1-ansi.StringWidth(title)))+"╮")
	goal := oneLine(residentText(c.goal))
	if c.aimed {
		goal = "↪ " + goal
	}
	if c.goal == "" {
		goal = "no goal yet"
	}
	worked := "worked " // (the word follows the state: an idle card's time is over)
	if c.status == "working" {
		worked = "working "
	}
	speed := "–"
	if c.tokPerS > 0 {
		speed = fmt.Sprintf("%.0f tok/s", c.tokPerS)
	}
	gh := "no GitHub account"
	if c.github != "" {
		gh = "GitHub " + c.github
	}
	ctx := ""
	if c.ctx > 0 {
		ctx = " · " + shortTokens(c.ctxUsed) + "/" + shortTokens(c.ctx)
	}
	rows := []string{
		lipgloss.NewStyle().Foreground(cText).Render(emberClip(goal, inner-2)),
		emberDim.Render(emberClip(worked+shortDur(c.workingS)+fmt.Sprintf(" · $%.2f on this goal", c.spendGoal), inner-2)),
		emberDim.Render(emberClip(residentText(c.model)+" · "+speed+ctx, inner-2)),
		emberDim.Render(emberClip(residentText(gh), inner-2)),
		emberDim.Render(emberClip(fmt.Sprintf("$%.2f today in this session", c.spendToday), inner-2)),
	}
	out := []string{top}
	for _, r := range rows {
		cell := " " + r
		out = append(out, edge.Render("│")+seq+replate(cell, seq)+strings.Repeat(" ", max(0, inner-ansi.StringWidth(cell)))+"\x1b[m"+edge.Render("│"))
	}
	return append(out, edge.Render("╰"+strings.Repeat("─", inner)+"╯"))
}

// switcherCards are the cards in order: the sessions found, the one in view when none were read yet.
func (m emberModel) switcherCards() []sessionCard {
	var out []sessionCard // one per session name, its card when loaded (a name alone while it loads)
	if len(m.sessionNames) == 0 {
		out = append(out, m.sessionCards...)
	}
	for _, n := range m.sessionNames {
		c := sessionCard{name: n, status: "loading"}
		for _, have := range m.sessionCards {
			if have.name == n {
				c = have
			}
		}
		out = append(out, c)
	}
	if len(out) == 0 {
		out = []sessionCard{{name: m.agentName, status: "unknown"}}
	}
	for i := range out {
		if out[i].name == m.agentName && out[i].status != "loading" { // the session in view: the status bar's live speed
			out[i].tokPerS = m.resident.TokPerS
		}
	}
	return out
}

// cardCount is the cards plus the last one, "+ new session".
func (m emberModel) cardCount() int { return len(m.switcherCards()) + 1 }

// newSessionCard is the last card: Enter or a click on it asks for the new session's name.
func (m emberModel) newSessionCard(selected bool, cw int) []string {
	edge := lipgloss.NewStyle().Foreground(cDim)
	if selected {
		_, border := overlayColors()
		edge = lipgloss.NewStyle().Foreground(lipgloss.Color(border)).Bold(true)
	}
	inner := cw - 2
	body := []string{"", lipgloss.NewStyle().Foreground(cAccent).Bold(true).Render("+ new session"), emberDim.Render("Enter or click names it"), "", ""}
	out := []string{edge.Render("╭" + strings.Repeat("─", inner) + "╮")}
	for _, r := range body {
		cell := " " + r
		out = append(out, edge.Render("│")+cell+strings.Repeat(" ", max(0, inner-ansi.StringWidth(cell)))+edge.Render("│"))
	}
	return append(out, edge.Render("╰"+strings.Repeat("─", inner)+"╯"))
}

// pickerBox is the switcher: a heading, the cards in rows, the keys.
func (m emberModel) pickerBox() string {
	w := max(18, m.width-2)
	cards, cols, cw := m.switcherCards(), cardCols(w), cardWidth(w)
	lines := []string{" " + lipgloss.NewStyle().Foreground(cAccent).Bold(true).Render("sessions") + emberDim.Render(fmt.Sprintf("  %d", len(cards)))}
	first, rows := m.cardRows()
	for i := first * cols; i < min(m.cardCount(), (first+rows)*cols); i += cols {
		var row []string
		for j := i; j < min(i+cols, m.cardCount()); j++ {
			if j == len(cards) {
				row = append(row, strings.Join(m.newSessionCard(j == m.chat.pickerSel, cw), "\n"))
				continue
			}
			row = append(row, strings.Join(m.sessionCardView(cards[j], j == m.chat.pickerSel, cw), "\n"))
		}
		lines = append(lines, lipgloss.JoinHorizontal(lipgloss.Top, intersperse(row, " ")...))
	}
	if m.chat.pickerDestroy != "" {
		lines = append(lines, emberClip(" "+emberCtxHot.Render("x again destroys "+residentText(m.chat.pickerDestroy)+":")+
			emberDim.Render(" stops it, cancels its goal, removes the session (its home is kept in removed-sessions)"), w))
	} else if m.chat.pickerNew {
		lines = append(lines, emberClip(" "+emberWarn.Render("new session:")+" "+residentText(m.chat.pickerName)+"█"+emberDim.Render("  Enter creates · Esc cancels"), w))
	} else {
		hint := " ←↑↓→ choose · Enter or click switches · n new · x x destroy · s start/stop · Esc back"
		if ansi.StringWidth(hint) > w { // a narrow terminal cut it before "Esc back": the short form keeps every key
			hint = " ↵ switch · n new · x x destroy · s start/stop · Esc back"
		}
		lines = append(lines, emberDim.Render(emberClip(hint, w)))
	}
	return strings.Join(lines, "\n")
}

func intersperse(xs []string, sep string) []string {
	var out []string
	for i, x := range xs {
		if i > 0 {
			out = append(out, sep)
		}
		out = append(out, x)
	}
	return out
}

// cardRows is the first card row shown and how many fit: the rows around the selection, as many as the height holds.
func (m emberModel) cardRows() (first, rows int) {
	cols := cardCols(max(18, m.width-2))
	total := (m.cardCount() + cols - 1) / cols
	rows = max(1, min(total, (m.height-8)/7))
	first = max(0, min(m.chat.pickerSel/cols-rows+1, total-rows))
	return first, rows
}

// cardAt is the card under a click at (x, y) on the screen, or -1: the switcher sits directly above the status line.
func (m emberModel) cardAt(x, y int) int {
	top := m.height - lipgloss.Height(m.chatFoot(max(20, m.width))) - lipgloss.Height(m.pickerBox())
	row, cardH := y-top-1, 7
	if row < 0 || x < 0 {
		return -1
	}
	cols, cw := cardCols(max(18, m.width-2)), cardWidth(max(18, m.width-2))
	first, rows := m.cardRows()
	r, c := row/cardH, x/(cw+1)
	if c >= cols || r >= rows {
		return -1
	}
	r += first
	if i := r*cols + c; i < m.cardCount() {
		return i
	}
	return -1
}
