// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envpages.go: the old tabbed screens' content as overlays over the chat. The chat is the only screen: memory and
// context, audit and settings open in the shared overlay box (shell.go) with its keys (↑↓ move or scroll,
// PgUp/PgDn, Esc closes, q never quits), and a typed prompt (edit a root command, answer a question, leave a note,
// name a session) sits in the composer's slot. Nothing draws the old tab bar any more.

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// pageOverlays maps the slash command to the old page whose content it shows, and its title.
var pageOverlays = map[string]struct {
	tab   int
	title string
}{
	"/context": {10, "CONTEXT"},
}

// pageTitle is the overlay title for a page index.
func pageTitle(tab int) string {
	for _, p := range pageOverlays {
		if p.tab == tab {
			return p.title
		}
	}
	return ""
}

// openPage opens an old page's content as an overlay; it fetches what the page needs, as switching to it did.
func (m emberModel) openPage(tab int) (emberModel, tea.Cmd) {
	m.chat.page, m.chat.pageTop = tab, 0
	m.chat.compose.Blur()
	saved := m.tab
	m.tab = tab
	m.refreshLocal()
	cmds := sshRefresh(m)
	if tab == 3 || tab == 7 {
		cmds = append(cmds, residentSessions(m))
	}
	m.tab = saved
	return m, tea.Batch(cmds...)
}

// pageOverlay draws the open page in the overlay box, scrolled to pageTop.
func (m emberModel) pageOverlay(w, h int) string {
	saved := m.tab
	m.tab = m.chat.page
	content := m.pane()
	m.tab = saved
	inner := max(10, w-6)
	// Every pane sanitises what came from the agent and styles the rest; escaping the whole page here printed its
	// colours as text (\x1b[…m on every line). Its first line is its title, which the box already carries.
	if _, rest, ok := strings.Cut(content, "\n"); ok {
		content = strings.TrimLeft(rest, "\n")
	}
	lines := strings.Split(ansi.Hardwrap(content, inner, true), "\n")
	room := max(1, h-6)
	top := max(0, min(m.chat.pageTop, len(lines)-room))
	body := strings.Join(lines[top:min(top+room, len(lines))], "\n")
	keys := "↑↓ PgUp PgDn scroll"
	return overlayBox(pageTitle(m.chat.page)+emberDim.Render(" · "+residentText(m.agentName)), body, keys, w, h)
}

func (m emberModel) pageKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "esc":
		m.chat.page = -1
		return m, m.chat.compose.Focus()
	case "pgup":
		m.chat.pageTop = max(0, m.chat.pageTop-max(1, m.height-10))
		return m, nil
	case "pgdown":
		m.chat.pageTop += max(1, m.height-10)
		return m, nil
	}
	switch k {
	case "up", "k":
		m.chat.pageTop = max(0, m.chat.pageTop-1)
	case "down", "j":
		m.chat.pageTop++
	}
	return m, nil
}

// promptTitles names the typed prompts that replace the composer while they are open.
var promptTitles = map[string]string{
	"edit":       "edit the root command · Enter runs it under sudo · Esc cancels",
	"answer":     "answer the question · Enter sends · a number picks a choice · Esc cancels",
	"denywhy":    "deny · what should it do instead? · Enter sends it as a steer · Esc cancels",
	"sessionnew": "new session name · Enter creates it · Esc cancels",
}

// promptBox is a typed prompt in the composer's slot, in the composer's style with its purpose on the border.
func (m emberModel) promptBox(w int) string {
	title, ok := promptTitles[m.mode]
	if !ok {
		return ""
	}
	text := emberBusy.Render(g.user) + " " + terminalText(m.edit, false) + "█"
	if m.mode == "answer" && m.editRequest != nil {
		text = emberDim.Render(ansi.Truncate(residentText(m.editRequest.Text), max(8, w-6), "…")) + "\n" + text
	}
	return borderTitle(lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cYellow).Padding(0, 1).Width(w).Render(text), title, emberWarn)
}

// contextPane is what the agent's own context holds: its window, the measured use, where compaction starts and how far
// it goes, and what the brain has cost (the servers' windows are /servers).
func (m emberModel) contextPane() string {
	r := m.resident
	pct := func(f float64) string { return fmt.Sprintf("%.1f %%", f*100) }
	lines := []string{emberHead.Render("CONTEXT · " + residentText(m.agentName))}
	// a long value wraps under its own column (it ran back to column 0 at 80)
	vw := max(16, m.width-8-16)
	add := func(k, v string) {
		ls := strings.Split(ansi.Wordwrap(v, vw, " "), "\n")
		lines = append(lines, emberDim.Render(fmt.Sprintf("  %-14s", k))+ls[0])
		for _, l := range ls[1:] {
			lines = append(lines, strings.Repeat(" ", 16)+l)
		}
	}
	add("brain", residentText(r.Model))
	// (window 0: the brain's server reports none right now; the fractions below stay, with no token counts to show)
	of := func(f float64) string {
		if r.Context > 0 {
			return fmt.Sprintf("%s (%s)", pct(f), short(int(f*float64(r.Context))))
		}
		return pct(f)
	}
	if r.Context > 0 {
		add("window", fmt.Sprintf("%s tokens", short(r.Context)))
	} else {
		add("window", "unknown: the brain's server reports none right now")
	}
	switch {
	case r.ContextUsed > 0 && r.Context > 0:
		add("in use", fmt.Sprintf("%s (%s), measured at the last turn", short(r.ContextUsed), pct(float64(r.ContextUsed)/float64(r.Context))))
	case r.ContextUsed > 0:
		add("in use", fmt.Sprintf("%s, measured at the last turn", short(r.ContextUsed)))
	default:
		add("in use", "not measured yet")
	}
	if r.CompactThreshold > 0 {
		add("compacts at", of(r.CompactThreshold))
	}
	if r.CompactTarget > 0 {
		add("down to", of(r.CompactTarget)+"; old tool output is trimmed first, a summary only if needed")
	}
	if r.SpendToday > 0 {
		add("spent today", fmt.Sprintf("$%.4f", r.SpendToday))
	}
	if sp := r.speed().long(); sp != "" {
		add("speed", sp)
	}
	return strings.Join(lines, "\n")
}
