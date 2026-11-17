// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberactivity.go: the activity line above the composer while the agent works, like a coding agent's own: an ember
// spinner pulsing through the theme's heat colours, a verb that changes every few seconds, what it is doing right now
// (thinking, writing, the command it runs), and a few numbers for this stretch of work: how long, about how many
// tokens it has written, how many actions, what the goal has cost.

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type spinTickMsg time.Time

const spinEvery = 110 * time.Millisecond

func spinCmd() tea.Cmd {
	return tea.Tick(spinEvery, func(t time.Time) tea.Msg { return spinTickMsg(t) })
}

var (
	emberPulse      = []string{"·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"}
	emberPulseASCII = []string{".", "o", "O", "@", "O", "o"}
	emberVerbs      = []string{"Kindling", "Stoking", "Smouldering", "Forging", "Tempering", "Fanning the flames",
		"Sparking", "Glowing", "Banking the coals", "Rekindling", "Smelting", "Annealing"}
)

// working: the agent is on a goal right now (not waiting on you).
func (m emberModel) working() *residentGoal {
	if m.daemonDown() {
		return nil // the header says stopped: no spinner counting under it
	}
	if g := m.activeGoal(); g != nil && (g.State == "working" || g.State == "waiting_model") && m.parkedUntil(g.ID) == "" {
		return g
	}
	return nil
}

// daemonDown: the status has arrived and says the session's loop is not running.
func (m emberModel) daemonDown() bool { return !m.statusAt.IsZero() && !m.resident.DaemonAlive }

// parkedUntil is "HH:MM" while the goal sleeps on schedule (its newest scheduled event has no wake-up, step or
// message after it), else "". The status still says working then.
func (m emberModel) parkedUntil(goal string) string {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != goal {
			continue
		}
		switch e.Kind {
		case "scheduled":
			if at, _ := e.Data["wake_at"].(float64); at > float64(time.Now().Unix()) {
				return time.Unix(int64(at), 0).Format("15:04")
			}
			return ""
		case "woke", "assistant", "user_message", "cancelled", "paused", "finished":
			return ""
		}
	}
	return ""
}

// spinStep advances the spinner while the agent works; the chain stops by itself when it is idle.
func (m emberModel) spinStep() (emberModel, tea.Cmd) {
	if m.working() == nil {
		m.spinning = false
		return m, nil
	}
	m.spinN++
	return m, spinCmd()
}

// spinStart starts the spinner's tick when work begins (called from the slow tick).
func (m emberModel) spinStart() (emberModel, tea.Cmd) {
	if m.spinning || m.working() == nil {
		return m, nil
	}
	m.spinning = true
	return m, spinCmd()
}

func compactCount(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1e3)
	}
	return fmt.Sprint(n)
}

func compactDuration(d time.Duration) string {
	s := int(d.Seconds())
	switch {
	case s >= 3600:
		return fmt.Sprintf("%dh %02dm", s/3600, s/60%60)
	case s >= 60:
		return fmt.Sprintf("%dm %02ds", s/60, s%60)
	}
	return fmt.Sprintf("%ds", s)
}

// stretch is this stretch of work on goal gid: when it began (your message, the goal's start or a wake-up), and
// since then the actions taken and about how many tokens were written (characters / 4, the live reply included).
func (m emberModel) stretch(gid string) (since time.Time, actions, tokens int) {
	chars := 0
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != gid {
			continue
		}
		switch e.Kind {
		case "goal_queued", "user_message", "peer_message", "woke":
			since = time.Unix(int64(e.Ts), 0)
		case "tool_result":
			actions++
		case "assistant":
			for _, k := range []string{"content", "reasoning"} {
				s, _ := e.Data[k].(string)
				chars += len(s)
			}
			calls, _ := e.Data["tool_calls"].([]any)
			for _, raw := range calls {
				c, _ := raw.(map[string]any)
				f, _ := c["function"].(map[string]any)
				a, _ := f["arguments"].(string)
				chars += len(a)
			}
		}
		if !since.IsZero() {
			break
		}
	}
	if m.chat.live.active {
		chars += len(m.chat.live.text) + len(m.chat.live.reasoning)
	}
	return since, actions, (chars + 3) / 4
}

// activityLine is the line over the composer while the agent works; "" otherwise.
func (m emberModel) activityLine(w int) string {
	if g := m.activeGoal(); g != nil && !m.daemonDown() {
		if at := m.parkedUntil(g.ID); at != "" {
			return ansi.Truncate(" "+emberDim.Render("◌ parked until "+at+" · a message wakes it sooner"), max(10, w), "…")
		}
	}
	gl := m.working()
	if gl == nil {
		return ""
	}
	frames := emberPulse
	if g.name == "ascii" {
		frames = emberPulseASCII
	}
	f := m.spinN % len(frames)
	// the pulse runs hot to bright and back: the theme's hot colour blended toward its warn colour by frame
	heat := float64(f) / float64(len(frames)/2)
	if heat > 1 {
		heat = 2 - heat
	}
	spin := lipgloss.NewStyle().Foreground(lipgloss.Color(blendHex(themeRole("hot"), themeRole("warn"), heat))).Bold(true).Render(frames[f])
	if m.chat.question != nil { // the prompt below asks: the activity line must not say "type below"
		return ansi.Truncate(" "+emberWarn.Render("◌ waiting for your answer"), max(10, w), "…")
	}
	if m.waitingForYou(gl.ID) { // parked on wait(until=message): no work, no spinner; it waits for your words
		hint := "  · type below to answer"
		if len(m.sessionRequests()) > 0 { // a request was put off: it is answered with Tab, not by typing
			hint = "  · Tab answers its request"
		}
		return ansi.Truncate(" "+emberWarn.Render("◌ waiting for your message")+emberDim.Render(hint), max(10, w), "…")
	}
	verb := emberVerbs[(m.spinN/55+len(gl.ID))%len(emberVerbs)] + "…" // a new word every ~6 s
	doing := ""
	l := m.chat.live
	switch {
	case gl.State == "waiting_model":
		doing = "the model is busy; waiting it out"
	case m.chat.group.running != "":
		doing = "$ " + m.chat.group.running
	case l.active && l.text == "" && l.reasoning != "":
		doing = "thinking"
	case l.active && l.text != "":
		doing = "writing"
	case l.active && len(l.tools) > 0:
		doing = "calling " + l.tools[len(l.tools)-1]
	}
	since, actions, tokens := m.stretch(gl.ID)
	var stats []string
	if !since.IsZero() {
		stats = append(stats, compactDuration(time.Since(since)))
	}
	if tokens > 0 {
		stats = append(stats, "↓ "+compactCount(tokens)+" tokens")
	}
	if actions > 0 {
		stats = append(stats, plural(actions, "action", "actions"))
	}
	if usd := m.resident.SpendGoals[gl.ID]; usd >= 0.005 {
		stats = append(stats, fmt.Sprintf("$%.2f", usd))
	}
	line := " " + spin + " " + lipgloss.NewStyle().Foreground(themeColor("warn")).Bold(true).Render(verb)
	if doing != "" {
		line += " " + lipgloss.NewStyle().Foreground(cText).Render(oneLine(doing))
	}
	if len(stats) > 0 {
		line += emberDim.Render("  (" + strings.Join(stats, " · ") + ")")
	}
	if m.chat.live.active { // the one moment Esc does something here: say so where the eye already is
		line += emberDim.Render("  · esc interrupts")
	}
	return ansi.Truncate(line, max(10, w), "…")
}

// waitingForYou: the goal's newest call is wait(until=message) and it has not returned: the agent is parked until you
// write (the header said "working" with the spinner going).
func (m emberModel) waitingForYou(goal string) bool {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != goal {
			continue
		}
		switch e.Kind {
		case "tool_result", "user_message":
			return false
		case "assistant":
			calls, _ := e.Data["tool_calls"].([]any)
			if len(calls) == 0 {
				return false
			}
			c, _ := calls[len(calls)-1].(map[string]any)
			f, _ := c["function"].(map[string]any)
			if name, _ := f["name"].(string); name != "wait" {
				return false
			}
			raw, _ := f["arguments"].(string)
			var args struct {
				Until string `json:"until"`
			}
			return json.Unmarshal([]byte(raw), &args) == nil && args.Until == "message"
		}
	}
	return false
}
