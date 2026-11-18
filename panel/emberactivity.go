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
		if _, held := m.interruptedUntil(g.ID); held {
			return nil // an Esc dropped its turn: it waits for your message, so nothing counts (it kept counting)
		}
		return g
	}
	return nil
}

// interruptedUntil: an Esc dropped the goal's turn and it now waits for your message (its stored state stays
// working). It holds while its newest turn event is turn_interrupted, no new reply is being generated and the hold has
// not run out; until is when the runtime carries on by itself.
func (m emberModel) interruptedUntil(goal string) (until time.Time, held bool) {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Goal != goal {
			continue
		}
		switch e.Kind {
		case "turn_interrupted":
			hold, _ := e.Data["hold_s"].(float64)
			turn, _ := e.Data["turn"].(string)
			if hold <= 0 {
				hold = 120 // (an older runtime's event: its hold)
			}
			until = time.Unix(int64(e.Ts), 0).Add(time.Duration(hold) * time.Second)
			if l := m.chat.live; l.active && l.turn != "" && l.turn != turn {
				return time.Time{}, false // a new call is out: the agent carries on
			}
			return until, time.Now().Before(until)
		case "assistant", "user_message", "peer_message", "tool_result", "goal_queued", "goal_continued", "woke",
			"turn_retried", "cancelled", "paused", "finished", "rewound", "emergency_stop", "scheduled", "compacting":
			return time.Time{}, false
		}
	}
	return time.Time{}, false
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

// stretch is this stretch of work on goal gid (a turn): when it began (your message, the goal's start or a wake-up),
// and since then the actions taken and the tokens written: each finished call's own count as its provider reported
// it (the assistant event's tokens), else its characters / 4, plus the call in flight's live text / 4. The live
// snapshot restarts with every call, so counting it alone dropped back at each step of a many-call turn.
func (m emberModel) stretch(gid string) (since time.Time, actions, tokens int) {
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
			if n, _ := e.Data["tokens"].(float64); n > 0 {
				tokens += int(n)
				break
			}
			chars := 0
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
			tokens += (chars + 3) / 4
		}
		if !since.IsZero() {
			break
		}
	}
	if m.chat.live.active {
		tokens += (len(m.chat.live.text) + len(m.chat.live.reasoning) + 3) / 4
	}
	return since, actions, tokens
}

// turnPeak is the most tokens the activity line has shown for one turn (a goal and its stretch's start).
type turnPeak struct {
	goal   string
	since  time.Time
	tokens int
}

// noteTurnTokens raises the turn's peak to what stretch counts now; a new turn (another goal, a new start) starts it
// over. Called where the events or the live reply change, not per frame.
func (m *emberModel) noteTurnTokens() {
	gl := m.working()
	if gl == nil {
		return
	}
	since, _, tokens := m.stretch(gl.ID)
	if p := &m.chat.turnPeak; p.goal != gl.ID || !p.since.Equal(since) {
		*p = turnPeak{gl.ID, since, tokens}
	} else if tokens > p.tokens {
		p.tokens = tokens
	}
}

// turnTokens is stretch with the count held at the turn's peak: it only goes up within a turn. Between a call's live
// text clearing and its event arriving the sum dips for a moment, and a provider's count can come in under the
// estimate it replaces; the line holds its number until the real sum passes it.
func (m emberModel) turnTokens(gid string) (since time.Time, actions, tokens int) {
	since, actions, tokens = m.stretch(gid)
	if p := m.chat.turnPeak; p.goal == gid && p.since.Equal(since) && p.tokens > tokens {
		tokens = p.tokens
	}
	return since, actions, tokens
}

// activityLine is the line over the composer while the agent works; "" otherwise.
func (m emberModel) activityLine(w int) string {
	if g := m.activeGoal(); g != nil && !m.daemonDown() {
		if at := m.parkedUntil(g.ID); at != "" {
			return ansi.Truncate(" "+emberDim.Render("◌ parked until "+at+" · a message wakes it sooner"), max(10, w), "…")
		}
	}
	if g := m.activeGoal(); g != nil && !m.daemonDown() {
		if until, held := m.interruptedUntil(g.ID); held {
			// no spinner and no clock: nothing runs until you write (or the hold runs out and it carries on)
			return ansi.Truncate(" "+emberWarn.Render("◌ turn interrupted · nothing from it was kept")+
				emberDim.Render("  · type below to steer · it carries on by itself at "+until.Format("15:04")), max(10, w), "…")
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
	since, actions, tokens := m.turnTokens(gl.ID)
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
