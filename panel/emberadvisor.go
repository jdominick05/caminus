// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberadvisor.go: /advisor, the agent's advisor (`caminus ember agent advisor NAME`): the stronger model it consults
// when stuck, before a risky step or before a hard finish, on your key and within a daily budget. The screen shows the
// model, its endpoint, the budget and calls per goal, and today's spend; m sets the model, b the budget, c the calls per
// goal, o turns it off (two presses). With one set, Ember also consults it on its own before the watchdog pauses a goal
// caught looping, and the judge's advising rounds use it.

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// the relay `caminus ember relay install --provider anthropic` runs: where a new advisor points by default
const advisorDefaultEndpoint = "http://127.0.0.1:8765/v1"

type advisorInfo struct {
	Advisor *struct {
		Model        string  `json:"model"`
		Endpoint     string  `json:"endpoint"`
		BudgetUSD    float64 `json:"budget_usd"`
		CallsPerGoal int     `json:"calls_per_goal"`
		Effort       string  `json:"effort"`
	} `json:"advisor"`
	Today struct {
		USD   float64 `json:"usd"`
		Calls int     `json:"calls"`
	} `json:"today"`
}

type advisorLoadedMsg struct {
	info advisorInfo
	err  string
	verb string
}

type advisorState struct {
	open, loading bool
	info          advisorInfo
	err           string
	field         string // "" browsing, or the flag being typed: model, budget-usd, calls-per-goal
	input         textinput.Model
	armed         time.Time
}

func advisorCmd(m emberModel, verb string, args ...string) tea.Cmd {
	return func() tea.Msg {
		var out advisorInfo
		if err := residentJSON(m, append(append([]string{"advisor", m.agentName}, args...), "--json"), &out); err != nil {
			return advisorLoadedMsg{err: err.Error()}
		}
		return advisorLoadedMsg{info: out, verb: verb}
	}
}

func (m emberModel) openAdvisor() (emberModel, tea.Cmd) {
	st := &m.chat.advisor
	st.open, st.loading, st.err, st.field = true, true, "", ""
	st.input = textinput.New()
	return m, advisorCmd(m, "")
}

func (m emberModel) advisorLoaded(msg advisorLoadedMsg) emberModel {
	st := &m.chat.advisor
	st.loading, st.err = false, msg.err
	if msg.err == "" {
		st.info = msg.info
		if msg.verb != "" {
			m.msg = "advisor " + msg.verb
		}
	}
	return m
}

func (m emberModel) advisorOverlay(w, h int) string {
	st := m.chat.advisor
	inner := max(10, w-6)
	var rows []string
	row := func(k, v string) {
		rows = append(rows, emberDim.Render(fmt.Sprintf("%-16s", k))+ansi.Truncate(v, max(4, inner-16), "…"))
	}
	a := st.info.Advisor
	switch {
	case st.loading:
		rows = append(rows, emberDim.Render("reading the advisor…"))
	case st.err != "":
		rows = append(rows, emberWarn.Render(ansi.Truncate(st.err, inner, "…")))
	case a == nil:
		rows = append(rows, emberDim.Render("no advisor: the agent decides alone, and the judge's advising rounds use its own model"),
			"", emberDim.Render("m sets one (a model the endpoint serves; the Anthropic relay by default)"))
	default:
		row("model", emberCtxOK.Render(g.done+" ")+terminalText(a.Model, false))
		row("endpoint", terminalText(a.Endpoint, false))
		row("budget", fmt.Sprintf("$%.2f a day · $%.2f spent today in %d calls", a.BudgetUSD, st.info.Today.USD, st.info.Today.Calls))
		row("calls per goal", strconv.Itoa(a.CallsPerGoal))
		row("used", "when the agent asks · before the watchdog pauses a looping goal · in the judge's advising rounds")
	}
	if st.field != "" {
		rows = append(rows, "", emberTitle.Render(st.field)+emberDim.Render("  Enter sets it · esc cancels"), st.input.View())
	}
	note := "m model · b budget · c calls per goal · o off · r read again · esc closes"
	body := strings.Join(rows[:min(len(rows), max(1, h-6))], "\n") + "\n\n" + emberDim.Render(ansi.Truncate(note, inner, "…"))
	return overlayBox("ADVISOR  ·  "+terminalText(m.agentName, false), body, "m model  ·  b budget", w, h)
}

func (m emberModel) advisorKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	st := &m.chat.advisor
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	if st.field != "" {
		switch k {
		case "esc":
			st.field = ""
			return m, nil
		case "enter":
			v := strings.TrimSpace(st.input.Value())
			field := st.field
			st.field = ""
			if v == "" {
				return m, nil
			}
			args := []string{"--" + field, v}
			if field == "model" && (st.info.Advisor == nil || st.info.Advisor.Endpoint == "") {
				args = append(args, "--endpoint", advisorDefaultEndpoint)
			}
			st.loading = true
			return m, advisorCmd(m, "set: "+field+" "+v, args...)
		}
		var cmd tea.Cmd
		st.input, cmd = st.input.Update(msg)
		return m, cmd
	}
	armed := st.armed
	st.armed = time.Time{}
	edit := func(field, value string) (tea.Model, tea.Cmd) {
		st.field = field
		st.input.SetValue(value)
		return m, st.input.Focus()
	}
	a := st.info.Advisor
	switch k {
	case "esc", "q":
		st.open = false
		return m, m.chat.compose.Focus()
	case "r":
		st.loading = true
		return m, advisorCmd(m, "")
	case "m":
		cur := "claude-opus-5-5"
		if a != nil {
			cur = a.Model
		}
		return edit("model", cur)
	case "b":
		if a != nil {
			return edit("budget-usd", fmt.Sprintf("%.2f", a.BudgetUSD))
		}
	case "c":
		if a != nil {
			return edit("calls-per-goal", strconv.Itoa(a.CallsPerGoal))
		}
	case "o":
		if a == nil {
			return m, nil
		}
		if armed.IsZero() || time.Since(armed) > 10*time.Second {
			st.armed = time.Now()
			m.msg = "turn the advisor off? the agent then decides alone · press o again"
			return m, nil
		}
		st.loading = true
		return m, advisorCmd(m, "off", "--off")
	}
	return m, nil
}
