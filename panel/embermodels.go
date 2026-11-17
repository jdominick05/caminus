// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envmodels.go: /model and /effort. The Models overlay lists what the resident's providers serve (its catalog, from
// `caminus ember agent models`): window, effort levels and price per model, the current one marked. Enter picks one
// (`agent model`), r asks the providers again. /effort LEVEL sets the reasoning depth among the levels the current
// model offers; alone it names them. A running resident takes either change at its next turn.

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type catalogPrices struct {
	Input     *float64 `json:"input"`
	Output    *float64 `json:"output"`
	CacheRead *float64 `json:"cache_read"`
}

type catalogModel struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Provider  string         `json:"provider"`
	Context   *int           `json:"context"`
	MaxOutput *int           `json:"max_output"`
	Efforts   []string       `json:"efforts"`
	Thinking  string         `json:"thinking"`
	Vision    bool           `json:"vision"`
	Prices    *catalogPrices `json:"prices"`
	Stale     bool           `json:"stale"`
}

type modelCurrent struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Context  *int   `json:"context"`
	Effort   string `json:"effort"`
}

type modelsLoadedMsg struct {
	models  []catalogModel
	current modelCurrent
	err     string
}

type modelsState struct {
	open, loading bool
	sel           int
	list          []catalogModel
	current       modelCurrent
	err           string
}

func residentJSON(m emberModel, args []string, into any, input ...string) error {
	if len(m.cli) == 0 {
		return fmt.Errorf("resident controls need the caminus CLI (--cli)")
	}
	stdin := ""
	if len(input) > 0 {
		stdin = input[0] // (a description or a question, kept off the command line)
	}
	out, err := runResident(residentCommand(m, args, stdin))
	if err != nil {
		text := strings.TrimSpace(string(out))
		if i := strings.LastIndex(text, "\n"); i >= 0 {
			text = text[i+1:]
		}
		return fmt.Errorf("%s", strings.TrimSpace(strings.TrimPrefix(text, "error:")))
	}
	return json.Unmarshal(out, into)
}

func fetchModels(m emberModel, refresh bool) tea.Cmd {
	return func() tea.Msg {
		args := []string{"models", m.agentName, "--json"}
		if refresh {
			args = append(args, "--refresh")
		}
		var out struct {
			Models  []catalogModel `json:"models"`
			Current modelCurrent   `json:"current"`
		}
		if err := residentJSON(m, args, &out); err != nil {
			return modelsLoadedMsg{err: err.Error()}
		}
		return modelsLoadedMsg{models: out.Models, current: out.Current}
	}
}

// pickModel and setEffort report the outcome in the status line.
func pickModel(m emberModel, id string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Current modelCurrent `json:"current"`
		}
		if err := residentJSON(m, []string{"model", m.agentName, id, "--json"}, &out); err != nil {
			return residentActionMsg{"model: " + err.Error()}
		}
		c := out.Current
		effort := ""
		if c.Effort != "" {
			effort = " · effort " + c.Effort
		}
		return residentActionMsg{fmt.Sprintf("model %s (%s) · %s window%s · from the next turn", c.Model, c.Provider,
			tokensShort(c.Context), effort)}
	}
}

func setEffort(m emberModel, level string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"effort", m.agentName, "--json"}
		if level != "" {
			args = []string{"effort", m.agentName, level, "--json"}
		}
		var out struct {
			Effort *string  `json:"effort"`
			Levels []string `json:"levels"`
			Model  string   `json:"model"`
		}
		if err := residentJSON(m, args, &out); err != nil {
			return residentActionMsg{"effort: " + err.Error()}
		}
		cur := "the model's default"
		if out.Effort != nil && *out.Effort != "" {
			cur = *out.Effort
		}
		offers := "no effort control"
		if len(out.Levels) > 0 {
			offers = "offers " + strings.Join(out.Levels, ", ")
		}
		verb := "effort"
		if level != "" {
			verb = "effort set:"
		}
		return residentActionMsg{fmt.Sprintf("%s %s · %s %s", verb, cur, out.Model, offers)}
	}
}

func tokensShort(n *int) string {
	if n == nil || *n <= 0 {
		return "?"
	}
	v := *n
	switch {
	case v >= 1_000_000 && v%1_000_000 < 50_000:
		return fmt.Sprintf("%dM", (v+500_000)/1_000_000)
	case v >= 1000:
		return fmt.Sprintf("%dK", (v+500)/1000)
	}
	return fmt.Sprint(v)
}

func priceShort(p *catalogPrices) string {
	if p == nil || p.Input == nil || p.Output == nil {
		return ""
	}
	return fmt.Sprintf("$%g/$%g", *p.Input, *p.Output)
}

func (m emberModel) modelRow(c catalogModel, selected bool, width int) string {
	current := c.ID == m.chat.models.current.Model &&
		(m.chat.models.current.Provider == "" || c.Provider == m.chat.models.current.Provider)
	mark := "  "
	if current {
		mark = g.done + " "
	}
	meta := []string{c.Provider, tokensShort(c.Context)}
	if len(c.Efforts) > 0 {
		meta = append(meta, "effort "+strings.Join(c.Efforts, "/"))
	} else if c.Thinking != "" {
		meta = append(meta, "thinks")
	}
	if c.Vision {
		meta = append(meta, "vision")
	}
	if p := priceShort(c.Prices); p != "" {
		meta = append(meta, p)
	}
	if c.Stale {
		meta = append(meta, "stale")
	}
	if selected {
		return emberSel.Render(ansi.Truncate(" "+g.collapsed+" "+mark+c.ID+"  ·  "+strings.Join(meta, " · "), max(8, width-1), "…"))
	}
	row := "   " + emberCtxOK.Render(mark) + lipgloss.NewStyle().Foreground(cText).Render(c.ID) +
		emberDim.Render("  ·  "+strings.Join(meta, " · "))
	return ansi.Truncate(row, width, "…")
}

func (m emberModel) modelsOverlay(w, h int) string {
	st := m.chat.models
	inner := max(10, w-6)
	listH := max(1, h-6)
	sel := max(0, min(st.sel, len(st.list)-1))
	first := 0
	if len(st.list) > listH {
		first = max(0, min(sel-listH+1, len(st.list)-listH))
	}
	var rows []string
	switch {
	case st.loading && len(st.list) == 0:
		rows = append(rows, emberDim.Render("asking the providers…"))
	case st.err != "" && len(st.list) == 0:
		rows = append(rows, emberWarn.Render(ansi.Truncate(st.err, inner, "…")))
	case len(st.list) == 0:
		rows = append(rows, emberDim.Render("no models: add a provider (caminus ember agent provider NAME add …), then r"))
	}
	for i := first; i < min(first+listH, len(st.list)); i++ {
		rows = append(rows, m.modelRow(st.list[i], i == sel, inner))
	}
	note := "Enter picks it (from the next turn)  ·  r asks the providers again  ·  /effort sets the depth"
	if st.loading && len(st.list) > 0 {
		note = "refreshing…"
	} else if st.err != "" && len(st.list) > 0 {
		note = st.err
	}
	body := strings.Join(rows, "\n") + "\n\n" + emberDim.Render(ansi.Truncate(note, inner, "…"))
	title := fmt.Sprintf("MODELS  %d", len(st.list))
	if st.current.Model != "" {
		title += "  ·  now " + st.current.Model
	}
	return overlayBox(title, body, "↑↓ move  ·  Enter pick", w, h)
}

func (m emberModel) modelsKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	st := &m.chat.models
	switch msg.String() {
	case "ctrl+c":
		return m, tea.Quit
	case "esc", "q":
		st.open = false
		return m, m.chat.compose.Focus()
	case "up", "k":
		st.sel = max(0, st.sel-1)
	case "down", "j":
		st.sel = min(max(0, len(st.list)-1), st.sel+1)
	case "r":
		st.loading, st.err = true, ""
		return m, fetchModels(m, true)
	case "enter":
		if len(st.list) == 0 {
			return m, nil
		}
		c := st.list[max(0, min(st.sel, len(st.list)-1))]
		st.open = false
		m.msg = "switching to " + c.ID + "…"
		return m, tea.Batch(pickModel(m, c.Provider+"/"+c.ID), m.chat.compose.Focus())
	}
	return m, nil
}

// modelsLoaded keeps the selection on the current model when the list arrives.
func (m emberModel) modelsLoaded(msg modelsLoadedMsg) emberModel {
	st := &m.chat.models
	st.loading, st.err = false, msg.err
	if msg.err != "" {
		return m
	}
	st.list, st.current = msg.models, msg.current
	for i, c := range st.list {
		if c.ID == st.current.Model {
			st.sel = i
			break
		}
	}
	return m
}

// emergencyStop (/halt) runs `caminus ember stop-all`: a STOP file in the sticky bridge folder that every resident reads before
// each step and cannot remove. /halt clear lifts it; paused goals still wait for /resume.
func emergencyStop(m emberModel, arg string) tea.Cmd {
	return func() tea.Msg {
		if len(m.cli) == 0 {
			return residentActionMsg{"the emergency stop needs the caminus CLI (--cli)"}
		}
		argv := append(append([]string{}, m.cli[1:]...), "ember", "stop-all")
		if arg == "clear" {
			argv = append(argv, "--clear")
		} else if arg != "" {
			argv = append(argv, "--reason", arg)
		}
		out, err := exec.Command(m.cli[0], argv...).CombinedOutput()
		text := strings.TrimSpace(string(out))
		if err != nil {
			text = "stop-all failed: " + text
		}
		return residentActionMsg{text}
	}
}
