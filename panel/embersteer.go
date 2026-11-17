// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envsteer.go: steering and inspection from the chat.
//   Esc while a reply is being written   interrupts that turn only (nothing stored; the next message steers)
//   ↑ on an empty composer               pulls back your newest message the agent has not read yet
//   /btw QUESTION                        a side question with no tools; the answer opens here, the goal keeps nothing
//   /diff                                what the goal changed in its git repositories, coloured, in an overlay
//   /export                              the goal as markdown, in the agent's ~/exports
//   /rewind                              back to an earlier step, files included (two presses: it rewrites files)
//   /plan TEXT, /plan approve|revise [NOTE]   a goal that explores and proposes before it changes anything
//   Shift+↑/↓ in the Goals overlay       move a queued goal up or down the queue

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// btwState is the /btw side answer: a section of its own above the composer (with a small Ember), not an overlay.
type btwState struct {
	open, loading bool
	question      string
	lines         []string // the answer rendered (Markdown), sanitised
	err           string
	top           int // scrolled with Shift+Up / Shift+Down
}

type btwMsg struct {
	question string
	answer   string
	err      string
}

// textView is a read-only overlay of lines (the /btw answer, the /diff).
type textView struct {
	open        bool
	title, keys string
	lines       []string
	top         int
	colourDiff  bool
}

type textViewMsg struct {
	title string
	lines []string
	diff  bool
	err   string
}

type rewindStep struct {
	Step  int    `json:"step"`
	Label string `json:"label"`
}

type rewindState struct {
	open  bool
	goal  string
	steps []rewindStep
	sel   int
	err   string
}

type rewindListMsg struct {
	goal  string
	steps []rewindStep
	err   string
}

type unsendMsg struct {
	text *string
	err  string
}

func (m emberModel) targetGoal() string {
	if m.chat.target != "" {
		return m.chat.target
	}
	if g := m.selectedResidentGoal(); g != nil {
		return g.ID
	}
	return ""
}

func interruptTurn(m emberModel, goal string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Queued bool `json:"queued"`
		}
		if err := residentJSON(m, []string{"interrupt", m.agentName, goal, "--json"}, &out); err != nil {
			return residentActionMsg{"interrupt: " + err.Error()}
		}
		return residentActionMsg{"turn interrupted: nothing from it is kept; your next message steers"}
	}
}

func unsendLast(m emberModel, goal string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Text *string `json:"text"`
		}
		if err := residentJSON(m, []string{"unsend", m.agentName, goal, "--json"}, &out); err != nil {
			return unsendMsg{err: err.Error()}
		}
		return unsendMsg{text: out.Text}
	}
}

// adviseNow runs `agent advise NAME GOAL [FOCUS]`: the advisor's review of the goal, delivered to the agent with the
// reply it is writing dropped. The outcome lands in the status line; the review shows in the conversation.
func adviseNow(m emberModel, goal, focus string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"advise", m.agentName, goal}
		if focus != "" {
			args = append(args, focus)
		}
		var out struct {
			Model string  `json:"model"`
			USD   float64 `json:"usd"`
		}
		if err := residentJSON(m, append(args, "--json"), &out); err != nil {
			text := err.Error()
			if strings.Contains(text, "no advisor") {
				text = "no advisor set up: /advisor setup picks one"
			}
			return residentActionMsg{"advisor: " + terminalText(text, false)}
		}
		return residentActionMsg{fmt.Sprintf("advisor (%s) reviewed the goal; the agent reads it before its next step ($%.2f)",
			terminalText(out.Model, false), out.USD)}
	}
}

func askBtw(m emberModel, goal, question string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Answer string `json:"answer"`
		}
		args := []string{"btw", m.agentName, goal, "--stdin", "--json"}
		if len(m.cli) == 0 {
			return textViewMsg{err: "needs the caminus CLI (--cli)"}
		}
		raw, err := runResident(residentCommand(m, args, question))
		if err == nil {
			if i := strings.Index(string(raw), "{"); i >= 0 {
				err = jsonInto(raw[i:], &out)
			}
		}
		if err != nil {
			return btwMsg{question: question, err: strings.TrimSpace(string(raw))}
		}
		return btwMsg{question: question, answer: out.Answer}
	}
}

func fetchDiff(m emberModel, goal string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Repos []struct {
				Path      string `json:"path"`
				Stat      string `json:"stat"`
				Diff      string `json:"diff"`
				Truncated bool   `json:"truncated"`
			} `json:"repos"`
		}
		args := []string{"diff", m.agentName, "--json"}
		if goal != "" {
			args = []string{"diff", m.agentName, goal, "--json"}
		}
		if err := residentJSON(m, args, &out); err != nil {
			return textViewMsg{err: "diff: " + err.Error()}
		}
		var lines []string
		for _, r := range out.Repos {
			lines = append(lines, "### "+r.Path)
			lines = append(lines, strings.Split(strings.TrimRight(r.Stat, "\n"), "\n")...)
			lines = append(lines, "")
			lines = append(lines, strings.Split(strings.TrimRight(r.Diff, "\n"), "\n")...)
			if r.Truncated {
				lines = append(lines, "… (cut at 200 KB)")
			}
			lines = append(lines, "")
		}
		if len(out.Repos) == 0 {
			lines = []string{"no changes in a git repository this goal touched"}
		}
		for i, l := range lines { // (the agent's files: sanitised here, the overlay draws lines as they are)
			lines[i] = residentText(l)
		}
		return textViewMsg{title: "diff", lines: lines, diff: true}
	}
}

func exportGoal(m emberModel, goal string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Path  string `json:"path"`
			Bytes int    `json:"bytes"`
		}
		if err := residentJSON(m, []string{"export", m.agentName, goal, "--json"}, &out); err != nil {
			return residentActionMsg{"export: " + err.Error()}
		}
		return residentActionMsg{fmt.Sprintf("exported to %s (%d bytes, in the agent's account)", out.Path, out.Bytes)}
	}
}

func fetchRewind(m emberModel, goal string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Steps []rewindStep `json:"steps"`
		}
		if err := residentJSON(m, []string{"rewind", m.agentName, goal, "--list", "--json"}, &out); err != nil {
			return rewindListMsg{goal: goal, err: err.Error()}
		}
		return rewindListMsg{goal: goal, steps: out.Steps}
	}
}

func doRewind(m emberModel, goal string, step int) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			To       int      `json:"to"`
			Queued   bool     `json:"queued"`
			Restored []string `json:"restored"`
			Removed  []string `json:"removed"`
			Note     string   `json:"note"`
		}
		if err := residentJSON(m, []string{"rewind", m.agentName, goal, "--to", strconv.Itoa(step), "--json"}, &out); err != nil {
			return residentActionMsg{"rewind: " + err.Error()}
		}
		if out.Queued { // the agent is running: it rewinds between steps, and the rewound event shows what changed
			return residentActionMsg{fmt.Sprintf("rewind to step %d queued: it happens between steps", out.To)}
		}
		return residentActionMsg{fmt.Sprintf("rewound to step %d: %d files restored, %d removed. %s", out.To,
			len(out.Restored), len(out.Removed), out.Note)}
	}
}

func decidePlan(m emberModel, goal, verdict, note string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"plan", m.agentName, goal, verdict, "--json"}
		if note != "" {
			args = append(args, "--note", note)
		}
		var out struct {
			Plan string `json:"plan"`
		}
		if err := residentJSON(m, args, &out); err != nil {
			return residentActionMsg{"plan: " + err.Error()}
		}
		if out.Plan == "approved" {
			return residentActionMsg{"plan approved: the agent goes ahead"}
		}
		return residentActionMsg{"sent back for revision: it stays in plan mode"}
	}
}

func moveQueued(m emberModel, goal string, to int) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Queue []struct {
				ID string `json:"id"`
			} `json:"queue"`
		}
		if err := residentJSON(m, []string{"queue", m.agentName, "move", goal, "--to", strconv.Itoa(to), "--json"}, &out); err != nil {
			return residentActionMsg{"queue: " + err.Error()}
		}
		return residentActionMsg{fmt.Sprintf("goal %s is now %d in the queue", shortGoal(goal), to)}
	}
}

func (m emberModel) sendPlanGoal(text string) (tea.Model, tea.Cmd) {
	name, epoch := m.agentName, m.residentEpoch
	m.chat.sending = true
	return m, func() tea.Msg {
		out, err := runResident(residentCommand(m, []string{"send", name, "--plan", "--stdin", "--json"}, text))
		var reply struct {
			ID string `json:"id"`
		}
		if err != nil || jsonInto(out, &reply) != nil || reply.ID == "" {
			return residentComposeResult{agent: name, epoch: epoch, err: "plan goal not sent: " + strings.TrimSpace(string(out))}
		}
		return residentComposeResult{agent: name, epoch: epoch, id: reply.ID}
	}
}

// planCard is the plan a plan-first goal proposed, waiting for the owner.
func (m emberModel) planCard(w int) string {
	var g *residentGoal
	for i := range m.resident.Goals {
		if m.resident.Goals[i].State == "waiting_plan" {
			g = &m.resident.Goals[i]
		}
	}
	if g == nil {
		return ""
	}
	inner := max(10, w-4)
	rows := []string{emberWarn.Render("PLAN · goal " + shortGoal(g.ID) + " waits for your OK")}
	plan := strings.Split(ansi.Hardwrap(residentText(g.PlanText), inner, true), "\n")
	if len(plan) > 14 {
		plan = append(plan[:14], "… (/diff, /export or the goals overlay show more)")
	}
	for _, l := range plan {
		rows = append(rows, lipgloss.NewStyle().Foreground(cText).Render(l))
	}
	rows = append(rows, emberDim.Render("/plan approve  ·  /plan revise WHAT TO CHANGE"))
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cYellow).Padding(0, 1).
		Width(max(12, w-2)).Render(strings.Join(rows, "\n"))
}

func colourDiffLine(l string) string {
	switch {
	case strings.HasPrefix(l, "### "):
		return emberHead.Render(l)
	case strings.HasPrefix(l, "+++") || strings.HasPrefix(l, "---") || strings.HasPrefix(l, "diff --git"):
		return emberDim.Render(l)
	case strings.HasPrefix(l, "+"):
		return lipgloss.NewStyle().Foreground(cGreen).Render(l)
	case strings.HasPrefix(l, "-"):
		return lipgloss.NewStyle().Foreground(cRed).Render(l)
	case strings.HasPrefix(l, "@@"):
		return emberBusy.Render(l)
	}
	return lipgloss.NewStyle().Foreground(cText).Render(l)
}

func (m emberModel) textOverlay(w, h int) string {
	tv := m.chat.view
	inner := max(10, w-6)
	room := max(1, h-6)
	var lines []string
	for _, l := range tv.lines {
		// (every producer sanitises what came from the agent; the lines carry their own styling)
		lines = append(lines, strings.Split(ansi.Hardwrap(l, inner, true), "\n")...)
	}
	top := max(0, min(tv.top, max(0, len(lines)-room)))
	var rows []string
	for _, l := range lines[top:min(len(lines), top+room)] {
		if tv.colourDiff {
			rows = append(rows, colourDiffLine(l))
		} else {
			rows = append(rows, l)
		}
	}
	return overlayBox(tv.title, strings.Join(rows, "\n"), "↑↓ PgUp PgDn scroll", w, h)
}

func (m emberModel) textKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	tv := &m.chat.view
	switch msg.String() {
	case "ctrl+c":
		return m, tea.Quit
	case "esc", "q":
		tv.open = false
		return m, m.chat.compose.Focus()
	case "up", "k":
		tv.top = max(0, tv.top-1)
	case "down", "j":
		tv.top++
	case "pgup":
		tv.top = max(0, tv.top-10)
	case "pgdown", "space", " ":
		tv.top += 10
	}
	return m, nil
}

func (m emberModel) rewindOverlay(w, h int) string {
	st := m.chat.rewind
	inner := max(10, w-6)
	var rows []string
	if st.err != "" {
		rows = append(rows, emberWarn.Render(ansi.Truncate(st.err, inner, "…")))
	} else if len(st.steps) == 0 {
		rows = append(rows, emberDim.Render("reading the goal's steps…"))
	}
	listH := max(1, h-7)
	sel := max(0, min(st.sel, len(st.steps)-1))
	first := 0
	if len(st.steps) > listH {
		first = max(0, min(sel-listH+1, len(st.steps)-listH))
	}
	for i := first; i < min(first+listH, len(st.steps)); i++ {
		s := st.steps[i]
		label := fmt.Sprintf("undo from step %d on  (%s)", s.Step, residentText(s.Label))
		if i == sel {
			rows = append(rows, emberSel.Render(ansi.Truncate(" "+g.collapsed+" "+label, max(8, inner-1), "…")))
		} else {
			rows = append(rows, ansi.Truncate("   "+emberDim.Render(label), inner, "…"))
		}
	}
	rows = append(rows, "", emberDim.Render(ansi.Truncate("edited files are restored; terminal side effects are not undone", inner, "…")))
	return overlayBox("REWIND  goal "+shortGoal(st.goal), strings.Join(rows, "\n"), "↑↓ move  ·  Enter twice rewinds", w, h)
}

func (m emberModel) rewindKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	st := &m.chat.rewind
	armed := m.chat.armed
	m.chat.armed = ""
	switch msg.String() {
	case "ctrl+c":
		return m, tea.Quit
	case "esc", "q":
		st.open = false
		return m, m.chat.compose.Focus()
	case "up", "k":
		st.sel = max(0, st.sel-1)
	case "down", "j":
		st.sel = min(max(0, len(st.steps)-1), st.sel+1)
	case "enter":
		if len(st.steps) == 0 {
			return m, nil
		}
		s := st.steps[max(0, min(st.sel, len(st.steps)-1))]
		key := fmt.Sprintf("rewind:%s:%d", st.goal, s.Step)
		if armed != key || time.Since(m.chat.armedAt) > 10*time.Second {
			m.chat.armed, m.chat.armedAt = key, time.Now()
			m.msg = fmt.Sprintf("undo goal %s from step %d on (%s)? it restores files: press Enter again",
				shortGoal(st.goal), s.Step, residentText(s.Label))
			return m, nil
		}
		st.open = false
		// "undo from step N on" keeps the work of the steps before it: the rewind goes to step N-1
		return m, tea.Batch(doRewind(m, st.goal, s.Step-1), m.chat.compose.Focus())
	}
	return m, nil
}

// steerSlash handles the commands in this file; ok reports whether it took the command.
func (m emberModel) steerSlash(command, arg string) (emberModel, tea.Cmd, bool) {
	goal := m.targetGoal()
	need := func() bool {
		if goal == "" {
			m.msg = "no goal selected: /goals picks one"
			return false
		}
		return true
	}
	switch command {
	case "/advisor":
		// a review now: the advisor reads the goal and its advice reaches the agent with its current reply dropped, so
		// it reads the advice before its next step. /advisor setup is the settings screen (model, budget, off).
		if strings.TrimSpace(arg) == "setup" {
			next, cmd := m.openAdvisor()
			return next, cmd, true
		}
		if !need() {
			return m, nil, true
		}
		m.msg = "the advisor is reviewing goal " + shortGoal(goal) + "…"
		return m, adviseNow(m, goal, strings.TrimSpace(arg)), true
	case "/btw":
		if strings.TrimSpace(arg) == "" || !need() {
			if strings.TrimSpace(arg) == "" {
				m.msg = "usage: /btw QUESTION — a side question; nothing enters the goal"
			}
			return m, nil, true
		}
		m.chat.btw = btwState{open: true, loading: true, question: arg}
		m.syncChat()
		return m, askBtw(m, goal, arg), true
	case "/diff":
		m.msg = "reading the changes…"
		return m, fetchDiff(m, goal), true
	case "/export":
		if !need() {
			return m, nil, true
		}
		return m, exportGoal(m, goal), true
	case "/rewind":
		if !need() {
			return m, nil, true
		}
		m.chat.rewind = rewindState{open: true, goal: goal}
		return m, fetchRewind(m, goal), true
	case "/plan":
		verdict, note, _ := strings.Cut(strings.TrimSpace(arg), " ")
		planWaiting := false
		for _, gl := range m.resident.Goals {
			planWaiting = planWaiting || gl.State == "waiting_plan"
		}
		// approve / revise answer a waiting plan; with none waiting, all of TEXT is a new goal's ("/plan revise the
		// README" starts one, it does not report that no plan waits)
		if !planWaiting || (verdict != "approve" && verdict != "revise") {
			if strings.TrimSpace(arg) == "" {
				m.msg = "usage: /plan TEXT starts a goal that plans first  ·  /plan approve  ·  /plan revise WHAT TO CHANGE"
				return m, nil, true
			}
			m.chat.newGoal, m.chat.target = true, "" // /plan TEXT: a goal that explores and proposes before it changes anything
			next, cmd := m.sendPlanGoal(strings.TrimSpace(arg))
			return next.(emberModel), cmd, true
		}
		var waiting string
		for _, gl := range m.resident.Goals {
			if gl.State == "waiting_plan" {
				waiting = gl.ID
			}
		}
		if waiting == "" {
			m.msg = "no plan is waiting"
			return m, nil, true
		}
		if verdict == "revise" {
			verdict = "reject"
		}
		return m, decidePlan(m, waiting, verdict, strings.TrimSpace(note)), true
	}
	return m, nil, false
}

func jsonInto(b []byte, v any) error {
	if i := strings.Index(string(b), "{"); i > 0 {
		b = b[i:]
	}
	return json.Unmarshal(b, v)
}
