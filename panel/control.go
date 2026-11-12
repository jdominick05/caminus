// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// control.go: the Caminus control panel, in llm-top's design language (its palette, panel() and fit() from top_main.go).
// It never runs the engine or edits a config itself: every read and every action is one `caminus <command> --json` run
// (backend.go), so the panel and the command line can never disagree.
//
//	caminus panel                 basic layout: pick a model, start it, launch Qwen Code
//	caminus panel -advanced       every option `caminus serve` accepts, several servers, the cards
//	caminus panel -caminus CMD    the caminus command line to drive (default: CAMINUS_CLI, else caminus on PATH)
//	caminus panel -once [W H]     one frame to stdout once every read answered (tests), then exit
//
// Keys: tab / shift+tab = next / previous panel, up / down (k / j) = move, a = basic <-> advanced, R = refresh, q = quit;
// the footer lists the keys of the panel in focus.
package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

func bold(c color.Color) lipgloss.Style { return lipgloss.NewStyle().Foreground(c).Bold(true) }

func pill(bg color.Color, s string) string {
	return lipgloss.NewStyle().Background(bg).Foreground(lipgloss.Color("#1b1f27")).Bold(true).Padding(0, 1).Render(s)
}

// bar is a used / total gauge of w cells, coloured by how close it is to the line.
func bar(used, total float64, w int) string {
	if total <= 0 || w <= 0 {
		return fg(cGray).Render(strings.Repeat("·", max(0, w)))
	}
	f := min(1, max(0, used/total))
	n := int(f*float64(w) + 0.5)
	c := cGreen
	switch {
	case f > 0.97:
		c = cRed
	case f > 0.9:
		c = cOrange
	case f > 0.75:
		c = cYellow
	}
	return fg(c).Render(strings.Repeat("█", n)) + fg(cGray).Render(strings.Repeat("░", w-n))
}

const (
	paneModels = iota
	paneServers
	paneHarness
	paneOptions
)

var paneNames = []string{"MODELS", "SERVERS", "QWEN CODE", "OPTIONS"}

// ctl is the control panel's Bubble Tea model.
type ctl struct {
	be       *ctlCLI
	w, h     int
	frame    int
	advanced bool
	focus    int

	models  []camModel
	ps      psInfo
	hs      harnessInfo
	opts    []option
	loaded  map[string]bool   // which reads answered at least once
	vals    map[string]string // advanced: option values for the next start (dest -> text); absent = the default
	card    string            // one-card models: "auto", "0", "1" (advanced; basic: auto)
	curM    int
	curS    int
	curO    int
	editing bool
	edit    string
	confirm *pending // a y / n question
	busy    string   // the action in flight ("" = none)
	note    string   // the last result line
	noteErr bool
	prefs   string // where the layout choice and the option values are remembered ("" = nowhere: tests)
}

// pending is an action waiting for y.
type pending struct {
	q   string
	run tea.Cmd
}

type (
	modelsMsg struct {
		v   []camModel
		err error
	}
	psMsg struct {
		v   psInfo
		err error
	}
	harnessMsg struct {
		v   harnessInfo
		err error
	}
	optsMsg struct {
		v   []option
		err error
	}
	actMsg struct {
		what, note string
		err        error
	}
	ctlTick time.Time
	ctlPoll time.Time
)

func ctlTickCmd() tea.Cmd {
	return tea.Tick(150*time.Millisecond, func(t time.Time) tea.Msg { return ctlTick(t) })
}
func ctlPollCmd() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg { return ctlPoll(t) })
}

func (m ctl) loadModels() tea.Cmd {
	return func() tea.Msg {
		var v struct {
			Models []camModel `json:"models"`
		}
		err := m.be.call(&v, 30*time.Second, "models", "--json")
		return modelsMsg{v.Models, err}
	}
}

func (m ctl) loadPS() tea.Cmd {
	return func() tea.Msg {
		var v psInfo
		err := m.be.call(&v, 15*time.Second, "ps", "--json")
		return psMsg{v, err}
	}
}

func (m ctl) loadHarness() tea.Cmd {
	return func() tea.Msg {
		var v harnessInfo
		err := m.be.call(&v, 15*time.Second, "qwen-code", "status", "--json")
		return harnessMsg{v, err}
	}
}

func (m ctl) loadOpts() tea.Cmd {
	return func() tea.Msg {
		var v []option
		err := m.be.call(&v, 30*time.Second, "serve", "--options-json")
		return optsMsg{v, err}
	}
}

// action runs one `caminus ... --json` action off the UI thread; the reads are repeated after it.
func (m ctl) action(what string, timeout time.Duration, args ...string) tea.Cmd {
	// (--json goes before a "--": what follows it is passed on to caminus serve as it is)
	a := append([]string{}, args...)
	at := len(a)
	for i, x := range a {
		if x == "--" {
			at = i
			break
		}
	}
	a = append(a[:at], append([]string{"--json"}, args[at:]...)...)
	return func() tea.Msg {
		note, err := m.be.act(timeout, a...)
		return actMsg{what, note, err}
	}
}

// interactive hands the terminal to `caminus args...` (Qwen Code, the log pager, the monitor) and comes back when it exits.
func (m ctl) interactive(what string, args ...string) tea.Cmd {
	return tea.ExecProcess(m.be.cmd(args...), func(err error) tea.Msg { return actMsg{what, "", err} })
}

func (m ctl) Init() tea.Cmd {
	return tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadOpts(), ctlTickCmd(), ctlPollCmd())
}

func (m *ctl) panes() []int {
	if m.advanced {
		return []int{paneModels, paneServers, paneHarness, paneOptions}
	}
	return []int{paneModels, paneServers, paneHarness}
}

func (m *ctl) cycle(d int) {
	p := m.panes()
	i := 0
	for k, v := range p {
		if v == m.focus {
			i = k
		}
	}
	m.focus = p[(i+d+len(p))%len(p)]
}

func (m *ctl) setNote(s string, isErr bool) { m.note, m.noteErr = s, isErr }

func (m ctl) selModel() *camModel {
	if m.curM >= 0 && m.curM < len(m.models) {
		return &m.models[m.curM]
	}
	return nil
}

func (m ctl) selServer() *serverInfo {
	if m.curS >= 0 && m.curS < len(m.ps.Servers) {
		return &m.ps.Servers[m.curS]
	}
	return nil
}

// startArgs is `caminus start NAME ...` with the card and, in the advanced layout, every edited option after `--` (passed
// to caminus serve as they are).
func (m ctl) startArgs(mi camModel) []string {
	a := []string{"start", mi.Name}
	if !m.advanced {
		return a
	}
	if mi.Cards == 1 && m.card != "" && m.card != "auto" {
		a = append(a, "--card", m.card)
	}
	var keys []string
	for k := range m.vals {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var extra []string
	for _, k := range keys {
		o := m.optByDest(k)
		if o == nil {
			continue
		}
		v := m.vals[k]
		if o.Type == "bool" {
			if v == "true" {
				extra = append(extra, o.Flag)
			}
			continue
		}
		extra = append(extra, o.Flag, v)
	}
	if len(extra) > 0 {
		a = append(append(a, "--"), extra...)
	}
	return a
}

func (m ctl) optByDest(d string) *option {
	for i := range m.opts {
		if m.opts[i].Dest == d {
			return &m.opts[i]
		}
	}
	return nil
}

// visibleOpts: the options the advanced layout lists (--model is the MODELS selection, --port is picked by caminus start,
// --config would replace the panel's own values).
func (m ctl) visibleOpts() []option {
	var out []option
	for _, o := range m.opts {
		switch o.Dest {
		case "model", "config", "port":
			continue
		}
		out = append(out, o)
	}
	return out
}

func (m ctl) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.w, m.h = msg.Width, msg.Height
	case ctlTick:
		m.frame++
		return m, ctlTickCmd()
	case ctlPoll:
		return m, tea.Batch(m.loadPS(), ctlPollCmd())
	case modelsMsg:
		m.loaded["models"] = true
		if msg.err != nil {
			m.setNote(msg.err.Error(), true)
		} else {
			m.models = msg.v
			m.curM = min(m.curM, max(0, len(m.models)-1))
		}
	case psMsg:
		m.loaded["ps"] = true
		if msg.err == nil {
			m.ps = msg.v
			m.curS = min(m.curS, max(0, len(m.ps.Servers)-1))
		} else if m.busy == "" {
			m.setNote(msg.err.Error(), true)
		}
	case harnessMsg:
		m.loaded["harness"] = true
		if msg.err == nil {
			m.hs = msg.v
		}
	case optsMsg:
		m.loaded["opts"] = true
		if msg.err == nil {
			m.opts = msg.v
		}
	case actMsg:
		m.busy = ""
		if msg.err != nil {
			m.setNote(msg.what+": "+msg.err.Error(), true)
		} else if msg.note != "" {
			m.setNote(msg.what+": "+msg.note, false)
		} else {
			m.setNote(msg.what+": done", false)
		}
		return m, tea.Batch(m.loadPS(), m.loadModels(), m.loadHarness())
	case tea.KeyPressMsg:
		return m.key(msg)
	}
	return m, nil
}

func (m ctl) key(k tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	s := k.String()
	if m.confirm != nil {
		c := m.confirm
		m.confirm = nil
		if s == "y" || s == "Y" {
			m.busy = strings.TrimSuffix(c.q, "? y / n")
			return m, c.run
		}
		m.setNote("cancelled", false)
		return m, nil
	}
	if m.editing {
		return m.editKey(k)
	}
	switch s {
	case "ctrl+c", "q":
		return m, tea.Quit
	case "a":
		m.advanced = !m.advanced
		if !m.advanced && m.focus == paneOptions {
			m.focus = paneModels
		}
		m.savePrefs()
		return m, nil
	case "tab":
		m.cycle(1)
		return m, nil
	case "shift+tab":
		m.cycle(-1)
		return m, nil
	case "R":
		m.setNote("refreshing", false)
		return m, tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadOpts())
	case "up", "k":
		m.move(-1)
		return m, nil
	case "down", "j":
		m.move(1)
		return m, nil
	}
	if m.busy != "" && s != "l" && s != "m" {
		m.setNote("wait: "+m.busy+" is still running", false)
		return m, nil
	}
	switch m.focus {
	case paneModels:
		return m.modelsKey(s)
	case paneServers:
		return m.serversKey(s)
	case paneHarness:
		return m.harnessKey(s)
	case paneOptions:
		return m.optionsKey(s)
	}
	return m, nil
}

func (m *ctl) move(d int) {
	switch m.focus {
	case paneModels:
		m.curM = min(max(0, m.curM+d), max(0, len(m.models)-1))
	case paneServers:
		m.curS = min(max(0, m.curS+d), max(0, len(m.ps.Servers)-1))
	case paneOptions:
		m.curO = min(max(0, m.curO+d), max(0, len(m.visibleOpts())-1))
	}
}

func (m ctl) modelsKey(s string) (tea.Model, tea.Cmd) {
	mi := m.selModel()
	if mi == nil {
		return m, nil
	}
	switch s {
	case "enter", "s":
		if e := m.hwError(); e != "" {
			m.setNote(e, true)
			return m, nil
		}
		if !mi.Present {
			m.setNote(mi.Name+" is not downloaded: p to pull it", true)
			return m, nil
		}
		m.busy = "starting " + mi.Name
		return m, m.action("start "+mi.Name, 15*time.Minute, m.startArgs(*mi)...)
	case "p":
		if mi.Present {
			m.setNote(mi.Name+" is already on disk", false)
			return m, nil
		}
		m.confirm = &pending{q: fmt.Sprintf("download %s (%.1f GB)? y / n", mi.Name, mi.SizeGB),
			run: m.action("pull "+mi.Name, 12*time.Hour, "pull", mi.Name)}
	case "c":
		if m.advanced && mi.Cards == 1 {
			m.card = m.nextCard()
			m.savePrefs()
		}
	case "b":
		if m.advanced {
			m.busy = "starting the brain / hands team"
			return m, m.action("brain / hands", 15*time.Minute, "start", "--team", "brain-hands")
		}
	}
	return m, nil
}

func (m ctl) serversKey(s string) (tea.Model, tea.Cmd) {
	sv := m.selServer()
	if sv == nil {
		return m, nil
	}
	name := sv.Name
	switch s {
	case "x":
		m.confirm = &pending{q: "stop " + name + "? y / n", run: m.action("stop "+name, 5*time.Minute, "stop", name)}
	case "t":
		m.busy = "restarting " + name
		return m, m.action("restart "+name, 15*time.Minute, "restart", name)
	case "l":
		return m, m.interactive("log of "+name, "logs", name)
	case "m":
		return m, m.interactive("monitor of "+name, "top", name)
	case "enter", "L":
		return m, m.interactive("Qwen Code on "+name, "qwen", "--server", name)
	}
	return m, nil
}

func (m ctl) harnessKey(s string) (tea.Model, tea.Cmd) {
	target := ""
	if sv := m.selServer(); sv != nil {
		target = sv.Name
	}
	switch s {
	case "enter", "L":
		if !m.hs.Installed {
			m.setNote("Qwen Code is not installed: i to install it", true)
			return m, nil
		}
		a := []string{"qwen"}
		if target != "" {
			a = append(a, "--server", target)
		}
		return m, m.interactive("Qwen Code", a...)
	case "u":
		a := []string{"qwen-code", "setup"}
		if target != "" {
			a = append(a, "--server", target)
		}
		m.busy = "setting up Qwen Code"
		return m, m.action("Qwen Code setup", 2*time.Minute, a...)
	case "i":
		v := m.hs.Pinned
		if v == "" {
			v = "(the pinned version)"
		}
		m.confirm = &pending{q: fmt.Sprintf("install Qwen Code %s into Caminus's own npm prefix (your own installs stay as they are)? y / n", v),
			run: m.action("Qwen Code install", 20*time.Minute, "qwen-code", "setup", "--install")}
	}
	return m, nil
}

func (m ctl) optionsKey(s string) (tea.Model, tea.Cmd) {
	vo := m.visibleOpts()
	if m.curO >= len(vo) {
		return m, nil
	}
	o := vo[m.curO]
	switch s {
	case "enter", "e":
		if o.Type == "bool" {
			if m.vals[o.Dest] == "true" {
				delete(m.vals, o.Dest)
			} else {
				m.vals[o.Dest] = "true"
			}
			m.savePrefs()
			return m, nil
		}
		m.editing, m.edit = true, m.valueOf(o)
	case "d", "backspace", "delete":
		delete(m.vals, o.Dest)
		m.savePrefs()
	}
	return m, nil
}

func (m ctl) editKey(k tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	vo := m.visibleOpts()
	if len(vo) == 0 {
		m.editing = false
		return m, nil
	}
	o := vo[min(m.curO, len(vo)-1)]
	switch k.String() {
	case "esc":
		m.editing = false
	case "enter":
		m.editing = false
		v := strings.TrimSpace(m.edit)
		if err := checkValue(o, v); err != nil {
			m.setNote(o.Flag+": "+err.Error(), true)
			return m, nil
		}
		if v == "" || v == defaultText(o) {
			delete(m.vals, o.Dest)
		} else {
			m.vals[o.Dest] = v
		}
		m.savePrefs()
	case "backspace":
		if r := []rune(m.edit); len(r) > 0 {
			m.edit = string(r[:len(r)-1])
		}
	case "ctrl+u":
		m.edit = ""
	default:
		if t := k.Text; t != "" {
			m.edit += t
		}
	}
	return m, nil
}

// checkValue checks the option's type before anything is saved (the server checks again when it starts).
func checkValue(o option, v string) error {
	if v == "" {
		return nil
	}
	switch o.Type {
	case "int":
		if _, err := strconv.Atoi(v); err != nil {
			return fmt.Errorf("a whole number")
		}
	case "float":
		if _, err := strconv.ParseFloat(v, 64); err != nil {
			return fmt.Errorf("a number")
		}
	case "ctx":
		if v != "auto" {
			if n, err := strconv.Atoi(v); err != nil || n < 1 {
				return fmt.Errorf("auto or a positive number")
			}
		}
	}
	if len(o.Choices) > 0 {
		for _, c := range o.Choices {
			if c == v {
				return nil
			}
		}
		return fmt.Errorf("one of %s", strings.Join(o.Choices, ", "))
	}
	return nil
}

func defaultText(o option) string {
	switch v := o.Default.(type) {
	case nil:
		return ""
	case float64:
		if v == float64(int64(v)) {
			return strconv.FormatInt(int64(v), 10)
		}
		return strconv.FormatFloat(v, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprint(v)
	}
}

func (m ctl) valueOf(o option) string {
	if v, ok := m.vals[o.Dest]; ok {
		return v
	}
	return defaultText(o)
}

// ---------- remembered choices ----------

type prefs struct {
	Advanced bool              `json:"advanced"`
	Card     string            `json:"card"`
	Values   map[string]string `json:"values"`
}

func prefsPath() string {
	d := os.Getenv("XDG_CONFIG_HOME")
	if d == "" {
		h, _ := os.UserHomeDir()
		d = filepath.Join(h, ".config")
	}
	return filepath.Join(d, "caminus", "panel.json")
}

func (m *ctl) loadPrefs() {
	if m.prefs == "" {
		return
	}
	b, err := os.ReadFile(m.prefs)
	if err != nil {
		return
	}
	var p prefs
	if json.Unmarshal(b, &p) == nil {
		m.advanced = m.advanced || p.Advanced
		if p.Card != "" {
			m.card = p.Card
		}
		for k, v := range p.Values {
			m.vals[k] = v
		}
	}
}

func (m ctl) savePrefs() {
	if m.prefs == "" {
		return
	}
	b, _ := json.MarshalIndent(prefs{m.advanced, m.card, m.vals}, "", "  ")
	_ = os.MkdirAll(filepath.Dir(m.prefs), 0o700)
	_ = os.WriteFile(m.prefs, append(b, '\n'), 0o600)
}

// ---------- rendering ----------

func (m ctl) View() tea.View {
	v := tea.NewView(m.render())
	v.AltScreen = true
	v.WindowTitle = "caminus panel"
	return v
}

func stateColor(s string) color.Color {
	switch s {
	case "ready":
		return cGreen
	case "loading", "starting":
		return cYellow
	case "failed":
		return cRed
	}
	return cMuted
}

func (m ctl) header(W int) string {
	title := bold(cAccent).Render("◆ caminus") + fg(cMuted).Render(" control panel "+version)
	mode := pill(cTeal, "BASIC")
	if m.advanced {
		mode = pill(cPurple, "ADVANCED")
	}
	state, c := "NO SERVER", cGray
	for _, s := range m.ps.Servers {
		switch {
		case s.State == "ready":
			state, c = "READY", cGreen
		case s.State == "loading" && state != "READY":
			state, c = "LOADING", cYellow
		case s.State == "failed" && state == "NO SERVER":
			state, c = "FAILED", cRed
		}
	}
	right := ""
	if m.busy != "" {
		right = fg(cYellow).Render(spin[m.frame%len(spin)] + " " + m.busy)
	}
	l := title + "  " + mode + " " + pill(c, state)
	return l + strings.Repeat(" ", max(1, W-lipgloss.Width(l)-lipgloss.Width(right))) + right
}

func (m ctl) cursor(pane, i, cur int) string {
	if m.focus == pane && i == cur {
		return bold(cAccent).Render("› ")
	}
	return "  "
}

func gbText(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) + " GB" }

func (m ctl) modelRows() []string {
	if !m.loaded["models"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " reading the model list")}
	}
	if len(m.models) == 0 {
		return []string{fg(cMuted).Render("no models known (caminus models)")}
	}
	var rows []string
	for i, mi := range m.models {
		disk := fg(cGreen).Render("✓ on disk")
		if !mi.Present {
			disk = fg(cYellow).Render("not downloaded")
		}
		cards := "1 card"
		if mi.Cards > 1 {
			cards = strconv.Itoa(mi.Cards) + " cards"
		}
		name := bold(cText).Render(mi.Name)
		if m.focus == paneModels && i == m.curM {
			name = bold(cAccent).Render(mi.Name)
		}
		rows = append(rows, m.cursor(paneModels, i, m.curM)+fit(name, 18)+fit(fg(cText).Render(mi.Title), 40)+" "+
			fit(fg(cMuted).Render(cards+" · "+gbText(mi.SizeGB)), 18)+" "+disk)
		if mi.Speed != "" && (m.advanced || (m.focus == paneModels && i == m.curM)) {
			rows = append(rows, "    "+fg(cMuted).Render(mi.Speed))
		}
	}
	if m.advanced {
		if mi := m.selModel(); mi != nil && mi.Cards == 1 {
			rows = append(rows, "", fg(cMuted).Render("card for one-card models: ")+bold(cAccent).Render(m.card)+
				fg(cMuted).Render("  (c to change; auto = a free card)"))
		}
	}
	return rows
}

func (m ctl) serverRows() []string {
	if !m.loaded["ps"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " looking for servers")}
	}
	if len(m.ps.Servers) == 0 {
		return []string{fg(cMuted).Render("no server running: pick a model above and press enter")}
	}
	// the name column fits the longest name (servers take their model's name: up to 16 characters), 9 at least and
	// 18 at most, and a space always follows it
	nw := 9
	for _, s := range m.ps.Servers {
		nw = max(nw, min(lipgloss.Width(s.Name), 18))
	}
	var rows []string
	for i, s := range m.ps.Servers {
		st := fg(stateColor(s.State)).Render(s.State)
		extra := ""
		if s.MaxCtx > 0 {
			extra += " · " + humanInt(s.MaxCtx) + " ctx"
		}
		if s.DecodeTPS != nil {
			extra += fmt.Sprintf(" · %.1f tok/s decode", *s.DecodeTPS)
		}
		rows = append(rows, m.cursor(paneServers, i, m.curS)+fg(stateColor(s.State)).Render("●")+" "+fit(bold(cText).Render(s.Name), nw)+" "+
			fit(fg(cText).Render(s.Model), 20)+fit(st, 9)+fg(cAccent).Render(s.URL)+fg(cMuted).Render(extra))
		switch {
		case s.State == "failed" && s.Error != "":
			rows = append(rows, "    "+fg(cRed).Render(s.Error))
		case m.advanced:
			d := fmt.Sprintf("    cards %s · pid %d · up %s", strings.Join(s.Cards, "+"), s.Pid, upFor(s.UptimeS))
			if s.Alias != "" {
				d += " · model id " + s.Alias
			}
			rows = append(rows, fg(cMuted).Render(d))
		}
	}
	return rows
}

// nextCard is the advanced layout's card choice after the current one: auto, then each usable card's position, then auto.
func (m ctl) nextCard() string {
	order := []string{"auto"}
	for _, c := range m.ps.Cards {
		order = append(order, strconv.Itoa(c.Index))
	}
	for i, k := range order {
		if k == m.card {
			return order[(i+1)%len(order)]
		}
	}
	return "auto"
}

// hwError is the host's blocking hardware problem (no supported GPU), or "": the panel refuses to start anything then.
func (m ctl) hwError() string {
	for _, p := range m.ps.Problems {
		if p.Level == "error" {
			return p.Text
		}
	}
	return ""
}

// hwWarn: a hardware line the basic layout shows too (anything but notes).
func (m ctl) hwWarn() bool {
	for _, p := range m.ps.Problems {
		if p.Level == "warn" {
			return true
		}
	}
	return false
}

// hwRows are the hardware lines (unsupported cards, slow slots, untested cards; the error when nothing is supported).
func (m ctl) hwRows() []string {
	var rows []string
	for _, p := range m.ps.Problems {
		switch p.Level {
		case "error":
			rows = append(rows, bold(cRed).Render("✗ "+p.Text))
		case "warn":
			rows = append(rows, fg(cYellow).Render("! "+p.Text))
		default:
			rows = append(rows, fg(cMuted).Render("· "+p.Text))
		}
	}
	return rows
}

func (m ctl) cardRows() []string {
	var rows []string
	for _, c := range m.ps.Cards {
		extra := ""
		if c.Link != "" {
			extra = " · " + c.Link
		}
		if c.Display {
			extra += " · display"
		}
		rows = append(rows, fit(bold(cText).Render(fmt.Sprintf("%d %s", c.Index, c.Name)), 8)+bar(c.UsedMiB, c.LineMiB, 28)+
			fg(cMuted).Render(fmt.Sprintf("  %s / %s MiB (the guard's line)%s", humanInt(int(c.UsedMiB)), humanInt(int(c.LineMiB)), extra)))
	}
	if len(rows) == 0 {
		rows = append(rows, fg(cMuted).Render("no card readings"))
	}
	return rows
}

func (m ctl) harnessRows() []string {
	if !m.loaded["harness"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " checking Qwen Code")}
	}
	h := m.hs
	var rows []string
	if h.Installed {
		rows = append(rows, fg(cGreen).Render("✓ installed ")+fg(cText).Render(h.Version)+fg(cMuted).Render("  "+h.Binary))
	} else {
		n := "not installed"
		if !h.Npm {
			n += " (needs Node.js / npm)"
		}
		rows = append(rows, fg(cYellow).Render(n)+fg(cMuted).Render("  i installs "+h.Pinned+" into Caminus's own prefix"))
	}
	if h.Configured {
		t := ""
		if h.Target != "" {
			t = " for " + h.Target
		}
		rows = append(rows, fg(cGreen).Render("✓ set up")+fg(cMuted).Render(t+" · its own home "+h.Home+" (your ~/.qwen is untouched)"))
	} else {
		rows = append(rows, fg(cMuted).Render("not set up yet: u writes Caminus's tested config into its own home"))
	}
	return rows
}

func (m ctl) optionRows(height int) []string {
	if !m.loaded["opts"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " reading caminus serve's options")}
	}
	vo := m.visibleOpts()
	if len(vo) == 0 {
		return []string{fg(cMuted).Render("caminus serve --options-json gave no options")}
	}
	rows := []string{fg(cMuted).Render("every caminus serve option; edited ones (") + bold(cAccent).Render("this colour") +
		fg(cMuted).Render(") go to the next start")}
	n := max(3, height-4)
	lo := max(0, min(m.curO-n/2, len(vo)-n))
	for i := lo; i < min(len(vo), lo+n); i++ {
		o := vo[i]
		v, edited := m.vals[o.Dest]
		val := fg(cMuted).Render(defaultText(o))
		if edited {
			val = bold(cAccent).Render(v)
		}
		if o.Type == "bool" {
			val = fg(cMuted).Render("off")
			if edited && v == "true" {
				val = bold(cAccent).Render("on")
			}
		}
		if m.editing && i == m.curO {
			val = lipgloss.NewStyle().Foreground(cText).Background(cDim).Render(m.edit + "▏")
		}
		rows = append(rows, m.cursor(paneOptions, i, m.curO)+fit(fg(cText).Render(o.Flag), 26)+val)
	}
	if m.curO < len(vo) {
		rows = append(rows, "", fg(cMuted).Render(oneLine(vo[m.curO].Help)))
	}
	return rows
}

func (m ctl) keysLine() string {
	k := func(key, what string) string { return bold(cAccent).Render(key) + fg(cMuted).Render(" "+what) }
	var p []string
	switch m.focus {
	case paneModels:
		p = []string{k("enter", "start"), k("p", "pull")}
		if m.advanced {
			p = append(p, k("c", "card"), k("b", "brain / hands team"))
		}
	case paneServers:
		p = []string{k("enter", "Qwen Code on it"), k("x", "stop"), k("t", "restart"), k("l", "log"), k("m", "monitor")}
	case paneHarness:
		p = []string{k("enter", "launch"), k("u", "set up"), k("i", "install")}
	case paneOptions:
		p = []string{k("enter", "edit / toggle"), k("d", "default")}
		if m.editing {
			p = []string{k("enter", "save"), k("esc", "cancel"), k("ctrl+u", "clear")}
		}
	}
	other := "advanced"
	if m.advanced {
		other = "basic"
	}
	p = append(p, k("tab", "next panel"), k("a", other), k("R", "refresh"), k("q", "quit"))
	return strings.Join(p, fg(cDim).Render(" · "))
}

func (m ctl) render() string {
	W := m.w
	if W <= 0 {
		W = 120
	}
	W = min(max(W, 60), 220)
	H := m.h
	if H <= 0 {
		H = 40
	}
	title := func(p int) string {
		if m.focus == p {
			return "▸ " + paneNames[p]
		}
		return paneNames[p]
	}
	acc := func(p int, c color.Color) color.Color {
		if m.focus == p {
			return cAccent
		}
		return c
	}
	out := []string{m.header(W), ""}
	if hw := m.hwRows(); len(hw) > 0 && (m.advanced || m.hwError() != "" || m.hwWarn()) {
		c := cYellow
		if m.hwError() != "" {
			c = cRed
		}
		out = append(out, panel("HARDWARE", c, W, hw), "")
	}
	two := m.advanced && W >= 150
	left := W
	if two {
		left = W * 3 / 5
	}
	col := []string{
		panel(title(paneModels), acc(paneModels, cBlue), left, m.modelRows()),
		panel(title(paneServers), acc(paneServers, cGreen), left, m.serverRows()),
	}
	if m.advanced {
		col = append(col, panel("CARDS", cOrange, left, m.cardRows()))
	}
	col = append(col, panel(title(paneHarness), acc(paneHarness, cPurple), left, m.harnessRows()))
	body := strings.Join(col, "\n")
	if m.advanced {
		if two {
			right := panel(title(paneOptions), acc(paneOptions, cTeal), W-left-1, m.optionRows(max(8, H-6)))
			body = lipgloss.JoinHorizontal(lipgloss.Top, body, " ", right)
		} else {
			body += "\n" + panel(title(paneOptions), acc(paneOptions, cTeal), W, m.optionRows(14))
		}
	}
	out = append(out, body, "")
	switch {
	case m.confirm != nil:
		out = append(out, bold(cYellow).Render("? "+m.confirm.q))
	case m.note != "":
		c := cGreen
		if m.noteErr {
			c = cRed
		}
		out = append(out, fg(c).Render(m.note))
	default:
		out = append(out, "")
	}
	return strings.Join(append(out, m.keysLine()), "\n")
}

func humanInt(n int) string {
	s := strconv.Itoa(n)
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	return b.String()
}

func upFor(s float64) string {
	d := time.Duration(s * float64(time.Second))
	switch {
	case d >= time.Hour:
		return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
	case d >= time.Minute:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%ds", int(d.Seconds()))
}

func newCtl(be *ctlCLI, advanced bool, prefsFile string) ctl {
	m := ctl{be: be, advanced: advanced, loaded: map[string]bool{}, vals: map[string]string{}, card: "auto", prefs: prefsFile}
	m.loadPrefs()
	return m
}

// ctlMain runs the control panel: `caminus-panel [-advanced] [-caminus CMD]`.
func ctlMain(args []string) {
	adv, cli, once, w, h := false, "", false, 140, 45
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-advanced", "--advanced":
			adv = true
		case "-once", "--once": // one frame to stdout after every read answered (tests, screenshots), then exit
			once = true
			if i+2 < len(args) {
				if a, err := strconv.Atoi(args[i+1]); err == nil {
					if b, err := strconv.Atoi(args[i+2]); err == nil {
						w, h, i = a, b, i+2
					}
				}
			}
		case "-caminus", "--caminus":
			if i+1 < len(args) {
				i++
				cli = args[i]
			}
		default:
			fmt.Fprintf(os.Stderr, "caminus panel: unknown argument %q (-advanced, -caminus CMD; `top` for the monitor)\n", args[i])
			os.Exit(2)
		}
	}
	if once {
		m := newCtl(newCtlCLI(cli), adv, "") // (no prefs file: a one-off frame changes nothing)
		m.w, m.h = w, h
		for _, c := range []tea.Cmd{m.loadModels(), m.loadPS(), m.loadHarness(), m.loadOpts()} {
			nm, _ := m.Update(c())
			m = nm.(ctl)
		}
		fmt.Println(m.render())
		return
	}
	if _, err := tea.NewProgram(newCtl(newCtlCLI(cli), adv, prefsPath())).Run(); err != nil {
		fmt.Fprintln(os.Stderr, "caminus panel:", err)
		os.Exit(1)
	}
}
