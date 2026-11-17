// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// control.go: the Caminus control panel, one status-first screen (panelview.go draws it). It never runs the engine or
// edits a config itself: every read and every action is one `caminus <command> --json` run (backend.go), so the panel
// and the command line can never disagree. Its live numbers come from the running server's own monitoring surface
// through the monitor's readers (server.go, caminus.go; panellive.go).
//
//	caminus panel                 the screen: what runs, the models with their choices, the agents
//	caminus panel -caminus CMD    the caminus command line to drive (default: CAMINUS_CLI, else caminus on PATH)
//	caminus panel -once [W H]     one frame to stdout once every read answered (tests), then exit
//
// The panel is a component (ctl) a host runs: `caminus panel` hosts it full screen (panelHost) and Ember can host the
// same component in its own frame. The component never quits the program, owns no alt screen and reads no window
// size: View(w, h) gets the host's size, and Esc / q send panelCloseMsg for the host to act on.
package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

func bold(c color.Color) lipgloss.Style { return lipgloss.NewStyle().Foreground(c).Bold(true) }

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
	return fg(c).Render(strings.Repeat(pg().bar, n)) + fg(cGray).Render(strings.Repeat(pg().barEmpty, w-n))
}

// What the screen shows besides the main view: the options editor (o) or the key list (?).
const (
	viewMain = iota
	viewOptions
	viewHelp
)

// panelOptions configures the component.
type panelOptions struct {
	CLI    string // the caminus command line ("" = CAMINUS_CLI, else caminus on PATH)
	Prefs  string // where the choices are remembered ("" = nowhere: tests, one-off frames)
	Hosted bool   // inside another program (Ember): e goes back to it (close) instead of opening a second Ember
}

// panelSeq numbers the components: a tick of a closed one never drives a newer one.
var panelSeq int

// ctl is the panel component.
type ctl struct {
	be     *ctlCLI
	id     int // this component
	epoch  int // bumped by each Init's open message: ticks of an earlier Init are dropped
	hosted bool
	w, h   int // the last View's size (key lines and widths read it)
	frame  int

	models  []camModel
	ps      psInfo
	hs      harnessInfo
	ember   emberMsg // whether Ember is set up
	opts    []option
	loaded  map[string]bool      // which reads answered at least once
	readErr map[string]error     // the last failed read of models / ps / harness / ember / opts (nil once one answers): said as "couldn't read" in its own area
	retry   map[string]bool      // the failed reads the poll is reading again (one at a time each: a slow failure never piles up)
	failAt  map[string]time.Time // when each failed read's answer came: the poll reads it again readRetry later
	vals    map[string]string    // serve option values for the next start (dest -> text); absent = the default
	card    string               // one-card models: "auto", "0", "1" (the options editor's first row)
	picks   map[string]modelPick // model name -> card count and drafter choice at each count
	fits    map[string]fitEntry  // caminus fit --json per model, card count, drafter and options (display only)
	fitGen  int                  // bumped by R and by every action: a reply from an older generation is dropped
	curM    int
	curO    int // the options editor's row: 0 = the card pin, then visibleOpts
	choice  int // the selected model's choice row under the cursor (tab / shift+tab), its value changed with ← / →
	view    int
	editing bool
	edit    string
	// editFresh: the field still holds the value it opened with, as if selected: the first typed character replaces it
	editFresh bool
	helpPage  int        // the key list's page (?)
	seen      *panelSeen // what the last View drew (View has a value receiver; the keys read it from here)
	confirm   *pending   // a y / n question
	busy      string     // the action in flight ("" = none)
	note      string     // the last result line (a key's or an action's: a read's failure is said in that read's own area)
	noteErr   bool
	noteRead  string // the read a key's "couldn't read" note is about ("" = none): cleared when that read next answers
	helpDrawn int    // the key list's page count in the View being drawn (the key line under it reads it)
	prefs     string

	live *liveMon // the running server's live numbers and its clients (panellive.go)
}

// panelSeen is what the last View drew that a key needs: how many pages the key list took at that size.
type panelSeen struct{ helpPages int }

// pending is an action waiting for y.
type pending struct {
	q   string
	run tea.Cmd
}

// panelCloseMsg is the component asking its host to close it (Esc or q, or e inside Ember). The panel's own host
// quits; Ember goes back to its chat.
type panelCloseMsg struct{}

func closePanel() tea.Msg { return panelCloseMsg{} }

type (
	// panelOpenMsg starts the ticks of one Init (id: the component); a second Init starts a new epoch.
	panelOpenMsg struct{ id int }
	modelsMsg    struct {
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
	// emberMsg: whether Ember, the agent environment, is set up (caminus ember status --json); err is a read that failed
	// (no JSON came back), which is not "not set up": status answers ok false for that
	emberMsg struct {
		ok, known bool
		err       error
	}
	optsMsg struct {
		v   []option
		err error
	}
	actMsg struct {
		what, note string
		err        error
	}
	fitMsg struct {
		key string
		gen int
		v   fitInfo
		err error
	}
	// ctlTick redraws the spinner (150 ms); ctlPoll reads caminus ps again (2 s). Both carry the component and epoch.
	ctlTick struct{ id, epoch int }
	ctlPoll struct{ id, epoch int }
)

// fitInfo is `caminus fit --json` (serve's own projection; the panel shows it and does no memory math). max_ctx is per KV
// format, each the largest context that fits with that cache: q8 is null when the model has no 8-bit cache (kv_formats
// lacks "q8") or when no cards hold it. kv_fixed names the cache of a model that always keeps its own (the Coder's
// "q8_0"; empty for the others): its f16 slot is that cache's figure, and the row shows it as the only one.
type fitInfo struct {
	OK        bool     `json:"ok"`
	Error     string   `json:"error"`
	MaxCtx    fitCtx   `json:"max_ctx"`
	KVFormats []string `json:"kv_formats"`
	KVFixed   string   `json:"kv_fixed"`
}

// fitCtx is fit's max_ctx: positions per KV format, nil where none fits (or, for q8, where it is not offered).
type fitCtx struct {
	F16 *int `json:"f16"`
	Q8  *int `json:"q8"`
}

// hasQ8 is whether fit projected the 8-bit cache for this model (kv_formats lists "q8").
func (v fitInfo) hasQ8() bool {
	for _, f := range v.KVFormats {
		if f == "q8" {
			return true
		}
	}
	return false
}

// q8Text is the fit row's Q8 part: the figure, "none fits" for a model that has the 8-bit cache but no room for it, or
// "not offered" for a model without one.
func (v fitInfo) q8Text() string {
	if v.MaxCtx.Q8 != nil {
		return strconv.Itoa(*v.MaxCtx.Q8)
	}
	if v.hasQ8() {
		return "none fits"
	}
	return "not offered"
}

// fixedText is the fit row's label for a model that keeps its own cache ("its own 8-bit cache" for q8_0), or "" for
// a model with the F16 / Q8 choice.
func (v fitInfo) fixedText() string {
	switch {
	case v.KVFixed == "":
		return ""
	case strings.HasPrefix(v.KVFixed, "q8"):
		return "its own 8-bit cache"
	}
	return "its own " + v.KVFixed + " cache"
}

// fitEntry is one fit read: done once its answer (or error) has arrived.
type fitEntry struct {
	done     bool
	info     fitInfo
	err      error
	failedAt time.Time // a failed read (error or ok:false) is retried after fitRetry, not kept
}

// fitRetry is how long a failed fit read waits before the picker asks again (the picker redraws every 150 ms).
const fitRetry = 10 * time.Second

// fitArgs is the fit query for the model's next start: its file, card count, drafter, cache and the edited serve options
// that `caminus fit` takes (fit uses serve's parser, so these are the flags start would pass). The key is the whole query.
func (m ctl) fitArgs(mi camModel) (string, []string) {
	mode, draft := m.modelMode(mi)
	a := []string{"fit", "--model", mi.Path, "--cards", strconv.Itoa(mode.Cards)}
	if draft != "auto" {
		a = append(a, "--draft", draft)
	}
	a = append(a, m.extraArgs(mi)...)
	a = append(a, m.optionFlags("model", "config", "port")...)
	return strings.Join(a, " "), append(a, "--json")
}

// fitKey is the cache key of the selected model's fit query (fitArgs' key).
func (m ctl) fitKey(mi camModel) string {
	k, _ := m.fitArgs(mi)
	return k
}

// fitCmd reads the selected model's fit once per query; nil while that read is in flight, when a failed read is still
// inside its retry wait, or when the model is not on disk (nothing to project).
func (m ctl) fitCmd() tea.Cmd {
	mi := m.selModel()
	if mi == nil || !mi.Present || m.fits == nil || m.liveOf(mi.Name) != nil { // (a running model: no projection shown)
		return nil
	}
	key, args := m.fitArgs(*mi)
	if e, seen := m.fits[key]; seen {
		if !e.done || e.failedAt.IsZero() || time.Since(e.failedAt) < fitRetry {
			return nil
		}
	}
	m.fits[key] = fitEntry{}
	gen, be := m.fitGen, m.be
	return func() tea.Msg {
		var v fitInfo
		err := be.call(&v, 60*time.Second, args...)
		return fitMsg{key: key, gen: gen, v: v, err: err}
	}
}

// refreshFits forgets every fit read: fit depends on the cards' memory now, so R and every start, stop or restart
// (actMsg) make the picker ask again. A reply still in flight from before is dropped by its generation.
func (m *ctl) refreshFits() {
	m.fitGen++
	m.fits = map[string]fitEntry{}
}

// settle runs c and every command its messages return, feeding each message back into the model before the next:
// the -once frame renders only after its reads have been answered, never with one still in flight.
func settle(m ctl, c tea.Cmd, depth int) ctl {
	if c == nil || depth > 32 {
		return m
	}
	msg := c()
	if b, ok := msg.(tea.BatchMsg); ok {
		for _, sub := range b {
			m = settle(m, sub, depth+1)
		}
		return m
	}
	nm, next := m.Update(msg)
	return settle(nm, next, depth+1)
}

// fitRow is the picker's line for the context that fits at F16 and with the 8-bit cache, or with the model's own cache
// for one without the choice (empty when there is no read for the selection).
func (m ctl) fitRow(mi camModel) string {
	key, _ := m.fitArgs(mi)
	e, seen := m.fits[key]
	if !seen {
		return ""
	}
	fixed := e.info.fixedText()
	switch {
	case !e.done:
		return fg(cMuted).Render("context that fits: reading caminus fit")
	case e.err != nil:
		return fg(cYellow).Render("context that fits: " + e.err.Error())
	case !e.info.OK && fixed != "":
		return fg(cYellow).Render("context that fits (" + fixed + "): " + e.info.Error)
	case !e.info.OK:
		line := fg(cYellow).Render("context that fits: " + e.info.Error)
		if q8 := e.info.MaxCtx.Q8; q8 != nil {
			// (no cards for the selected cache, but the 8-bit one fits: fit keeps that figure in a refusal)
			line += fg(cMuted).Render(" · Q8: ") + bold(cAccent).Render(strconv.Itoa(*q8)) + fg(cMuted).Render(" positions")
		}
		return line
	case fixed != "" && e.info.MaxCtx.F16 == nil:
		return fg(cYellow).Render("context that fits (" + fixed + "): none with this card count")
	case e.info.MaxCtx.F16 == nil && e.info.MaxCtx.Q8 == nil && e.info.hasQ8():
		// (both caches were projected: say Q8 fails too, so switching to --kv q8 is not the way out)
		return fg(cYellow).Render("context that fits: no context fits at F16 or Q8 with this card count")
	case e.info.MaxCtx.F16 == nil && e.info.MaxCtx.Q8 == nil:
		return fg(cYellow).Render("context that fits: no context fits at F16 with this card count")
	}
	var line string
	if fixed != "" {
		// (no F16 / Q8 choice: f16 is the figure of the cache the model always builds, and there is no Q8 part)
		line = fg(cMuted).Render("context that fits ("+fixed+"): ") + bold(cAccent).Render(strconv.Itoa(*e.info.MaxCtx.F16)) +
			fg(cMuted).Render(" positions")
	} else if e.info.MaxCtx.F16 == nil {
		// (only the 8-bit cache fits: `caminus start --kv q8` is the way to serve this choice)
		line = fg(cYellow).Render("context that fits: none at F16") + fg(cMuted).Render(" · Q8: ") +
			bold(cAccent).Render(strconv.Itoa(*e.info.MaxCtx.Q8)) + fg(cMuted).Render(" positions")
	} else {
		q8 := e.info.q8Text()
		line = fg(cMuted).Render("context that fits at F16: ") + bold(cAccent).Render(strconv.Itoa(*e.info.MaxCtx.F16)) +
			fg(cMuted).Render(" positions · Q8: ")
		if e.info.MaxCtx.Q8 != nil {
			line += bold(cAccent).Render(q8)
		} else {
			line += fg(cMuted).Render(q8)
		}
	}
	if mode, _ := m.modelMode(mi); mode.Cards == 1 && m.card != "" && m.card != "auto" {
		// caminus fit has no --card: it places the model itself, so a pinned card does not change this figure
		line += fg(cYellow).Render(" · card " + m.card + " pinned: not in this figure")
	}
	return line
}

func (m ctl) tickCmd() tea.Cmd {
	id, ep := m.id, m.epoch
	return tea.Tick(150*time.Millisecond, func(time.Time) tea.Msg { return ctlTick{id, ep} })
}

func (m ctl) pollCmd() tea.Cmd {
	id, ep := m.id, m.epoch
	return tea.Tick(2*time.Second, func(time.Time) tea.Msg { return ctlPoll{id, ep} })
}

// mine is whether a tick belongs to this component's current Init.
func (m ctl) mine(id, epoch int) bool { return id == m.id && epoch == m.epoch }

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

// loadEmber reads whether Ember is set up: ok false is not set up, a failed read is said as one (the row says which).
func (m ctl) loadEmber() tea.Cmd {
	return func() tea.Msg {
		var v struct {
			OK bool `json:"ok"`
		}
		err := m.be.call(&v, 20*time.Second, "ember", "status", "--json")
		return emberMsg{ok: err == nil && v.OK, known: true, err: err}
	}
}

// readRetry is how long after a failed read of the models, Qwen Code, Ember or the serve options the poll reads it
// again (ps is read on every poll anyway): its area clears on the next read that answers, without R.
const readRetry = 4 * time.Second

// retryReads is the failed reads (not ps) whose wait is over and that are not already being read again.
func (m *ctl) retryReads(now time.Time) tea.Cmd {
	load := map[string]func() tea.Cmd{"models": m.loadModels, "harness": m.loadHarness, "ember": m.loadEmber, "opts": m.loadOpts}
	var cmds []tea.Cmd
	for _, k := range []string{"models", "harness", "ember", "opts"} {
		if m.readErr[k] == nil || m.retry[k] || now.Sub(m.failAt[k]) < readRetry {
			continue
		}
		m.retry[k] = true
		cmds = append(cmds, load[k]())
	}
	return tea.Batch(cmds...)
}

// readDone records a read's answer: its error (said in its area) or, when it answered, no error and no key note about it.
func (m *ctl) readDone(k string, err error) {
	m.loaded[k] = true
	m.readErr[k] = err
	delete(m.retry, k)
	if err != nil {
		m.failAt[k] = time.Now()
		return
	}
	if m.noteRead == k {
		m.setNote("", false)
	}
}

// psDown is whether the last caminus ps read failed: what runs is not known, so nothing is offered a start.
func (m ctl) psDown() bool { return m.readErr["ps"] != nil }

// psDownNote is the note of a key that needs to know what runs while caminus ps can't be read.
func (m *ctl) psDownNote(what string) {
	m.setNote(what+"couldn't read the running servers · R reads again: "+m.readErr["ps"].Error(), true)
	m.noteRead = "ps"
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

// Init reads everything once and starts the ticks (through panelOpenMsg, so each Init starts one set).
func (m ctl) Init() tea.Cmd {
	id := m.id
	return tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadEmber(), m.loadOpts(),
		func() tea.Msg { return panelOpenMsg{id} })
}

func (m *ctl) setNote(s string, isErr bool) { m.note, m.noteErr, m.noteRead = s, isErr, "" }

func (m ctl) selModel() *camModel {
	if m.curM >= 0 && m.curM < len(m.models) {
		return &m.models[m.curM]
	}
	return nil
}

type modelPick struct {
	Cards  int            `json:"cards"`
	Drafts map[int]string `json:"drafts"`
	// KV and Long are the cache and long-context choices at each card count ("" / false: the server's default).
	KV   map[int]string `json:"kv,omitempty"`
	Long map[int]bool   `json:"long,omitempty"`
}

// modelExtras is the next start's cache and long-context choice for the mode: kv "auto" (no --kv: F16, or in the
// long-context mode the format the mode picks for the cards), "f16" or "q8"; long true adds --long-context. A choice the
// mode does not offer reads as the default.
func (m ctl) modelExtras(mi camModel, mode serveMode) (string, bool) {
	p, saved := m.picks[mi.Name]
	kv := p.KV[mode.Cards]
	long := p.Long[mode.Cards]
	if !saved { // (saved --kv / --long-context option values, from before the picker owned them)
		kv, long = m.vals["kv"], m.vals["long_context"] == "true"
	}
	ok := false
	for _, f := range mode.KV {
		if f == kv {
			ok = true
		}
	}
	if !ok {
		kv = "auto"
	}
	return kv, mode.Long && long
}

// cycleExtra steps the cache choice (auto, then each format) or flips the long-context mode, for the current card count.
func (m *ctl) cycleExtra(mi camModel, long bool) {
	mode, draft := m.modelMode(mi)
	kv, lc := m.modelExtras(mi, mode)
	p := m.picks[mi.Name]
	if p.Drafts == nil {
		p.Drafts = make(map[int]string)
	}
	if p.KV == nil {
		p.KV = make(map[int]string)
	}
	if p.Long == nil {
		p.Long = make(map[int]bool)
	}
	// (the first saved pick carries what the next start shows now: the card count, drafter, cache and mode, which may come
	//  from saved option values; modelMode and modelExtras read only the pick once it exists)
	p.Cards, p.Drafts[mode.Cards], p.Long[mode.Cards] = mode.Cards, draft, lc
	if kv != "auto" {
		p.KV[mode.Cards] = kv
	}
	if long {
		if !mode.Long {
			return
		}
		p.Long[mode.Cards] = !lc
	} else {
		if len(mode.KV) < 2 {
			return
		}
		choices := append([]string{"auto"}, mode.KV...)
		for i, v := range choices {
			if v == kv {
				p.KV[mode.Cards] = choices[(i+1)%len(choices)]
				break
			}
		}
		if p.KV[mode.Cards] == "auto" {
			delete(p.KV, mode.Cards)
		}
	}
	if m.picks == nil {
		m.picks = make(map[string]modelPick)
	}
	m.picks[mi.Name] = p
	m.savePrefs()
}

// extraArgs is the cache and long-context flags of the next start (caminus start and caminus fit take the same ones).
func (m ctl) extraArgs(mi camModel) []string {
	mode, _ := m.modelMode(mi)
	kv, lc := m.modelExtras(mi, mode)
	var a []string
	if kv != "auto" {
		a = append(a, "--kv", kv)
	}
	if lc {
		a = append(a, "--long-context")
	}
	return a
}

func (mi camModel) servingModes() []serveMode {
	if len(mi.Modes) != 0 {
		return mi.Modes
	}
	// An older CLI supplies no capabilities: retain its catalog card count and let serve choose its default.
	return []serveMode{{Cards: mi.Cards}}
}

func hasDraft(mode serveMode, draft string) bool {
	if draft == "auto" {
		return true
	}
	for _, d := range mode.Drafts {
		if d == draft {
			return true
		}
	}
	return false
}

func (m ctl) modelMode(mi camModel) (serveMode, string) {
	p, saved := m.picks[mi.Name]
	nc := p.Cards
	if !saved {
		nc, _ = strconv.Atoi(m.vals["cards"])
	}
	if nc == 0 {
		nc = mi.Cards
	}
	modes := mi.servingModes()
	mode := modes[0]
	for _, v := range modes {
		if v.Cards == nc {
			mode = v
			break
		}
	}
	draft := p.Drafts[mode.Cards]
	if draft == "" && !saved {
		draft = m.vals["draft"]
		if m.vals["no_drafter"] == "true" {
			draft = "off"
		}
	}
	if !hasDraft(mode, draft) {
		draft = "auto"
	}
	return mode, draft
}

func (m *ctl) cycleModelMode(mi camModel, cards bool) {
	mode, draft := m.modelMode(mi)
	p := m.picks[mi.Name]
	if p.Drafts == nil {
		p.Drafts = make(map[int]string)
	}
	p.Drafts[mode.Cards] = draft
	if cards {
		modes := mi.servingModes()
		for i, v := range modes {
			if v.Cards == mode.Cards {
				mode = modes[(i+1)%len(modes)]
				break
			}
		}
	} else {
		choices := append([]string{"auto"}, mode.Drafts...)
		for i, v := range choices {
			if v == draft {
				p.Drafts[mode.Cards] = choices[(i+1)%len(choices)]
				break
			}
		}
	}
	p.Cards = mode.Cards
	if m.picks == nil {
		m.picks = make(map[string]modelPick)
	}
	m.picks[mi.Name] = p
	m.savePrefs()
}

// startArgs is `caminus start NAME ...` with the model's choices, the one-card pin and the edited serve options.
func (m ctl) startArgs(mi camModel) []string {
	a := []string{"start", mi.Name}
	mode, draft := m.modelMode(mi)
	if mode.Cards != mi.Cards {
		a = append(a, "--cards", strconv.Itoa(mode.Cards))
	}
	if mode.Cards == 1 && m.card != "" && m.card != "auto" {
		a = append(a, "--card", m.card)
	}
	a = append(a, m.extraArgs(mi)...) // (caminus start's own --kv / --long-context: it checks them before a card is picked)
	var extra []string
	if draft != "auto" {
		extra = append(extra, "--draft", draft)
	}
	extra = append(extra, m.optionFlags()...)
	if len(extra) > 0 {
		a = append(append(a, "--"), extra...)
	}
	return a
}

// optionFlags is the edited serve options as flags, in option-name order; the picker's own choices (cards, drafter,
// cache, long context) and the names in skip are left out.
func (m ctl) optionFlags(skip ...string) []string {
	var keys []string
	for k := range m.vals {
		switch k {
		case "cards", "draft", "no_drafter", "kv", "long_context":
			continue
		}
		if slices.Contains(skip, k) {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var a []string
	for _, k := range keys {
		o := m.optByDest(k)
		if o == nil {
			continue
		}
		v := m.vals[k]
		if o.Type == "bool" {
			if v == "true" {
				a = append(a, o.Flag)
			}
			continue
		}
		if v != "" {
			a = append(a, o.Flag, v)
		}
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

// visibleOpts: the options the options editor lists (--model is the selected model, --port is picked by caminus start,
// --config would replace the panel's own values, and the picker owns cards, drafter, cache and long context).
func (m ctl) visibleOpts() []option {
	var out []option
	for _, o := range m.opts {
		switch o.Dest {
		case "model", "config", "port", "cards", "draft", "no_drafter", "kv", "long_context":
			continue
		}
		out = append(out, o)
	}
	return out
}

// Update runs the message and then reads caminus fit for the selection when it is not yet known (cached per choice).
func (m ctl) Update(msg tea.Msg) (ctl, tea.Cmd) {
	mm, cmd := m.update(msg)
	if mm.loaded["models"] {
		return mm, tea.Batch(cmd, mm.fitCmd())
	}
	return mm, cmd
}

func (m ctl) update(msg tea.Msg) (ctl, tea.Cmd) {
	switch msg := msg.(type) {
	case panelOpenMsg:
		if msg.id != m.id {
			return m, nil
		}
		m.epoch++
		return m, tea.Batch(m.tickCmd(), m.pollCmd(), m.liveTickCmd(0), m.emberTickCmd(0))
	case fitMsg:
		if msg.gen != m.fitGen {
			return m, nil // a read from before R or a start, stop or restart: its cards' memory is no longer the current one
		}
		e := fitEntry{done: true, info: msg.v, err: msg.err}
		if msg.err != nil || !msg.v.OK {
			e.failedAt = time.Now() // not kept: asked again after fitRetry
		}
		m.fits[msg.key] = e
	case ctlTick:
		if !m.mine(msg.id, msg.epoch) {
			return m, nil
		}
		m.frame++
		return m, m.tickCmd()
	case ctlPoll:
		if !m.mine(msg.id, msg.epoch) {
			return m, nil
		}
		return m, tea.Batch(m.loadPS(), m.retryReads(time.Now()), m.pollCmd())
	case liveTick, liveMsg, sessionsMsg, emberTick, emberCountMsg:
		return m.liveUpdate(msg)
	// (a read's failure goes to its own area, never the note line: a poll that keeps failing never hides a key's note)
	case modelsMsg:
		m.readDone("models", msg.err)
		if msg.err == nil {
			m.models = msg.v
			m.curM = min(m.curM, max(0, len(m.models)-1))
		}
	case psMsg:
		m.readDone("ps", msg.err)
		if msg.err == nil {
			msg.v.Servers = m.keepExternal(msg.v.Servers)
			m.ps = msg.v
		}
	case emberMsg:
		m.readDone("ember", msg.err)
		if msg.err == nil || !m.ember.known { // (a failed read keeps what the last answer said)
			m.ember = msg
		}
	case harnessMsg:
		m.readDone("harness", msg.err)
		if msg.err == nil {
			m.hs = msg.v
		}
	case optsMsg:
		m.readDone("opts", msg.err)
		if msg.err == nil {
			m.opts = msg.v
		}
	case actMsg:
		m.busy = ""
		m.refreshFits() // a start, stop or restart changes the cards' memory: fit is read again
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

// panelKey is one key of the screen: the help (?) lists them, and a test keeps every key bound once per view.
type panelKey struct{ keys, what string }

// mainKeys are the main view's keys, in the help's order. The key line shows the ones the selection allows.
func (m ctl) mainKeys() []panelKey {
	ember := "open Ember"
	if m.hosted {
		ember = "back to Ember"
	}
	closeWhat := "quit (running models keep running)"
	if m.hosted {
		closeWhat = "close the panel"
	}
	return []panelKey{
		{"↑ ↓", "choose a model"}, {"tab", "next choice"}, {"← →", "change the choice"}, {"enter", "start the model"},
		{"p", "download the model or its drafter"}, {"s", "stop the running server"}, {"t", "restart it"},
		{"L", "its log"}, {"m", "the full monitor (caminus top)"}, {"e", ember}, {"c", "Qwen Code on the running server"},
		{"u", "set up Qwen Code"}, {"i", "install Qwen Code"}, {"o", "serve options and cards"}, {"n", "card count"},
		{"d", "drafter"}, {"v", "KV cache"}, {"l", "long context"}, {"R", "refresh"}, {"?", "these keys"},
		{"q esc", closeWhat},
	}
}

// optionKeys are the options editor's keys.
func optionKeys() []panelKey {
	return []panelKey{{"↑ ↓", "choose"}, {"enter", "edit or switch"}, {"d", "back to the default"}, {"esc o", "back"}}
}

func (m ctl) key(k tea.KeyPressMsg) (ctl, tea.Cmd) {
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
	switch m.view {
	case viewHelp: // ? shows the next page while there is one; any other key closes the list (esc and q too, not the panel)
		if s == "?" && m.seen != nil && m.helpPage+1 < m.seen.helpPages {
			m.helpPage++
			return m, nil
		}
		m.view, m.helpPage = viewMain, 0
		return m, nil
	case viewOptions:
		if m.editing {
			return m.editKey(k)
		}
		return m.optionsKey(s)
	}
	switch s {
	case "ctrl+c", "q", "esc":
		return m, closePanel
	case "?":
		m.view, m.helpPage = viewHelp, 0
		return m, nil
	case "o":
		m.view, m.curO = viewOptions, 0
		return m, nil
	case "R":
		m.setNote("refreshing", false)
		m.refreshFits()
		return m, tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadOpts(), m.loadEmber())
	case "up", "k":
		m.curM, m.choice = max(0, m.curM-1), 0
		return m, nil
	case "down", "j":
		m.curM, m.choice = min(m.curM+1, max(0, len(m.models)-1)), 0
		return m, nil
	case "tab", "shift+tab":
		if mi := m.selModel(); mi != nil {
			if n := len(m.choiceRows(*mi)); n > 0 {
				d := 1
				if s == "shift+tab" {
					d = n - 1
				}
				m.choice = (m.choice + d) % n
			}
		}
		return m, nil
	case "L", "m":
		return m.serverKey(s)
	}
	if m.busy != "" {
		m.setNote("wait: "+m.busy+" is still running", false)
		return m, nil
	}
	switch s {
	case "s", "t":
		return m.serverKey(s)
	case "e", "c", "u", "i":
		return m.agentKey(s)
	}
	return m.modelsKey(s)
}

// choiceRow is one of the selected model's choices the cursor can rest on.
type choiceRow int

const (
	choiceCards choiceRow = iota
	choiceDrafter
	choiceKV
	choiceLong
)

// choiceRows are the selected model's choices with more than one value, in the order the details show them.
func (m ctl) choiceRows(mi camModel) []choiceRow {
	mode, _ := m.modelMode(mi)
	var r []choiceRow
	if len(mi.servingModes()) > 1 {
		r = append(r, choiceCards)
	}
	if len(mode.Drafts) > 0 {
		r = append(r, choiceDrafter)
	}
	if len(mode.KV) > 1 {
		r = append(r, choiceKV)
	}
	if mode.Long {
		r = append(r, choiceLong)
	}
	return r
}

// change steps a choice forward (d 1) or back (d -1).
func (m *ctl) change(mi camModel, c choiceRow, d int) {
	steps := 1
	if d < 0 {
		mode, _ := m.modelMode(mi)
		switch c { // back = forward by one less than the number of values
		case choiceCards:
			steps = len(mi.servingModes()) - 1
		case choiceDrafter:
			steps = len(mode.Drafts)
		case choiceKV:
			steps = len(mode.KV)
		}
	}
	for i := 0; i < steps; i++ {
		switch c {
		case choiceCards:
			m.cycleModelMode(mi, true)
		case choiceDrafter:
			m.cycleModelMode(mi, false)
		case choiceKV:
			m.cycleExtra(mi, false)
		case choiceLong:
			m.cycleExtra(mi, true)
		}
	}
}

func (m ctl) modelsKey(s string) (ctl, tea.Cmd) {
	mi := m.selModel()
	if mi == nil {
		return m, nil
	}
	switch s {
	case "enter":
		if m.psDown() { // (what runs is not known: a start could land on a busy card or start a model twice)
			m.psDownNote("can't start " + mi.Name + ": ")
			return m, nil
		}
		if e := m.hwError(); e != "" {
			m.setNote(e, true)
			return m, nil
		}
		if !mi.Present {
			m.setNote(mi.Name+" is not downloaded: p to pull it", true)
			return m, nil
		}
		if sv := m.liveOf(mi.Name); sv != nil {
			why := " (s stops it)"
			if sv.External {
				why = " (started outside the panel)"
			}
			m.setNote(fmt.Sprintf("%s is already running on :%d%s", mi.Name, sv.Port, why), false)
			return m, nil
		}
		_, draft := m.modelMode(*mi)
		customDrafter := m.optByDest("drafter") != nil && strings.TrimSpace(m.vals["drafter"]) != ""
		if mi.MissingDrafter != "" && draft == mi.MissingDrafter && !customDrafter {
			m.setNote(mi.Name+"'s "+draft+" drafter is not downloaded: p to pull it", true)
			return m, nil
		}
		m.busy = "starting " + mi.Name
		return m, m.action("start "+mi.Name, 15*time.Minute, m.startArgs(*mi)...)
	case "p":
		if mi.Present && mi.MissingDrafter == "" {
			m.setNote(mi.Name+" is already on disk", false)
			return m, nil
		}
		question := fmt.Sprintf("download %s (%.1f GB)? y / n", mi.Name, mi.SizeGB)
		if mi.Present {
			question = fmt.Sprintf("download %s's %s drafter (%.1f GB)? y / n", mi.Name, mi.MissingDrafter, mi.MissingDrafterGB)
		}
		m.confirm = &pending{q: question, run: m.action("pull "+mi.Name, 12*time.Hour, "pull", mi.Name)}
	case "left", "right":
		rows := m.choiceRows(*mi)
		if len(rows) == 0 {
			return m, nil
		}
		d := 1
		if s == "left" {
			d = -1
		}
		m.change(*mi, rows[min(m.choice, len(rows)-1)], d)
	case "n":
		m.cycleModelMode(*mi, true)
	case "d":
		m.cycleModelMode(*mi, false)
	case "v":
		m.cycleExtra(*mi, false)
	case "l":
		m.cycleExtra(*mi, true)
	}
	if rows := m.choiceRows(*mi); m.choice >= len(rows) { // (a card count can take a choice row away)
		m.choice = max(0, len(rows)-1)
	}
	return m, nil
}

// serverKey acts on the server in the running card: stop, restart, its log, the full monitor.
func (m ctl) serverKey(s string) (ctl, tea.Cmd) {
	sv := m.primary()
	if sv == nil && m.psDown() {
		m.psDownNote("")
		return m, nil
	}
	if sv == nil {
		m.setNote("no server running", false)
		return m, nil
	}
	name := sv.Name
	if sv.External && s == "L" {
		// (caminus ps keeps no log for a server started outside caminus start, and caminus logs finds only theirs)
		shown := name
		if sv.Model != "" {
			shown = sv.Model
		}
		m.setNote(shown+" was started outside the panel, so its log is wherever it was started", false)
		return m, nil
	}
	if sv.External && s != "m" {
		m.setNote(name+" was started outside the panel: stop or restart it where it was started", false)
		return m, nil
	}
	switch s {
	case "s":
		m.confirm = &pending{q: "stop " + name + "? y / n", run: m.action("stop "+name, 5*time.Minute, "stop", name)}
	case "t":
		m.confirm = &pending{q: "restart " + name + "? y / n", run: m.action("restart "+name, 15*time.Minute, "restart", name)}
	case "L":
		return m, m.interactive("log of "+name, "logs", name)
	case "m":
		return m, m.interactive("monitor of "+name, "top", name)
	}
	return m, nil
}

// agentKey: Ember and Qwen Code.
func (m ctl) agentKey(s string) (ctl, tea.Cmd) {
	target := ""
	sv := m.primary()
	if sv != nil && !sv.External {
		target = sv.Name
	}
	if sv != nil && sv.External && (s == "c" || s == "u") && !m.anyManaged() {
		// (caminus qwen and its setup follow the servers caminus start records; this one has no record)
		m.setNote("Qwen Code follows servers started by the panel or caminus start: "+sv.Model+" was started outside it", false)
		return m, nil
	}
	switch s {
	case "e":
		if m.hosted {
			return m, closePanel // (inside Ember: e goes back to it)
		}
		if err := m.readErr["ember"]; err != nil && !m.ember.ok {
			m.setNote("couldn't read Ember's status: "+err.Error(), true)
			m.noteRead = "ember"
			return m, m.loadEmber()
		}
		if !m.ember.ok {
			m.setNote("Ember is not set up: run caminus ember setup (docs/ember-quickstart.md walks through it)", true)
			return m, m.loadEmber()
		}
		return m, m.interactive("Ember", "ember", "ui")
	case "c":
		if err := m.readErr["harness"]; err != nil && !m.hs.Installed {
			m.setNote("couldn't read Qwen Code's status: "+err.Error(), true)
			m.noteRead = "harness"
			return m, m.loadHarness()
		}
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

// anyManaged is whether a server caminus start recorded runs.
func (m ctl) anyManaged() bool {
	for _, s := range m.liveServers() {
		if !s.External {
			return true
		}
	}
	return false
}

// optionsKey: the options editor (o). Row 0 is the card for one-card models, then every serve option.
func (m ctl) optionsKey(s string) (ctl, tea.Cmd) {
	vo := m.visibleOpts()
	switch s {
	case "esc", "o", "q":
		m.view = viewMain
		return m, nil
	case "ctrl+c":
		return m, closePanel
	case "up", "k":
		m.curO = max(0, m.curO-1)
		return m, nil
	case "down", "j":
		m.curO = min(m.curO+1, len(vo))
		return m, nil
	}
	if m.curO == 0 { // the card pin
		switch s {
		case "enter", "right":
			m.card = m.nextCard()
			m.savePrefs()
		case "d", "backspace", "delete":
			m.card = "auto"
			m.savePrefs()
		}
		return m, nil
	}
	if m.curO-1 >= len(vo) {
		return m, nil
	}
	o := vo[m.curO-1]
	switch s {
	case "enter":
		if o.Type == "bool" {
			if m.vals[o.Dest] == "true" {
				delete(m.vals, o.Dest)
			} else {
				m.vals[o.Dest] = "true"
			}
			m.savePrefs()
			return m, nil
		}
		m.editing, m.edit, m.editFresh = true, m.valueOf(o), true
	case "d", "backspace", "delete":
		delete(m.vals, o.Dest)
		m.savePrefs()
	}
	return m, nil
}

func (m ctl) editKey(k tea.KeyPressMsg) (ctl, tea.Cmd) {
	vo := m.visibleOpts()
	if len(vo) == 0 || m.curO < 1 {
		m.editing = false
		return m, nil
	}
	o := vo[min(m.curO-1, len(vo)-1)]
	fresh := m.editFresh
	m.editFresh = false // (any key ends the selection: arrows keep the value, backspace edits it as before)
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
			if fresh { // the first character replaces the value the field opened with
				m.edit = ""
			}
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

// prefs is panel.json. Advanced is read and dropped: the panel had a basic and an advanced layout, and the saved option
// values now always reach the next start (the details show them).
type prefs struct {
	Advanced bool                 `json:"advanced,omitempty"`
	Card     string               `json:"card"`
	Values   map[string]string    `json:"values"`
	Picks    map[string]modelPick `json:"models,omitempty"`
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
		if p.Card != "" {
			m.card = p.Card
		}
		for k, v := range p.Values {
			m.vals[k] = v
		}
		m.picks = p.Picks
	}
}

func (m ctl) savePrefs() {
	if m.prefs == "" {
		return
	}
	b, _ := json.MarshalIndent(prefs{Card: m.card, Values: m.vals, Picks: m.picks}, "", "  ")
	_ = os.MkdirAll(filepath.Dir(m.prefs), 0o700)
	_ = os.WriteFile(m.prefs, append(b, '\n'), 0o600)
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

// newPanel is the component for a host: Init, Update(msg) (ctl, tea.Cmd), View(w, h) string; it sends panelCloseMsg
// when it wants to be closed. Make a new one for each open (its choices come back from the prefs file).
func newPanel(o panelOptions) ctl {
	m := newCtl(newCtlCLI(o.CLI), o.Prefs)
	m.hosted = o.Hosted
	return m
}

func newCtl(be *ctlCLI, prefsFile string) ctl {
	panelSeq++
	m := ctl{be: be, id: panelSeq, loaded: map[string]bool{}, readErr: map[string]error{}, retry: map[string]bool{},
		failAt: map[string]time.Time{}, vals: map[string]string{}, card: "auto",
		prefs: prefsFile, fits: map[string]fitEntry{}, live: newLiveMon(), seen: &panelSeen{}}
	m.loadPrefs()
	return m
}

// panelHost runs the component full screen: the alt screen, the window size, the theme's background in every cell,
// and a close is a quit.
type panelHost struct {
	p    ctl
	w, h int
}

func (h panelHost) Init() tea.Cmd { return h.p.Init() }

func (h panelHost) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		h.w, h.h = msg.Width, msg.Height
		return h, nil
	case panelCloseMsg:
		return h, tea.Quit
	}
	var cmd tea.Cmd
	h.p, cmd = h.p.Update(msg)
	return h, cmd
}

func (h panelHost) View() tea.View {
	w, ht := h.w, h.h
	if w <= 0 {
		w = 120
	}
	if ht <= 0 {
		ht = 40
	}
	frame := h.p.View(w, ht)
	if themeBackground() != nil { // the theme's background in every cell, as Ember paints it: the two share one look
		frame = paintBackground(frame, 0, w, max(ht, lipgloss.Height(frame)), false)
	}
	v := tea.NewView(frame)
	v.AltScreen = true
	v.WindowTitle = "caminus panel"
	return v
}

// onceFrame is the -once frame: every read answered, the selected model's fit and one live round, then drawn.
func onceFrame(m ctl, w, h int) string {
	m = settle(m, tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadOpts(), m.loadEmber()), 0)
	m = settle(m, m.fitCmd(), 0) // the selected model's fit is answered before the frame is drawn
	m = settle(m, m.liveRound(), 0)
	m = settle(m, m.sessionScan(), 0)
	if m.ember.ok {
		m = settle(m, m.emberCount(), 0)
	}
	return m.View(w, h)
}

// ctlMain runs the control panel: `caminus-panel [-caminus CMD] [-once [W H]]`.
func ctlMain(args []string) {
	cli, once, w, h := "", false, 120, 40
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-advanced", "--advanced": // (the old advanced layout: everything is on the one screen now)
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
			fmt.Fprintf(os.Stderr, "caminus panel: unknown argument %q (-caminus CMD, -once [W H]; `top` for the monitor)\n", args[i])
			os.Exit(2)
		}
	}
	if once {
		fmt.Println(onceFrame(newPanel(panelOptions{CLI: cli}), w, h)) // (no prefs file: a one-off frame changes nothing)
		return
	}
	if _, err := tea.NewProgram(panelHost{p: newPanel(panelOptions{CLI: cli, Prefs: prefsPath()})}).Run(); err != nil {
		fmt.Fprintln(os.Stderr, "caminus panel:", err)
		os.Exit(1)
	}
}
