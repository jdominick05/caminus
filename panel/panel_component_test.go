// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// dummyHost hosts the panel the way Ember will: it forwards every message while the panel is open, draws it at its own
// size, and catches the close.
type dummyHost struct {
	panel  ctl
	open   bool
	closed int
	w, h   int
}

func (d dummyHost) Update(msg tea.Msg) (dummyHost, tea.Cmd) {
	if _, ok := msg.(panelCloseMsg); ok {
		d.open = false
		d.closed++
		return d, nil
	}
	if !d.open {
		return d, nil
	}
	var cmd tea.Cmd
	d.panel, cmd = d.panel.Update(msg)
	return d, cmd
}

// fixturePanel is a component with a made-up host: two cards, five models (one not downloaded), a ready server.
func fixturePanel() ctl {
	tps := 104.0
	m := newPanel(panelOptions{CLI: "caminus-not-run"})
	for _, k := range []string{"models", "ps", "harness", "opts"} {
		m.loaded[k] = true
	}
	m.models = []camModel{
		{Name: "coder", Title: "Qwen3.8-Flash-Next Coder IQ1_M", Cards: 2, SizeGB: 59, Present: true, Path: "/m/coder.gguf",
			Speed: "~104 tok/s, 123-203 with DFlash", Modes: []serveMode{{Cards: 2, Drafts: []string{"off", "dflash"}}}},
		{Name: "swift-1.5", Title: "Swift 1.5 27B Q4_K_M", Cards: 1, SizeGB: 20, Present: true, Path: "/m/swift.gguf",
			Modes: []serveMode{{Cards: 1, Drafts: []string{"off", "mtp", "dflash"}}, {Cards: 2, Drafts: []string{"off", "mtp", "dflash"}}}},
		{Name: "thinkingcap", Title: "ThinkingCap 27B", Cards: 1, SizeGB: 16, Present: true, MissingDrafter: "dflash", MissingDrafterGB: 2.1},
		{Name: "qwen3.8-27b", Title: "Qwen3.8-27B", Cards: 1, SizeGB: 14, Present: false},
		{Name: "qwen3-30b-a3b", Title: "Qwen3-30B-A3B", Cards: 1, SizeGB: 19, Present: true,
			Modes: []serveMode{{Cards: 1, Drafts: []string{"off"}, KV: []string{"f16", "q8"}, Long: true}}},
	}
	m.ps = psInfo{
		Servers: []serverInfo{{Name: "coder", Model: "coder", State: "ready", URL: "http://127.0.0.1:1", Port: 8092,
			Cards: []string{"XTX", "XT"}, MaxCtx: 1048576, UptimeS: 8040, DecodeTPS: &tps}},
		Cards: []cardInfo{{Index: 0, Name: "XTX", UsedMiB: 21800, LineMiB: 23800, TotalMiB: 24560},
			{Index: 1, Name: "XT", UsedMiB: 15460, LineMiB: 19850, TotalMiB: 20464}},
	}
	m.hs = harnessInfo{Installed: true, Configured: true, Version: "0.26.9", Npm: true}
	m.ember = emberMsg{ok: true, known: true}
	m.opts = []option{{Flag: "--max-ctx", Dest: "max_ctx", Type: "ctx", Default: "auto", Help: "the engine's KV positions"}}
	return m
}

// TestThePanelInAHost: the component opens in a host, receives its own ticks, draws at the host's two sizes, and Esc
// closes it without quitting the host's program.
func TestThePanelInAHost(t *testing.T) {
	d := dummyHost{panel: fixturePanel(), open: true}
	open := d.panel.Init()()
	batch, ok := open.(tea.BatchMsg)
	if !ok {
		t.Fatalf("Init is a batch of reads and the open message: %T", open)
	}
	var opened tea.Msg
	for _, c := range batch {
		if c == nil {
			continue
		}
		if msg, ok := callFast(c); ok {
			if _, is := msg.(panelOpenMsg); is {
				opened = msg
			}
		}
	}
	if opened == nil {
		t.Fatal("Init sends panelOpenMsg")
	}
	d, cmd := d.Update(opened)
	if d.panel.epoch != 1 || cmd == nil {
		t.Fatalf("the open message starts the ticks: epoch %d", d.panel.epoch)
	}
	frame0 := d.panel.frame
	d, cmd = d.Update(ctlTick{d.panel.id, d.panel.epoch})
	if d.panel.frame != frame0+1 || cmd == nil {
		t.Fatal("a tick of this panel advances its frame and asks for the next")
	}
	d, cmd = d.Update(ctlTick{d.panel.id, d.panel.epoch - 1}) // a tick of an earlier Init
	if d.panel.frame != frame0+1 || cmd != nil {
		t.Fatal("a tick from an earlier Init is dropped and starts nothing")
	}
	d, _ = d.Update(ctlTick{d.panel.id + 1000, d.panel.epoch}) // another panel's
	if d.panel.frame != frame0+1 {
		t.Fatal("another component's tick is dropped")
	}
	d, _ = d.Update(tea.WindowSizeMsg{Width: 10, Height: 5}) // the host's size message changes nothing in the panel
	for _, sz := range [][2]int{{80, 24}, {160, 48}} {
		v := d.panel.View(sz[0], sz[1])
		ls := strings.Split(v, "\n")
		if len(ls) > sz[1] {
			t.Fatalf("%dx%d: %d lines", sz[0], sz[1], len(ls))
		}
		for _, l := range ls {
			if lipgloss.Width(l) > sz[0] {
				t.Fatalf("%dx%d: a line of %d cells: %q", sz[0], sz[1], lipgloss.Width(l), ansi.Strip(l))
			}
		}
		if !strings.Contains(ansi.Strip(v), "CAMINUS") || !strings.Contains(ansi.Strip(v), "coder") {
			t.Fatalf("%dx%d: the frame:\n%s", sz[0], sz[1], ansi.Strip(v))
		}
	}
	// Esc in the options editor goes back to the main view; Esc there closes the panel
	d, _ = d.Update(tea.KeyPressMsg{Code: 'o', Text: "o"})
	if d.panel.view != viewOptions {
		t.Fatal("o opens the options editor")
	}
	d, cmd = d.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
	if d.panel.view != viewMain || cmd != nil || !d.open {
		t.Fatal("Esc leaves the options editor, not the panel")
	}
	d, cmd = d.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
	if cmd == nil {
		t.Fatal("Esc asks to close")
	}
	msg := cmd()
	if _, quit := msg.(tea.QuitMsg); quit {
		t.Fatal("the component never quits the host's program")
	}
	d, _ = d.Update(msg)
	if d.open || d.closed != 1 {
		t.Fatal("the host caught the close")
	}
	// the panel's own host turns the close into a quit
	h := panelHost{p: fixturePanel()}
	if _, c := h.Update(panelCloseMsg{}); c == nil {
		t.Fatal("caminus panel quits on close")
	} else if _, quit := c().(tea.QuitMsg); !quit {
		t.Fatal("caminus panel's close is a quit")
	}
}

// callFast runs a command that answers at once (the open message); reads that run the CLI are skipped.
func callFast(c tea.Cmd) (tea.Msg, bool) {
	ch := make(chan tea.Msg, 1)
	go func() { ch <- c() }()
	select {
	case m := <-ch:
		return m, true
	case <-time.After(2 * time.Second):
		return nil, false
	}
}

// Every key is bound once in each view, and none of the main view's is a key the host could need (no ctrl keys but
// ctrl+c, which closes).
func TestPanelKeysAreBoundOnce(t *testing.T) {
	for name, keys := range map[string][]panelKey{"main": fixturePanel().mainKeys(), "options": optionKeys()} {
		seen := map[string]string{}
		for _, k := range keys {
			for _, key := range strings.Fields(k.keys) {
				if prev, dup := seen[key]; dup {
					t.Fatalf("%s view: %q is bound twice (%s, %s)", name, key, prev, k.what)
				}
				seen[key] = k.what
			}
		}
	}
	// each main-view letter does what the list says it does: nothing unhandled falls through silently
	m := fixturePanel()
	for _, k := range m.mainKeys() {
		for _, key := range strings.Fields(k.keys) {
			if len([]rune(key)) != 1 || strings.ContainsAny(key, "↑↓←→?") {
				continue
			}
			n, cmd := m.key(tea.KeyPressMsg{Code: []rune(key)[0], Text: key})
			if cmd == nil && n.confirm == nil && n.note == m.note && n.view == m.view && n.picks == nil &&
				!strings.ContainsAny(key, "ndvl") {
				t.Fatalf("%q (%s) did nothing", key, k.what)
			}
		}
	}
}

// The options editor holds what the advanced layout had: the card pin, every serve option, the cards and the notes.
// An edited option shows on the main screen and reaches the next start.
func TestOptionsEditorKeepsTheAdvancedChoices(t *testing.T) {
	m := fixturePanel()
	m.ps.Problems = []problemInfo{{Level: "info", Text: "two supported cards"}}
	m.curM = 1 // swift-1.5: one card by default
	m, _ = m.key(tea.KeyPressMsg{Code: 'o', Text: "o"})
	out := ansi.Strip(m.View(120, 40))
	for _, want := range []string{"card for one-card models", "--max-ctx", "the guard's line", "two supported cards"} {
		if !strings.Contains(out, want) {
			t.Fatalf("the options editor shows %q:\n%s", want, out)
		}
	}
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyEnter}) // the card pin: auto -> 0
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyDown})
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyEnter}) // edit --max-ctx
	if !m.editing {
		t.Fatal("enter edits the option")
	}
	m, _ = m.key(tea.KeyPressMsg{Code: 'u', Mod: tea.ModCtrl})
	for _, r := range "8192" {
		m, _ = m.key(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyEnter})
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyEscape})
	if m.view != viewMain || m.vals["max_ctx"] != "8192" || m.card != "0" {
		t.Fatalf("the edits are kept: view %d vals %v card %q", m.view, m.vals, m.card)
	}
	if d := ansi.Strip(strings.Join(m.detailRows(90), "\n")); !strings.Contains(d, "options  --max-ctx 8192 --card 0") {
		t.Fatalf("what goes to the next start is on the main screen:\n%s", d)
	}
	if got := strings.Join(m.startArgs(m.models[1]), " "); got != "start swift-1.5 --card 0 -- --max-ctx 8192" {
		t.Fatalf("and reaches the start: %s", got)
	}
}

// ← / → change the choice under the cursor; tab moves the cursor between the selected model's choices.
func TestChoicesChangeWithTheArrows(t *testing.T) {
	m := fixturePanel()
	m.curM = 1 // swift-1.5: cards and drafter
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyRight})
	if mode, _ := m.modelMode(m.models[1]); mode.Cards != 2 {
		t.Fatalf("→ on cards: two cards (%d)", mode.Cards)
	}
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyTab})
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyRight})
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyRight})
	if _, d := m.modelMode(m.models[1]); d != "mtp" {
		t.Fatalf("→ → on the drafter: auto -> off -> mtp (%s)", d)
	}
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyLeft})
	if _, d := m.modelMode(m.models[1]); d != "off" {
		t.Fatalf("← steps back: %s", d)
	}
	if d := ansi.Strip(strings.Join(m.detailRows(90), "\n")); !strings.Contains(d, "cards    1 [2]") || !strings.Contains(d, "drafter  auto [off] mtp dflash") {
		t.Fatalf("the choices show their values:\n%s", d)
	}
}

// The live round fills the running card: decode and prefill from the status, the cache share from /metrics, the
// clients from the session scan.
func TestLiveNumbersFillTheRunningCard(t *testing.T) {
	m := fixturePanel()
	if c := m.liveRound(); c == nil {
		t.Fatal("a running server is followed")
	}
	mpt, pre := 9.6, 2800.0
	cs := &camStatus{Version: 1}
	cs.Decode.MsPerTok = &mpt
	cs.Prefill.Rate.P50 = &pre
	p := probe{T: time.Now(), Base: m.live.url, Health: 200, MetricsCode: 200, CamCode: 200, Cam: cs,
		Metrics: map[string]float64{"prompt_tokens_total": 18, "prompt_tokens_cached_total": 82}}
	m, _ = m.Update(liveMsg{id: m.id, url: m.live.url, p: p})
	m, _ = m.Update(sessionsMsg{id: m.id, n: 3})
	card := ansi.Strip(strings.Join(m.runningRows(110), "\n"))
	for _, want := range []string{"104 tok/s decode", "2,800 tok/s prefill", "3 clients", "82% cached", "decode ", "idle"} {
		if !strings.Contains(card, want) {
			t.Fatalf("the card shows %q:\n%s", want, card)
		}
	}
	m, _ = m.Update(liveMsg{id: m.id + 1, url: m.live.url, p: probe{T: time.Now(), Health: 0}})
	if !m.live.srv.Up {
		t.Fatal("another panel's live answer is dropped")
	}
}

// Empty states: nothing running, nothing on disk, nothing read yet.
func TestEmptyStates(t *testing.T) {
	m := fixturePanel()
	m.ps.Servers = nil
	for i := range m.models {
		m.models[i].Present = false
	}
	out := ansi.Strip(m.View(80, 24))
	for _, want := range []string{"no server running · enter starts the selected model", "not downloaded · p downloads it (59 GB)", "↓"} {
		if !strings.Contains(out, want) {
			t.Fatalf("empty state shows %q:\n%s", want, out)
		}
	}
	m, _ = m.key(tea.KeyPressMsg{Code: tea.KeyEnter})
	if !strings.Contains(m.note, "not downloaded") {
		t.Fatalf("enter on a model not on disk says so: %q", m.note)
	}
	fresh := newPanel(panelOptions{CLI: "caminus-not-run"})
	if out := ansi.Strip(fresh.View(100, 30)); !strings.Contains(out, "looking for servers") || !strings.Contains(out, "reading the model list") {
		t.Fatalf("before the first reads:\n%s", out)
	}
}

// TestPanelGoldenFrames: the fixture's frames at the sizes the panel is checked at (LLMTOP_GOLDEN=update after a
// deliberate change; the files are what a reviewer reads to audit the screen).
func TestPanelGoldenFrames(t *testing.T) {
	old, oldG := version, g
	version, g = "0.1.0", glyphTiers["unicode"]
	defer func() { version, g = old, oldG }()
	stacked := fixturePanel()
	stacked.curM = 4
	for _, c := range []struct {
		name string
		m    ctl
		w, h int
	}{{"panel_80x24.txt", fixturePanel(), 80, 24}, {"panel_100x30.txt", fixturePanel(), 100, 30},
		{"panel_160x48.txt", fixturePanel(), 160, 48}, {"panel_240x60.txt", fixturePanel(), 240, 60},
		{"panel_70x30_stacked.txt", stacked, 70, 30}} {
		got := ansi.Strip(c.m.View(c.w, c.h)) + "\n"
		for i, l := range strings.Split(got, "\n") {
			if lipgloss.Width(l) > c.w {
				t.Fatalf("%s line %d: %d cells", c.name, i+1, lipgloss.Width(l))
			}
		}
		p := filepath.Join("testdata", c.name)
		if os.Getenv("LLMTOP_GOLDEN") == "update" {
			if err := os.WriteFile(p, []byte(got), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("%s: %v (LLMTOP_GOLDEN=update writes it)", c.name, err)
		}
		if got != string(want) {
			t.Errorf("%s differs (LLMTOP_GOLDEN=update after a deliberate change):\n%s", c.name, got)
		}
	}
}

// "clients" counts the Qwen Code sessions on the server and the running Ember sessions whose brain is it: their status'
// model is one of the server's /v1/models ids. A session on a paid brain is not one of its clients. The Ember reads are
// the ones the panel already makes (peers, then one status each); /v1/models comes with the live round.
func TestClientsCountEmberSessionsOnThisServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/models":
			fmt.Fprint(w, `{"object":"list","data":[{"id":"k3-coder"},{"id":"k3-coder-xhigh"},{"id":"k3-coder-low"}]}`)
		case "/props":
			fmt.Fprint(w, `{"engine":"caminus","model_alias":"k3-coder","default_generation_settings":{"n_ctx":1048576}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	dir := t.TempDir()
	script := filepath.Join(dir, "caminus.sh")
	body := `#!/bin/sh
case "$3" in
peers) echo '[{"name":"a"},{"name":"b"},{"name":"paid"},{"name":"stopped"}]' ;;
status) case "$4" in
  a) echo '{"daemon_alive":true,"model":"k3-coder","goals":[{"state":"working"}]}' ;;
  b) echo '{"daemon_alive":true,"model":"k3-coder-xhigh","goals":[]}' ;;
  paid) echo '{"daemon_alive":true,"model":"claude-haiku-5-5","goals":[{"state":"working"}]}' ;;
  stopped) echo '{"daemon_alive":false,"model":"k3-coder","goals":[]}' ;;
  esac ;;
esac
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	m := fixturePanel()
	m.be = &ctlCLI{argv: []string{"sh", script}, timeout: 10 * time.Second}
	m.ps.Servers[0].URL = srv.URL
	m = settle(m, m.liveRound(), 0) // one round: health, props, and the model ids
	if !m.live.srv.Up || len(m.live.ids) != 3 {
		t.Fatalf("the round reads the server's model ids: up %v ids %v", m.live.srv.Up, m.live.ids)
	}
	m = settle(m, m.emberCount(), 0)
	if c := m.live.ember; c.sessions != 4 || c.active != 2 || len(c.brains) != 3 {
		t.Fatalf("Ember's reads: %+v", c)
	}
	m, _ = m.Update(sessionsMsg{id: m.id, n: 1}) // one Qwen Code session
	if got := m.clientsText(); got != "3 clients (Qwen Code 1, Ember 2)" {
		t.Fatalf("2 Ember sessions on this server + 1 Qwen Code session, the paid and the stopped one not counted: %q", got)
	}
	if card := ansi.Strip(strings.Join(m.runningRows(110), "\n")); !strings.Contains(card, "3 clients (Qwen Code 1, Ember 2)") {
		t.Fatalf("the running card says it:\n%s", card)
	}
	m, _ = m.Update(sessionsMsg{id: m.id, n: 0})
	if got := m.clientsText(); got != "2 clients" {
		t.Fatalf("one kind only: no breakdown: %q", got)
	}
	m.live.ember.brains = []string{"claude-haiku-5-5"}
	if got := m.clientsText(); got != "0 clients" {
		t.Fatalf("a paid brain is not a client: %q", got)
	}
	m.live.ids = nil // before /v1/models answers: the alias and its variants still match
	m.live.ember.brains = []string{"k3-coder-low"}
	if got := m.clientsText(); got != "1 client" {
		t.Fatalf("the alias rule: %q", got)
	}
}
