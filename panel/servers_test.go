// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// The running card follows the server in use: the selected model's when it runs, else the first ready one. Others
// get one line each; a record that failed or stopped is never in the card.
func TestRunningCardFollowsTheServerInUse(t *testing.T) {
	tps := 136.0
	m := newCtl(newCtlCLI("caminus"), "")
	m.loaded["ps"], m.loaded["models"] = true, true
	m.models = []camModel{{Name: "swift-1.5", Title: "Swift 1.5 27B", Cards: 1, Present: true},
		{Name: "qwen3-30b-a3b", Title: "Qwen3-30B-A3B", Cards: 1, Present: true}}
	m.ps = psInfo{Servers: []serverInfo{
		{Name: "swift-1.5", Model: "swift-1.5", State: "failed", Error: "RuntimeError: old", Port: 8094},
		{Name: "qwen3-30b-a3b", Model: "qwen3-30b-a3b", State: "ready", URL: "http://127.0.0.1:8093", Port: 8093, Cards: []string{"XT"},
			MaxCtx: 40960, DecodeTPS: &tps, UptimeS: 600},
	}}
	card := ansi.Strip(strings.Join(m.runningRows(96), "\n"))
	if !strings.Contains(card, "qwen3-30b-a3b") || !strings.Contains(card, ":8093") || !strings.Contains(card, "136 tok/s decode") ||
		strings.Contains(card, "failed") {
		t.Fatalf("the card shows the ready server, never the failed record:\n%s", card)
	}
	if p := m.primary(); p == nil || p.Name != "qwen3-30b-a3b" {
		t.Fatalf("stop / restart act on the ready server: %+v", p)
	}
	if d := ansi.Strip(strings.Join(m.detailRows(80), "\n")); !strings.Contains(d, "last start failed: RuntimeError: old") {
		t.Fatalf("the failed record stays readable on its own model's details:\n%s", d)
	}
	m.ps.Servers = append(m.ps.Servers, serverInfo{Name: "swift-1.5", Model: "swift-1.5", State: "ready", URL: "http://127.0.0.1:8092", Port: 8092})
	if p := m.primary(); p == nil || p.Name != "swift-1.5" {
		t.Fatalf("with its model selected, a running server is the one in use: %+v", p)
	}
	if card := ansi.Strip(strings.Join(m.runningRows(96), "\n")); !strings.Contains(card, "qwen3-30b-a3b · :8093") {
		t.Fatalf("the other running server keeps a line:\n%s", card)
	}
	for _, w := range []int{80, 100, 160, 240} {
		for i, l := range strings.Split(m.View(w, 40), "\n") {
			if lw := lipgloss.Width(l); lw > w {
				t.Fatalf("%d columns: line %d is %d cells: %q", w, i, lw, ansi.Strip(l))
			}
		}
	}
}

// One old record must never paint the panel failed: the header carries no server state, and with nothing live the card
// says so quietly.
func TestAFailedRecordNeverPaintsThePanel(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), "")
	m.loaded["ps"], m.loaded["models"] = true, true
	m.models = []camModel{{Name: "coder", Title: "Coder", Cards: 2, Present: true}}
	m.ps = psInfo{Servers: []serverInfo{{Name: "brain", Model: "arex-2", State: "failed",
		Error: "GUARD REFUSED: a GPU fault was recorded (FAULT_FILE). Stop all GPU work and report to the lead."},
		{Name: "coder", Model: "coder", State: "stopped"}}}
	out := ansi.Strip(m.View(100, 30))
	head := strings.SplitN(out, "\n", 2)[0]
	if strings.Contains(head, "failed") || strings.Contains(out, "GUARD REFUSED") {
		t.Fatalf("a record of another model paints nothing:\n%s", out)
	}
	if !strings.Contains(out, "no server running · enter starts the selected model") {
		t.Fatalf("nothing live: the quiet line:\n%s", out)
	}
	if m.primary() != nil {
		t.Fatal("a failed or stopped record is not a server in use")
	}
}

// A healthy server started outside the panel (caminus ps --json lists it external) shows as running, on its model's
// row too, and s / t refuse to touch it.
func TestAServerStartedOutsideThePanelRuns(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), "")
	m.loaded["ps"], m.loaded["models"] = true, true
	m.models = []camModel{{Name: "coder", Title: "Qwen3.8-Flash-Next Coder", Cards: 2, Present: true}}
	m.ps = psInfo{Servers: []serverInfo{{Name: "coder", Model: "coder", State: "stopped", Port: 8092},
		{Name: "observed-8092", Model: "coder", State: "ready", URL: "http://127.0.0.1:8092", Port: 8092, MaxCtx: 1048576,
			External: true}}}
	card := ansi.Strip(strings.Join(m.runningRows(96), "\n"))
	if !strings.Contains(card, "coder · Qwen3.8-Flash-Next Coder") || !strings.Contains(card, "1M ctx") ||
		!strings.Contains(card, "started outside the panel") || strings.Contains(card, "observed-8092") {
		t.Fatalf("the outside server runs under its model's name:\n%s", card)
	}
	if l := ansi.Strip(m.listRows()[0]); !strings.Contains(l, pg().working) {
		t.Fatalf("its model's row is marked running: %q", l)
	}
	for _, k := range []string{"s", "t"} {
		n, cmd := m.serverKey(k)
		if cmd != nil || n.confirm != nil || !strings.Contains(n.note, "started outside the panel") {
			t.Fatalf("%s on an outside server does nothing but say so: cmd %v confirm %+v note %q", k, cmd != nil, n.confirm, n.note)
		}
	}
	if _, cmd := m.serverKey("m"); cmd == nil {
		t.Fatal("m opens the full monitor on it")
	}
	if n, cmd := m.modelsKey("enter"); cmd != nil || !strings.Contains(n.note, "already running on :8092") {
		t.Fatalf("enter on its model starts nothing: cmd %v note %q", cmd != nil, n.note)
	}
	m.hs = harnessInfo{Installed: true, Configured: true, Version: "0.26.9"}
	for _, k := range []string{"c", "u"} {
		if n, cmd := m.agentKey(k); cmd != nil || n.busy != "" || !strings.Contains(n.note, "started outside it") {
			t.Fatalf("%s with only an outside server runs nothing and says why: cmd %v note %q", k, cmd != nil, n.note)
		}
	}
	// ps asks it /health with a 1 s limit: a read that misses it keeps it while the live round still reaches it, for a
	// few reads, then lets it go
	m.live.url, m.live.srv.Up = "http://127.0.0.1:8092", true
	for i := 0; i < extGrace; i++ {
		m, _ = m.Update(psMsg{v: psInfo{Servers: []serverInfo{{Name: "coder", Model: "coder", State: "stopped", Port: 8092}}}})
		if m.primary() == nil {
			t.Fatalf("read %d: a missed answer keeps the outside server running", i+1)
		}
	}
	m, _ = m.Update(psMsg{v: psInfo{Servers: []serverInfo{{Name: "coder", Model: "coder", State: "stopped", Port: 8092}}}})
	if m.primary() != nil {
		t.Fatal("past the grace the outside server is gone")
	}
	m.live.srv.Up = false
	m.ps.Servers = append(m.ps.Servers, serverInfo{Name: "observed-8092", Model: "coder", State: "ready", URL: "http://127.0.0.1:8092",
		Port: 8092, External: true})
	m, _ = m.Update(psMsg{v: psInfo{}})
	if m.primary() != nil {
		t.Fatal("a server the live round no longer reaches is not kept")
	}
}

func TestThePanelHasAWayIntoEmber(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), "")
	m.loaded["harness"] = true
	m.ember = emberMsg{known: true}
	if v := ansi.Strip(strings.Join(m.agentLines(120), "\n")); !strings.Contains(v, "Ember not set up") || !strings.Contains(v, "caminus ember setup") {
		t.Fatalf("not set up, the line says how:\n%s", v)
	}
	next, _ := m.agentKey("e")
	if !strings.Contains(next.note, "caminus ember setup") {
		t.Fatalf("e without Ember names the setup: %q", next.note)
	}
	m.ember = emberMsg{ok: true, known: true}
	if v := ansi.Strip(m.emberText()); !strings.Contains(v, "Ember set up · e") {
		t.Fatalf("set up, the line offers it: %s", v)
	}
	m.live.ember = emberCounts{known: true, sessions: 3, active: 1}
	if v := ansi.Strip(m.emberText()); v != "Ember 3 sessions, 1 working · e" {
		t.Fatalf("its sessions are counted: %q", v)
	}
	if _, cmd := m.agentKey("e"); cmd == nil {
		t.Fatal("e opens caminus ember ui")
	}
	m.hosted = true // inside Ember, e goes back to it
	_, cmd := m.agentKey("e")
	if cmd == nil {
		t.Fatal("e inside Ember closes the panel")
	}
	if _, ok := cmd().(panelCloseMsg); !ok {
		t.Fatal("e inside Ember is a close, not a second Ember")
	}
}

func TestThePanelAndTopPaintTheThemeAsEmberDoes(t *testing.T) {
	saved := themeCurrent
	defer func() { themeCurrent = saved; applyThemeStyles() }()
	if err := themeLoad("caminus"); err != nil {
		t.Fatal(err)
	}
	if themeRole("bg") != "#15181c" || themeRole("accent") != "#5fc4b0" {
		t.Fatalf("the default is Iron & Patina: bg %s accent %s", themeRole("bg"), themeRole("accent"))
	}
	h := panelHost{p: newCtl(newCtlCLI("caminus"), ""), w: 100, h: 30}
	if v := h.View(); !strings.Contains(v.Content, bgSeq("#15181c")) {
		t.Fatal("the panel's host paints the theme's background, as Ember does")
	}
	if strings.Contains(h.p.View(100, 30), bgSeq("#15181c")) {
		t.Fatal("the component leaves the background to its host (Ember paints its own)")
	}
	if err := themeLoad("nord"); err != nil {
		t.Fatal(err)
	}
	if themeCurrent["bg"] != "" && themeCurrent["bg"] == "#15181c" {
		t.Fatal("a theme without a background does not inherit the default's")
	}
}
