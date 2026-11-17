// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// The server list's name column: a server takes its model's name, and a long one (qwen3-coder-next: 16 characters)
// stays whole and apart from the model column, the model columns of all rows line up, a list of short names keeps
// the old 10-cell column, and the panel still fits an 80-column terminal.
func TestServerRowNames(t *testing.T) {
	tps := 136.0
	srv := func(name string) serverInfo {
		return serverInfo{Name: name, Model: name, State: "ready", URL: "http://127.0.0.1:8092", MaxCtx: 262144, DecodeTPS: &tps}
	}
	modelAt := func(row, model string) int { // the cell where the model column starts
		plain := ansi.Strip(row)
		dot := strings.Index(plain, "● ")
		i := strings.Index(plain[dot+len("● "):], " "+model+" ")
		if dot < 0 || i < 0 {
			t.Fatalf("no model %q in %q", model, plain)
		}
		return lipgloss.Width(plain[:dot+len("● ")+i+1])
	}
	m := newCtl(newCtlCLI("caminus"), false, "")
	m.w, m.h = 80, 40
	m.loaded["ps"] = true

	m.ps = psInfo{Servers: []serverInfo{srv("qwen3-coder-next"), srv("qwen3.6")}}
	rows := m.serverRows()
	if p := ansi.Strip(rows[0]); !strings.Contains(p, "qwen3-coder-next qwen3-coder-next") || strings.Contains(p, "…") {
		t.Fatalf("a 16-character name is whole and apart from its model: %q", p)
	}
	if a, b := modelAt(rows[0], "qwen3-coder-next"), modelAt(rows[1], "qwen3.6"); a != b {
		t.Fatalf("the model columns line up: %d and %d", a, b)
	}

	m.ps = psInfo{Servers: []serverInfo{srv("qwen3.6")}}
	// the cursor's 2 cells, "● ", then the name in 10 cells (9 and its space), as before
	if got := modelAt(m.serverRows()[0], "qwen3.6"); got != 2+2+10 {
		t.Fatalf("short names keep the 10-cell column: the model starts at cell %d", got)
	}

	m.ps = psInfo{Servers: []serverInfo{srv("qwen3-coder-next"), srv("thinkingcap")}}
	for _, l := range strings.Split(m.render(), "\n") {
		if w := lipgloss.Width(l); w > 80 {
			t.Fatalf("a line of %d cells in an 80-column terminal: %q", w, ansi.Strip(l))
		}
	}
}

func TestThePanelHasAWayIntoEmber(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	m.loaded["harness"] = true
	m.ember = emberMsg{known: true}
	if v := ansi.Strip(strings.Join(m.harnessRows(), "\n")); !strings.Contains(v, "Ember not set up") || !strings.Contains(v, "caminus ember setup") {
		t.Fatalf("not set up, the row says how:\n%s", v)
	}
	next, _ := m.harnessKey("e")
	if n := next.(ctl); !strings.Contains(n.note, "caminus ember setup") {
		t.Fatalf("e without Ember names the setup: %q", n.note)
	}
	m.ember = emberMsg{ok: true, known: true}
	if v := ansi.Strip(strings.Join(m.harnessRows(), "\n")); !strings.Contains(v, "e opens the agent chat") {
		t.Fatalf("set up, the row offers it:\n%s", v)
	}
	if _, cmd := m.harnessKey("e"); cmd == nil {
		t.Fatal("e opens caminus ember ui")
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
	m := newCtl(newCtlCLI("caminus"), false, "")
	m.w, m.h = 100, 30
	if v := m.View(); !strings.Contains(v.Content, bgSeq("#15181c")) {
		t.Fatal("the panel paints the theme's background, as Ember does")
	}
	if err := themeLoad("nord"); err != nil {
		t.Fatal(err)
	}
	if themeCurrent["bg"] != "" && themeCurrent["bg"] == "#15181c" {
		t.Fatal("a theme without a background does not inherit the default's")
	}
}
