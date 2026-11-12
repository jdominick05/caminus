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
