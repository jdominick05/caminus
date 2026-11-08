// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// TestRenderAllSizes renders every terminal size from 1x1 up, for a server that is down with no sessions, a busy one
// with demo sessions, a session with hundreds of requests, two cards and the Caminus panels. frame() fails on any line
// wider than the terminal; a panic fails the test.
func TestRenderAllSizes(t *testing.T) {
	down := &state{cfg: defaultConfig, base: defaultURL, known: map[string]*session{}}
	busy := demoState(t)
	busy.srv.Slot = slotJSON{NCtx: 131072, Processing: true, Task: 7, NPrompt: 61400, NProcessed: 9000, NCache: 52800}
	busy.srv.Busy, busy.srv.curRead = 1, 2400
	many := demoState(t)
	s := many.sessions[0]
	for i := 0; i < 300; i++ {
		s.prompts = append(s.prompts, 1000+i*300)
		if i%50 == 0 {
			s.compactBefore[len(s.prompts)-1] = true
		}
	}
	dual := demoState(t)
	dual.sys.GPUs = nil
	dual.sys.useCards(dualCards([4]float64{15032385536, 13958643712, 343932928, 35}, [4]float64{12884901888, 11811160064, 322122547, 30}))
	if f := ansi.Strip(frame(t, dual, 120, 40)); !strings.Contains(f, "VRAM2") || !strings.Contains(f, "1 RX 7900 XTX · 2 RX 7900 XT · model on 1+2") {
		t.Errorf("two cards:\n%s", f)
	}
	cam := camState(t, "caminus_status_busy.json")
	cam.srv.Slot.Processing, cam.srv.Busy = true, 1
	camLone := camState(t, "caminus_status.json")
	camLone.sessions = nil
	camNoHW := camState(t, "caminus_status_nohw.json")
	camNoHW.srv.Cam.Agent.Loop.Live, camNoHW.srv.Cam.Agent.Loop.Threshold = 4, 3
	for name, st := range map[string]*state{"down": down, "busy": busy, "many": many, "dual": dual,
		"caminus": cam, "caminus alone": camLone, "caminus without hardware": camNoHW} {
		for w := 1; w <= 130; w++ {
			for _, h := range []int{1, 2, 5, 12, 30, 60} {
				frame(t, st, w, h)
			}
		}
		for h := 1; h <= 60; h++ {
			frame(t, st, 50, h)
		}
		if t.Failed() {
			t.Fatalf("%s: see above", name)
		}
	}
}

func TestBoxCut(t *testing.T) {
	rows := func(b *box) string { return strings.Join(b.rows, "|") }
	// a whole section goes rather than half of one, and no blank line is left at the end
	b := &box{rows: []string{"gauge", "legend", "", "waffle1", "waffle2", "waffle3"}}
	if n := b.cut(1, 0); n != 4 || rows(b) != "gauge|legend" {
		t.Errorf("section cut: took %d, left %q", n, rows(b))
	}
	// floor: rows that stay; the section rule never goes below it
	b = &box{rows: []string{"gauge", "legend", "", "a", "b"}}
	if b.cut(10, 2); rows(b) != "gauge|legend" {
		t.Errorf("floor 2: left %q", rows(b))
	}
	// the kept tail stays while the list shrinks; the blank line before it isn't doubled
	b = &box{rows: []string{"s1", "s2", "s3", "", "now"}, keep: 2}
	if b.cut(2, 0); rows(b) != "s1||now" {
		t.Errorf("keep: left %q", rows(b))
	}
	b = &box{rows: []string{"s1", "", "now"}, keep: 2}
	if b.cut(1, 0); rows(b) != "now" {
		t.Errorf("list gone: left %q", rows(b))
	}
	// cutting everything takes the border too
	b = &box{rows: []string{"x", "y"}}
	if n := b.cut(2, 0); n != 4 || b.height() != 0 || b.render() != "" {
		t.Errorf("all: took %d, height %d", n, b.height())
	}
	// shrink honours min before cutting further, in order
	a, c := &box{rows: []string{"1", "2", "3", "4"}, min: 1}, &box{rows: []string{"1", "2", "3", "4"}, min: 3}
	shrink(8, a, c)
	if rows(a) != "1" || len(c.rows) != 3 {
		t.Errorf("shrink: %q %q", rows(a), rows(c))
	}
}

func TestColumnsDegenerate(t *testing.T) {
	for _, w := range []int{-3, 0, 1} {
		for _, h := range []int{0, 1, 5} {
			out := columns([]float64{1, 2, 3}, []bool{false, true, false}, w, h, 3, cAccent, cRed)
			if len(out) != h {
				t.Errorf("columns(w=%d, h=%d) gave %d rows", w, h, len(out))
			}
		}
	}
	if out := columns(nil, nil, -1, 5, 1, cAccent, cRed); len(out) != 5 {
		t.Errorf("no values: %d rows", len(out))
	}
}

// The session list is re-sorted by activity every scan; the session picked with tab must stay picked.
func TestSelectionSticks(t *testing.T) {
	st := demoState(t)
	m := model{st: st, w: 120, h: 40, sel: new(string)}
	m.pick(1)
	if *m.sel != st.sessions[1].Path {
		t.Fatalf("tab picked %q", *m.sel)
	}
	want := st.sessions[1]
	st.sessions[0], st.sessions[1] = st.sessions[1], st.sessions[0] // another agent became the newest
	if got := st.sessions[st.selIndex(*m.sel)]; got != want {
		t.Errorf("selection moved to %s", got.Project)
	}
	m.pick(1)
	if *m.sel != st.sessions[1].Path {
		t.Errorf("tab after re-sort picked %q", *m.sel)
	}
	m.pick(-2)
	if *m.sel != st.sessions[2].Path {
		t.Errorf("shift+tab wrap picked %q", *m.sel)
	}
	*m.sel = "gone"
	if st.selIndex(*m.sel) != 0 {
		t.Error("a vanished session should fall back to the newest")
	}
}

// With no live model every panel says why, instead of zeros or the last picture as if it were current.
func TestOfflineStates(t *testing.T) {
	cases := []struct {
		name string
		srv  serverState
		pill string
		why  string
	}{
		{"before the first poll", serverState{}, "CONNECTING", "looking for the server"},
		{"no server, its process still there", serverState{Polled: true}, "DOWN", "caminus is running but not answering"},
		{"loading", serverState{Polled: true, Loading: true}, "LOADING", "model loading"},
	}
	for _, c := range cases {
		st := demoState(t)
		st.srv = c.srv
		st.srv.ReadRate, st.srv.GenRate = ring{}, ring{}
		st.sessions[0].lastT = time.Now().Add(-27 * time.Minute)
		out := frame(t, st, 120, 50)
		for _, want := range []string{c.pill, "no live context: " + c.why, "▸ " + c.why, "— tok/s", "what filled it · rocket-api · 27m ago"} {
			if !strings.Contains(out, want) {
				t.Errorf("%s: missing %q", c.name, want)
			}
		}
		if strings.Contains(out, "GPU-bound") || strings.Contains(out, "idle · no fallbacks") || strings.Contains(out, "131k?") {
			t.Errorf("%s: a verdict or context size for a model that isn't there", c.name)
		}
	}
	// no server and no process
	st0 := demoState(t)
	st0.srv, st0.sys.Pid = serverState{Polled: true}, 0
	if out := frame(t, st0, 120, 50); !strings.Contains(out, "no live context: no server running") || !strings.Contains(out, "▸ no server running") {
		t.Errorf("no server, no process:\n%s", out)
	}
	// live and recent: no age, real numbers
	st := demoState(t)
	st.sessions[0].lastT = time.Now()
	out := frame(t, st, 120, 50)
	if strings.Contains(out, "no live context") || strings.Contains(out, "ago ·") || strings.Contains(out, "— tok/s") {
		t.Errorf("live frame marked offline")
	}
}

// mesloGlyphs is every non-ASCII character llm-top draws, each checked to exist in MesloLGS NF (the recommended
// font). A glyph the font lacks is drawn from a fallback font, often at the wrong width, which breaks the boxes.
// Adding a glyph here means checking it is in MesloLGS NF first.
const mesloGlyphs = "·—•…⇣⇶−≈⌕─│╭╮╯╰▁▂▃▄▅▆▇█░■▤▶▸◂◆●◧◴◵◶◷★☑✂✎✓✗✚✦❯"

func TestGlyphsInFont(t *testing.T) {
	ok := map[rune]bool{}
	for _, r := range mesloGlyphs {
		ok[r] = true
	}
	busy := demoState(t)
	busy.srv.Slot = slotJSON{NCtx: 131072, Processing: true, Task: 7, NPrompt: 61400, NProcessed: 9000, NCache: 52800}
	busy.srv.Busy = 1
	comp := demoState(t)
	comp.srv.Slot = slotJSON{NCtx: 131072, Processing: true, Task: 8, NPrompt: 61400, NProcessed: 61400}
	comp.srv.Slot.Params.NPredict = compactBudget
	comp.srv.Busy, comp.srv.Decoded = 1, 300
	hold := demoState(t)
	hold.sys.Hold = "gpu job"
	cam := camState(t, "caminus_status_busy.json")
	cam.sessions = nil
	cs := cam.srv.Cam
	cs.Reliability.Bails, cs.Reliability.Errors = map[string]float64{"failed": 1}, map[string]float64{"engine": 1}
	cs.Agent.Failures, cs.Agent.Loop.Live, cs.Agent.Loop.Flagged = map[string]float64{"cut": 1}, 9, 1
	cam.srv.Slot.Processing, cam.srv.Busy = true, 1
	states := []*state{busy, comp, hold, cam, camState(t, "caminus_status_nohw.json"),
		{cfg: defaultConfig, base: defaultURL, known: map[string]*session{}}}
	for _, k := range []string{"run_shell_command", "read_file", "edit", "write_file", "grep_search", "list_directory",
		"web_fetch", "ask_user_question", "task", "todo_write", "save_memory", "skill", "mcp__x__y"} {
		ic, _ := iconFor(event{Kind: "tool", Name: k})
		for _, r := range ic {
			if r > 127 && !ok[r] {
				t.Errorf("tool %s icon %q (U+%04X) is not in MesloLGS NF", k, r, r)
			}
		}
	}
	for _, st := range states {
		for f := 0; f < len(spin); f++ {
			out := model{st: st, w: 140, h: 50, frame: f}.render()
			for _, r := range ansi.Strip(out) {
				if r > 127 && !ok[r] {
					t.Errorf("frame draws %q (U+%04X), not in MesloLGS NF", r, r)
					ok[r] = true // once
				}
			}
		}
	}
}

// An agent session that has been quiet for a while is history: CONTEXT and ACTIVITY don't show it until it is
// picked with tab, and then say how old it is.
func TestStaleSessionNotShown(t *testing.T) {
	st := demoState(t)
	for _, s := range st.sessions {
		s.Mod, s.lastT = time.Now().Add(-48*time.Minute), time.Now().Add(-48*time.Minute)
	}
	m := model{st: st, w: 120, h: 50, sel: new(string)}
	out := frame(t, st, 120, 50)
	if !strings.Contains(out, "no agent active in the last 10 min") || strings.Contains(out, "what fill") ||
		strings.Contains(out, "read_file") || !strings.Contains(out, "no agent active · tab shows earlier ones") {
		t.Errorf("stale session shown:\n%s", out)
	}
	m.pick(1)
	if *m.sel != st.sessions[0].Path {
		t.Fatalf("first tab picked %q, want the newest", *m.sel)
	}
	out = ansi.Strip(m.render())
	if !strings.Contains(out, "what filled it · rocket-api · 48m ago") || !strings.Contains(out, "read_file") {
		t.Errorf("picked session not shown:\n%s", out)
	}
	// busy server: the newest session stays shown even if its log is old (a long request writes nothing)
	st2 := demoState(t)
	st2.sessions[0].Mod = time.Now().Add(-30 * time.Minute)
	st2.srv.Slot.Processing, st2.srv.Busy = true, 1
	if out := frame(t, st2, 120, 50); !strings.Contains(out, "rocket-api ·") {
		t.Errorf("busy server lost its session")
	}
}
