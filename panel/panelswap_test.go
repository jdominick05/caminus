// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// stubPanel is a panel on testdata/caminus_stub.sh: a made-up host with one card, two models (alpha, beta) and,
// as files in its state folder, the server that runs. Nothing real starts or stops.
func stubPanel(t *testing.T, hosted bool, files map[string]string) (ctl, string) {
	t.Helper()
	dir := t.TempDir()
	for k, v := range files {
		if err := os.WriteFile(filepath.Join(dir, k), []byte(v), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("STUB_STATE", dir)
	t.Setenv("STUB_DELAY", "0")
	stub, err := filepath.Abs(filepath.Join("testdata", "caminus_stub.sh"))
	if err != nil {
		t.Fatal(err)
	}
	m := newPanel(panelOptions{CLI: "caminus-not-run", Hosted: hosted})
	m.be = &ctlCLI{argv: []string{"sh", stub}, timeout: 10 * time.Second}
	m = settle(m, tea.Batch(m.loadModels(), m.loadPS(), m.loadHarness(), m.loadEmber(), m.loadOpts()), 0)
	if len(m.models) != 2 {
		t.Fatalf("the stub's models: %+v (%v)", m.models, m.readErr)
	}
	return m, dir
}

// actions are the stub's recorded commands that change something (the reads left out), and the number of ps reads
// between the first stop and the first start.
func stubActions(t *testing.T, dir string) ([]string, int) {
	t.Helper()
	b, _ := os.ReadFile(filepath.Join(dir, "calls.log"))
	var acts []string
	psBetween, stopped := 0, false
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		switch f := strings.Fields(l); {
		case len(f) == 0:
		case f[0] == "stop" || f[0] == "start" || f[0] == "restart":
			acts = append(acts, l)
			stopped = stopped || f[0] == "stop"
		case f[0] == "ps" && stopped && !strings.Contains(strings.Join(acts, "\n"), "start "):
			psBetween++
		}
	}
	return acts, psBetween
}

// press sends one key to the component (through Update, as a host does).
func pkey(m ctl, k string) (ctl, tea.Cmd) {
	var msg tea.KeyPressMsg
	switch k {
	case "enter":
		msg = tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		msg = tea.KeyPressMsg{Code: tea.KeyEscape}
	case "down":
		msg = tea.KeyPressMsg{Code: tea.KeyDown}
	default:
		msg = tea.KeyPressMsg{Code: rune(k[0]), Text: k}
	}
	return m.Update(msg)
}

// fits checks that every line of the frame is inside the screen and returns it without colour.
func frameFits(t *testing.T, m ctl, w, h int) string {
	t.Helper()
	v := m.View(w, h)
	ls := strings.Split(v, "\n")
	if len(ls) > h {
		t.Fatalf("%dx%d: %d lines", w, h, len(ls))
	}
	for i, l := range ls {
		if lipgloss.Width(l) > w {
			t.Fatalf("%dx%d line %d: %d cells: %q", w, h, i+1, lipgloss.Width(l), ansi.Strip(l))
		}
	}
	return ansi.Strip(v)
}

func bottomLine(v string) string {
	ls := strings.Split(strings.TrimRight(v, "\n"), "\n")
	return strings.TrimSpace(ls[len(ls)-1])
}

func fastSwap(t *testing.T, wait time.Duration) {
	oldW, oldP := swapWait, swapPoll
	swapWait, swapPoll = wait, 5*time.Millisecond
	t.Cleanup(func() { swapWait, swapPoll = oldW, oldP })
}

// Enter on a model while another server runs asks once (enter / esc), then stops the running server, waits until
// caminus ps shows it gone and its card's memory free, and only then starts the selected model, each step in the
// header. Esc asks nothing more and runs nothing.
func TestEnterStopsTheRunningServerThenStarts(t *testing.T) {
	fastSwap(t, 5*time.Second)
	t.Setenv("STUB_HOLD", "2") // two ps reads after the stop still show the card's memory in use
	m, dir := stubPanel(t, false, map[string]string{"running": "alpha"})
	m, _ = pkey(m, "down")
	if mi := m.selModel(); mi == nil || mi.Name != "beta" {
		t.Fatal("down selects beta")
	}
	for _, sz := range [][2]int{{100, 30}, {80, 24}} {
		if out := frameFits(t, m, sz[0], sz[1]); !strings.Contains(out, "enter stops alpha and starts it (asks first)") {
			t.Fatalf("%dx%d: the details say what enter does:\n%s", sz[0], sz[1], out)
		}
	}
	m, cmd := pkey(m, "enter")
	if cmd != nil || m.confirm == nil || m.confirm.q != "stop alpha and start beta? enter / esc" {
		t.Fatalf("enter asks first: %+v", m.confirm)
	}
	for _, sz := range [][2]int{{100, 30}, {80, 24}} {
		out := frameFits(t, m, sz[0], sz[1])
		if !strings.Contains(out, "? stop alpha and start beta? enter / esc") || bottomLine(out) != "enter yes  esc no" {
			t.Fatalf("%dx%d: the question and its keys:\n%s", sz[0], sz[1], out)
		}
	}
	m, _ = pkey(m, "esc")
	if m.confirm != nil || m.busy != "" || m.note != "cancelled" {
		t.Fatalf("esc cancels: busy %q note %q", m.busy, m.note)
	}
	if acts, _ := stubActions(t, dir); len(acts) != 0 {
		t.Fatalf("esc ran nothing: %v", acts)
	}

	m, _ = pkey(m, "enter")
	m, cmd = pkey(m, "enter")
	if m.swap == nil || m.busy != "stopping alpha (then beta starts)" || cmd == nil {
		t.Fatalf("enter answers yes: busy %q", m.busy)
	}
	var busy []string
	for i := 0; i < 20 && m.swap != nil; i++ { // (one step at a time: each answer through update, as Bubble Tea does)
		msg := cmd()
		m, cmd = m.update(msg)
		busy = append(busy, m.busy)
	}
	if m.swap != nil || m.busy != "starting beta" {
		t.Fatalf("the swap reached the start: busy %q, steps %q", m.busy, busy)
	}
	waited := strings.Join(busy, "\n")
	if !strings.Contains(waited, "waiting: XTX still has 19.3 GB in use") {
		t.Fatalf("the header says what it waits for: %q", busy)
	}
	out := frameFits(t, m, 80, 24)
	if !strings.Contains(out, "starting beta") {
		t.Fatalf("the header shows the start:\n%s", out)
	}
	m = settle(m, cmd, 0)
	if m.busy != "" || m.noteErr || m.note != "alpha stopped · start beta: beta on XTX at http://127.0.0.1:59991" {
		t.Fatalf("the start's answer: busy %q note %q", m.busy, m.note)
	}
	acts, psBetween := stubActions(t, dir)
	if strings.Join(acts, "; ") != "stop alpha --json; start beta --json" || psBetween < 3 {
		t.Fatalf("stop, then ps until the card is free (%d reads), then start: %v", psBetween, acts)
	}
	if sv := m.primary(); sv == nil || sv.Model != "beta" {
		t.Fatalf("beta runs now: %+v", sv)
	}
}

// A stop that fails ends the swap there: the note names the step and its reason, and nothing is started.
func TestSwapStopFailureStartsNothing(t *testing.T) {
	fastSwap(t, 5*time.Second)
	m, dir := stubPanel(t, false, map[string]string{"running": "alpha", "fail_stop": ""})
	m, _ = pkey(m, "down")
	m, _ = pkey(m, "enter")
	m, cmd := pkey(m, "enter")
	m = settle(m, cmd, 0)
	want := "beta not started: couldn't stop alpha: alpha still running after 170 s (guard pid 4242): caminus logs alpha"
	if m.swap != nil || m.busy != "" || !m.noteErr || m.note != want {
		t.Fatalf("a failed stop: busy %q note %q", m.busy, m.note)
	}
	if acts, _ := stubActions(t, dir); strings.Join(acts, "; ") != "stop alpha --json" {
		t.Fatalf("no start after a failed stop: %v", acts)
	}
}

// A card that stays busy after the stop ends the swap when the wait runs out, with the card and its memory.
func TestSwapWaitRunsOut(t *testing.T) {
	fastSwap(t, 100*time.Millisecond)
	t.Setenv("STUB_HOLD", "100000")
	m, dir := stubPanel(t, false, map[string]string{"running": "alpha"})
	m, _ = pkey(m, "down")
	m, _ = pkey(m, "enter")
	m, cmd := pkey(m, "enter")
	for i := 0; i < 200 && m.swap != nil; i++ {
		m, cmd = m.update(cmd())
	}
	if m.swap != nil || !m.noteErr || !strings.HasPrefix(m.note, "beta not started: alpha stopped, but after ") ||
		!strings.HasSuffix(m.note, "XTX still has 19.3 GB in use") {
		t.Fatalf("the wait ran out: note %q", m.note)
	}
	if acts, _ := stubActions(t, dir); strings.Join(acts, "; ") != "stop alpha --json" {
		t.Fatalf("no start while the card is busy: %v", acts)
	}
	if out := frameFits(t, m, 80, 24); !strings.Contains(out, "XTX still has 19.3 GB") {
		t.Fatalf("the note keeps the reason at 80 columns:\n%s", out)
	}
}

// A start that fails after the stop says both: the server was stopped, and why the new one did not start.
func TestSwapStartFailure(t *testing.T) {
	fastSwap(t, 5*time.Second)
	t.Setenv("STUB_HOLD", "0")
	m, _ := stubPanel(t, false, map[string]string{"running": "alpha", "fail_start": ""})
	m, _ = pkey(m, "down")
	m, _ = pkey(m, "enter")
	m, cmd := pkey(m, "enter")
	m = settle(m, cmd, 0)
	if !m.noteErr || m.note != "alpha stopped · start beta: beta exited during startup: out of memory" {
		t.Fatalf("a failed start: %q", m.note)
	}
}

// A server started outside the panel is never stopped by it (caminus stop only knows its own): s and enter say the
// command that stops it (kill and the pid caminus ps found), else where it was started. Nothing runs against it.
func TestOutsideServerSaysWhatStopsIt(t *testing.T) {
	sleep := exec.Command("sleep", "60")
	if err := sleep.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sleep.Process.Kill(); _ = sleep.Wait() }()
	pid := strconv.Itoa(sleep.Process.Pid)

	m, dir := stubPanel(t, false, map[string]string{"ext_pid": pid})
	if sv := m.primary(); sv == nil || !sv.External || sv.Model != "alpha" {
		t.Fatalf("the stub's outside server: %+v", sv)
	}
	if line := bottomLine(frameFits(t, m, 100, 30)); !strings.Contains(line, "s stop") {
		t.Fatalf("the key line offers s on an outside server: %q", line)
	}
	n, cmd := pkey(m, "s")
	want := "kill " + pid + " stops alpha (started outside the panel, so caminus stop can't)"
	if cmd != nil || n.confirm != nil || n.note != want {
		t.Fatalf("s: %q, want %q", n.note, want)
	}
	for _, sz := range [][2]int{{100, 30}, {80, 24}} {
		if out := frameFits(t, n, sz[0], sz[1]); !strings.Contains(out, "kill "+pid+" stops alpha") {
			t.Fatalf("%dx%d: the command survives the note's width:\n%s", sz[0], sz[1], out)
		}
	}
	m, _ = pkey(m, "down")
	if out := frameFits(t, m, 100, 30); !strings.Contains(out, "enter starts it once alpha (started outside the panel) is stopped") {
		t.Fatalf("beta's details say enter waits for the outside server:\n%s", out)
	}
	n, cmd = pkey(m, "enter")
	want = "kill " + pid + " stops alpha first, then enter starts beta (started outside the panel, so caminus stop can't)"
	if cmd != nil || n.confirm != nil || n.note != want || !n.noteErr {
		t.Fatalf("enter with an outside server: %q, want %q", n.note, want)
	}
	if acts, _ := stubActions(t, dir); len(acts) != 0 {
		t.Fatalf("nothing ran against the outside server: %v", acts)
	}
	if err := sleep.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("the outside process was never signalled: %v", err)
	}
	// no pid from caminus ps: no command at all (kill 0 would signal a whole process group)
	if outsideStop(serverInfo{External: true}) != "" {
		t.Fatal("no pid, no command")
	}
	if got := outsideStopNote(serverInfo{External: true, Model: "alpha"}, "first, then enter starts beta"); got !=
		"alpha was started outside the panel: stop it where it was started, then enter starts beta" {
		t.Fatalf("enter's note without a pid: %q", got)
	}
}

// s stops a server the panel started after one y; inside Ember (/panel) the same keys do the same, and esc or q
// can't close the panel halfway through a swap (the start would be lost).
func TestStopAndSwapInsideEmber(t *testing.T) {
	fastSwap(t, 5*time.Second)
	t.Setenv("STUB_HOLD", "1")
	p, dir := stubPanel(t, true, map[string]string{"running": "alpha"})
	d := dummyHost{panel: p, open: true}
	key := func(k string) tea.Cmd {
		var c tea.Cmd
		d.panel, c = pkey(d.panel, k)
		return c
	}
	key("down")
	key("enter")
	c := key("enter")
	if d.panel.swap == nil {
		t.Fatal("the swap runs hosted")
	}
	for _, k := range []string{"esc", "q"} {
		if cc := key(k); cc != nil || !strings.HasPrefix(d.panel.note, "wait: stopping alpha") {
			t.Fatalf("%s during a swap closes nothing: %q", k, d.panel.note)
		}
	}
	for i := 0; i < 20 && d.panel.swap != nil; i++ {
		d.panel, c = d.panel.update(c())
	}
	d.panel = settle(d.panel, c, 0)
	if acts, _ := stubActions(t, dir); strings.Join(acts, "; ") != "stop alpha --json; start beta --json" {
		t.Fatalf("hosted: stop then start: %v", acts)
	}
	// s on the server the panel started: one y, then caminus stop
	c = key("s")
	if c != nil || d.panel.confirm == nil || d.panel.confirm.q != "stop beta? y / n" {
		t.Fatalf("s asks: %+v", d.panel.confirm)
	}
	if line := bottomLine(frameFits(t, d.panel, 80, 24)); line != "y yes  any other key no" {
		t.Fatalf("the y / n keys: %q", line)
	}
	c = key("y")
	d.panel = settle(d.panel, c, 0)
	if acts, _ := stubActions(t, dir); acts[len(acts)-1] != "stop beta --json" || d.panel.note != "stop beta: stopped beta" {
		t.Fatalf("s stopped beta: %v, note %q", acts, d.panel.note)
	}
	if c = key("esc"); c == nil {
		t.Fatal("esc closes once nothing runs")
	}
	d, _ = d.Update(c())
	if d.open {
		t.Fatal("the host caught the close")
	}
}

// runEmber feeds a command's messages into Ember, one at a time, the way Bubble Tea does: a batch's commands each run,
// and one that does not answer within a second (a tick of Ember's own) is left out.
func runEmber(m emberModel, c tea.Cmd, depth int) emberModel {
	if c == nil || depth > 40 {
		return m
	}
	msg, ok := callWithin(c, time.Second)
	if !ok {
		return m
	}
	if b, is := msg.(tea.BatchMsg); is {
		for _, sub := range b {
			m = runEmber(m, sub, depth+1)
		}
		return m
	}
	next, nc := m.Update(msg)
	return runEmber(next.(emberModel), nc, depth+1)
}

func callWithin(c tea.Cmd, d time.Duration) (tea.Msg, bool) {
	ch := make(chan tea.Msg, 1)
	go func() { ch <- c() }()
	select {
	case msg := <-ch:
		return msg, true
	case <-time.After(d):
		return nil, false
	}
}

// /panel inside Ember: enter's question and the swap's steps are the same as in caminus panel, with the keys going
// through Ember, and esc can't close the panel halfway.
func TestSwapThroughEmbersPanel(t *testing.T) {
	fastSwap(t, 5*time.Second)
	t.Setenv("STUB_HOLD", "1")
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	p, dir := stubPanel(t, true, map[string]string{"running": "alpha"})
	m := chatFixture(t)
	m.width, m.height = 100, 30
	m.cli = p.be.argv
	m, _ = m.openPanel()
	m.hosted = settle(m.hosted, tea.Batch(m.hosted.loadModels(), m.hosted.loadPS(), m.hosted.loadOpts()), 0)
	send := func(k tea.KeyPressMsg) tea.Cmd {
		next, c := m.Update(k)
		m = next.(emberModel)
		return c
	}
	send(tea.KeyPressMsg{Code: tea.KeyDown})
	send(tea.KeyPressMsg{Code: tea.KeyEnter})
	if m.hosted.confirm == nil || m.hosted.confirm.q != "stop alpha and start beta? enter / esc" {
		t.Fatalf("enter asks inside Ember: %+v", m.hosted.confirm)
	}
	if v := ansi.Strip(m.render()); !strings.Contains(v, "stop alpha and start beta? enter / esc") {
		t.Fatalf("Ember's frame shows the question:\n%s", v)
	}
	c := send(tea.KeyPressMsg{Code: tea.KeyEnter})
	send(tea.KeyPressMsg{Code: tea.KeyEscape})
	if !m.hostedOpen || !strings.HasPrefix(m.hosted.note, "wait: ") {
		t.Fatalf("esc mid-swap keeps the panel open: open %v note %q", m.hostedOpen, m.hosted.note)
	}
	m = runEmber(m, c, 0)
	if acts, _ := stubActions(t, dir); strings.Join(acts, "; ") != "stop alpha --json; start beta --json" {
		t.Fatalf("Ember's /panel: stop then start: %v (busy %q note %q)", acts, m.hosted.busy, m.hosted.note)
	}
	if m.hosted.note != "alpha stopped · start beta: beta on XTX at http://127.0.0.1:59991" || !m.hostedOpen {
		t.Fatalf("the start's answer, the panel still open: %q", m.hosted.note)
	}
}
