// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// a small Qwen Code log: prompt, tool call + result, reply, compaction, thinking
const sessionLog = `{"type":"system","subtype":"slash_command","timestamp":"2026-10-17T10:00:00.000Z","cwd":"/work/me/src/My Proj"}
{"type":"user","timestamp":"2026-10-17T10:00:01.000Z","cwd":"/work/me/src/My Proj","message":{"role":"user","parts":[{"text":"<system-reminder>x</system-reminder>"},{"text":"fix the bug"}]}}
{"type":"assistant","timestamp":"2026-10-17T10:00:05.000Z","message":{"role":"model","parts":[{"text":"hmm","thought":true},{"functionCall":{"id":"c1","name":"read_file","args":{"absolute_path":"C:\\src\\main.go"}}}]},"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":50,"thoughtsTokenCount":20,"cachedContentTokenCount":800},"executionContext":{"modelId":"qwen3.8"}}
{"type":"tool_result","timestamp":"2026-10-17T10:00:06.500Z","message":{"role":"user","parts":[{"functionResponse":{"id":"c1","name":"read_file","response":{"output":"package main"}}}]}}
{"type":"assistant","timestamp":"2026-10-17T10:00:09.000Z","message":{"role":"model","parts":[{"text":"Fixed it."}]},"usageMetadata":{"promptTokenCount":1500,"candidatesTokenCount":10,"cachedContentTokenCount":1000}}
{"type":"system","subtype":"chat_compression","timestamp":"2026-10-17T10:01:00.000Z","systemPayload":{"compressedHistory":[{"parts":[{"text":"summary of everything"}]}]}}
{"type":"assistant","timestamp":"2026-10-17T10:01:05.000Z","message":{"role":"model","parts":[{"functionCall":{"id":"c2","name":"run_shell_command","args":{"command":"go test","description":"Run tests"}}}]},"usageMetadata":{"promptTokenCount":400}}
`

func TestSessionParse(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "projects", "-home-me-src-my-proj", "chats", "a.jsonl")
	write(t, p, sessionLog)
	s := newSession(p, projectName("-home-me-src-my-proj", "/home/me"))
	if s.Project != "my-proj" {
		t.Errorf("dir name: %q", s.Project)
	}
	s.tail()
	if s.Project != "My Proj" {
		t.Errorf("cwd name: %q", s.Project)
	}
	if s.turns != 1 || s.tools != 2 || s.compactions != 1 || s.model != "qwen3.8" {
		t.Errorf("turns %d tools %d compactions %d model %q", s.turns, s.tools, s.compactions, s.model)
	}
	if len(s.prompts) != 3 || !s.compactBefore[2] || s.lastUsage.Prompt != 400 {
		t.Errorf("prompts %v compact %v", s.prompts, s.compactBefore)
	}
	var kinds []string
	for _, e := range s.events {
		kinds = append(kinds, e.Kind+":"+e.Detail)
	}
	want := "user:fix the bug|tool:main.go|reply:Fixed it.|compact:context compacted|tool:Run tests"
	if strings.Join(kinds, "|") != want {
		t.Errorf("events\n got %s\nwant %s", strings.Join(kinds, "|"), want)
	}
	if e := s.events[1]; !e.Done || e.Err || e.DoneT.Sub(e.T) != 1500*time.Millisecond {
		t.Errorf("tool result not matched: %+v", e)
	}
	if s.events[4].Done {
		t.Error("c2 has no result yet")
	}
	est := s.estimate(400)
	sum := 0
	for _, n := range est {
		sum += n
	}
	if sum != 400 || est[catHist] == 0 {
		t.Errorf("estimate %v", est)
	}
}

func TestSessionTailIncremental(t *testing.T) {
	p := filepath.Join(t.TempDir(), "s.jsonl")
	lines := strings.SplitAfter(sessionLog, "\n")
	half := lines[2][:40]
	write(t, p, lines[0]+lines[1]+half)
	s := newSession(p, "x")
	s.tail()
	if s.turns != 1 || s.tools != 0 {
		t.Fatalf("first part: turns %d tools %d", s.turns, s.tools)
	}
	f, _ := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString(lines[2][40:] + lines[3])
	f.Close()
	s.tail()
	if s.tools != 1 || !s.events[1].Done {
		t.Errorf("split line not joined: tools %d", s.tools)
	}
	write(t, p, lines[0]) // rewritten shorter: start over
	s.tail()
	if s.turns != 0 || len(s.events) != 0 {
		t.Errorf("not reset: %d turns", s.turns)
	}
}

func TestProjectName(t *testing.T) {
	cases := []struct{ dir, home, want string }{
		{"c--users-dev-src-rocket-api", `C:\Users\dev`, "rocket-api"},
		{"c--users-dev", `C:\Users\dev`, "home"},
		{"c--users-dev-notes", `C:\Users\dev`, "notes"},
		{"d--work-x", `C:\Users\dev`, "d--work-x"},
		{"c--users-devbot-x", `C:\Users\dev`, "c--users-devbot-x"}, // another user whose name starts the same
		{"-home-me-src-p", "/home/me", "p"},
	}
	for _, c := range cases {
		if got := projectName(c.dir, c.home); got != c.want {
			t.Errorf("projectName(%q, %q) = %q, want %q", c.dir, c.home, got, c.want)
		}
	}
}

func TestScanSessionsAndSavings(t *testing.T) {
	d := t.TempDir()
	now := time.Now()
	for i, name := range []string{"a", "b", "c", "d", "e", "f", "g", "old"} {
		p := filepath.Join(d, "proj-"+name, "chats", name+".jsonl")
		write(t, p, sessionLog)
		mt := now.Add(-time.Duration(i) * time.Minute)
		if name == "old" {
			mt = now.Add(-7 * time.Hour)
		}
		os.Chtimes(p, mt, mt)
	}
	glob := filepath.Join(d, "*", "chats", "*.jsonl")
	st := &state{known: map[string]*session{}}
	st.scanSessions(glob, "/nowhere", now)
	if len(st.sessions) != 6 || filepath.Base(st.sessions[0].Path) != "a.jsonl" {
		t.Fatalf("%d sessions, first %s", len(st.sessions), st.sessions[0].Path)
	}
	l := &ledger{files: map[string]*usageFile{}, days: map[string]*tokTotals{}}
	sv := l.update(glob, time.Date(2026, 10, 17, 23, 0, 0, 0, time.Local))
	if sv.All.Reqs != 24 || sv.All.Prompt != 8*2900 || sv.All.Cached != 8*1800 || sv.All.Out != 8*80 {
		t.Errorf("ledger %+v", sv.All)
	}
	// a frame with sessions renders at every width
	st.cfg, st.base = defaultConfig, "http://127.0.0.1:1"
	st.sav = sv
	for _, w := range []int{80, 90, 120, 200} {
		out := frame(t, st, w, 50)
		if !strings.Contains(out, "My Proj") {
			t.Errorf("%d wide: no session in frame", w)
		}
	}
}

func TestCost(t *testing.T) {
	p := defaultPrices
	api, pow, net := p.cost(tokTotals{Prompt: 2_000_000, Cached: 1_000_000, Out: 100_000}, 1000, 25)
	// 1M fresh at cache-write 2.50 + 1M hits at 0.10 + 0.1M out at 10
	if !near(api, 2.5+0.1+1.0) {
		t.Errorf("api %v", api)
	}
	// (1e6/1000 + 1e5/25) s = 5000 s = 1.389 h * 0.38 kW * 0.17
	if !near(pow, 5000.0/3600*0.38*0.17) || !near(net, api-pow) {
		t.Errorf("power %v net %v", pow, net)
	}
	p.WritePerM = 0
	if api, _, _ := p.cost(tokTotals{Prompt: 1_000_000}, 0, 0); !near(api, 2.0) {
		t.Errorf("no cache write price: %v", api)
	}
}

// A log that is truncated and written again is read from the start; earlier counts stay.
func TestLedgerTruncated(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "p", "chats", "s.jsonl")
	line := `{"type":"assistant","timestamp":"2026-10-17T10:00:05.000Z","usageMetadata":{"promptTokenCount":1000,"cachedContentTokenCount":800}}` + "\n"
	write(t, p, line+line+line)
	glob := filepath.Join(d, "*", "chats", "*.jsonl")
	l := &ledger{files: map[string]*usageFile{}, days: map[string]*tokTotals{}}
	now := time.Date(2026, 10, 17, 23, 0, 0, 0, time.Local)
	if sv := l.update(glob, now); sv.All.Reqs != 3 {
		t.Fatalf("first scan %+v", sv.All)
	}
	write(t, p, line) // rewritten shorter
	if sv := l.update(glob, now); sv.All.Reqs != 4 {
		t.Errorf("after truncate %+v", sv.All)
	}
	write(t, p, line+line)
	if sv := l.update(glob, now); sv.All.Reqs != 5 || sv.All.Prompt != 5000 {
		t.Errorf("after regrow %+v", sv.All)
	}
}

func TestSavingsDays(t *testing.T) {
	l := &ledger{files: map[string]*usageFile{}, days: map[string]*tokTotals{}}
	first := time.Date(2026, 10, 10, 22, 0, 0, 0, time.Local)
	l.first = first
	l.days[first.Format("2006-01-02")] = &tokTotals{Reqs: 1}
	l.days["2026-10-17"] = &tokTotals{Reqs: 2} // a week with no use in between
	sv := l.update(filepath.Join(t.TempDir(), "none"), time.Date(2026, 10, 17, 9, 0, 0, 0, time.Local))
	if sv.Days != 8 {
		t.Errorf("days %d, want 8 (Oct 10..17)", sv.Days)
	}
	if sv.Today.Reqs != 2 {
		t.Errorf("today %+v", sv.Today)
	}
	// "today" is the local date of now, whatever zone now carries
	if sv := l.update("none", time.Date(2026, 10, 17, 9, 0, 0, 0, time.Local).UTC()); sv.Today.Reqs != 2 {
		t.Errorf("now in UTC: today %+v", sv.Today)
	}
}

func TestCostGuards(t *testing.T) {
	p := defaultPrices
	p.ReadTokSec, p.WriteTokSec = 0, -1
	for _, tt := range []tokTotals{{}, {Prompt: 1000, Cached: 5000, Out: 10}, {Prompt: 1_000_000, Out: 1000}} {
		api, pow, net := p.cost(tt, 0, 0)
		for _, x := range []float64{api, pow, net} {
			if math.IsNaN(x) || math.IsInf(x, 0) || x < -1e-9 && x != net {
				t.Errorf("%+v: api %v power %v net %v", tt, api, pow, net)
			}
		}
		if api < 0 {
			t.Errorf("%+v: negative api cost %v", tt, api)
		}
	}
	p = defaultPrices
	p.InPerM, p.WritePerM, p.OutPerM, p.BusyWatts = -5, 0, -1, -100
	if api, pow, _ := p.cost(tokTotals{Prompt: 1_000_000, Out: 1_000_000}, 1000, 25); api != 0 || pow != 0 {
		t.Errorf("negative prices: api %v power %v", api, pow)
	}
}

// Right after a compaction (no usage line yet) the estimate isn't scaled to the size before it: that put nearly
// the whole old prompt into "system + tools".
func TestEstimateRightAfterCompaction(t *testing.T) {
	p := filepath.Join(t.TempDir(), "s.jsonl")
	lines := strings.SplitAfter(sessionLog, "\n")
	write(t, p, strings.Join(lines[:6], ""))
	s := newSession(p, "x")
	s.tail()
	if s.lastKind != "compact" || s.lastUsage.Prompt != 0 {
		t.Fatalf("last %q, prompt %d", s.lastKind, s.lastUsage.Prompt)
	}
	if est := s.estimate(s.lastUsage.Prompt); est[catSys] != 0 || est[catHist] == 0 {
		t.Errorf("estimate %v", est)
	}
	// every compaction now renders with no prompt size: no NaN or Inf in CONTEXT
	st := demoState(t)
	s.Mod, s.lastT = time.Now(), time.Now()
	st.sessions[0] = s
	if out := frame(t, st, 120, 50); strings.Contains(out, "NaN") || strings.Contains(out, "Inf") || !strings.Contains(out, "what fills it") {
		t.Errorf("CONTEXT after a compaction:\n%s", out)
	}
}

// A line appended within the same mtime tick as the one before is still read; sessions gone from the last 6 h
// are forgotten.
func TestScanSessionsSameMtimeAndPrune(t *testing.T) {
	d := t.TempDir()
	lines := strings.SplitAfter(sessionLog, "\n")
	p := filepath.Join(d, "proj", "chats", "a.jsonl")
	write(t, p, strings.Join(lines[:3], ""))
	now := time.Now()
	mt := now.Add(-time.Second)
	os.Chtimes(p, mt, mt)
	glob := filepath.Join(d, "*", "chats", "*.jsonl")
	st := &state{known: map[string]*session{}}
	st.scanSessions(glob, "/nowhere", now)
	if st.sessions[0].events[1].Done {
		t.Fatal("tool done before its result")
	}
	f, _ := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString(lines[3])
	f.Close()
	os.Chtimes(p, mt, mt) // same mtime as at the last scan
	st.scanSessions(glob, "/nowhere", now)
	if !st.sessions[0].events[1].Done {
		t.Error("tool result appended in the same mtime tick was missed")
	}
	old := now.Add(-7 * time.Hour)
	os.Chtimes(p, old, old)
	st.scanSessions(glob, "/nowhere", now)
	if len(st.sessions) != 0 || len(st.known) != 0 {
		t.Errorf("%d sessions, %d known after 6 h", len(st.sessions), len(st.known))
	}
}
