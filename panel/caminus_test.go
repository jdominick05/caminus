// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"strings"
	"testing"
)

// The caminus_* fixtures were captured from Caminus' mock server (tools/k3_serve.py --draft --hw-fixture: synthetic
// engine numbers and a synthetic two-card sysfs tree) under synthetic traffic; paths and the build hash replaced.

// camMock answers like the Caminus mock server, idle.
func camMock(t *testing.T) *mock {
	return &mock{props: readFixture(t, "caminus_props.json"), metrics: readFixture(t, "caminus_metrics.txt"),
		slots:  strings.Replace(readFixture(t, "caminus_slots.json"), `"is_processing":true`, `"is_processing":false`, 1),
		status: readFixture(t, "caminus_status.json")}
}

// camBusy makes it busy: a request runs, /slots and the status say so.
func camBusy(t *testing.T, m *mock) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metrics = strings.Replace(readFixture(t, "caminus_metrics.txt"), "llamacpp:requests_processing 0", "llamacpp:requests_processing 1", 1)
	m.slots, m.status = readFixture(t, "caminus_slots.json"), readFixture(t, "caminus_status_busy.json")
}

// camState is the demo state watching a Caminus server whose status is the fixture file.
func camState(t *testing.T, status string) *state {
	t.Helper()
	st := demoState(t)
	cs, err := parseCam([]byte(readFixture(t, status)))
	if err != nil {
		t.Fatal(err)
	}
	var props propsJSON
	if err := json.Unmarshal([]byte(readFixture(t, "caminus_props.json")), &props); err != nil {
		t.Fatal(err)
	}
	var slots []slotJSON
	if err := json.Unmarshal([]byte(readFixture(t, "caminus_slots.json")), &slots); err != nil {
		t.Fatal(err)
	}
	sl := slots[0]
	sl.Processing = false
	st.srv = serverState{Base: "http://127.0.0.1:8092", Polled: true, Up: true, HaveSlots: true, NSlots: 1,
		Props: props, Slot: sl, Decoded: sl.next().NDecoded, Cam: cs, CamCode: 200,
		Metrics: parseMetrics(strings.NewReader(readFixture(t, "caminus_metrics.txt")))}
	st.base = st.srv.Base
	st.sys.VramGB, st.sys.VramTot = 35.7, 44
	st.sys.GPUs = []gpuCard{{Name: "AMD Radeon RX 7900 XTX", Used: 20.2, Tot: 24, Proc: 19.1, Util: 97, Model: true},
		{Name: "AMD Radeon RX 7900 XT", Used: 15.5, Tot: 20, Proc: 14.6, Util: 95, Model: true}}
	return st
}

// ---------- the status ----------

func TestParseCam(t *testing.T) {
	cs, err := parseCam([]byte(readFixture(t, "caminus_status.json")))
	if err != nil {
		t.Fatal(err)
	}
	if f, _ := val(cs.Physics.FloorMs); math.Abs(f-4.0863) > 1e-9 || cs.Physics.TokenPct == nil || cs.Decode.PerPass.Mean == nil {
		t.Errorf("physics %+v", cs.Physics)
	}
	if p := cs.DFlash.Classes["prose"]; p.Drafts == 0 || p.Acceptance == nil || len(p.PerPosRate) == 0 || cs.DFlash.Classes["tool"].Acceptance != nil {
		t.Errorf("dflash classes %+v", cs.DFlash.Classes)
	}
	if cs.Cache.Tokens["vram"] == 0 || cs.Cache.Tokens["disk"] == 0 || cs.Cache.Restore.P95 == nil || cs.Cache.Disk.CapGB == nil {
		t.Errorf("cache %+v", cs.Cache)
	}
	if len(cs.Cards) != 2 || cs.Cards[0].Name != "XTX" || cs.Cards[0].PowerW == nil || cs.Cards[1].PassMs.P50 == nil || cs.Cards[0].Margin == nil {
		t.Errorf("cards %+v", cs.Cards)
	}
	if cs.Context.Attention != "dense" || len(cs.Context.KVBytes) != 2 || cs.Agent.Harness["qwen-code"] == 0 || cs.Agent.Loop.Threshold != 3 {
		t.Errorf("context %+v agent %+v", cs.Context, cs.Agent)
	}
	// without a hardware reader and a drafter: no power, no drafts, the engine's per-card timings still there
	cs, err = parseCam([]byte(readFixture(t, "caminus_status_nohw.json")))
	if err != nil {
		t.Fatal(err)
	}
	if cs.Cards[0].PowerW != nil || cs.Cards[0].PassMs.P50 == nil || cs.Server.Spec != "none" || cs.DFlash.Drafts != 0 || cs.Energy.TPJ.Decode != nil {
		t.Errorf("no hardware: %+v", cs)
	}
	for in, want := range map[string]string{`{"version":2,"physics":{}}`: "status version 2, this llm-top reads 1",
		`{"engine":"caminus"}`: "status version none", `<html>`: "unreadable status"} {
		if _, err := parseCam([]byte(in)); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v", in, err)
		}
	}
	// a field of another type only loses that field (its pointer may be left at zero: the floor must be above 0)
	cs, err = parseCam([]byte(`{"version":1,"physics":{"floor_ms":"fast"},"cache":{"tokens":{"vram":5}}}`))
	if err != nil || cs.Cache.Tokens["vram"] != 5 {
		t.Errorf("drift: %v %+v", err, cs)
	}
	st := camState(t, "caminus_status.json")
	st.srv.Cam = cs
	if f := frame(t, st, 120, 40); strings.Contains(f, "floor") || !strings.Contains(f, "cache   5 reused") {
		t.Errorf("drifted status:\n%s", f)
	}
}

// ---------- the client ----------

// The first round reads /props' engine marker and the status; then the status comes at /props' pace while idle
// (every 2 s) and every round while a request runs, /slots every round either way.
func TestCaminusPolling(t *testing.T) {
	m := camMock(t)
	srv := httptest.NewServer(m)
	defer srv.Close()
	st := run(t, srv.URL, 1, nil)
	if st.srv.NotCam || st.srv.Cam == nil || m.count("/caminus/status") != 1 {
		t.Fatalf("first round: not caminus %v, status %v, %d GETs", st.srv.NotCam, st.srv.Cam != nil, m.count("/caminus/status"))
	}
	slots := m.count("/slots")
	st = run(t, srv.URL, 12, nil) // another client: 6 s idle
	if n, s := m.count("/caminus/status")-1, m.count("/slots")-slots; n != 3 || s != 12 {
		t.Errorf("idle: %d status and %d /slots GETs in 6 s, want 3 and 12", n, s)
	}
	if n := m.count("!"); n != 0 {
		t.Errorf("%d non-GET requests", n)
	}
	f := frame(t, st, 120, 40)
	for _, want := range []string{"k3-coder · caminus-mock · :", "CAMINUS", "of 4.09 ms floor", "engine  dense attention", "build   caminus-0123abc"} {
		if !strings.Contains(f, want) {
			t.Errorf("missing %q:\n%s", want, f)
		}
	}
	camBusy(t, m)
	before, slots := m.count("/caminus/status"), m.count("/slots")
	st = run(t, srv.URL, 12, nil)
	if n, s := m.count("/caminus/status")-before, m.count("/slots")-slots; n != 12 || s != 12 {
		t.Errorf("busy: %d status and %d /slots GETs in 12 rounds, want 12 each", n, s)
	}
	if !st.srv.busy() || st.srv.Cam == nil || !st.srv.Cam.Stats {
		t.Errorf("busy state %+v", st.srv)
	}
}

// A server whose /props has no engine marker is not Caminus: it is never asked for the status, even one that
// would answer it, and the panel says what it is not.
func TestNotCaminus(t *testing.T) {
	m := &mock{props: `{"default_generation_settings":{"n_ctx":32768},"model_path":"/m/other.gguf"}`, metrics: metricsText(0),
		slots: slotsJSON(slot(0, false, 1, 100, 100, 0, 5)), status: readFixture(t, "caminus_status.json")}
	srv := httptest.NewServer(m)
	defer srv.Close()
	st := run(t, srv.URL, 12, nil)
	if !st.srv.NotCam || st.srv.Cam != nil || m.count("/caminus/status") != 0 {
		t.Errorf("not caminus %v, %d status GETs", st.srv.NotCam, m.count("/caminus/status"))
	}
	f := frame(t, st, 120, 40)
	if !strings.Contains(f, `not a Caminus server: its /props has no "engine": "caminus"`) || strings.Contains(f, "engine  ") ||
		!strings.Contains(f, "other") {
		t.Errorf("frame:\n%s", f)
	}
}

// Caminus without the status route: the panel says so, and the 404 is asked again only at /props' pace, busy or not.
func TestCaminusWithoutStatus(t *testing.T) {
	m := &mock{metrics: metricsText(1), slots: slotsJSON(slot(0, true, 5, 9000, 4000, 0, 0))}
	srv := httptest.NewServer(m)
	defer srv.Close()
	st := run(t, srv.URL, 12, nil)
	if st.srv.NotCam || st.srv.Cam != nil || st.srv.CamCode != 404 || m.count("/caminus/status") != 3 {
		t.Errorf("not caminus %v code %d, %d status GETs", st.srv.NotCam, st.srv.CamCode, m.count("/caminus/status"))
	}
	if f := frame(t, st, 120, 40); !strings.Contains(f, "no /caminus/status (HTTP 404)") {
		t.Errorf("frame:\n%s", f)
	}
}

// Another status version is named, not misread.
func TestCaminusVersion(t *testing.T) {
	m := camMock(t)
	m.status = `{"version":2,"engine":"caminus","physics":{"floor_ms":"x"}}`
	srv := httptest.NewServer(m)
	defer srv.Close()
	st := run(t, srv.URL, 3, nil)
	if st.srv.Cam != nil || !strings.Contains(frame(t, st, 120, 40), "status version 2, this llm-top reads 1") {
		t.Errorf("version 2: %+v", st.srv.CamNote)
	}
}

// The port's server restarts as another server without going down: the marker is gone from the next /props, the
// old status goes and is not asked again. A /props that does not decode changes nothing.
func TestCaminusMarkerGone(t *testing.T) {
	m := camMock(t)
	srv := httptest.NewServer(m)
	defer srv.Close()
	s := newStepper(t, srv.URL)
	s.step(4)
	if s.st.srv.NotCam || s.st.srv.Cam == nil {
		t.Fatalf("not caminus %v", s.st.srv.NotCam)
	}
	m.mu.Lock()
	m.props = "{broken"
	m.mu.Unlock()
	s.step(5)
	if s.st.srv.NotCam || s.st.srv.Props.Engine != engineCaminus {
		t.Errorf("an unreadable /props changed the server: %+v", s.st.srv.Props)
	}
	m.mu.Lock()
	m.props = `{"default_generation_settings":{"n_ctx":32768},"model_path":"/m/Other-27B.gguf"}`
	m.mu.Unlock()
	s.step(5)
	n := m.count("/caminus/status")
	s.step(8)
	if !s.st.srv.NotCam || s.st.srv.Cam != nil || m.count("/caminus/status") != n {
		t.Errorf("after the restart: not caminus %v, %d more status GETs", s.st.srv.NotCam, m.count("/caminus/status")-n)
	}
	if f := frame(t, s.st, 120, 40); !strings.Contains(f, "not a Caminus server") || !strings.Contains(f, "Other-27B") || strings.Contains(f, "floor") {
		t.Errorf("frame:\n%s", f)
	}
}

// ---------- the panels ----------

// Every group the status gives has its row; without hardware readings or a drafter the rows say so or go.
func TestCaminusPanels(t *testing.T) {
	st := camState(t, "caminus_status.json")
	lone := camState(t, "caminus_status.json") // no agent session: the whole panel fits 120 x 40
	lone.sessions = nil
	for _, c := range []struct {
		st   *state
		w, h int
	}{{lone, 120, 40}, {st, 200, 60}, {lone, 90, 80}} {
		size := [2]int{c.w, c.h}
		f := frame(t, c.st, c.w, c.h)
		for _, want := range []string{
			"CAMINUS", "speed", "of 4.09 ms floor · pass", "ms/token · 1.09 tok/pass", "■ k3 ", "■ sampler", // physics, ms/token
			"ttft", "■ queue", "■ batched", "■ passes", // TTFT
			"dflash  3.2% accepted", "prose 3% ", "code —", // DFlash by class
			"cache   7.1k reused · vram 6.3k · disk 813 · exact 0", "vram ", "8/8 1.2 GB", "5.4 / 30 GB", "restore p50 0.67ms p95 260ms", // the cache
			"cards   XTX 19.8 ms + 1.02 ms wait · XT 18.6 ms",                                    // the two-card split
			"engine  dense attention (QSA from 2.1k) · KV 12.5 MB per card",                      // context / QSA
			"health  ✓ no bails, retries, KFD evictions, refusals or errors",                     // reliability
			"power 517 W · 0.094 tok/J decode", "XTX   286 W 2482 MHz  78C high · margin 3.0 GB", // power
			"agents  32 requests · 0 tool calls · 0 parse failures", "0 loops · thinking 2.4k / answer 240", // agents
		} {
			if !strings.Contains(f, want) {
				t.Errorf("%dx%d: missing %q", size[0], size[1], want)
			}
		}
		if t.Failed() {
			t.Fatalf("frame:\n%s", f)
		}
	}
	// an agent session at 120 x 40: its waffle stays, the prompt-size chart goes, CAMINUS keeps speed, TTFT and DFlash
	f := frame(t, st, 120, 40)
	for _, want := range []string{"what fills it · rocket-api", "■ tool output", "of 4.09 ms floor", "■ queue", "prose 3% "} {
		if !strings.Contains(f, want) {
			t.Errorf("120x40 with a session: missing %q:\n%s", want, f)
		}
	}
	if strings.Contains(f, "prompt size per request") {
		t.Errorf("120x40 with a session: the prompt-size chart stayed:\n%s", f)
	}
	nohw := camState(t, "caminus_status_nohw.json")
	nohw.sessions = nil
	f = frame(t, nohw, 120, 40)
	for _, want := range []string{"dflash  off (speculative.types none)", "cards   XTX 11.7 ms + 0.60 ms wait"} {
		if !strings.Contains(f, want) {
			t.Errorf("no hardware: missing %q:\n%s", want, f)
		}
	}
	if strings.Contains(f, "tok/J") || strings.Contains(f, "margin") {
		t.Errorf("power rows without hardware readings:\n%s", f)
	}
	// offline: no panel and no engine rows, like every other panel's numbers
	off := camState(t, "caminus_status.json")
	off.srv.Up = false
	if f := frame(t, off, 120, 40); strings.Contains(f, "CAMINUS") || strings.Contains(f, "engine  ") || strings.Contains(f, "tok/J") {
		t.Errorf("offline:\n%s", f)
	}
	// before the first status
	first := camState(t, "caminus_status.json")
	first.srv.Cam, first.srv.CamCode = nil, 0
	if f := frame(t, first, 120, 40); !strings.Contains(f, "reading /caminus/status…") {
		t.Errorf("first:\n%s", f)
	}
	// a per-position rate outside [0, 1] is clamped (it used to index past sparkTo's blocks), the status left as sent
	for _, c := range []struct {
		r    []float64
		want string
	}{{[]float64{1.5}, "prose 3% █ "}, {[]float64{-0.2}, "prose 3%   "}, {[]float64{1e300, -1e300, 0.5}, "prose 3% █ ▄ "}} {
		bad := camState(t, "caminus_status.json")
		bad.sessions = nil
		p := bad.srv.Cam.DFlash.Classes["prose"]
		p.PerPosRate = c.r
		bad.srv.Cam.DFlash.Classes["prose"] = p
		if f := frame(t, bad, 120, 40); !strings.Contains(f, c.want) {
			t.Errorf("rates %v: missing %q:\n%s", c.r, c.want, f)
		}
		if got := bad.srv.Cam.DFlash.Classes["prose"].PerPosRate; got[0] != c.r[0] {
			t.Errorf("rates %v: the status was changed to %v", c.r, got)
		}
	}
}

// What went wrong is counted and colored; a reasoning loop shows in the panel and, while busy, in the status line.
func TestCaminusAlerts(t *testing.T) {
	st := camState(t, "caminus_status_busy.json")
	cs := st.srv.Cam
	cs.Reliability.Bails = map[string]float64{"recovered": 2, "failed": 1}
	cs.Reliability.Retries, cs.Reliability.Evicted = 4, 40
	cs.Reliability.Refusals = map[string]float64{"context": 1, "bounds": 0}
	cs.Agent.Failures = map[string]float64{"cut": 2, "unclosed": 1}
	cs.Agent.Loop.Flagged, cs.Agent.Loop.Live = 3, 5
	cs.Physics.TokenPct = new(float64)
	*cs.Physics.TokenPct = 140 // drafts accepted: above the floor's speed; the gauge stays full
	st.srv.Slot.Processing, st.srv.Busy, st.srv.Decoded = true, 1, 40
	f := frame(t, st, 200, 60)
	for _, want := range []string{"health  bails 3 (1 failed, 2 recovered) · refusals 1 (1 context) · retries 4 · KFD evicted 40.0 ms",
		"3 parse failures (2 cut, 1 unclosed)", "3 loops", "▸ loop: the running request's reasoning repeats one line 5x", "140.0% of 4.09 ms floor"} {
		if !strings.Contains(f, want) {
			t.Errorf("missing %q", want)
		}
	}
	if h := header(t, st); !strings.Contains(h, "▸ loop") {
		t.Errorf("no loop alert in the status line:\n%s", h)
	}
	st.srv.Slot.Processing, st.srv.Busy = false, 0
	if h := header(t, st); strings.Contains(h, "▸ loop") {
		t.Errorf("loop alert while idle:\n%s", h)
	}
}

// The DOWN line names a caminus process that runs without answering.
func TestCaminusNotAnswering(t *testing.T) {
	st := camState(t, "caminus_status.json")
	st.srv = serverState{Polled: true}
	st.sys.Pid = 3100
	if out := frame(t, st, 120, 50); !strings.Contains(out, "caminus is running but not answering") || strings.Contains(out, "CAMINUS") {
		t.Errorf("down:\n%s", out)
	}
}

func TestCamFormats(t *testing.T) {
	for x, want := range map[float64]string{0.6661: "0.67 ms", 17.806: "17.8 ms", 260.03: "260 ms", 3604: "3.60 s"} {
		if got := msf(x); got != want {
			t.Errorf("msf(%v) = %q", x, got)
		}
	}
	for x, want := range map[float64]string{6228936: "6.2 MB", 381580400: "382 MB", 900: "900 B", 1.342e9: "1.3 GB"} {
		if got := bytesf(x); got != want {
			t.Errorf("bytesf(%v) = %q", x, got)
		}
	}
	if got := nonzero(map[string]float64{"b": 1, "a_b": 2, "c": 0}); got != "2 a b, 1 b" {
		t.Errorf("nonzero %q", got)
	}
}
