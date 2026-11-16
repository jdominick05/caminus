// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// fakeFitCLI is a `caminus` stand-in that answers every call with the same stdout (and exit status): the real
// `caminus fit --json` reply shapes, so the panel decodes what serve prints.
func fakeFitCLI(t *testing.T, stdout string, exit int) *ctlCLI {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "caminus.sh")
	body := "#!/bin/sh\ncat <<'JSON'\n" + stdout + "\nJSON\nexit " + string(rune('0'+exit)) + "\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return &ctlCLI{argv: []string{"sh", script}, timeout: 10 * time.Second}
}

// fitOK is serve's reply for a model with the 8-bit cache that fits 32768 positions at F16 and its whole 40960 at Q8.
const fitOK = `{"native_ctx": 40960, "max_ctx": {"f16": 32768, "q8": 40960}, "step": 4096, "room_mib": [1000.0], "slot_mib": 4.0, "need_mib": [900.0], "ok": true, "arch": "qwen3moe", "draft": "off", "cards": ["0000:03:00.0"], "kv": "f16", "kv_formats": ["f16", "q8"]}`

// fitF16Only is serve's reply for a model without the 8-bit cache: q8 null, kv_formats only f16.
const fitF16Only = `{"native_ctx": 131072, "max_ctx": {"f16": 65536, "q8": null}, "step": 4096, "room_mib": [1000.0], "slot_mib": 4.0, "need_mib": [900.0], "ok": true, "arch": "qwen35-own", "draft": "off", "cards": ["0000:03:00.0"], "kv": "f16", "kv_formats": ["f16"]}`

// fitQ8Only is serve's reply (with --kv q8) when only the 8-bit cache fits: f16 null, q8 a number.
const fitQ8Only = `{"native_ctx": 40960, "max_ctx": {"f16": null, "q8": 40960}, "step": 4096, "room_mib": [1000.0], "slot_mib": 4.0, "need_mib": [900.0], "ok": true, "arch": "qwen3moe", "draft": "off", "cards": ["0000:03:00.0"], "kv": "q8", "kv_formats": ["f16", "q8"]}`

// fitRefusedQ8 is serve's reply (exit 1) when no cards hold the F16 cache but the 8-bit one fits: ok false, the Q8
// figure kept.
const fitRefusedQ8 = `{"ok": false, "max_ctx": {"f16": null, "q8": 40960}, "kv": "f16", "kv_formats": ["f16", "q8"], "error": "no card holds qwen3-30b-a3b"}`

// fitQ8NoRoom is serve's reply when the model has the 8-bit cache but no cards hold it (q8 null, "q8" listed).
const fitQ8NoRoom = `{"native_ctx": 40960, "max_ctx": {"f16": 32768, "q8": null}, "step": 4096, "room_mib": [1000.0], "slot_mib": 4.0, "need_mib": [900.0], "ok": true, "arch": "qwen3moe", "draft": "off", "cards": ["0000:03:00.0"], "kv": "f16", "kv_formats": ["f16", "q8"]}`

// fitFixed is serve's reply for the Coder, which always keeps its own q8_0 cache: kv_fixed names it, f16 is its figure,
// q8 null and kv_formats only f16 (no --kv choice).
const fitFixed = `{"native_ctx": 262144, "max_ctx": {"f16": 131072, "q8": null}, "step": 4096, "room_mib": [1000.0, 1000.0], "slot_mib": 4.0, "need_mib": [900.0, 900.0], "ok": true, "arch": "qwen4exp", "draft": "dflash", "cards": ["0000:03:00.0", "0000:06:00.0"], "kv": "f16", "kv_formats": ["f16"], "kv_fixed": "q8_0"}`

// fitFixedNone is the Coder's reply when no context fits its cards, and fitFixedFail its refusal (exit 1) after the
// recipe (kv_fixed is in it).
const fitFixedNone = `{"native_ctx": 262144, "max_ctx": {"f16": null, "q8": null}, "step": 4096, "room_mib": [1000.0, 1000.0], "slot_mib": 4.0, "need_mib": null, "ok": true, "arch": "qwen4exp", "draft": "off", "cards": ["0000:03:00.0", "0000:06:00.0"], "kv": "f16", "kv_formats": ["f16"], "kv_fixed": "q8_0"}`
const fitFixedFail = `{"ok": false, "max_ctx": {"f16": null, "q8": null}, "kv": "f16", "kv_formats": ["f16"], "kv_fixed": "q8_0", "error": "no two supported cards"}`

// fitNone is serve's reply when no context fits: ok true, f16 null.
const fitNone = `{"native_ctx": 40960, "max_ctx": {"f16": null, "q8": null}, "step": 4096, "room_mib": [1000.0], "slot_mib": 4.0, "need_mib": null, "ok": true, "arch": "qwen3moe", "draft": "off", "cards": ["0000:03:00.0"]}`

// fitFail is serve's reply on a refused query (exit 1, JSON on stdout).
const fitFail = `{"ok": false, "max_ctx": {"f16": null, "q8": null}, "error": "the DFlash GGUF does not exist"}`

func presentPicker(path string) camModel {
	mi := pickerModel()
	mi.Present = true
	mi.Path = path
	return mi
}

// The real path: Update -> fitCmd -> fitMsg -> render, with the fake caminus answering the real JSON shape.
func TestFitRowRealPathOK(t *testing.T) {
	m := newCtl(fakeFitCLI(t, fitOK, 0), false, "")
	mi := presentPicker("/models/dense.gguf")
	m.models, m.loaded["models"] = []camModel{mi}, true
	m = settle(m, m.fitCmd(), 0)
	row := ansi.Strip(m.fitRow(mi))
	if !strings.Contains(row, "at F16: 32768 positions") || !strings.Contains(row, "Q8: 40960") {
		t.Fatalf("F16 and Q8 maxima from the real reply: %q", row)
	}
	if strings.Contains(row, "reading") {
		t.Fatalf("the row is stuck reading: %q", row)
	}
}

// The Q8 part of the row for each reply shape: a figure, "none fits" (the model has the 8-bit cache, no room for it),
// "not offered" (no 8-bit cache), a row that still shows Q8 when nothing fits at F16, and a refusal that keeps it.
func TestFitRowQ8Shapes(t *testing.T) {
	mi := presentPicker("/models/m.gguf")
	for _, c := range []struct{ reply, want, not string }{
		{fitF16Only, "at F16: 65536 positions · Q8: not offered", "none fits"},
		{fitQ8NoRoom, "at F16: 32768 positions · Q8: none fits", "not offered"},
		{fitQ8Only, "none at F16 · Q8: 40960 positions", "no context fits"},
		{fitRefusedQ8, "no card holds qwen3-30b-a3b · Q8: 40960 positions", "reading"},
		{fitFail, "the DFlash GGUF does not exist", "Q8"},
	} {
		exit := 0
		if c.reply == fitRefusedQ8 || c.reply == fitFail {
			exit = 1
		}
		m := newCtl(fakeFitCLI(t, c.reply, exit), false, "")
		m.models, m.loaded["models"] = []camModel{mi}, true
		m = settle(m, m.fitCmd(), 0)
		row := ansi.Strip(m.fitRow(mi))
		if !strings.Contains(row, c.want) || strings.Contains(row, c.not) {
			t.Fatalf("want %q (not %q) in %q", c.want, c.not, row)
		}
	}
}

// A model that keeps its own cache (kv_fixed: the Coder's q8_0) gets one figure, named as its own 8-bit cache: never
// "at F16" and no Q8 part ("not offered" would be wrong: its cache is the 8-bit one).
func TestFitRowOwnCache(t *testing.T) {
	mi := presentPicker("/models/coder.gguf")
	for _, c := range []struct {
		reply string
		exit  int
		want  string
	}{
		{fitFixed, 0, "context that fits (its own 8-bit cache): 131072 positions"},
		{fitFixedNone, 0, "context that fits (its own 8-bit cache): none with this card count"},
		{fitFixedFail, 1, "context that fits (its own 8-bit cache): no two supported cards"},
	} {
		m := newCtl(fakeFitCLI(t, c.reply, c.exit), false, "")
		m.models, m.loaded["models"] = []camModel{mi}, true
		m = settle(m, m.fitCmd(), 0)
		row := ansi.Strip(m.fitRow(mi))
		if !strings.Contains(row, c.want) {
			t.Fatalf("want %q in %q", c.want, row)
		}
		for _, not := range []string{"F16", "Q8", "not offered", "none fits", "reading"} {
			if strings.Contains(row, not) {
				t.Fatalf("%q in the own-cache row %q", not, row)
			}
		}
	}
}

func TestFitRowRealPathNoFitAndRefusal(t *testing.T) {
	mi := presentPicker("/models/dense.gguf")

	none := newCtl(fakeFitCLI(t, fitNone, 0), false, "")
	none.models, none.loaded["models"] = []camModel{mi}, true
	none = settle(none, none.fitCmd(), 0)
	if r := ansi.Strip(none.fitRow(mi)); !strings.Contains(r, "no context fits at F16") {
		t.Fatalf("no F16 maximum is said, not a number: %q", r)
	}

	bad := newCtl(fakeFitCLI(t, fitFail, 1), false, "")
	bad.models, bad.loaded["models"] = []camModel{mi}, true
	bad = settle(bad, bad.fitCmd(), 0)
	r := ansi.Strip(bad.fitRow(mi))
	if !strings.Contains(r, "the DFlash GGUF does not exist") {
		t.Fatalf("the serve error text survives a nonzero exit: %q", r)
	}
	if strings.Contains(r, "reading") {
		t.Fatalf("a refusal is shown, not read forever: %q", r)
	}
	// a refused read is not kept: it is asked again after the retry wait, not on every frame
	if cmd := bad.fitCmd(); cmd != nil {
		t.Fatal("a failed read waits before it is asked again")
	}
	e := bad.fits[bad.fitKey(mi)]
	e.failedAt = time.Now().Add(-2 * fitRetry)
	bad.fits[bad.fitKey(mi)] = e
	if cmd := bad.fitCmd(); cmd == nil {
		t.Fatal("a failed read is asked again after the retry wait")
	}
}

// A model that is not on disk gets no read and no line.
func TestFitNotOnDiskNoReadNoLine(t *testing.T) {
	m := newCtl(fakeFitCLI(t, fitOK, 0), false, "")
	mi := pickerModel()
	mi.Present = false // (pickerModel is on disk by default: this one is not)
	m.models, m.loaded["models"] = []camModel{mi}, true
	if cmd := m.fitCmd(); cmd != nil {
		t.Fatal("a model not on disk is not read")
	}
	m = settle(m, m.fitCmd(), 0)
	if r := m.fitRow(mi); r != "" {
		t.Fatalf("a model not on disk shows no line: %q", r)
	}
	if len(m.fits) != 0 {
		t.Fatalf("nothing is cached for it: %v", m.fits)
	}
}

// R and every start, stop or restart (actMsg) forget the reads; a reply from before is dropped.
func TestFitCacheClearsAndDropsStaleReply(t *testing.T) {
	m := newCtl(fakeFitCLI(t, fitOK, 0), false, "")
	mi := presentPicker("/models/dense.gguf")
	m.models, m.loaded["models"] = []camModel{mi}, true
	m = settle(m, m.fitCmd(), 0)
	if m.fitRow(mi) == "" {
		t.Fatal("first read shown")
	}
	gen := m.fitGen
	next, _ := m.Update(actMsg{what: "start dense"})
	m = next.(ctl)
	if m.fitGen == gen {
		t.Fatal("a start, stop or restart advances the fit generation")
	}
	if e := m.fits[m.fitKey(mi)]; e.done {
		t.Fatalf("a start clears the answered read: %+v", e)
	}
	// a reply from the generation before the action is dropped, not kept
	stale := fitMsg{key: m.fitKey(mi), gen: gen, v: fitInfo{OK: true}}
	next, _ = m.Update(stale)
	m = next.(ctl)
	if e := m.fits[m.fitKey(mi)]; e.done {
		t.Fatal("a stale reply is dropped, not kept")
	}
	// R does the same
	m.fits[m.fitKey(mi)] = fitEntry{done: true, info: fitInfo{OK: true}}
	m.refreshFits()
	if len(m.fits) != 0 {
		t.Fatal("R (refreshFits) forgets every read")
	}
}

// Advanced options that change the start reach fit; the card choice is labelled as not in the figure.
func TestFitAdvancedOptionsPassThrough(t *testing.T) {
	m := newCtl(fakeFitCLI(t, fitOK, 0), true, "")
	m.opts = []option{
		{Flag: "--auto-cache-slots", Dest: "auto_cache_slots", Type: "int"},
		{Flag: "--kv-host", Dest: "kv_host", Type: "bool"},
		{Flag: "--max-ctx", Dest: "max_ctx", Type: "ctx"},
	}
	m.vals = map[string]string{"auto_cache_slots": "8", "kv_host": "false", "max_ctx": "16384"}
	mi := presentPicker("/models/dense.gguf")
	m.models, m.loaded["models"] = []camModel{mi}, true
	_, args := m.fitArgs(mi)
	got := strings.Join(args, " ")
	for _, want := range []string{"--auto-cache-slots 8", "--max-ctx 16384"} {
		if !strings.Contains(got, want) {
			t.Fatalf("fit gets %q: %s", want, got)
		}
	}
	if strings.Contains(got, "--kv-host") {
		t.Fatalf("a false bool is not passed: %s", got)
	}
	// the start command gets the same options (the line and the start agree)
	if s := strings.Join(m.startArgs(mi), " "); !strings.Contains(s, "--auto-cache-slots 8") || !strings.Contains(s, "--max-ctx 16384") {
		t.Fatalf("start gets the same options: %s", s)
	}
}

// -once: the frame is drawn only after the fit read is answered (no "reading" forever).
func TestOnceFrameAnswersFitRead(t *testing.T) {
	m := newCtl(fakeFitCLI(t, fitOK, 0), false, "")
	mi := presentPicker("/models/dense.gguf")
	m.models, m.loaded["models"] = []camModel{mi}, true
	m = settle(m, m.fitCmd(), 0)
	if r := ansi.Strip(m.fitRow(mi)); strings.Contains(r, "reading") || !strings.Contains(r, "32768") {
		t.Fatalf("the frame's row is answered: %q", r)
	}
}
