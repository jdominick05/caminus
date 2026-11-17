// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func pickerModel() camModel {
	return camModel{Name: "dense", Cards: 1, Present: true, Modes: []serveMode{
		{Cards: 1, Drafts: []string{"off", "mtp", "dflash"}},
		{Cards: 2, Drafts: []string{"off"}},
	}}
}

func TestModelPickerBasicChoices(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	mi := pickerModel()
	m.models, m.loaded["models"] = []camModel{mi}, true
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "dense"}) {
		t.Fatalf("auto leaves the measured default to serve: %v", got)
	}
	if !strings.Contains(m.keysLine(), "drafter") || !strings.Contains(m.keysLine(), "card count") {
		t.Fatal("both choices are discoverable in the basic model picker")
	}
	for _, want := range []string{"off", "mtp", "dflash", "auto"} {
		next, cmd := m.modelsKey("d")
		m = next.(ctl)
		if cmd != nil {
			t.Fatal("choosing a drafter must not launch anything")
		}
		_, got := m.modelMode(mi)
		if got != want {
			t.Fatalf("drafter cycle: got %s, want %s", got, want)
		}
		args := strings.Join(m.startArgs(mi), " ")
		if want == "auto" && strings.Contains(args, "--draft") {
			t.Fatal("auto became explicit off")
		}
		if want != "auto" && !strings.Contains(args, "-- --draft "+want) {
			t.Fatalf("drafter not passed: %s", args)
		}
	}
	if rows := strings.Join(m.modelRows(), "\n"); !strings.Contains(rows, "auto (server default)") {
		t.Fatalf("the next start is visible: %s", rows)
	}
}

func TestModelPickerPerCardCountAndModel(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	mi := pickerModel()
	m.cycleModelMode(mi, false) // off
	m.cycleModelMode(mi, false) // mtp
	m.cycleModelMode(mi, true)  // two cards: only off/auto are supported
	mode, draft := m.modelMode(mi)
	if mode.Cards != 2 || draft != "auto" {
		t.Fatalf("two-card selection: %+v %s", mode, draft)
	}
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "dense", "--cards", "2"}) {
		t.Fatalf("two cards with server default: %v", got)
	}
	m.cycleModelMode(mi, false) // explicit off for two cards
	m.cycleModelMode(mi, true)  // back to one card
	mode, draft = m.modelMode(mi)
	if mode.Cards != 1 || draft != "mtp" {
		t.Fatalf("one-card choice was lost: %+v %s", mode, draft)
	}
	m.cycleModelMode(mi, true)
	_, draft = m.modelMode(mi)
	if draft != "off" {
		t.Fatalf("two-card off was lost: %s", draft)
	}
	other := camModel{Name: "other", Cards: 2, Modes: []serveMode{{Cards: 2, Drafts: []string{"off", "dflash"}}}}
	if got := m.startArgs(other); !reflect.DeepEqual(got, []string{"start", "other"}) {
		t.Fatalf("another model inherited this model's options: %v", got)
	}
}

func TestModelPickerCapabilitiesAndLegacyOptions(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), true, "")
	m.card = "1"
	m.vals = map[string]string{"cards": "2", "draft": "mtp", "no_drafter": "true", "max_ctx": "8192"}
	m.opts = []option{
		{Dest: "cards", Flag: "--cards", Type: "int"}, {Dest: "draft", Flag: "--draft", Type: "str"},
		{Dest: "no_drafter", Flag: "--no-drafter", Type: "bool"}, {Dest: "max_ctx", Flag: "--max-ctx", Type: "ctx"},
	}
	mi := pickerModel()
	want := []string{"start", "dense", "--cards", "2", "--", "--draft", "off", "--max-ctx", "8192"}
	if got := m.startArgs(mi); !reflect.DeepEqual(got, want) {
		t.Fatalf("legacy choices/conflicting flags: %v", got)
	}
	if got := m.visibleOpts(); len(got) != 1 || got[0].Dest != "max_ctx" {
		t.Fatalf("duplicate options: %v", got)
	}
	// Restoring a choice from a newer engine cannot enable an unsupported mode on the current one.
	m.picks = map[string]modelPick{mi.Name: {Cards: 2, Drafts: map[int]string{2: "mtp"}}}
	if got := strings.Join(m.startArgs(mi), " "); strings.Contains(got, "--draft") || strings.Contains(got, "--card 1") {
		t.Fatalf("unsupported drafting or a one-card placement leaked into TP: %s", got)
	}
	old := camModel{Name: "old-cli", Cards: 1}
	if got := strings.Join(m.startArgs(old), " "); strings.Contains(got, "--cards") || strings.Contains(got, "--draft") {
		t.Fatalf("an old CLI supplied no capability for an override: %s", got)
	}
}

func TestModelPickerRememberedChoices(t *testing.T) {
	p := filepath.Join(t.TempDir(), "panel.json")
	m := newCtl(newCtlCLI("caminus"), false, p)
	mi := pickerModel()
	m.cycleModelMode(mi, false) // explicit off
	m.cycleModelMode(mi, true)
	m.cycleModelMode(mi, false) // explicit off for two cards
	loaded := newCtl(newCtlCLI("caminus"), false, p)
	if got, want := loaded.startArgs(mi), m.startArgs(mi); !reflect.DeepEqual(got, want) {
		t.Fatalf("saved model/card choices: got %v want %v", got, want)
	}
	var parsed camModel
	if err := json.Unmarshal([]byte(`{"name":"metadata","cards":1,"serve_modes":[{"cards":1,"drafts":["off","mtp"]},{"cards":2,"drafts":["off","dflash"]}]}`), &parsed); err != nil {
		t.Fatal(err)
	}
	loaded.cycleModelMode(parsed, true)
	loaded.cycleModelMode(parsed, false)
	loaded.cycleModelMode(parsed, false)
	if got := strings.Join(loaded.startArgs(parsed), " "); got != "start metadata --cards 2 -- --draft dflash" {
		t.Fatalf("capabilities from the registry control TP drafting: %s", got)
	}
}

func TestModelPickerMissingOptionalDrafter(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	mi := pickerModel()
	mi.MissingDrafter, mi.MissingDrafterGB = "dflash", 2.1
	m.models, m.loaded["models"] = []camModel{mi}, true
	m.cycleModelMode(mi, false) // off: starts plain, without fetching the optional file
	if _, cmd := m.modelsKey("enter"); cmd == nil {
		t.Fatal("off must start without its optional drafter")
	}
	m.cycleModelMode(mi, false) // MTP: its block is in the model, not the optional file
	if _, cmd := m.modelsKey("enter"); cmd == nil {
		t.Fatal("MTP must start without the DFlash file")
	}
	m.cycleModelMode(mi, false) // DFlash: refuse the start before a command is made
	next, cmd := m.modelsKey("enter")
	m = next.(ctl)
	if cmd != nil || !m.noteErr || !strings.Contains(m.note, "not downloaded") {
		t.Fatalf("missing DFlash did not refuse the start: command=%v note=%q", cmd != nil, m.note)
	}
	next, cmd = m.modelsKey("p")
	m = next.(ctl)
	if cmd != nil || m.confirm == nil || m.confirm.run == nil || !strings.Contains(m.confirm.q, "dflash drafter (2.1 GB)") {
		t.Fatalf("an installed model can fetch its missing drafter after confirmation: %+v", m.confirm)
	}
	if rows := strings.Join(m.modelRows(), "\n"); !strings.Contains(rows, "dflash missing") {
		t.Fatalf("the missing optional file is visible: %s", rows)
	}
}

func TestModelPickerCustomDrafter(t *testing.T) {
	for _, advanced := range []bool{false, true} {
		for _, path := range []string{"", "  ", "/fixture/custom.gguf"} {
			m := newCtl(newCtlCLI("caminus"), advanced, "")
			mi := pickerModel()
			mi.MissingDrafter = "dflash"
			m.models = []camModel{mi}
			m.opts = []option{{Dest: "drafter", Flag: "--drafter", Type: "str"}}
			m.vals["drafter"] = path
			m.picks = map[string]modelPick{mi.Name: {Cards: 1, Drafts: map[int]string{1: "dflash"}}}
			_, cmd := m.modelsKey("enter")
			wantStart := advanced && strings.TrimSpace(path) != ""
			if (cmd != nil) != wantStart {
				t.Fatalf("advanced %v, custom %q: start=%v want %v", advanced, path, cmd != nil, wantStart)
			}
			if wantStart {
				args := strings.Join(m.startArgs(mi), " ")
				if !strings.Contains(args, "--draft dflash") || !strings.Contains(args, "--drafter "+path) || strings.Contains(args, "--no-drafter") {
					t.Fatalf("custom drafter did not reach serve unchanged: %s", args)
				}
			}
		}
	}
}

// The picker shows serve's F16 and Q8 maxima from `caminus fit --json` for the selected model, card count and drafter
// (q8 null: "not offered" for a model without the 8-bit cache). The panel does no memory math of its own.
func TestModelPickerFitRowFromServeProjection(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	mi := pickerModel()
	m.models, m.loaded["models"] = []camModel{mi}, true
	key, args := m.fitArgs(mi)
	if key != "fit --model  --cards 1" {
		t.Fatalf("fit key is the whole query: %q", key)
	}
	if got := strings.Join(args, " "); got != "fit --model  --cards 1 --json" {
		t.Fatalf("fit args (auto drafter: serve's default): %q", got)
	}
	if m.fitRow(mi) != "" {
		t.Fatal("no row before the fit read")
	}
	if cmd := m.fitCmd(); cmd == nil {
		t.Fatal("a present model's fit is read once")
	}
	if cmd := m.fitCmd(); cmd != nil {
		t.Fatal("the same choice is not read twice")
	}
	f16 := 40960
	next, _ := m.update(fitMsg{key: key, v: fitInfo{OK: true, MaxCtx: fitCtx{F16: &f16}, KVFormats: []string{"f16"}}})
	row := ansi.Strip(next.(ctl).fitRow(mi))
	if !strings.Contains(row, "40960") || !strings.Contains(row, "Q8: not offered") {
		t.Fatalf("F16 maximum shown from fit, Q8 not offered: %q", row)
	}
	// a drafter change is another choice with its own read
	m2 := next.(ctl)
	m2.picks = map[string]modelPick{"dense": {Cards: 1, Drafts: map[int]string{1: "mtp"}}}
	if k2, a2 := m2.fitArgs(mi); k2 == key || !strings.Contains(strings.Join(a2, " "), "--draft mtp") {
		t.Fatalf("drafter choice: key %s args %v", k2, a2)
	}
	// serve says nothing fits at F16: shown as no context, not as a number
	none := ctl{}
	none = newCtl(newCtlCLI("caminus"), false, "")
	none.models, none.loaded["models"] = []camModel{mi}, true
	nk, _ := none.fitArgs(mi)
	nn, _ := none.update(fitMsg{key: nk, v: fitInfo{OK: true}})
	if r := nn.(ctl).fitRow(mi); !strings.Contains(r, "no context fits") {
		t.Fatalf("no F16 maximum row: %q", r)
	}
}

func q3mModel() camModel {
	return camModel{Name: "qwen3-30b-a3b", Cards: 1, Present: true, Path: "/m/q3.gguf", Modes: []serveMode{
		{Cards: 1, Drafts: []string{"off"}, KV: []string{"f16", "q8"}, Long: true},
		{Cards: 2, Drafts: []string{"off"}, KV: []string{"f16", "q8"}, Long: true},
	}}
}

func TestModelPickerKVAndLongContext(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	m.prefs = filepath.Join(t.TempDir(), "prefs.json")
	mi := q3mModel()
	m.models, m.loaded["models"] = []camModel{mi}, true
	keys := m.keysLine()
	if !strings.Contains(keys, "v KV cache") || !strings.Contains(keys, "l long context") {
		t.Fatalf("both choices are discoverable for the 30B: %s", keys)
	}
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "qwen3-30b-a3b"}) {
		t.Fatalf("defaults pass no flags (the server's F16, native): %v", got)
	}
	for _, want := range []string{"f16", "q8", "auto"} {
		next, cmd := m.modelsKey("v")
		m = next.(ctl)
		if cmd != nil {
			t.Fatal("choosing a cache must not launch anything")
		}
		mode, _ := m.modelMode(mi)
		if kv, _ := m.modelExtras(mi, mode); kv != want {
			t.Fatalf("KV cycle: got %s, want %s", kv, want)
		}
	}
	next, _ := m.modelsKey("l")
	m = next.(ctl)
	next, _ = m.modelsKey("v") // f16
	m = next.(ctl)
	next, _ = m.modelsKey("v") // q8
	m = next.(ctl)
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "qwen3-30b-a3b", "--kv", "q8", "--long-context"}) {
		t.Fatalf("start takes its own --kv and --long-context: %v", got)
	}
	if _, fa := m.fitArgs(mi); !strings.Contains(strings.Join(fa, " "), "--cards 1 --kv q8 --long-context --json") {
		t.Fatalf("fit projects the same choice: %v", fa)
	}
	if rows := strings.Join(m.modelRows(), "\n"); !strings.Contains(ansi.Strip(rows), "KV q8 · long context (YaRN x4, as far as the card holds") {
		t.Fatalf("the next start shows the choice: %s", ansi.Strip(rows))
	}
	// the choice survives a restart of the panel (KV and Long saved with the picks)
	re := newCtl(newCtlCLI("caminus"), false, m.prefs)
	if got := re.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "qwen3-30b-a3b", "--kv", "q8", "--long-context"}) {
		t.Fatalf("the saved choice was not reloaded: %v", got)
	}
	// per card count: two cards start from the defaults, and the one-card choice comes back
	m.cycleModelMode(mi, true)
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "qwen3-30b-a3b", "--cards", "2"}) {
		t.Fatalf("two cards keep their own (default) choice: %v", got)
	}
	m.cycleModelMode(mi, true)
	if got := m.startArgs(mi); !reflect.DeepEqual(got, []string{"start", "qwen3-30b-a3b", "--kv", "q8", "--long-context"}) {
		t.Fatalf("the one-card choice was lost: %v", got)
	}
	// a model without the choices shows neither key and ignores both
	d := pickerModel()
	m.models = []camModel{d}
	if k := m.keysLine(); strings.Contains(k, "KV cache") || strings.Contains(k, "long context") {
		t.Fatalf("no KV or long-context keys for a model without them: %s", k)
	}
	m.cycleExtra(d, false)
	m.cycleExtra(d, true)
	if got := m.startArgs(d); !reflect.DeepEqual(got, []string{"start", "dense"}) {
		t.Fatalf("a model without the choices passes neither flag: %v", got)
	}
	var parsed camModel
	if err := json.Unmarshal([]byte(`{"name":"q","cards":1,"serve_modes":[{"cards":1,"drafts":["off"],"kv":["f16","q8"],"long_context":true}]}`), &parsed); err != nil {
		t.Fatal(err)
	}
	if md := parsed.servingModes()[0]; !md.Long || len(md.KV) != 2 {
		t.Fatalf("the CLI's kv / long_context keys reach the picker: %+v", md)
	}
}

func TestModelPickerAdvancedValuesCarryIntoThePick(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), true, "")
	m.prefs = filepath.Join(t.TempDir(), "prefs.json")
	mi := q3mModel()
	m.models, m.loaded["models"] = []camModel{mi}, true
	m.vals["kv"], m.vals["long_context"], m.vals["draft"] = "q8", "true", "off"
	if got := strings.Join(m.startArgs(mi), " "); !strings.Contains(got, "--kv q8 --long-context") || !strings.Contains(got, "-- --draft off") {
		t.Fatalf("saved advanced --kv / --long-context / --draft values still apply before any pick: %s", got)
	}
	m.cycleExtra(mi, true) // long context off: the pick now exists and must keep the drafter and cache shown before
	if got := strings.Join(m.startArgs(mi), " "); strings.Contains(got, "--long-context") || !strings.Contains(got, "--kv q8") || !strings.Contains(got, "-- --draft off") {
		t.Fatalf("the first pick dropped the advanced drafter or cache: %s", got)
	}
	m.busy = "starting x"
	m.focus = paneModels
	before := strings.Join(m.startArgs(mi), " ")
	next, _ := m.key(tea.KeyPressMsg{Code: 'l', Text: "l"})
	if got := strings.Join(next.(ctl).startArgs(mi), " "); got != before {
		t.Fatalf("l in MODELS while a start runs must be refused like the other choices: %s -> %s", before, got)
	}
}
