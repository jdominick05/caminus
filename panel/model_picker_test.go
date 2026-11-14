// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
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
