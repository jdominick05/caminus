package main

import (
	"strings"
	"testing"
)

// The hardware lines (caminus ps --json's problems, from caminus.device.host): an error blocks every start and shows in
// both layouts; warnings show in both; notes only in the advanced one. The card choice cycles over the cards ps lists.
func TestHardwareLines(t *testing.T) {
	m := newCtl(newCtlCLI("caminus"), false, "")
	m.w, m.h = 120, 40
	m.models = []camModel{{Name: "qwen3.6", Cards: 1, Present: true}}
	m.ps = psInfo{Problems: []problemInfo{{Level: "error", Text: "no supported GPU: Caminus runs on AMD RX 7900 XTX / XT (gfx1100) cards"}}}
	out := m.render()
	if !strings.Contains(out, "HARDWARE") || !strings.Contains(out, "no supported GPU") {
		t.Fatalf("an error is shown in the basic layout:\n%s", out)
	}
	if m.hwError() == "" {
		t.Fatal("hwError reports the error")
	}
	m.ps = psInfo{Problems: []problemInfo{{Level: "info", Text: "one supported card: the one-card models run; the two-card models need two"}}}
	if strings.Contains(m.render(), "HARDWARE") {
		t.Fatal("a note alone is not shown in the basic layout")
	}
	m.advanced = true
	if !strings.Contains(m.render(), "one supported card") {
		t.Fatal("a note is shown in the advanced layout")
	}
	m.advanced = false
	m.ps = psInfo{Problems: []problemInfo{{Level: "warn", Card: "card2", Text: "XT card2 (0000:06:00.0) is on a PCIe 3.0 x1 link"}}}
	if out := m.render(); !strings.Contains(out, "PCIe 3.0 x1") || m.hwError() != "" {
		t.Fatalf("a warning shows in the basic layout and blocks nothing:\n%s", out)
	}
	m.ps = psInfo{Cards: []cardInfo{{Index: 0, Name: "XTX"}, {Index: 1, Name: "XT"}, {Index: 2, Name: "XT"}}}
	var seen []string
	for i := 0; i < 4; i++ {
		m.card = m.nextCard()
		seen = append(seen, m.card)
	}
	if strings.Join(seen, ",") != "0,1,2,auto" {
		t.Fatalf("the card cycle follows the cards ps lists: %v", seen)
	}
}
