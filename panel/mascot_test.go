package main

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// The ember is five rows of ten cells in every tier, both frames, and the empty chat shows it.
func TestMascotShapeAndEmptyChat(t *testing.T) {
	saved := g
	defer func() { g = saved }()
	for _, tier := range []string{"unicode", "nerd", "ascii"} {
		g = glyphTiers[tier]
		for frame := 0; frame < 2; frame++ {
			rows := mascot(frame)
			if len(rows) != 5 {
				t.Fatalf("%s frame %d: %d rows", tier, frame, len(rows))
			}
			for _, r := range rows {
				if w := lipgloss.Width(r); w > 10 {
					t.Fatalf("%s frame %d: row %q is %d wide", tier, frame, ansi.Strip(r), w)
				}
			}
		}
	}
	g = glyphTiers["unicode"]
	m := chatFixture(t)
	m.residentEvents, m.chat.content, m.chat.rendered = nil, "", 0
	m.chat.transcript.SetContent("")
	m.syncChat()
	if v := ansi.Strip(m.chat.transcript.View()); !strings.Contains(v, "▀") || !strings.Contains(v, "CAMINUS") {
		t.Fatalf("the empty chat does not show the ember:\n%s", v)
	}
}
