package main

import (
	"os"
	"testing"
)

// TestMain pins the shipped defaults, so a developer's own saved glyph tier (~/.config/caminus/panel-glyphs) cannot
// change what the tests see.
func TestMain(m *testing.M) {
	g = glyphTiers["unicode"]
	os.Exit(m.Run())
}
