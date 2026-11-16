// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// mascot.go: the Caminus ember, the project's mascot. Caminus is Latin for furnace; the mascot is a coal from it with
// one flame on top. A 10x10 pixel map, printed with half-block characters (two pixel rows per text row) in the unicode
// and nerd tiers and as plain text in the ascii tier. Frame 1 is the flame flickering, for "working".

import (
	"strings"

	"charm.land/lipgloss/v2"
)

// emberFrames are the sprite's pixel maps: k coal, w eye, o ember, y flame tip, . empty.
var emberFrames = [2][10]string{
	// (the empty pixel row is on top: the feet fill the bottom half of the last cell, so Ember stands on what is under
	// it instead of hovering half a row above it)
	{
		"..........",
		"....o.....",
		"...oyo....",
		"..kkkkkk..",
		".kkkkkkkk.",
		".kwkkkkwk.",
		".kwkkkkwk.",
		".kkkkkkkk.",
		".kkkkkkkk.",
		"..k....k..",
	},
	{
		"..........",
		".....o....",
		"...oyyo...",
		"..kkkkkk..",
		".kkkkkkkk.",
		".kwkkkkwk.",
		".kwkkkkwk.",
		".kkkkkkkk.",
		".kkkkkkkk.",
		"..k....k..",
	},
}

// emberASCII is the ascii tier's mascot: no block characters, the same shape.
// emberSmall is Ember at 8x6 pixels (three rows): the /btw answer's own, smaller Ember. The empty pixel row is on
// top, so it stands on the line under it.
var emberSmall = [2][6]string{
	{"........", "...o....", "..kkkk..", ".kwkkwk.", ".kkkkkk.", "..k..k.."},
	{"........", "....o...", "..kkkk..", ".kwkkwk.", ".kkkkkk.", "..k..k.."},
}

// mascotSmall is the small Ember as three rows of half blocks (a plain face with the ascii glyphs).
func mascotSmall(frame int) []string {
	if g.name == "ascii" {
		return []string{" ,^. ", "(o o)", " ' ' "}
	}
	px := emberSmall[frame%2]
	var rows []string
	for y := 0; y < len(px); y += 2 {
		var b strings.Builder
		for x := 0; x < len(px[y]); x++ {
			top, okT := emberColor(px[y][x])
			bot, okB := emberColor(px[y+1][x])
			switch {
			case !okT && !okB:
				b.WriteString(" ")
			case okT && !okB:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(top)).Render("▀"))
			case !okT && okB:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(bot)).Render("▄"))
			default:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(top)).Background(lipgloss.Color(bot)).Render("▀"))
			}
		}
		rows = append(rows, b.String())
	}
	return rows
}

var emberASCII = []string{"   ,^.  ", " .----. ", " |o  o| ", " '----' ", "  '  '  "}

// emberColor maps a pixel to its colour. The tones are fixed, so the mascot looks the same in every theme; the coal is
// a warm charcoal rather than black, so it still shows on a black terminal.
func emberColor(ch byte) (string, bool) {
	switch ch {
	case 'k':
		return "#4a3e39", true
	case 'w':
		return "#fff6ea", true
	case 'o':
		return "#ff7a2f", true
	case 'y':
		return "#ffd75e", true
	}
	return "", false
}

// mascot renders the ember (frame 0 at rest, 1 while working) as five text rows.
func mascot(frame int) []string {
	if g.name == "ascii" {
		return emberASCII
	}
	m := emberFrames[frame%2]
	rows := make([]string, 0, 5)
	for y := 0; y < 10; y += 2 {
		var b strings.Builder
		for x := 0; x < 10; x++ {
			top, okT := emberColor(m[y][x])
			bot, okB := emberColor(m[y+1][x])
			switch {
			case !okT && !okB:
				b.WriteString(" ")
			case okT && !okB:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(top)).Render("▀"))
			case !okT && okB:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(bot)).Render("▄"))
			default:
				b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(top)).Background(lipgloss.Color(bot)).Render("▀"))
			}
		}
		rows = append(rows, b.String())
	}
	return rows
}

// mascotBeside puts the ember to the left of a few text lines (the empty chat, the splash).
func mascotBeside(frame int, lines ...string) string {
	art := mascot(frame)
	out := make([]string, len(art))
	for i := range art {
		text := ""
		if i < len(lines) {
			text = lines[i]
		}
		out[i] = art[i] + "  " + text
	}
	return strings.Join(out, "\n")
}
