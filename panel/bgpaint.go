// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// bgpaint.go: a theme's background. The solid colour (bg) is painted into every cell (terminals may ignore a request
// to change their own background), light and pink themes included. On top of it:
//   - gradient: each row shaded from bg to bg2, re-applied after every reset that clears a background, so text sits on
//     the gradient instead of in boxes of the flat colour;
//   - dots, grid, diagonal, waves, stars: a faint pattern in the pattern colour, drawn only in runs of three or more
//     blank cells (one blank kept beside any text), never in a cell that has a background of its own (a selection, a
//     code plate, a pill), so the gaps between words and every element keep their look.
// Computed from the cell's position: it fits any terminal size (a vector background, as a terminal can draw one).

import (
	"fmt"
	"image/color"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/charmbracelet/x/ansi"
)

// paintBackground paints the theme's background into a rendered frame: every cell (many terminals, GNOME Terminal
// among them, ignore a program's request to change their background colour, so a pink theme drew dark text on the
// terminal's own dark background), as a gradient when the theme has one, and its pattern in rows above protectFrom
// (not the composer and status line; patterns false for overlays). Rows are padded to width and the frame to
// height, so the colour reaches the edges.
func paintBackground(frame string, protectFrom, width, height int, patterns bool) string {
	pat := themeRole("bg_pattern")
	bg, bg2 := themeRole("bg"), themeRole("bg2")
	if _, _, _, ok := hexRGB(bg); !ok {
		return frame
	}
	gradient := pat == "gradient" && bg2 != ""
	glyphs := patterns && pat != "" && pat != "none" && pat != "gradient"
	ink := themeRole("pattern")
	if ink == "" {
		ink = themeRole("dim")
	}
	inkSeq := sgrColor(38, ink)
	lines := strings.Split(frame, "\n")
	for len(lines) < height {
		lines = append(lines, "")
	}
	n := len(lines)
	for y, line := range lines {
		rowBg := sgrColor(48, bg)
		if gradient {
			rowBg = sgrColor(48, mixHex(bg, bg2, float64(y)/math.Max(1, float64(n-1))))
		}
		rowBg += sgrColor(38, themeRole("text")) // and the theme's text as the default colour (re-applied with it)
		if pad := width - ansi.StringWidth(line); pad > 0 {
			line += "\x1b[0m" + strings.Repeat(" ", pad) // (the padding is plain blanks: painted, and patterned)
		}
		lines[y] = paintLine(line, y, rowBg, inkSeq, pat, glyphs && y < protectFrom) + "\x1b[0m"
	}
	return strings.Join(lines, "\n")
}

// bgCell is one written piece of a row: a rune (x its column) or an escape sequence (x -1, no width).
type bgCell struct {
	text  string
	blank bool // a space with no background of its own
	x     int
}

// paintLine walks one row: SGR state is tracked (what is active since the last reset, whether a background is set),
// the row's gradient is put back after any reset, and blank runs get pattern glyphs.
func paintLine(line string, y int, rowBg, inkSeq, pat string, glyphs bool) string {
	var cells []bgCell
	var active []string // SGR sequences since the last reset
	hasBg := false
	x := 0
	for i := 0; i < len(line); {
		if line[i] == 0x1b {
			j := i + 1
			if j < len(line) && line[j] == '[' {
				j++
				for j < len(line) && (line[j] < 0x40 || line[j] > 0x7e) {
					j++
				}
				if j < len(line) {
					j++
				}
			} else {
				j = i + 2
			}
			seq := line[i:min(j, len(line))]
			out := seq
			if strings.HasSuffix(seq, "m") && strings.HasPrefix(seq, "\x1b[") {
				params := seq[2 : len(seq)-1]
				if params == "" || params == "0" {
					active, hasBg = nil, false
					out += rowBg // (the gradient survives a reset)
				} else {
					active = append(active, seq)
					if sgrSetsBg(params) {
						hasBg = true
					}
					if sgrClearsBg(params) || sgrClearsFg(params) {
						if sgrClearsBg(params) {
							hasBg = false
						}
						out += rowBg // (the default colours return after a 49 or 39)
					}
				}
			}
			cells = append(cells, bgCell{text: out, x: -1})
			i = min(j, len(line))
			continue
		}
		r, size := utf8.DecodeRuneInString(line[i:])
		cells = append(cells, bgCell{text: line[i : i+size], blank: r == ' ' && !hasBg, x: x})
		x += max(0, ansi.StringWidth(string(r)))
		i += size
	}
	if glyphs {
		// runs of blank cells (escape sequences in between do not break a run); the first and last blank of a run
		// stay plain, so text keeps a margin
		var run []int
		flush := func() {
			if len(run) >= 3 {
				for _, k := range run[1 : len(run)-1] {
					if g := patternGlyph(pat, cells[k].x, y); g != "" {
						// the pattern's ink, the glyph, then the state as it was (a reset, the gradient, the active SGRs)
						cells[k].text = inkSeq + g + "\x1b[0m" + rowBg + strings.Join(activeAt(cells, k), "")
					}
				}
			}
			run = run[:0]
		}
		for k, c := range cells {
			switch {
			case c.x < 0:
			case c.blank:
				run = append(run, k)
			default:
				flush()
			}
		}
		flush()
	}
	var b strings.Builder
	b.WriteString(rowBg)
	for _, c := range cells {
		b.WriteString(c.text)
	}
	return b.String()
}

// activeAt replays the SGR sequences active at cell k (since the last reset before it).
func activeAt(cells []bgCell, k int) []string {
	var active []string
	for i := 0; i < k; i++ {
		t := cells[i].text
		if cells[i].x >= 0 || !strings.HasPrefix(t, "\x1b[") {
			continue
		}
		seq := t
		if j := strings.IndexByte(t, 'm'); j >= 0 {
			seq = t[:j+1]
		}
		params := strings.TrimSuffix(strings.TrimPrefix(seq, "\x1b["), "m")
		if params == "" || params == "0" {
			active = nil
		} else if strings.HasSuffix(seq, "m") {
			active = append(active, seq)
		}
	}
	return active
}

// patternGlyph is the pattern's mark at a cell, or "" for a plain blank.
func patternGlyph(pat string, x, y int) string {
	switch pat {
	case "dots":
		if x%4 == 0 && y%2 == 0 {
			return "·"
		}
	case "grid":
		if y%4 == 0 && x%8 == 0 {
			return "+"
		}
		if y%4 == 0 && x%2 == 0 {
			return "·"
		}
	case "diagonal":
		if (x+y*2)%8 == 0 {
			return "╱"
		}
	case "waves":
		if y%3 == 0 && (x+int(3*math.Sin(float64(y)/2)))%6 == 0 {
			return "∼"
		}
	case "stars":
		h := uint32(x*73856093) ^ uint32(y*19349663)
		h ^= h >> 13
		h *= 0x5bd1e995
		h ^= h >> 15
		switch {
		case h%401 == 0: // sparse: a sky, not noise
			return "✦"
		case h%119 == 0:
			return "·"
		}
	}
	return ""
}

// sgrAttrs is an SGR parameter string's attribute codes, without the colour arguments that follow 38 / 48 / 58
// (5;n or 2;r;g;b): read field by field, \x1b[38;2;39;200;100m looked like "default foreground" (39).
func sgrAttrs(p string) []string {
	f := strings.Split(p, ";")
	var out []string
	for i := 0; i < len(f); i++ {
		out = append(out, f[i])
		if (f[i] == "38" || f[i] == "48" || f[i] == "58") && i+1 < len(f) {
			switch f[i+1] {
			case "5":
				i += 2
			case "2":
				i += 4
			}
		}
	}
	return out
}

func sgrSetsBg(p string) bool {
	for _, f := range sgrAttrs(p) {
		if f == "48" || (len(f) == 2 && f[0] == '4' && f[1] >= '0' && f[1] <= '7') || (len(f) == 3 && f[:2] == "10") {
			return true
		}
	}
	return false
}

func sgrClearsFg(p string) bool {
	for _, f := range sgrAttrs(p) {
		if f == "39" {
			return true
		}
	}
	return false
}

func sgrClearsBg(p string) bool {
	for _, f := range sgrAttrs(p) {
		if f == "49" {
			return true
		}
	}
	return false
}

// sgrColor is the truecolor SGR for a #rrggbb hex (base 38 foreground, 48 background); "" for anything else.
func sgrColor(base int, hex string) string {
	r, g, b, ok := hexRGB(hex)
	if !ok {
		return ""
	}
	return fmt.Sprintf("\x1b[%d;2;%d;%d;%dm", base, r, g, b)
}

func hexRGB(hex string) (r, g, b uint8, ok bool) {
	if len(hex) != 7 || hex[0] != '#' {
		return 0, 0, 0, false
	}
	var v uint32
	if _, err := fmt.Sscanf(hex[1:], "%06x", &v); err != nil {
		return 0, 0, 0, false
	}
	return uint8(v >> 16), uint8(v >> 8), uint8(v), true
}

// mixHex blends two #rrggbb colours (t from 0, a, to 1, b).
func mixHex(a, b string, t float64) string {
	ar, ag, ab, ok1 := hexRGB(a)
	br, bg, bb, ok2 := hexRGB(b)
	if !ok1 || !ok2 {
		return a
	}
	m := func(x, y uint8) uint8 { return uint8(math.Round(float64(x) + (float64(y)-float64(x))*t)) }
	return fmt.Sprintf("#%02x%02x%02x", m(ar, br), m(ag, bg), m(ab, bb))
}

// themeBackground is the theme's background colour (nil: none, the terminal's own shows).
func themeBackground() color.Color {
	if _, _, _, ok := hexRGB(themeRole("bg")); ok {
		return themeColor("bg")
	}
	return nil
}
