// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberdiff.go: an edit card's diff the way a code review shows it: the file's line numbers (from the hunk headers),
// added and removed lines on a tinted plate, and the code highlighted for the file's language with the theme's
// code_theme. Before, an edit was a column of dim unified-diff text with no numbers.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/x/ansi"
)

var hunkRe = regexp.MustCompile(`^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

// diffPlates are the added / removed backgrounds: the theme's ok and hot colours blended into its background (or into
// a dark one, for a theme that keeps the terminal's), so the tint reads on light themes and dark ones alike.
func diffPlates() (add, del string) {
	bg := themeRole("bg")
	if bg == "" {
		bg = "#16181d"
	}
	return blendHex(bg, themeRole("ok"), 0.22), blendHex(bg, themeRole("hot"), 0.22)
}

func blendHex(a, b string, t float64) string {
	pa, pb := diffHex(a), diffHex(b)
	if pa == nil || pb == nil {
		return a
	}
	var out [3]int
	for i := range out {
		out[i] = int(float64(pa[i])*(1-t) + float64(pb[i])*t + 0.5)
	}
	return fmt.Sprintf("#%02x%02x%02x", out[0], out[1], out[2])
}

// bgSeq is the 24-bit background SGR for a #rrggbb colour (empty when it is not one).
func bgSeq(hex string) string {
	c := diffHex(hex)
	if c == nil {
		return ""
	}
	return fmt.Sprintf("\x1b[48;2;%d;%d;%dm", c[0], c[1], c[2])
}

// editCard is toolCard for an edit: the same header, then the diff as diffLines draws it under the card's rule.
func editCard(path, meta string, expanded bool, diff string, w int) string {
	body := diffLines(path, diff, max(10, w-5))
	head := toolCard("edit", path, meta, true, false, nil, w)
	if !expanded || len(body) == 0 {
		return head // folded (Ctrl+E opens every card), or nothing changed
	}
	rule := "   " + emberDim.Render("│ ")
	out := []string{strings.Replace(head, g.collapsed, g.expanded, 1)}
	for _, b := range body {
		out = append(out, rule+b)
	}
	return strings.Join(out, "\n")
}

func diffHex(h string) []int {
	h = strings.TrimPrefix(h, "#")
	if len(h) != 6 {
		return nil
	}
	v, err := strconv.ParseUint(h, 16, 32)
	if err != nil {
		return nil
	}
	return []int{int(v >> 16 & 255), int(v >> 8 & 255), int(v & 255)}
}

// highlightLine is one line of code in the file's language, as ANSI (no trailing reset newline); plain on any error.
func highlightLine(lexer chroma.Lexer, style *chroma.Style, code string) string {
	it, err := lexer.Tokenise(nil, code)
	if err != nil {
		return code
	}
	var b strings.Builder
	if formatters.TTY16m.Format(&b, style, it) != nil {
		return code
	}
	return strings.TrimRight(b.String(), "\n")
}

// diffLines renders a unified diff body (without its ---/+++ header) for the file at path, w columns wide.
func diffLines(path, diff string, w int) []string {
	lexer := lexers.Match(path)
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)
	style := styles.Get(themeRole("code_theme"))
	if style == nil {
		style = styles.Fallback
	}
	addBg, delBg := diffPlates()
	oldN, newN := 0, 0
	var out []string
	for _, ln := range strings.Split(diff, "\n") {
		if m := hunkRe.FindStringSubmatch(ln); m != nil {
			oldN, _ = strconv.Atoi(m[1])
			newN, _ = strconv.Atoi(m[2])
			if len(out) > 0 {
				out = append(out, emberDim.Render("     ⋮"))
			}
			continue
		}
		if ln == "" || strings.HasPrefix(ln, "…") {
			if ln != "" {
				out = append(out, emberDim.Render(ln))
			}
			continue
		}
		mark, code := ln[:1], ln[1:]
		num, bg := newN, ""
		switch mark {
		case "+":
			bg = addBg
			newN++
		case "-":
			num, bg = oldN, delBg
			oldN++
		default:
			mark = " "
			oldN++
			newN++
		}
		body := fmt.Sprintf("%4d %s ", num, mark) + highlightLine(lexer, style, terminalText(code, false))
		if bg != "" {
			// the plate runs the full width: re-applied after every reset inside the highlighted code
			seq := bgSeq(bg)
			body = seq + strings.ReplaceAll(body, "\x1b[0m", "\x1b[0m"+seq)
			if pad := w - ansi.StringWidth(body); pad > 0 {
				body += strings.Repeat(" ", pad)
			}
			body += "\x1b[0m"
		} else {
			body = emberDim.Render(fmt.Sprintf("%4d", num)) + body[4:]
		}
		out = append(out, ansi.Truncate(body, w, "…"))
	}
	return out
}
