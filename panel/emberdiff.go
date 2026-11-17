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

const editPreview = 8 // the lines a folded edit card still shows

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

// replate puts a background back after every full reset in s (chroma ends tokens with ESC[0m, lipgloss with ESC[m),
// so a plate drawn behind styled text has no holes.
func replate(s, seq string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\x1b[0m", "\x1b[m"), "\x1b[m", "\x1b[m"+seq)
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
func editCard(label, path, meta string, expanded bool, diff string, w int) string {
	body := diffLines(path, diff, max(10, w-5))
	head := toolCard(label, path, meta, true, false, nil, w)
	if len(body) == 0 {
		return head
	}
	rule := "   " + emberDim.Render("│ ")
	shown := body
	if !expanded && len(body) > editPreview {
		// a long change shows its first lines, as a code review does, instead of only its header: the code the
		// agent writes was invisible until Ctrl+O
		shown = body[:editPreview]
	}
	out := []string{strings.Replace(head, g.collapsed, g.expanded, 1)}
	for _, b := range shown {
		out = append(out, rule+b)
	}
	if len(shown) < len(body) {
		out = append(out, rule+emberDim.Render(fmt.Sprintf("… %d more lines · Ctrl+O shows everything", len(body)-len(shown))))
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
	// (a lexer's comment token carries the line's newline inside its colour span, before the reset: a trim at the end
	// missed it and the plate broke onto a second row)
	return strings.ReplaceAll(b.String(), "\n", "")
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
		body := fmt.Sprintf("%4d %s ", num, mark) + highlightLine(lexer, style, terminalText(strings.ReplaceAll(code, "\t", "    "), false))
		if bg != "" {
			// the plate runs the full width: re-applied after every reset inside the highlighted code
			seq := bgSeq(bg)
			body = seq + replate(body, seq)
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

type mdSegment struct {
	code       bool
	lang, text string
}

var fenceRe = regexp.MustCompile("^(\\s*)(```+|~~~+)\\s*([\\w+#.-]*)")

// splitFences cuts markdown into prose and fenced code blocks (``` or ~~~, with an optional language); an unclosed
// fence runs to the end, as markdown reads it.
func splitFences(text string) []mdSegment {
	var out []mdSegment
	var buf []string
	inCode, fence, lang := false, "", ""
	flush := func(code bool) {
		out = append(out, mdSegment{code: code, lang: lang, text: strings.Join(buf, "\n")})
		buf = nil
	}
	for _, ln := range strings.Split(text, "\n") {
		m := fenceRe.FindStringSubmatch(ln)
		switch {
		case !inCode && m != nil:
			flush(false)
			inCode, fence, lang = true, m[2], m[3]
		case inCode && m != nil && strings.HasPrefix(strings.TrimSpace(ln), fence) && strings.TrimSpace(ln) == strings.Repeat(fence[:1], len(strings.TrimSpace(ln))):
			flush(true)
			inCode, lang = false, ""
		default:
			buf = append(buf, ln)
		}
	}
	flush(inCode)
	return out
}

// codePlate draws a fenced block: the language in dim on the plate's first line, then the code highlighted for it
// (the theme's code_theme) on the theme's codeblock_bg, every line the block's full width; long lines wrap inside.
func codePlate(lang, code string, w int) string {
	lexer := lexers.Get(lang)
	if lexer == nil {
		lexer = lexers.Analyse(code)
	}
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)
	style := styles.Get(themeRole("code_theme"))
	if style == nil {
		style = styles.Fallback
	}
	seq := bgSeq(themeRole("codeblock_bg"))
	inner := max(4, w-2)
	plate := func(s string) string {
		s = seq + " " + replate(s, seq)
		if pad := w - ansi.StringWidth(s); pad > 0 {
			s += strings.Repeat(" ", pad)
		}
		return "  " + s + "\x1b[0m"
	}
	label := lang
	if label == "" {
		label = "code"
	}
	out := []string{plate(emberDim.Render(label))}
	for _, ln := range strings.Split(strings.TrimRight(code, "\n"), "\n") {
		hl := highlightLine(lexer, style, strings.ReplaceAll(ln, "\t", "    "))
		for _, piece := range strings.Split(ansi.Hardwrap(hl, inner, true), "\n") {
			out = append(out, plate(piece))
		}
	}
	return strings.Join(out, "\n")
}
