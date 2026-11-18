// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// terminalText leaves reviewed text readable while showing control and Unicode
// formatting characters as escapes. The raw request stays unchanged for claims.
func terminalText(s string, multiline bool) string {
	var out strings.Builder
	for _, r := range s {
		if unicode.IsPrint(r) || (multiline && (r == '\n' || r == '\t')) {
			out.WriteRune(r)
		} else {
			quoted := strconv.QuoteRune(r)
			out.WriteString(quoted[1 : len(quoted)-1])
		}
	}
	return out.String()
}

// screenSafe is what a renderer's output (glamour's, chroma's) may send the terminal: its text through terminalText
// (printable runes, \n and \t; every other control, C1 and bidi override or isolate shown as an escape) and its SGR
// colour sequences (CSI digits m). Every other escape sequence is dropped whole: cursor and screen CSIs, OSC (titles,
// the clipboard, OSC 8 links), DCS, APC, PM and SOS. The text was sanitised before rendering, but glamour turns HTML
// entities back into the bytes they name (&#27; is ESC), so its output is filtered again here.
func screenSafe(s string) string {
	var out strings.Builder
	for len(s) > 0 {
		i := strings.IndexByte(s, 0x1b)
		if i < 0 {
			out.WriteString(terminalText(s, true))
			break
		}
		out.WriteString(terminalText(s[:i], true))
		seq, n := escapeSequence(s[i:])
		if sgrRe.MatchString(seq) {
			out.WriteString(seq)
		}
		s = s[i+n:]
	}
	return out.String()
}

var sgrRe = regexp.MustCompile(`^\x1b\[[0-9;:]*m$`)

// escapeSequence is the escape sequence s starts with (s[0] is ESC) and its length in bytes; an unterminated one runs
// to the end of s.
func escapeSequence(s string) (string, int) {
	if len(s) < 2 {
		return s, len(s)
	}
	switch s[1] {
	case '[': // CSI: parameter and intermediate bytes, then one final byte
		for j := 2; j < len(s); j++ {
			if s[j] >= 0x40 && s[j] <= 0x7e {
				return s[:j+1], j + 1
			}
			if s[j] < 0x20 || s[j] > 0x7e {
				return s[:j], j // (a broken one ends before the byte that cannot belong to it)
			}
		}
	case ']', 'P', '_', '^', 'X': // OSC, DCS, APC, PM, SOS: up to BEL (OSC) or ST (ESC \)
		for j := 2; j < len(s); j++ {
			if s[j] == 0x07 && s[1] == ']' {
				return s[:j+1], j + 1
			}
			if s[j] == 0x1b && j+1 < len(s) && s[j+1] == '\\' {
				return s[:j+2], j + 2
			}
		}
	default:
		return s[:2], 2
	}
	return s, len(s)
}
