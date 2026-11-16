// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
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
