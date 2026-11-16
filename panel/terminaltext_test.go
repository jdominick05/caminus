// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"
)

func TestApprovalTextShowsControlsWithoutChangingRequest(t *testing.T) {
	command := "printf actual #\x1b[2K\rprintf harmless\u009b2K\u202e"
	safe := terminalText(command, false)
	for _, control := range []rune{'\x1b', '\r', '\u009b', '\u202e'} {
		if strings.ContainsRune(safe, control) {
			t.Fatalf("terminal control retained: %q", safe)
		}
	}
	for _, marker := range []string{"printf actual", `\x1b`, `\r`, `\u202e`} {
		if !strings.Contains(safe, marker) {
			t.Fatalf("missing visible command/escape %q in %q", marker, safe)
		}
	}
	m := emberModel{mode: "edit", edit: command, reqs: []emberReq{{ID: "review", Cmd: command,
		User: "agent\x1b[2J", CWD: "work\rhidden", Reason: "why\x1b[2K"}}}
	view := m.reqPane()
	if strings.Contains(view, command) || strings.Contains(view, "work\rhidden") || !strings.Contains(view, "printf actual") {
		t.Fatalf("approval text was not made visible: %q", view)
	}
	if m.reqs[0].Cmd != command {
		t.Fatal("display altered the command that will be claimed")
	}
	if !strings.Contains(m.render(), `\x1b`) {
		t.Fatal("edit view lost the visible escape")
	}
}

func TestTerminalOutputPreservesLinesWithoutTerminalInstructions(t *testing.T) {
	out := terminalText("first\nsecond\tcolumn\x1b[2K\rhidden", true)
	if !strings.Contains(out, "first\nsecond\tcolumn") || strings.ContainsAny(out, "\x1b\r") {
		t.Fatal(out)
	}
}
