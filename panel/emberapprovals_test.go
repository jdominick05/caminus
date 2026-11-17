// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// approvalsFixture is a conversation with a root request and a question waiting in the bridge.
func approvalsFixture(t *testing.T) emberModel {
	t.Helper()
	m := chatFixture(t)
	bridge := t.TempDir()
	for _, d := range []string{"req", "res"} {
		if err := os.MkdirAll(filepath.Join(bridge, d), 0o770); err != nil {
			t.Fatal(err)
		}
	}
	root, _ := json.Marshal(emberReq{ID: "r1", Pid: 42, User: "agent", CWD: "/work/project",
		Cmd: "apt-get install -y libcmocka-dev", Reason: "the new test needs the cmocka headers",
		Ts: float64(time.Now().Add(-2 * time.Minute).Unix())})
	ask, _ := json.Marshal(emberReq{ID: "q1", Pid: 43, User: "agent", Goal: "original", Kind: "question",
		Text: "Which parser should the empty-input test use?", Choices: []string{"the C one", "the vendored one"},
		Ts: float64(time.Now().Add(-40 * time.Minute).Unix())})
	if err := os.WriteFile(filepath.Join(bridge, "req", "r1.json"), root, 0o660); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bridge, "req", "q1.json"), ask, 0o660); err != nil {
		t.Fatal(err)
	}
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
	m.refreshLocal()
	return m
}

func press(m emberModel, code rune, mod tea.KeyMod) (emberModel, tea.Cmd) {
	msg := tea.KeyPressMsg{Code: code, Mod: mod}
	if mod == 0 && code >= ' ' { // a plain printable key carries its text; special keys and modifiers do not
		msg.Text = string(code)
	}
	next, cmd := m.chatKey(msg)
	return next.(emberModel), cmd
}

// The overlay opens with Ctrl+A and /approvals, draws one row per pending request (root first), closes on Esc, never
// quits on q, and its first a only arms the approval.
func TestApprovalsOverlayOpensListsAndCloses(t *testing.T) {
	for _, open := range []string{"ctrl+a", "/approvals"} {
		m := approvalsFixture(t)
		if open == "ctrl+a" {
			m, _ = press(m, 'a', tea.ModCtrl)
		} else {
			next, _ := m.chatSlash("/approvals")
			m = next.(emberModel)
		}
		if !m.chat.approvals || m.chat.compose.Focused() {
			t.Fatalf("%s: the overlay did not take the conversation (%v)", open, m.chat.approvals)
		}
		frame := ansi.Strip(m.render())
		for _, want := range []string{"approvals  2 pending", g.root + " root", "? ask", "apt-get install -y libcmocka-dev",
			"Which parser should the empty-input test use?", "esc close", "CAMINUS", "⇧Tab sessions"} {
			if !strings.Contains(frame, want) {
				t.Fatalf("%s: the overlay is missing %q:\n%s", open, want, frame)
			}
		}
		if strings.Index(frame, "apt-get install") > strings.Index(frame, "Which parser") {
			t.Fatalf("%s: the root request must be listed before the question:\n%s", open, frame)
		}
		// q is not a quit inside an overlay
		m, cmd := press(m, 'q', 0)
		if cmd != nil || !m.chat.approvals {
			t.Fatalf("%s: q quit the panel or closed the overlay", open)
		}
		// the first a only arms: nothing is decided
		m, _ = press(m, 'a', 0)
		if m.chat.armed != "a:r1" || !strings.Contains(m.msg, "press a again") {
			t.Fatalf("%s: the first a must only arm the approval: %q", open, m.msg)
		}
		if _, err := os.Stat(filepath.Join(m.bridge, "res", "r1.json")); err == nil {
			t.Fatalf("%s: an unconfirmed a wrote a decision", open)
		}
		// any other key disarms, exactly as the inline card does (k stays on the root row)
		m, _ = press(m, 'k', 0)
		if m.chat.armed != "" {
			t.Fatalf("%s: another key must disarm the approval", open)
		}
		// Enter opens the selected request in full
		m, _ = press(m, tea.KeyEnter, 0)
		if frame := ansi.Strip(m.render()); !strings.Contains(frame, "/work/project") ||
			!strings.Contains(frame, "why: the new test needs the cmocka headers") {
			t.Fatalf("%s: Enter did not show the full request:\n%s", open, frame)
		}
		// Esc closes it and gives the composer back
		m, _ = press(m, tea.KeyEscape, 0)
		if m.chat.approvals || !m.chat.compose.Focused() {
			t.Fatalf("%s: Esc did not close the overlay", open)
		}
		if frame := ansi.Strip(m.render()); strings.Contains(frame, "approvals  ") {
			t.Fatalf("%s: the overlay outlived Esc:\n%s", open, frame)
		}
	}
}

// d denies the selected request through the bridge and the audit log, as the Requests page does.
func TestApprovalsOverlayDeniesThroughTheBridge(t *testing.T) {
	m := approvalsFixture(t)
	m, _ = press(m, 'a', tea.ModCtrl)
	m, _ = press(m, 'd', 0)
	data, err := os.ReadFile(filepath.Join(m.bridge, "res", "r1.json"))
	if err != nil || !strings.Contains(string(data), `"denied"`) {
		t.Fatalf("no denial in the bridge: %s %v", data, err)
	}
	if !strings.Contains(mustRead(t, m.auditPath), `"denied"`) {
		t.Fatal("the denial is missing from the audit log")
	}
	if m.pendingRoot() != -1 {
		t.Fatal("a denied request still shows as waiting")
	}
	if frame := ansi.Strip(m.render()); !strings.Contains(frame, "approvals  1 pending") {
		t.Fatalf("the overlay lost the remaining request:\n%s", frame)
	}
}

// A question answers through the same answer mode the Requests page uses (a, then the answer box).
func TestApprovalsOverlayAnswersAQuestion(t *testing.T) {
	m := approvalsFixture(t)
	m, _ = press(m, 'a', tea.ModCtrl)
	m, _ = press(m, 'j', 0) // the question is the second row
	m, _ = press(m, 'a', 0)
	if m.mode != "answer" || m.editRequest == nil || m.editRequest.ID != "q1" {
		t.Fatalf("a on a question should open the answer box, mode=%q", m.mode)
	}
	if m.chat.approvals {
		t.Fatal("the overlay stayed over the page the answer box lives on")
	}
}

// The overlay is one box inside the frame: at 40x16 it must still fit, keys line included.
func TestApprovalsOverlayFitsTheTerminal(t *testing.T) {
	for _, size := range [][2]int{{40, 16}, {100, 30}} {
		m := approvalsFixture(t)
		m.width, m.height = size[0], size[1]
		m, _ = press(m, 'a', tea.ModCtrl)
		m, _ = press(m, tea.KeyEnter, 0) // the detail makes the box as tall as it gets
		rows := strings.Split(m.render(), "\n")
		if len(rows) > size[1] {
			t.Fatalf("%v: %d rows overflow the terminal:\n%s", size, len(rows), ansi.Strip(strings.Join(rows, "\n")))
		}
		for _, row := range rows {
			if ansi.StringWidth(row) > size[0] {
				t.Fatalf("%v: line too wide: %q", size, ansi.Strip(row))
			}
		}
		if frame := ansi.Strip(strings.Join(rows, "\n")); !strings.Contains(frame, "esc close") {
			t.Fatalf("%v: the overlay lost its keys line:\n%s", size, frame)
		}
	}
	// /help's extra footer lines must not push the box past a short terminal
	m := approvalsFixture(t)
	m.width, m.height, m.chat.help = 60, 12, true
	m, _ = press(m, 'a', tea.ModCtrl)
	if rows := strings.Split(m.render(), "\n"); len(rows) > 12 {
		t.Fatalf("help + overlay: %d rows overflow a 12-row terminal:\n%s", len(rows), ansi.Strip(strings.Join(rows, "\n")))
	}
}

// A row may cut a long command, but the opened request must show every character of what a person approves.
func TestApprovalDetailShowsTheWholeCommand(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 60, 30
	bridge := t.TempDir()
	long := "make -C /srv/build/" + strings.Repeat("deep/", 20) + "install-THE-END"
	writeReq(t, bridge, "r1", long, 1)
	m.bridge, m.seen = bridge, map[string]bool{}
	m.refreshLocal()
	m.chat.approvals, m.chat.approvalDetail = true, true
	frame := ansi.Strip(m.approvalOverlay(60, 28))
	joined := strings.ReplaceAll(strings.Join(strings.Fields(strings.NewReplacer("│", "", "cmd:", "").Replace(frame)), ""), " ", "")
	if !strings.Contains(joined, strings.ReplaceAll(long, " ", "")) {
		t.Fatalf("the opened request does not show the whole command:\n%s", frame)
	}
}
