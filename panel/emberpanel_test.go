// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

// /panel opens the panel component in Ember's frame, forwards its ticks, keeps the chat's messages arriving, and Esc
// brings the chat back with the draft and the focus as they were (TestThePanelInAHost, with Ember as the host).
func TestPanelOpensInsideEmberAndEscComesBack(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // (the panel's remembered choices: never the human's)
	m := chatFixture(t)
	m.width, m.height = 120, 40
	m.cli = []string{"caminus-not-run"}
	next, cmd := m.chatSlash("/panel")
	m = next.(emberModel)
	m.chat.compose.SetValue("a draft I keep") // (whatever the composer holds stays while the panel is open)
	if !m.hostedOpen || cmd == nil || m.chat.compose.Focused() {
		t.Fatal("/panel opens the component and takes the keys")
	}
	// the open message (one of Init's batch) starts the panel's ticks
	var opened tea.Msg
	if batch, ok := cmd().(tea.BatchMsg); ok {
		for _, c := range batch {
			if c == nil {
				continue
			}
			if msg, ok := callFast(c); ok {
				if _, is := msg.(panelOpenMsg); is {
					opened = msg
				}
			}
		}
	}
	if opened == nil {
		t.Fatal("Init sends panelOpenMsg")
	}
	next, _ = m.Update(opened)
	m = next.(emberModel)
	frame := m.hosted.frame
	next, _ = m.Update(ctlTick{m.hosted.id, m.hosted.epoch})
	m = next.(emberModel)
	if m.hosted.frame != frame+1 {
		t.Fatal("the panel's own tick reaches it through Ember")
	}
	if v := ansi.Strip(m.render()); strings.Contains(v, "a draft I keep") || strings.Contains(v, "⇧Tab sessions") {
		t.Fatalf("the panel fills the frame:\n%s", v)
	}
	// the chat's messages keep arriving underneath
	st := m.resident
	st.Model = "swift-1.5"
	next, _ = m.Update(residentMsg{agent: m.agentName, epoch: m.residentEpoch, status: st})
	m = next.(emberModel)
	if !m.hostedOpen || m.resident.Model != "swift-1.5" {
		t.Fatal("a status update reaches Ember while the panel is open, and the panel stays")
	}
	// a key is the panel's, never the composer's
	next, _ = m.Update(tea.KeyPressMsg{Code: 'x', Text: "x"})
	m = next.(emberModel)
	if m.chat.compose.Value() != "a draft I keep" {
		t.Fatalf("a key typed in the panel does not reach the draft: %q open=%v", m.chat.compose.Value(), m.hostedOpen)
	}
	// Esc: the panel asks to close; Ember shows the chat again with the draft and the focus
	next, cmd = m.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	if cmd == nil {
		t.Fatal("Esc asks to close")
	}
	closeMsg, ok := callFast(cmd)
	if _, is := closeMsg.(panelCloseMsg); !ok || !is {
		t.Fatalf("the close message: %T", closeMsg)
	}
	next, _ = m.Update(closeMsg)
	m = next.(emberModel)
	if m.hostedOpen || !m.chat.compose.Focused() || m.chat.compose.Value() != "a draft I keep" {
		t.Fatal("the chat comes back as it was")
	}
	if v := ansi.Strip(m.render()); !strings.Contains(v, "⇧Tab sessions") {
		t.Fatalf("the chat's frame is back:\n%s", v)
	}
	if !strings.Contains(ansi.Strip(strings.Join(helpLines(120), "\n")), "/panel") {
		t.Fatal("/help lists /panel")
	}
}
