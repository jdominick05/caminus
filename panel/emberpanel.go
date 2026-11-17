// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberpanel.go: /panel, the Caminus panel (models, servers, the cards) full screen inside Ember. It is the same
// component `caminus panel` hosts: Ember forwards every message to it while it is open and draws its View in
// Ember's whole frame; Esc, q or e close it and the chat comes back as it was (the draft, the focus; a request that
// arrived meanwhile asks or waits as usual).

import (
	"encoding/json"

	tea "charm.land/bubbletea/v2"
)

// openPanel builds a fresh component (each /panel starts from what is on the machine now) and starts it.
func (m emberModel) openPanel() (emberModel, tea.Cmd) {
	// as a JSON array: the argv may hold spaces (caminus ember ui's interpreter and boot script), and a joined string
	// was split apart again
	cli := ""
	if len(m.cli) > 0 {
		b, _ := json.Marshal(m.cli)
		cli = string(b)
	}
	p := newPanel(panelOptions{CLI: cli, Prefs: prefsPath(), Hosted: true})
	m.hosted, m.hostedOpen = p, true
	m.chat.compose.Blur()
	return m, p.Init()
}

// panelUpdate routes a message while /panel is open: keys and the mouse go to the panel only; everything else goes
// to the panel and on to Ember (the conversation, the requests and the status keep arriving underneath).
func (m emberModel) panelUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	if _, ok := msg.(panelCloseMsg); ok {
		m.hostedOpen = false
		m.hosted = ctl{}
		m.syncChat()
		if m.chat.question == nil {
			return m, m.chat.compose.Focus() // back to typing, the draft as it was
		}
		return m, nil
	}
	var pc tea.Cmd
	m.hosted, pc = m.hosted.Update(msg)
	switch msg.(type) {
	case tea.KeyPressMsg, tea.MouseClickMsg, tea.MouseWheelMsg, tea.MouseMotionMsg, tea.MouseReleaseMsg, tea.PasteMsg:
		return m, pc
	}
	m.hostedOpen = false // (Ember's own handling runs as if the panel were shut, then it is open again)
	next, ec := m.Update(msg)
	em := next.(emberModel)
	em.hostedOpen = true
	em.hosted = m.hosted
	return em, tea.Batch(pc, ec)
}
