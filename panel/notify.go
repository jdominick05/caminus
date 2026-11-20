// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// notify.go: desktop notifications for what needs the owner, once per item: a new root request or question, a
// need_you report, a goal that pauses or fails, a goal that completes. What is already pending when the panel starts
// is marked seen without a notification, so a restart does not repeat them. CAMINUS_PANEL_NOTIFY=0 turns them off.
// The badge (the pending count in the status line) and the OpenRGB state lights are the other two channels.

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// notifyRun shows one notification; tests replace it.
var notifyRun = func(urgency, title, body string) {
	// Never from a test binary: the suite drives fake root requests and goals, and each one reached the owner's
	// desktop as a real notification.
	if os.Getenv("CAMINUS_PANEL_NOTIFY") == "0" || testing.Testing() || demoMode {
		return
	}
	if path, err := exec.LookPath("notify-send"); err == nil {
		_ = exec.Command(path, "-a", "Ember", "-u", urgency, title, body).Start()
	}
}

type notifier struct {
	seen   map[string]bool
	primed map[string]bool // which sources have had their first, silent pass
}

// note notifies about key once; on a source's first pass it only records what is already there.
func (n *notifier) note(source, key, urgency, title, body string) {
	if n.seen == nil {
		n.seen, n.primed = map[string]bool{}, map[string]bool{}
	}
	if n.seen[key] {
		return
	}
	n.seen[key] = true
	if n.primed[source] {
		notifyRun(urgency, title, clip(body, 200))
	}
}

func (n *notifier) prime(source string) {
	if n.seen == nil {
		n.seen, n.primed = map[string]bool{}, map[string]bool{}
	}
	n.primed[source] = true
}

func clip(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

// notifyRequests: every bridge request not seen before.
func (m *emberModel) notifyRequests() {
	for _, r := range m.reqs {
		title := "Ember: a question from the agent"
		body := r.Text
		if r.Kind != "question" { // an empty kind is a root request
			title, body = "Ember: root approval needed", r.Cmd
		}
		m.notes.note("requests", "req:"+r.ID, "critical", title, body)
	}
	m.notes.prime("requests")
}

// notifyResident: need_you reports and goals reaching a stop.
func (m *emberModel) notifyResident(events []residentEvent) {
	agent := m.agentName
	for _, e := range events {
		if e.Kind == "report" && e.Data["level"] == "need_you" {
			text, _ := e.Data["text"].(string)
			m.notes.note("resident:"+agent, "ev:"+agent+":"+itoa(e.Seq), "critical", "Ember "+agent+" needs you", text)
		}
	}
	for _, g := range m.resident.Goals {
		key := "goal:" + agent + ":" + g.ID + ":" + g.State
		switch g.State {
		case "completed":
			m.notes.note("resident:"+agent, key, "normal", "Ember "+agent+": goal done", firstNonEmpty(g.Summary, g.Prompt))
		case "paused", "error":
			m.notes.note("resident:"+agent, key, "critical", "Ember "+agent+": goal "+g.State, firstNonEmpty(g.Error, g.Prompt))
		}
	}
	m.notes.prime("resident:" + agent)
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if strings.TrimSpace(x) != "" {
			return x
		}
	}
	return ""
}

func itoa(i int) string { return strconv.Itoa(i) }
