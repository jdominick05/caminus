// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"charm.land/lipgloss/v2"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

type residentSessionsMsg struct {
	agent, memory, err string
	names              []string
}

// sessionCardsMsg: the switcher's cards, loaded after the names so the session list never waits on them.
type sessionCardsMsg struct {
	agent string
	cards []sessionCard
}

// loadSessionCards reads every session's card at once (each command has residentCommand's own deadline).
func loadSessionCards(m emberModel, names []string) tea.Cmd {
	return func() tea.Msg {
		cards := make([]sessionCard, len(names))
		var wg sync.WaitGroup
		for i, n := range names {
			wg.Add(1)
			go func() { defer wg.Done(); cards[i] = loadSessionCard(m, n) }()
		}
		wg.Wait()
		return sessionCardsMsg{agent: m.agentName, cards: cards}
	}
}

type residentCreatedMsg struct{ name, err string }

// residentDestroyedMsg: a session the switcher removed, and where its home is kept.
type residentDestroyedMsg struct{ name, kept, err string }

// destroyResidentSession cancels every live goal of the session (which withdraws their open requests), then removes it:
// its service stops, its name leaves the mail bus, its home moves to removed-sessions.
func destroyResidentSession(m emberModel, name string, active []string) tea.Cmd {
	return func() tea.Msg {
		// a runtime from before `remove` refuses here, before the goal is cancelled: nothing changes half-way
		if out, err := runResident(residentCommand(m, []string{"remove", name, "--dry-run", "--json"}, "")); err != nil {
			text := strings.TrimSpace(string(out))
			if strings.Contains(text, "invalid choice") {
				text = "the agent's runtime predates destroy: run caminus ember agent install, then try again"
			}
			return residentDestroyedMsg{name: name, err: text}
		}
		for _, id := range active { // every live goal, so none leaves a request behind for a session that is gone
			_, _ = runResident(residentCommand(m, []string{"cancel", name, id, "--json"}, ""))
		}
		out, err := runResident(residentCommand(m, []string{"remove", name, "--json"}, ""))
		if err != nil {
			return residentDestroyedMsg{name: name, err: strings.TrimSpace(string(out))}
		}
		var r struct {
			Kept string `json:"kept"`
		}
		if i := strings.LastIndex(string(out), "{"); i >= 0 {
			_ = json.Unmarshal(out[i:], &r)
		}
		return residentDestroyedMsg{name: name, kept: r.Kept}
	}
}

func emberClip(text string, width int) string {
	if width <= 0 {
		width = 110
	}
	return ansi.Truncate(text, width, "")
}

func (m emberModel) detailViewport() viewport.Model {
	w, h := m.width, m.height
	if w <= 0 {
		w = 110
	}
	if h <= 0 {
		h = 32
	}
	reserve := 8
	if m.mode != "" && m.mode != "normal" {
		reserve += 4
	}
	view := viewport.New(viewport.WithWidth(max(1, w-2)), viewport.WithHeight(max(1, h-reserve)))
	view.SoftWrap = true
	view.SetContent(m.pane())
	view.SetYOffset(m.pageScroll)
	return view
}

func nextEnvTab(current, delta int) int {
	for i, id := range emberNavigation {
		if id == current {
			return emberNavigation[(i+delta+len(emberNavigation))%len(emberNavigation)]
		}
	}
	return emberNavigation[0]
}

func residentSessions(m emberModel) tea.Cmd {
	return func() tea.Msg {
		if len(m.cli) == 0 {
			return residentSessionsMsg{agent: m.agentName, err: "resident sessions need the caminus CLI (--cli)"}
		}
		out, err := runResident(residentCommand(m, []string{"peers", m.agentName, "--json"}, ""))
		if err != nil {
			return residentSessionsMsg{agent: m.agentName, err: strings.TrimSpace(string(out))}
		}
		var peers []struct {
			Name string `json:"name"`
		}
		if err = json.Unmarshal(out, &peers); err != nil {
			return residentSessionsMsg{agent: m.agentName, err: err.Error()}
		}
		result := residentSessionsMsg{agent: m.agentName}
		for _, p := range peers {
			result.names = append(result.names, p.Name)
		}
		out, err = runResident(residentCommand(m, []string{"memory", m.agentName, "--json"}, ""))
		var memory struct {
			Memory string `json:"memory"`
		}
		if err == nil && json.Unmarshal(out, &memory) == nil {
			result.memory = memory.Memory
		}
		return result
	}
}

func createResidentSession(m emberModel, name string) tea.Cmd {
	return func() tea.Msg {
		out, err := runResident(residentCommand(m, []string{"create", name, "--from", m.agentName, "--json"}, ""))
		if err != nil {
			return residentCreatedMsg{err: strings.TrimSpace(string(out)) + ": " + err.Error()}
		}
		out, err = runResident(residentCommand(m, []string{"start", name, "--json"}, ""))
		if err != nil {
			return residentCreatedMsg{err: "Session created; runtime start failed: " + strings.TrimSpace(string(out))}
		}
		return residentCreatedMsg{name: name}
	}
}

func (m *emberModel) openResident(name string) {
	if name != m.agentName {
		m.residentEpoch++
		m.agentName = name
		m.resident = residentStatus{}
		m.statusAt = time.Time{} // nothing known yet: "loading", not "stopped", until its first status lands
		m.residentEvents = nil
		m.chat = residentChat{}
		m.goalSel = 0
		m.sessionMemory = ""
	}
	m.pendingLast = ""
	m.tab = 6
	m.msg = ""
	m.initChat()
	m.syncChat()
	if validSessionName(m.agentName) {
		saveLastSession(m.agentName)
	}
}

// openLastSession reopens the last conversation at launch. The first session
// list message falls back to the homepage when the session no longer exists.
func openLastSession(m *emberModel) {
	last := loadLastSession()
	if last == "" {
		return
	}
	m.openResident(last)
	m.pendingLast = last
}

// validSessionName accepts the session names the panel feeds openResident; it
// only rules out values that could never come from the bus or CLI.
func validSessionName(name string) bool {
	if name == "" || len(name) > 128 {
		return false
	}
	for _, r := range name {
		if r == '\n' || r == '\x00' || r == '\r' || r < 0x20 {
			return false
		}
	}
	return strings.TrimSpace(name) == name
}

// The last opened conversation persists in the user's state dir so the next
// launch opens it directly. Failures fall back to the Sessions homepage silently.
func lastSessionPath() string {
	state := os.Getenv("XDG_STATE_HOME")
	if state == "" {
		if h, err := os.UserHomeDir(); err == nil {
			state = filepath.Join(h, ".local", "state")
		} else {
			return ""
		}
	}
	return filepath.Join(state, "caminus", "envui-last-session")
}

func loadLastSession() string {
	b, err := os.ReadFile(lastSessionPath())
	if err != nil {
		return ""
	}
	name := strings.TrimRight(string(b), "\n")
	if !validSessionName(name) {
		return ""
	}
	return name
}

func saveLastSession(name string) {
	if demoMode { // the demo never replaces the real last session
		return
	}
	path := lastSessionPath()
	if path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	_ = os.WriteFile(path, []byte(name+"\n"), 0o600)
}

func (m emberModel) sessionKeys(k string) (tea.Model, tea.Cmd) {
	switch k {
	case "up", "k":
		m.sessionSel = max(0, m.sessionSel-1)
	case "down", "j":
		m.sessionSel = min(max(0, len(m.sessionNames)-1), m.sessionSel+1)
	case "n":
		m.mode, m.edit = "sessionnew", ""
	case "enter":
		name := m.agentName
		if m.sessionSel >= 0 && m.sessionSel < len(m.sessionNames) {
			name = m.sessionNames[m.sessionSel]
		}
		m.openResident(name)
		return m, tea.Batch(residentRefresh(m), residentSessions(m))
	}
	return m, nil
}

func (m emberModel) sessionsPane() string {
	lines := []string{emberHead.Render("SESSIONS"), "Enter opens a conversation · n creates a session · ↑/↓ select", ""}
	names := m.sessionNames
	if len(names) == 0 {
		names = []string{m.agentName}
	}
	for i, name := range names {
		mark := "  "
		if i == m.sessionSel {
			mark = "› "
		}
		line := mark + residentText(name)
		if name == m.agentName {
			line += fmt.Sprintf(" · %d goals", len(m.resident.Goals))
			if m.resident.DaemonAlive {
				line += " · listening"
			}
		}
		lines = append(lines, line)
	}
	return strings.Join(append(lines, "", "Each named session keeps its own history and memory. Sessions share the local model server."), "\n")
}

func (m emberModel) residentSettingsPane() string {
	lines := []string{emberHead.Render("SESSION SETTINGS · " + residentText(m.agentName))}
	add := func(k, v string) {
		lines = append(lines, emberDim.Render(fmt.Sprintf("  %-14s", k))+lipgloss.NewStyle().Foreground(cText).Render(v))
	}
	add("brain", residentText(m.resident.Model))
	add("window", fmt.Sprintf("%s tokens", short(m.resident.Context)))
	add("compaction", fmt.Sprintf("at %.0f %%, down to %.0f %%", m.resident.CompactThreshold*100, m.resident.CompactTarget*100))
	lines = append(lines, "", emberDim.Render("  Read-only here: /model picks the brain, /effort its depth. New sessions take this one's settings."),
		emberDim.Render("  Root approvals are in /approvals; credentials are never shown."), "")
	key := func(k, v string) {
		lines = append(lines, "  "+emberTitle.Render(fmt.Sprintf("%-12s", k))+emberDim.Render(v))
	}
	key("Enter", "send · Alt+Enter a new line · ↑ ↓ your sent messages")
	key("Esc", "browse the history · PgUp PgDn scroll · Ctrl+L follow")
	key("/", "commands, with what each does")
	key("Ctrl+P", "sessions · Ctrl+G goals · Ctrl+A approvals")
	return strings.Join(lines, "\n")
}
