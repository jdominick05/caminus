// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envstudio.go: what the resident did while you were away. The recap ("while you were away": each free-time stretch
// since you last looked, from the event store, no model call) is a card above the composer when a session opens; Esc
// dismisses it and remembers when you looked. /recap shows it again. /studio opens the Studio overlay: the "for you"
// shelf, the journal and the studio, newest first; Enter reads one inside the overlay.

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type recapMsg struct {
	agent, text string
	now         float64
	err         string
}

type studioEntry struct {
	Path  string  `json:"path"`
	Name  string  `json:"name"`
	Size  int64   `json:"size"`
	Mtime float64 `json:"mtime"`
	Head  string  `json:"head"`
	group string
}

type studioMsg struct {
	entries  []studioEntry
	projects string
	err      string
}

type studioTextMsg struct {
	path, text, err string
	binary          bool
}

type studioState struct {
	open, loading bool
	sel, top      int
	list          []studioEntry
	projects      string
	reading       string // the path being read; its text below
	text          []string
	err           string
	img           image.Image // an image being shown (envstudioview.go)
	imgInfo       string
	armed         string // a file the owner pressed o on once that needs a second press (activeFormat)
}

func seenPath(agent string) string {
	last := lastSessionPath()
	if last == "" || !validSessionName(agent) {
		return ""
	}
	return filepath.Join(filepath.Dir(last), "envui-seen-"+agent)
}

func loadSeen(agent string) float64 {
	b, err := os.ReadFile(seenPath(agent))
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	return v
}

func saveSeen(agent string, t float64) {
	p := seenPath(agent)
	if p == "" || t <= 0 {
		return
	}
	_ = os.MkdirAll(filepath.Dir(p), 0o755)
	_ = os.WriteFile(p, []byte(strconv.FormatFloat(t, 'f', 3, 64)+"\n"), 0o600)
}

// fetchRecap: since > 0 asks for what happened after that time; 0 uses the resident's default (since your last goal).
func fetchRecap(m emberModel, since float64) tea.Cmd {
	agent := m.agentName
	return func() tea.Msg {
		args := []string{"heartbeat", agent, "recap", "--json"}
		if since > 0 {
			args = append(args, "--since", strconv.FormatFloat(since, 'f', 3, 64))
		}
		var out struct {
			Recap string  `json:"recap"`
			Now   float64 `json:"now"`
		}
		if err := residentJSON(m, args, &out); err != nil {
			return recapMsg{agent: agent, err: err.Error()}
		}
		return recapMsg{agent: agent, text: out.Recap, now: out.Now}
	}
}

func fetchStudio(m emberModel) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Journal  []studioEntry `json:"journal"`
			Studio   []studioEntry `json:"studio"`
			Shelf    []studioEntry `json:"shelf"`
			Projects string        `json:"projects"`
		}
		if err := residentJSON(m, []string{"studio", m.agentName, "--json"}, &out); err != nil {
			return studioMsg{err: err.Error()}
		}
		var all []studioEntry
		for _, g := range []struct {
			name string
			list []studioEntry
		}{{"for you", out.Shelf}, {"journal", out.Journal}, {"studio", out.Studio}} {
			for _, e := range g.list {
				e.group = g.name
				all = append(all, e)
			}
		}
		return studioMsg{entries: all, projects: out.Projects}
	}
}

func readStudio(m emberModel, path string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Text   string `json:"text"`
			Binary bool   `json:"binary"`
		}
		if err := residentJSON(m, []string{"studio", m.agentName, "--read", path, "--json"}, &out); err != nil {
			return studioTextMsg{path: path, err: err.Error()}
		}
		return studioTextMsg{path: path, text: out.Text, binary: out.Binary}
	}
}

// recapCard is the "while you were away" card above the composer.
func (m emberModel) recapCard(w int) string {
	if strings.TrimSpace(m.chat.recap) == "" {
		return ""
	}
	inner := max(10, w-4)
	var rows []string
	for i, line := range strings.Split(strings.TrimSpace(m.chat.recap), "\n") {
		st := lipgloss.NewStyle().Foreground(cText)
		if i == 0 {
			st = emberBusy
		}
		for _, piece := range strings.Split(ansi.Hardwrap(residentText(line), inner, true), "\n") {
			rows = append(rows, st.Render(piece))
		}
		if len(rows) > 10 {
			rows = append(rows[:10], emberDim.Render("…  /studio has the rest"))
			break
		}
	}
	rows = append(rows, emberDim.Render("Esc dismisses  ·  /studio opens what it made"))
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cAccent).Padding(0, 1).
		Width(max(12, w-2)).Render(strings.Join(rows, "\n"))
}

func (m emberModel) studioOverlay(w, h int) string {
	st := m.chat.studio
	inner := max(10, w-6)
	if st.reading != "" && st.img != nil && st.err == "" {
		bodyH := max(2, h-7)
		rows := append([]string{emberDim.Render(st.imgInfo)}, halfBlocks(st.img, inner, bodyH)...)
		return overlayBox("STUDIO  "+filepath.Base(st.reading), strings.Join(rows, "\n"),
			"o open full size on your desktop  ·  Esc back", w, h)
	}
	if st.reading != "" {
		bodyH := max(1, h-6)
		top := max(0, min(st.top, max(0, len(st.text)-bodyH)))
		var rows []string
		for _, l := range st.text[top:min(len(st.text), top+bodyH)] {
			rows = append(rows, ansi.Truncate(lipgloss.NewStyle().Foreground(cText).Render(residentText(l)), inner, "…"))
		}
		if st.err != "" {
			rows = []string{emberWarn.Render(st.err)}
		}
		return overlayBox("STUDIO  "+filepath.Base(st.reading), strings.Join(rows, "\n"), "↑↓ / PgUp PgDn scroll  ·  o open on your desktop  ·  Esc back", w, h)
	}
	listH := max(1, h-6)
	sel := max(0, min(st.sel, len(st.list)-1))
	first := 0
	if len(st.list) > listH {
		first = max(0, min(sel-listH+1, len(st.list)-listH))
	}
	var rows []string
	switch {
	case st.loading && len(st.list) == 0:
		rows = append(rows, emberDim.Render("reading its studio…"))
	case st.err != "":
		rows = append(rows, emberWarn.Render(ansi.Truncate(st.err, inner, "…")))
	case len(st.list) == 0:
		rows = append(rows, emberDim.Render("nothing yet: free time fills the journal, the studio and the shelf"))
	}
	for i := first; i < min(first+listH, len(st.list)); i++ {
		e := st.list[i]
		when := time.Unix(int64(e.Mtime), 0).Format("Jan 2 15:04")
		label := fmt.Sprintf("%-8s %s  %s", e.group, when, e.Name)
		if e.Head != "" && e.Head != e.Name {
			label += "  ·  " + e.Head
		}
		if i == sel {
			rows = append(rows, emberSel.Render(ansi.Truncate(" "+g.collapsed+" "+label, max(8, inner-1), "…")))
		} else {
			rows = append(rows, ansi.Truncate("   "+emberDim.Render(label), inner, "…"))
		}
	}
	return overlayBox(fmt.Sprintf("STUDIO  %d", len(st.list)), strings.Join(rows, "\n"), "↑↓ move  ·  Enter view  ·  o open on your desktop", w, h)
}

// studioOpen opens a studio file on the owner's desktop: at once for an image, text or media; for a format that can
// run code there (activeFormat), only on a second press after a warning.
func (m emberModel) studioOpen(path string) (tea.Model, tea.Cmd) {
	st := &m.chat.studio
	if activeFormat(path) && st.armed != path {
		st.armed = path
		m.msg = filepath.Base(path) + ": opening it runs your desktop app on a file the agent made, and this format " +
			"can carry scripts that would run as you, outside its account. Press o again to open it anyway."
		return m, nil
	}
	st.armed = ""
	m.msg = "copying " + filepath.Base(path) + " to open it…"
	return m, readStudioRaw(m, path, true)
}

func (m emberModel) studioKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	st := &m.chat.studio
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	if k != "o" {
		st.armed = "" // the second press must come right after the warning
	}
	if st.reading != "" {
		switch k {
		case "esc", "q", "left":
			st.reading, st.text, st.err, st.top, st.img, st.imgInfo = "", nil, "", 0, nil, ""
		case "o":
			return m.studioOpen(st.reading)
		case "up", "k":
			st.top = max(0, st.top-1)
		case "down", "j":
			st.top++
		case "pgup":
			st.top = max(0, st.top-10)
		case "pgdown", " ", "space":
			st.top += 10
		}
		return m, nil
	}
	switch k {
	case "esc", "q":
		st.open = false
		return m, m.chat.compose.Focus()
	case "up", "k":
		st.sel = max(0, st.sel-1)
	case "down", "j":
		st.sel = min(max(0, len(st.list)-1), st.sel+1)
	case "r":
		st.loading = true
		return m, fetchStudio(m)
	case "o":
		if len(st.list) == 0 {
			return m, nil
		}
		return m.studioOpen(st.list[max(0, min(st.sel, len(st.list)-1))].Path)
	case "enter":
		if len(st.list) == 0 {
			return m, nil
		}
		e := st.list[max(0, min(st.sel, len(st.list)-1))]
		st.reading, st.text, st.err, st.top, st.img, st.imgInfo = e.Path, []string{"reading…"}, "", 0, nil, ""
		switch studioKind(e.Path) {
		case "image":
			st.text = []string{"drawing the image…"}
			return m, readStudioRaw(m, e.Path, false)
		case "external":
			st.text = []string{fmt.Sprintf("%s  ·  %s", strings.TrimPrefix(filepath.Ext(e.Path), "."), humanBytes(int(e.Size))), "",
				"The panel cannot draw this format. Press o to open it in your desktop's viewer (a copy goes to",
				studioOpenDir(m.agentName) + ")."}
			return m, nil
		}
		return m, readStudio(m, e.Path)
	}
	return m, nil
}

func (m emberModel) receiveRecap(msg recapMsg) emberModel {
	if msg.agent != m.agentName {
		return m
	}
	if msg.err != "" {
		m.msg = "recap: " + msg.err
		return m
	}
	m.chat.recap, m.chat.recapNow = msg.text, msg.now
	if strings.TrimSpace(msg.text) == "" && m.chat.recapAsked == recapOnDemand {
		m.msg = "nothing from free time since your last goal"
	}
	return m
}

const (
	recapNotAsked = iota
	recapAtOpen
	recapOnDemand
)

// dismissRecap remembers when you looked, so the next session's card starts from here.
func (m emberModel) dismissRecap() emberModel {
	if m.chat.recapNow > 0 {
		saveSeen(m.agentName, m.chat.recapNow)
	}
	m.chat.recap = ""
	return m
}

var _ = json.Marshal
