// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberidentity.go: /identity, the agent's accounts. It shows the GitHub account the agent is linked to (`caminus
// ember agent github NAME status`): its login, the git identity its commits carry, its rights on each repository and
// the token file it reads (a path in the agent's own account; the panel never sees the token). l links: a token file
// path and, optionally, repositories to report rights on; u unlinks (two presses); r reads it again. What the token
// may not do upstream reaches you as a GITHUB approval card (github_handoff), not from here.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

type githubLink struct {
	Linked    bool              `json:"linked"`
	Login     string            `json:"login"`
	Email     string            `json:"email"`
	Repos     map[string]string `json:"repos"`
	TokenFile string            `json:"token_file"`
	Note      string            `json:"note"`
}

type identityLoadedMsg struct {
	link githubLink
	err  string
	verb string // "" a read, "linked", "unlinked"
}

type identityState struct {
	open, loading bool
	link          githubLink
	err           string
	editing       int // 0 browsing, 1 typing the token file, 2 typing the repositories
	token, repos  textinput.Model
	armed         time.Time // the first u of an unlink
}

// identityCmd runs one `agent github` action and reads the outcome back.
func identityCmd(m emberModel, verb string, args ...string) tea.Cmd {
	return func() tea.Msg {
		var out githubLink
		argv := append([]string{"github", m.agentName}, args...)
		if err := residentJSON(m, append(argv, "--json"), &out); err != nil {
			return identityLoadedMsg{err: err.Error()}
		}
		return identityLoadedMsg{link: out, verb: verb}
	}
}

func (m emberModel) openIdentity() (emberModel, tea.Cmd) {
	st := &m.chat.identity
	st.open, st.loading, st.err, st.editing = true, true, "", 0
	st.token, st.repos = textinput.New(), textinput.New()
	st.token.Placeholder = "~/homes/" + m.agentName + "/.config/github-token"
	st.repos.Placeholder = "OWNER/NAME, OWNER/NAME (optional: rights to report)"
	return m, identityCmd(m, "", "status")
}

func (m emberModel) identityLoaded(msg identityLoadedMsg) emberModel {
	st := &m.chat.identity
	st.loading, st.err = false, msg.err
	if msg.err == "" {
		st.link = msg.link
		switch msg.verb {
		case "linked":
			m.msg = "linked to GitHub as " + terminalText(msg.link.Login, false) + " · its terminals use it from the next command"
		case "unlinked":
			m.msg = "GitHub account unlinked"
		}
	}
	return m
}

func (m emberModel) identityOverlay(w, h int) string {
	st := m.chat.identity
	inner := max(10, w-6)
	var rows []string
	row := func(k, v string) {
		rows = append(rows, emberDim.Render(fmt.Sprintf("%-14s", k))+ansi.Truncate(v, max(4, inner-14), "…"))
	}
	rows = append(rows, emberTitle.Render("GitHub"))
	switch {
	case st.loading:
		rows = append(rows, emberDim.Render("reading the link…"))
	case st.err != "":
		rows = append(rows, emberWarn.Render(ansi.Truncate(st.err, inner, "…")))
	case !st.link.Linked:
		rows = append(rows, emberDim.Render("not linked: the agent's gh and git use no account, and its commits carry no identity"))
	default:
		row("account", emberCtxOK.Render(g.done+" ")+terminalText(st.link.Login, false))
		row("commits as", terminalText(st.link.Login+" <"+st.link.Email+">", false))
		row("token file", terminalText(st.link.TokenFile, false)+emberDim.Render("  (in the agent's account; never shown)"))
		names := make([]string, 0, len(st.link.Repos))
		for r := range st.link.Repos {
			names = append(names, r)
		}
		sort.Strings(names)
		for _, r := range names {
			right := st.link.Repos[r]
			note := ""
			if right != "write" {
				note = emberDim.Render("  · pull requests and comments reach you as GITHUB cards")
			}
			row("on "+r, terminalText(right, false)+note)
		}
		if len(names) == 0 {
			row("repositories", emberDim.Render("none checked (link again with repositories to see its rights)"))
		}
	}
	rows = append(rows, "")
	switch st.editing {
	case 1:
		rows = append(rows, emberTitle.Render("token file")+emberDim.Render("  a file in the agent's account holding one token"), st.token.View())
	case 2:
		rows = append(rows, emberTitle.Render("repositories")+emberDim.Render("  optional, comma separated"), st.repos.View())
	}
	note := "l link · u unlink · r read again · esc closes"
	if st.editing > 0 {
		note = "Enter next · esc cancels"
	}
	body := strings.Join(rows[:min(len(rows), max(1, h-6))], "\n") + "\n\n" + emberDim.Render(ansi.Truncate(note, inner, "…"))
	return overlayBox("IDENTITY  ·  "+terminalText(m.agentName, false), body, "l link  ·  u unlink", w, h)
}

func (m emberModel) identityKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	st := &m.chat.identity
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	if st.editing > 0 {
		switch k {
		case "esc":
			st.editing = 0
			return m, nil
		case "enter":
			if st.editing == 1 {
				if strings.TrimSpace(st.token.Value()) == "" {
					return m, nil
				}
				st.editing = 2
				st.token.Blur()
				return m, st.repos.Focus()
			}
			st.editing, st.loading = 0, true
			args := []string{"link", "--token-file", strings.TrimSpace(st.token.Value())}
			for _, r := range strings.Split(st.repos.Value(), ",") {
				if r = strings.TrimSpace(r); r != "" {
					args = append(args, "--repo", r)
				}
			}
			return m, identityCmd(m, "linked", args...)
		}
		var cmd tea.Cmd
		if st.editing == 1 {
			st.token, cmd = st.token.Update(msg)
		} else {
			st.repos, cmd = st.repos.Update(msg)
		}
		return m, cmd
	}
	armed := st.armed
	st.armed = time.Time{}
	switch k {
	case "esc", "q":
		st.open = false
		return m, m.chat.compose.Focus()
	case "r":
		st.loading = true
		return m, identityCmd(m, "", "status")
	case "l":
		st.editing = 1
		st.token.SetValue(st.link.TokenFile)
		return m, st.token.Focus()
	case "u":
		if !st.link.Linked {
			return m, nil
		}
		if armed.IsZero() || time.Since(armed) > 10*time.Second {
			st.armed = time.Now()
			m.msg = "unlink " + terminalText(st.link.Login, false) + "? the agent's gh and git lose the account · press u again"
			return m, nil
		}
		st.loading = true
		return m, identityCmd(m, "unlinked", "unlink")
	}
	return m, nil
}
