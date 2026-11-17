// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberpaths.go: @path names a path, +path attaches the file. Both complete from the agent's own filesystem (the file
// is read there), one listing per folder, cached for the session, so no SSH call runs per keystroke.

import (
	"path"
	"regexp"
	"strings"
	"unicode"

	tea "charm.land/bubbletea/v2"
)

// pathToken: the draft's last word when it starts with @ or + (at the start of a word: "C++" and "a@b.c" are not
// paths), without its mark; "" for the mark alone, which lists the agent's folder. A space after the word ends it.
func pathToken(draft string) (string, bool) {
	if draft == "" || strings.Contains(draft, "\n") || strings.HasSuffix(draft, " ") {
		return "", false
	}
	word := draft
	if i := strings.LastIndex(draft, " "); i >= 0 {
		word = draft[i+1:]
	}
	if word == "" || (word[0] != '@' && word[0] != '+') {
		return "", false
	}
	return word[1:], true
}

// agentCwdAbs is the agent's working folder as an absolute path (agentCwd without the "~" shortening).
func (m emberModel) agentCwdAbs() string {
	home := m.resident.Home
	if home == "" {
		return ""
	}
	cwd := m.agentCwd()
	acct := path.Dir(path.Dir(home))
	switch {
	case cwd == "~":
		return acct
	case strings.HasPrefix(cwd, "~/"):
		return path.Join(acct, cwd[2:])
	}
	return cwd
}

// pathDir splits a token into the folder it lists (absolute, on the agent's side) and the name prefix typed in it.
func (m emberModel) pathDir(token string) (dirPart, abs, prefix string) {
	dirPart, prefix = "", token
	if i := strings.LastIndex(token, "/"); i >= 0 {
		dirPart, prefix = token[:i+1], token[i+1:]
	}
	base := m.agentCwdAbs()
	if base == "" {
		return dirPart, "", prefix
	}
	if dirPart == "" {
		return "", base, prefix
	}
	return dirPart, resolveIn(m.resident.Home, base, dirPart), prefix
}

// pathMatches: the folder's entries the typed name starts with, folders first (marked by "/"), as they would be typed;
// nothing until the folder's listing has arrived. Hidden entries only once the name starts with a dot.
func (m emberModel) pathMatches(token string) []string {
	dirPart, abs, prefix := m.pathDir(token)
	entries, ok := m.chat.dirs[abs]
	if abs == "" || !ok {
		return nil
	}
	var dirs, files []string
	for _, e := range entries {
		if !strings.HasPrefix(e, prefix) || (strings.HasPrefix(e, ".") && !strings.HasPrefix(prefix, ".")) {
			continue
		}
		if strings.HasSuffix(e, "/") {
			dirs = append(dirs, dirPart+e)
		} else {
			files = append(files, dirPart+e)
		}
	}
	out := append(dirs, files...)
	if len(out) > 64 {
		out = out[:64]
	}
	return out
}

type dirListedMsg struct {
	agent, dir string
	entries    []string
	err        string
}

// ensureListing asks for the folder the current @ / + token lists when it is not cached or on its way.
func (m emberModel) ensureListing() (emberModel, tea.Cmd) {
	token, ok := pathToken(m.chat.compose.Value())
	if !ok || len(m.cli) == 0 {
		return m, nil
	}
	_, abs, _ := m.pathDir(token)
	if abs == "" {
		return m, nil
	}
	if _, have := m.chat.dirs[abs]; have || m.chat.dirsPending[abs] {
		return m, nil
	}
	if m.chat.dirsPending == nil {
		m.chat.dirsPending = map[string]bool{}
	}
	m.chat.dirsPending[abs] = true
	agent := m.agentName
	return m, func() tea.Msg {
		out, err := agentRun(m, "ls -1Ap -- "+shellQuote(abs)+" 2>&1 | head -n 400")
		msg := dirListedMsg{agent: agent, dir: abs}
		if err != nil {
			msg.err = strings.TrimSpace(out)
			return msg
		}
		for _, l := range strings.Split(out, "\n") {
			// a name with a control character is left out: the agent can name a file anything, and an escape
			// sequence in it would reach the owner's terminal through the menu
			if l = strings.TrimSpace(l); l != "" && !strings.HasPrefix(l, "ls:") && !strings.ContainsFunc(l, unicode.IsControl) {
				msg.entries = append(msg.entries, l)
			}
		}
		return msg
	}
}

// the words that attach (+) and name (@) paths in a sent message, at the start of a word only
var (
	attachRe = regexp.MustCompile(`(^|\s)\+((?:~/|/|\./)?[A-Za-z0-9_./~+-]+)`)
	nameRe   = regexp.MustCompile(`(^|\s)@((?:~/|/|\./)?[A-Za-z0-9_./~+-]+)`)
)
