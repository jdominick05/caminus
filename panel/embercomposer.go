// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envcomposer.go: the composer's extras.
//   Ctrl+X Ctrl+E   the draft in $VISUAL / $EDITOR; it comes back when the editor closes
//   Ctrl+R          your earlier prompts that contain the draft, newest first; each press goes one further back
//   Ctrl+O          every line of every open card (cards stop at their last 40 lines otherwise)
//   !cmd            runs cmd in the agent's account; its output lands in the draft for your next message
//   @path           in a message, attaches that file from the agent's account (up to 64 KB), fenced
//   /NAME args      a custom command: ~/.config/caminus/commands/NAME.md, $ARGUMENTS replaced, sent as a message

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

type editorDoneMsg struct {
	text string
	err  string
}

type bangDoneMsg struct {
	cmd, output string
	err         string
}

// composerKey handles the extras' keys before the composer's own; ok reports whether it took the key.
func (m emberModel) composerKey(msg tea.KeyPressMsg) (emberModel, tea.Cmd, bool) {
	k := msg.String()
	chord := m.chat.chordAt
	m.chat.chordAt = time.Time{}
	switch {
	case k == "ctrl+x":
		m.chat.chordAt = time.Now()
		m.msg = "Ctrl+X … (Ctrl+E edits the draft in $EDITOR)"
		return m, nil, true
	case k == "ctrl+e" && !chord.IsZero() && time.Since(chord) < 2*time.Second:
		cmd, err := m.openEditor()
		if err != "" {
			m.msg = err
			return m, nil, true
		}
		return m, cmd, true
	case k == "ctrl+r":
		return m.historyBack(), nil, true
	case k == "ctrl+o":
		return m.toggleLayout(), nil, true
	}
	if k != "ctrl+r" {
		m.chat.histQuery, m.chat.histIdx = "", 0
	}
	return m, nil, false
}

func editorCommand() string {
	for _, v := range []string{os.Getenv("VISUAL"), os.Getenv("EDITOR")} {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	for _, e := range []string{"nano", "vi"} {
		if _, err := exec.LookPath(e); err == nil {
			return e
		}
	}
	return ""
}

func (m emberModel) openEditor() (tea.Cmd, string) {
	editor := editorCommand()
	if editor == "" {
		return nil, "no editor: set $EDITOR"
	}
	f, err := os.CreateTemp("", "caminus-draft-*.md")
	if err != nil {
		return nil, "draft file: " + err.Error()
	}
	path := f.Name()
	_, _ = f.WriteString(m.chat.compose.Value())
	_ = f.Close()
	parts := strings.Fields(editor)
	c := exec.Command(parts[0], append(parts[1:], path)...)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		defer os.Remove(path)
		if err != nil {
			return editorDoneMsg{err: "editor: " + err.Error()}
		}
		b, rerr := os.ReadFile(path)
		if rerr != nil {
			return editorDoneMsg{err: "draft file: " + rerr.Error()}
		}
		return editorDoneMsg{text: strings.TrimRight(string(b), "\n")}
	}), ""
}

// pastPrompts: what you sent (goals and messages), newest first, without repeats.
func (m emberModel) pastPrompts() []string {
	var all []string
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		key := map[string]string{"goal_queued": "prompt", "user_message": "text"}[e.Kind]
		if key == "" {
			continue
		}
		if k, _ := e.Data["kind"].(string); k == "free" || k == "dream" {
			continue
		}
		if s, _ := e.Data[key].(string); strings.TrimSpace(s) != "" {
			all = append(all, s)
		}
	}
	for i := len(m.resident.Goals) - 1; i >= 0; i-- {
		all = append(all, m.resident.Goals[i].Prompt)
	}
	seen, out := map[string]bool{}, []string{}
	for _, s := range all {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func (m emberModel) historyBack() emberModel {
	if m.chat.histIdx == 0 {
		m.chat.histQuery = m.chat.compose.Value()
	}
	q := strings.ToLower(strings.TrimSpace(m.chat.histQuery))
	var hits []string
	for _, p := range m.pastPrompts() {
		if q == "" || strings.Contains(strings.ToLower(p), q) {
			hits = append(hits, p)
		}
	}
	if len(hits) == 0 {
		m.msg = "no earlier prompt contains that"
		return m
	}
	i := m.chat.histIdx % len(hits)
	m.chat.compose.SetValue(hits[i])
	m.chat.histIdx = i + 1
	m.msg = fmt.Sprintf("history %d/%d · Ctrl+R further back · edit or Enter to send", i+1, len(hits))
	return m
}

// agentRun runs one shell command as the agent (caminus ember run -s), bounded to 30 seconds.
func agentRun(m emberModel, script string) (string, error) {
	if len(m.cli) == 0 {
		return "", fmt.Errorf("needs the caminus CLI (--cli)")
	}
	argv := append(append([]string{}, m.cli[1:]...), "ember", "run", "-s", script)
	c := exec.Command(m.cli[0], argv...)
	done := make(chan struct{})
	var out []byte
	var err error
	go func() { out, err = c.CombinedOutput(); close(done) }()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		if c.Process != nil {
			_ = c.Process.Kill()
		}
		<-done
		return string(out), fmt.Errorf("timed out after 30 s")
	}
	return string(out), err
}

func runBang(m emberModel, cmd string) tea.Cmd {
	return func() tea.Msg {
		out, err := agentRun(m, cmd)
		if len(out) > 32*1024 {
			out = out[len(out)-32*1024:]
		}
		msg := bangDoneMsg{cmd: cmd, output: strings.TrimRight(out, "\n")}
		if err != nil {
			msg.err = err.Error()
		}
		return msg
	}
}

// bangDone puts the command and its output in the draft, fenced, for the next message.
func (m emberModel) bangDone(msg bangDoneMsg) emberModel {
	block := "```\n$ " + msg.cmd + "\n" + msg.output + "\n```"
	if msg.err != "" {
		block += "\n(" + msg.err + ")"
	}
	draft := strings.TrimSpace(m.chat.compose.Value())
	if draft != "" && !strings.HasPrefix(draft, "!") {
		block = draft + "\n\n" + block
	}
	m.chat.compose.SetValue(block + "\n")
	m.msg = "command output is in the draft: add your words and Enter to send"
	return m
}

var attachRe = regexp.MustCompile(`(^|\s)@((?:~/|/|\./)?[A-Za-z0-9_./~+-]+)`)

// expandAttachments appends each @path's file (read as the agent, at most 64 KB) to the message.
func expandAttachments(m emberModel, text string) string {
	found := attachRe.FindAllStringSubmatch(text, 8)
	var blocks []string
	for _, f := range found {
		path := f[2]
		if strings.Contains(path, "..") {
			continue
		}
		out, err := agentRun(m, "head -c 65536 -- "+shellQuote(path))
		if err != nil {
			blocks = append(blocks, fmt.Sprintf("[could not attach %s: %s]", path, strings.TrimSpace(out)))
			continue
		}
		blocks = append(blocks, fmt.Sprintf("Attached %s:\n```\n%s\n```", path, strings.TrimRight(out, "\n")))
	}
	if len(blocks) == 0 {
		return text
	}
	return text + "\n\n" + strings.Join(blocks, "\n\n")
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// customCommands: ~/.config/caminus/commands/*.md (or $XDG_CONFIG_HOME), name -> file.
func customCommands() map[string]string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		if h, err := os.UserHomeDir(); err == nil {
			dir = filepath.Join(h, ".config")
		}
	}
	out := map[string]string{}
	files, _ := filepath.Glob(filepath.Join(dir, "caminus", "commands", "*.md"))
	for _, f := range files {
		name := "/" + strings.TrimSuffix(filepath.Base(f), ".md")
		if regexp.MustCompile(`^/[a-z0-9][a-z0-9_-]{0,31}$`).MatchString(name) && !builtinCommand(name) {
			out[name] = f
		}
	}
	return out
}

func builtinCommand(name string) bool {
	for _, c := range slashCommands {
		if c.name == name {
			return true
		}
	}
	return false
}

func customNames() []string {
	var names []string
	for n := range customCommands() {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// expandCustom: the command file's text, $ARGUMENTS replaced (appended when the file does not use it).
func expandCustom(path, args string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	body := strings.TrimSpace(string(b))
	if strings.HasPrefix(body, "---") { // a front-matter block (description: ...) is for people, not the agent
		if end := strings.Index(body[3:], "\n---"); end >= 0 {
			body = strings.TrimSpace(body[3+end+4:])
		}
	}
	if strings.Contains(body, "$ARGUMENTS") {
		return strings.ReplaceAll(body, "$ARGUMENTS", args), nil
	}
	if strings.TrimSpace(args) != "" {
		body += "\n\n" + args
	}
	return body, nil
}
