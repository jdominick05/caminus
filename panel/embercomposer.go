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
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

type editorDoneMsg struct {
	text string
	err  string
}

type bangDoneMsg struct {
	cmd, output string
	err         string
	host        bool // run in your own shell (!!), not the agent's account
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

// hostShellStyle is the host shell's colour: copper, apart from the agent shell's purple.
func hostShellStyle() lipgloss.Style { return lipgloss.NewStyle().Bold(true).Foreground(cOrange) }

// hostRun runs one command in your own shell on this machine ($SHELL -c, else sh), bounded to 30 seconds: the !!
// shell mode. It runs only from your keystrokes; the agent has no way to reach it.
func hostRun(script string) (string, error) {
	sh := os.Getenv("SHELL")
	if sh == "" {
		sh = "/bin/sh"
	}
	ctx, cancel := context.WithTimeout(context.Background(), hostRunLimit)
	defer cancel()
	c := exec.CommandContext(ctx, sh, "-c", script)
	// the shell and everything it starts are one process group, killed together at the limit; and the output is not
	// waited on past it (a backgrounded child kept the pipe open, so the result never came)
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	c.Cancel = func() error { return syscall.Kill(-c.Process.Pid, syscall.SIGKILL) }
	c.WaitDelay = 2 * time.Second
	out, err := c.CombinedOutput()
	if ctx.Err() != nil {
		return string(out), fmt.Errorf("timed out after %s", hostRunLimit)
	}
	return string(out), err
}

// hostRunLimit bounds a !! command (a test shortens it).
var hostRunLimit = 30 * time.Second

func runHostBang(cmd string) tea.Cmd {
	return func() tea.Msg {
		out, err := hostRun(cmd)
		if len(out) > 32*1024 {
			out = out[len(out)-32*1024:]
		}
		msg := bangDoneMsg{cmd: cmd, output: strings.TrimRight(out, "\n"), host: true}
		if err != nil {
			msg.err = err.Error()
		}
		return msg
	}
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
	prompt := "$ "
	if msg.host {
		prompt = "(my shell) $ " // so the agent can tell your machine's output from its own account's
	}
	block := "```\n" + prompt + msg.cmd + "\n" + msg.output + "\n```"
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

// expandAttachments: each +path's file (read as the agent, at most 64 KB, at most 8, never "..") goes with the message,
// its word marked "(attached)"; an @path is a plain path, sent without the mark. Relative paths are the agent's
// working folder's, as completion lists them.
func expandAttachments(m emberModel, text string) string {
	var blocks []string
	base := m.agentCwdAbs()
	n := 0
	// only outside fenced blocks (a pasted diff's "+added" lines are text), and only a word that reads as a path: it
	// has a "/" or a "." ("+1 for that" is not a file)
	parts := strings.Split(text, "```")
	for k := 0; k < len(parts); k += 2 {
		parts[k] = attachRe.ReplaceAllStringFunc(parts[k], func(word string) string {
			g := attachRe.FindStringSubmatch(word)
			lead, p := g[1], g[2]
			if n >= 8 || strings.Contains(p, "..") || !strings.ContainsAny(p, "/.") {
				return word
			}
			n++
			full := p
			if base != "" {
				full = resolveIn(m.resident.Home, base, p)
			}
			out, err := agentRun(m, "head -c 65536 -- "+shellQuote(full))
			if err != nil {
				blocks = append(blocks, fmt.Sprintf("[could not attach %s: %s]", p, strings.TrimSpace(out)))
				return lead + p
			}
			blocks = append(blocks, fmt.Sprintf("Attached %s:\n```\n%s\n```", p, strings.TrimRight(out, "\n")))
			return lead + p + " (attached)"
		})
		parts[k] = nameRe.ReplaceAllString(parts[k], "${1}${2}")
	}
	text = strings.Join(parts, "```")
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
