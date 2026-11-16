// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberactions.go: the compact view's action cards. Every tool call the agent makes is one card (60 calls, 60 cards):
// a status dot, the tool in its family's colour, what it acted on, a dim note of what came back, and for some tools a
// few preview lines (a command's last output lines, a search's first hits, a message's first lines, an error's reason).
// Ctrl+O switches to everything: every card in full, the thinking, every gate run.

import (
	"fmt"
	"net/url"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type callInfo struct {
	name string
	args map[string]any
}

// toolFamily is the colour a tool's label takes: the shell, files, the web, memory, messages, the done check.
func toolFamily(name string) lipgloss.Style {
	switch name {
	case "terminal", "ask_root", "lease", "machine", "discover":
		return lipgloss.NewStyle().Foreground(cOrange).Bold(true)
	case "read", "edit", "write", "checkpoint", "rollback":
		return lipgloss.NewStyle().Foreground(cPurple).Bold(true)
	case "web_search", "web_fetch":
		return lipgloss.NewStyle().Foreground(cTeal).Bold(true)
	case "remember", "history_search", "todo":
		return lipgloss.NewStyle().Foreground(cBlue).Bold(true)
	case "send_message", "peers", "github_handoff", "advisor", "report":
		return lipgloss.NewStyle().Foreground(cYellow).Bold(true)
	case "define_done", "verify":
		return lipgloss.NewStyle().Foreground(cGreen).Bold(true)
	}
	return lipgloss.NewStyle().Foreground(cText).Bold(true)
}

func argStr(args map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := args[k].(string); ok && strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func hostOf(u string) string {
	if p, err := url.Parse(u); err == nil && p.Host != "" {
		return strings.TrimPrefix(p.Host, "www.")
	}
	return u
}

// actionCard is one tool call with its result, as a compact card w cells wide.
func actionCard(name string, args, result map[string]any, w int) string {
	if args == nil {
		args = map[string]any{}
	}
	rs := func(k string) string { v, _ := result[k].(string); return v }
	subject, meta := "", ""
	var preview []string
	state := "ok"
	if errText := rs("error"); errText != "" {
		state = "fail"
		meta = firstLine(residentText(errText))
		preview = lines(residentText(errText), 4)
		if len(preview) <= 1 {
			preview = nil
		}
	}
	switch name {
	case "terminal":
		if cmd := argStr(args, "command"); cmd != "" {
			subject = "$ " + firstLine(cmd)
		} else if argStr(args, "stdin") != "" {
			subject = "input → session"
		} else if _, sig := args["signal"]; sig {
			subject = "signal " + fmt.Sprint(args["signal"]) + " → session"
		} else {
			subject = "↻ checked on its command"
		}
		if state == "ok" {
			out := rs("output")
			n := 0
			if strings.TrimSpace(out) != "" {
				n = strings.Count(strings.TrimRight(out, "\n"), "\n") + 1
			}
			if code, ok := result["exit_code"].(float64); ok {
				meta = fmt.Sprintf("exit %d", int(code))
				if code != 0 {
					state = "warn"
				}
			} else if running, _ := result["running"].(bool); running {
				meta, state = "still running", "busy"
			}
			switch {
			case n > 0:
				meta = strings.TrimPrefix(meta+fmt.Sprintf(" · %d lines", n), " · ")
				all := strings.Split(strings.TrimRight(residentText(out), "\n"), "\n")
				preview = all[max(0, len(all)-3):]
			case result["no_new_output"] == true:
				meta = strings.TrimPrefix(meta+" · nothing new", " · ")
			}
		}
	case "read":
		subject = residentText(argStr(args, "path"))
		if state == "ok" {
			meta = "lines " + rs("shown")
			if d, _ := result["directory"].(bool); d {
				meta = "directory"
			}
		}
	case "web_search":
		subject = "“" + residentText(argStr(args, "query")) + "”"
		if hits, ok := result["results"].([]any); ok && state == "ok" {
			meta = fmt.Sprintf("%d results", len(hits))
			if e := rs("engine"); e != "" {
				meta += " · " + e
			}
			for _, h := range hits[:min(3, len(hits))] {
				r, _ := h.(map[string]any)
				t, _ := r["title"].(string)
				link, _ := r["href"].(string)
				if link == "" {
					link, _ = r["url"].(string)
				}
				preview = append(preview, residentText(firstLine(t))+"  "+hostOf(link))
			}
		}
	case "web_fetch":
		subject = residentText(argStr(args, "url"))
		if state == "ok" {
			meta = strings.TrimSpace(residentText(firstLine(rs("title"))))
			if t := rs("text"); t != "" {
				meta = strings.TrimPrefix(meta+fmt.Sprintf(" · %d chars", len(t)), " · ")
			}
		}
	case "remember":
		subject = residentText(argStr(args, "name"))
		meta = residentText(firstLine(argStr(args, "summary")))
		if p, _ := args["pin"].(bool); p {
			meta += " · pinned"
		}
	case "send_message":
		subject = "→ " + residentText(argStr(args, "to"))
		preview = lines(residentText(argStr(args, "text")), 3)
	case "github_handoff":
		subject = strings.TrimSpace(argStr(args, "action") + " " + argStr(args, "repo"))
		if t := argStr(args, "title"); t != "" {
			meta = residentText(t)
		}
	default:
		subject = residentText(firstLine(argStr(args, "query", "path", "check", "summary", "label", "id", "topic", "to",
			"until", "name", "text", "evidence", "question")))
		if state == "ok" {
			for _, k := range []string{"summary", "status", "state", "note"} {
				if v := rs(k); v != "" {
					meta = residentText(firstLine(v))
					break
				}
			}
			if p, ok := result["passed"].(bool); ok {
				meta = map[bool]string{true: "passes", false: "fails"}[p]
				if !p {
					state = "warn"
				}
			}
		}
	}
	dot := map[string]lipgloss.Style{"ok": emberCtxOK, "fail": emberCtxHot, "warn": emberWarn, "busy": emberBusy}[state]
	line := dot.Render(g.working) + " " + toolFamily(name).Render(name) + "  " + lipgloss.NewStyle().Foreground(cText).Render(oneLine(subject))
	if meta != "" {
		line += emberDim.Render("  ·  " + oneLine(meta))
	}
	out := []string{ansi.Truncate(line, w, "…")}
	rule := "  " + emberDim.Render("⎿ ")
	for i, p := range preview {
		if i > 0 {
			rule = "    "
		}
		out = append(out, ansi.Truncate(rule+emberDim.Render(terminalText(p, false)), w, "…"))
	}
	return strings.Join(out, "\n")
}

// actionBlock is the compact view's card for a tool_result event ("" when the result is drawn some other way: a
// question, the finish, the plan); edits and writes keep their diff cards.
func (c *residentChat) actionBlock(e residentEvent, w int) string {
	name, _ := e.Data["name"].(string)
	switch name {
	case "ask_user", "finish", "propose_plan", "":
		return ""
	}
	result, _ := e.Data["result"].(map[string]any)
	if (name == "edit" || name == "write") && result["error"] == nil {
		return c.eventBlock(e)
	}
	id, _ := e.Data["call_id"].(string)
	call := c.calls[id]
	return actionCard(name, call.args, result, w)
}

// firstLine is a text's first line, with " …" when more follows.
func firstLine(s string) string {
	s = strings.TrimSpace(residentText(s))
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i] + " …"
	}
	return s
}

// lines is a text's first limit lines, with a count of the rest.
func lines(s string, limit int) []string {
	s = strings.TrimRight(residentText(s), "\n")
	if s == "" {
		return nil
	}
	all := strings.Split(s, "\n")
	if len(all) <= limit {
		return all
	}
	return append(all[:limit], fmt.Sprintf("… %d more lines", len(all)-limit))
}
