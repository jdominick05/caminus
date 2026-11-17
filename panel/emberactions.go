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
	"time"

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
	case "read", "search", "edit", "write", "checkpoint", "rollback":
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

// shortPath shows a path the agent gave as it would say it: relative to its working folder, else ~/ under its home.
func shortPath(p, home, cwd string) string {
	if cwd != "" && strings.HasPrefix(p, cwd+"/") {
		return strings.TrimPrefix(p, cwd+"/")
	}
	if home != "" && (p == home || strings.HasPrefix(p, home+"/")) {
		return "~" + strings.TrimPrefix(p, home)
	}
	return p
}

// shortCommand drops a leading "cd WORKING-FOLDER &&" and writes the home as ~ (the card's subject only).
func shortCommand(cmd, home, cwd string) string {
	if cwd != "" {
		for _, lead := range []string{"cd " + cwd + " && ", "cd " + cwd + "; ", "cd " + cwd + "\n"} {
			cmd = strings.TrimPrefix(cmd, lead)
		}
	}
	if home != "" {
		cmd = strings.ReplaceAll(cmd, home+"/", "~/")
	}
	return cmd
}

func kOf(n float64) string {
	if n >= 1000 {
		return fmt.Sprintf("%.0fK", n/1000)
	}
	return fmt.Sprintf("%.0f", n)
}

// actionCard is one tool call with its result, as a compact card w cells wide; home and cwd shorten the paths.
func actionCard(name string, args, result map[string]any, w int, home, cwd string) string {
	if args == nil {
		args = map[string]any{}
	}
	rs := func(k string) string { v, _ := result[k].(string); return v }
	subject, meta := "", ""
	var preview []string
	state := "ok"
	if errText := rs("error"); errText != "" {
		state = "fail"
		// the agent's own paths read short (a raw error carried the whole home path)
		if home != "" {
			errText = strings.ReplaceAll(errText, home+"/", "~/")
		}
		meta = firstLine(residentText(errText))
		preview = lines(residentText(errText), 4)
		if len(preview) <= 1 {
			preview = nil
		}
	}
	switch name {
	case "terminal":
		if cmd := argStr(args, "command"); cmd != "" {
			subject = "$ " + firstLine(shortCommand(cmd, home, cwd))
		} else if argStr(args, "stdin") != "" {
			subject = "input → session"
		} else if _, sig := args["signal"]; sig {
			subject = "signal " + fmt.Sprint(args["signal"]) + " → session"
		} else {
			subject = "↻ checked on its command"
		}
		if state == "ok" {
			out := ansi.Strip(rs("output")) // (a program's colour codes are not text)
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
			} else if code, ok := result["ended_later"].(float64); ok {
				meta = fmt.Sprintf("ended later: exit %d", int(code))
				if code != 0 {
					state = "warn"
				}
			}
			switch {
			case n > 0:
				meta = strings.TrimPrefix(meta+" · "+plural(n, "line", "lines"), " · ")
				all := strings.Split(strings.TrimRight(residentText(out), "\n"), "\n")
				preview = all[max(0, len(all)-3):]
			case result["no_new_output"] == true:
				meta = strings.TrimPrefix(meta+" · nothing new", " · ")
			}
		}
	case "read":
		subject = residentText(shortPath(argStr(args, "path"), home, cwd))
		if state == "ok" {
			meta = "lines " + rs("shown")
			if d, _ := result["directory"].(bool); d {
				meta = "directory"
			}
		}
	case "search":
		s, m, hits := searchSummary(args, result, home, cwd)
		subject = s
		if state == "ok" { // (a failure keeps the error's meta from above, the home written as ~)
			meta, preview = m, hits[:min(3, len(hits))]
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
		if strings.Contains(rs("error"), "mail bus") {
			meta = "no session called " + residentText(argStr(args, "to")) + " gets mail here"
		}
	case "peers":
		var names []string
		ps, _ := result["peers"].([]any)
		for _, p := range ps {
			if d, ok := p.(map[string]any); ok {
				n, _ := d["name"].(string)
				names = append(names, residentText(n))
			} else if n, ok := p.(string); ok {
				names = append(names, residentText(n))
			}
		}
		if state == "ok" {
			meta = plural(len(names), "peer", "peers")
			if len(names) > 0 {
				meta += ": " + strings.Join(names, ", ")
			}
		}
	case "lease":
		subject = argStr(args, "action")
		if state == "ok" {
			if ls, ok := result["leases"].([]any); ok {
				meta = map[bool]string{true: "no leases", false: plural(len(ls), "lease", "leases")}[len(ls) == 0]
			} else if id := rs("lease"); id != "" {
				meta = fmt.Sprintf("lease %s · cards %v until %s", id, result["cards"], rs("until"))
			} else if rel, ok := result["released"].([]any); ok {
				meta = "released " + plural(len(rel), "lease", "leases")
			}
		}
	case "discover":
		subject = argStr(args, "topic")
		if subject == "" {
			subject = "overview"
		}
		if state == "ok" {
			var parts []string
			if acct, ok := result["account"].(map[string]any); ok {
				u, _ := acct["user"].(string)
				parts = append(parts, "account "+residentText(u))
			}
			n := 0
			if progs, ok := result["programs"].(map[string]any); ok {
				for _, cat := range progs {
					if m, ok := cat.(map[string]any); ok {
						n += len(m)
					}
				}
				parts = append(parts, plural(n, "program found", "programs found"))
			}
			if ms, ok := result["matches"].(map[string]any); ok {
				parts = append(parts, plural(len(ms), "match", "matches"))
			}
			meta = strings.Join(parts, " · ")
		}
	case "budget":
		if state == "ok" {
			var parts []string
			if gl, ok := result["goal"].(map[string]any); ok {
				st, _ := gl["steps"].(float64)
				parts = append(parts, plural(int(st), "step", "steps"))
			}
			if cx, ok := result["context"].(map[string]any); ok {
				t, _ := cx["tokens"].(float64)
				win, _ := cx["window"].(float64)
				parts = append(parts, short(int(t))+" / "+short(int(win))+" context") // (66K for a 65K window before)
			}
			meta = strings.Join(parts, " · ")
		}
	case "todo", "ledger":
		if state == "ok" {
			items, _ := result["items"].(float64)
			left, _ := result["left"].(float64)
			meta = plural(int(items), "task", "tasks") + fmt.Sprintf(" · %d left", int(left))
		}
	case "wait":
		subject = strings.TrimSpace(argStr(args, "until") + " " + argStr(args, "target"))
		if state == "ok" {
			s, _ := result["waited_s"].(float64)
			meta = residentText(rs("happened")) + fmt.Sprintf(" · waited %s", (time.Duration(s)*time.Second).String())
		}
	case "rollback":
		subject = argStr(args, "id")
		if r, ok := result["restored"].([]any); ok && state == "ok" {
			meta = "restored " + plural(len(r), "file", "files")
		}
	case "machine":
		if state == "ok" {
			var parts []string
			gpus, _ := result["gpus"].([]any)
			parts = append(parts, plural(len(gpus), "GPU", "GPUs"))
			for _, x := range gpus {
				gp, _ := x.(map[string]any)
				used, _ := gp["vram_used_mib"].(float64)
				total, _ := gp["vram_total_mib"].(float64)
				busy, _ := gp["busy_percent"].(float64)
				switch {
				case total >= 1024:
					parts = append(parts, fmt.Sprintf("%.0f/%.0f GB %d %%", used/1024, total/1024, int(busy)))
				case total > 0: // (a 512 MiB integrated GPU read "0/0 GB")
					parts = append(parts, fmt.Sprintf("%.0f/%.0f MiB %d %%", used, total, int(busy)))
				}
			}
			if mem, ok := result["memory"].(map[string]any); ok {
				av, _ := mem["available_mib"].(float64)
				parts = append(parts, fmt.Sprintf("RAM %.0f GB free", av/1024))
			}
			meta = strings.Join(parts, " · ")
		}
	case "schedule":
		if state == "ok" {
			meta = "parked until " + rs("sleeping_until")
		}
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
	// the outcome (exit 3, refused, still running) is what the card is for: a long subject is cut, never the meta
	tail := ""
	if meta != "" {
		tail = "  ·  " + oneLine(meta)
		if room := w - 4 - ansi.StringWidth(name) - 12; ansi.StringWidth(tail) > room {
			tail = ansi.Truncate(tail, max(12, room), "…")
		}
	}
	room := max(8, w-2-ansi.StringWidth(name)-2-ansi.StringWidth(tail))
	line := dot.Render(g.working) + " " + toolFamily(name).Render(name) + "  " +
		lipgloss.NewStyle().Foreground(cText).Render(ansi.Truncate(oneLine(subject), room, "…"))
	if tail != "" {
		line += emberDim.Render(tail)
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

// searchSummary is a search card's subject (the pattern or glob, and where), its count, and its hits: "file:line  text"
// for each matching line of a text search (the resident prints a file's name, then "  12:  line" per match and
// "  13-  line" per context line), the names for a glob search.
func searchSummary(args, result map[string]any, home, cwd string) (subject, meta string, hits []string) {
	subject = residentText(argStr(args, "glob"))
	if p := argStr(args, "pattern"); p != "" {
		subject = "“" + residentText(p) + "”" + strings.TrimSuffix(" in "+residentText(argStr(args, "glob")), " in ")
	}
	if p := argStr(args, "path"); p != "" && p != "." {
		subject += " under " + residentText(shortPath(p, home, cwd))
	}
	text, _ := result["text"].(string)
	files, _ := result["files"].(float64)
	m, isText := result["matches"].(float64)
	file := ""
	for _, ln := range strings.Split(residentText(text), "\n") {
		switch {
		case ln == "":
		case !isText:
			hits = append(hits, ln)
		case !strings.HasPrefix(ln, " "):
			file = ln
		default:
			if n, body, ok := strings.Cut(strings.TrimLeft(ln, " "), ":  "); ok && n != "" && strings.Trim(n, "0123456789") == "" {
				hits = append(hits, file+":"+n+"  "+strings.TrimSpace(body))
			}
		}
	}
	switch {
	case !isText:
		meta = plural(int(files), "file", "files")
	case m == 0:
		searched, _ := result["searched"].(float64)
		meta = "no matches in " + plural(int(searched), "file", "files")
	default:
		meta = plural(int(m), "match", "matches") + " in " + plural(int(files), "file", "files")
	}
	return subject, meta, hits
}

// actionBlock is the compact view's card for a tool_result event ("" when the result is drawn some other way: a
// question, the finish, the plan); edits and writes keep their diff cards.
func (c *residentChat) actionBlock(e residentEvent, w int) string {
	name, _ := e.Data["name"].(string)
	result, _ := e.Data["result"].(map[string]any)
	switch name {
	case "ask_user", "ask_root", "github_handoff", "":
		// a request's outcome is on its one-line record (approval_requested / question); only one never filed shows
		if err, _ := result["error"].(string); err != "" && name != "" {
			return recordLine(map[string]string{"ask_user": "?", "ask_root": g.root, "github_handoff": "⇡"}[name], "not sent: "+humanRefusal(err), w)
		}
		return ""
	case "finish":
		// what was left open is what the human most needs to see: compact showed it nowhere
		if open, _ := result["open"].(string); strings.TrimSpace(open) != "" {
			return recordLine("!", "left open: "+oneLine(residentText(open)), w)
		}
		return ""
	case "report": // the report's own line ("note fyi: …") says it; a card repeated it
		if err, _ := result["error"].(string); err == "" {
			return ""
		}
	case "propose_plan":
		if err, _ := result["error"].(string); err != "" { // a refused plan had no card at all
			return toolCard("error", "propose_plan", firstLine(err), false, false, nil, w)
		}
		return ""
	}
	if (name == "edit" || name == "write") && result["error"] == nil {
		return c.eventBlock(e)
	}
	id, _ := e.Data["call_id"].(string)
	call := c.calls[id]
	if running, _ := result["running"].(bool); running && name == "terminal" {
		if sid, _ := result["session"].(string); sid != "" {
			if code, done := c.sessionDone[sid]; done { // a later check saw it end: this card said "still running" for good
				r2 := map[string]any{}
				for k, v := range result {
					r2[k] = v
				}
				r2["running"], r2["ended_later"] = false, code
				result = r2
			}
		}
	}
	return actionCard(name, call.args, result, w, c.home, c.cwd)
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
