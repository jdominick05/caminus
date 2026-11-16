// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envgroup.go: the compact transcript. The agent's words stay whole; the tool work between two of its replies folds
// into one dim line ("ran 3 commands, edited 1 file · gate passes"), the way a coding agent's own UI shows its
// tool calls. The line grows while the work goes on and is fixed when the next reply, message or card arrives.
// Ctrl+E shows every card again (the full view draws each call, result and gate run as before).

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/charmbracelet/x/ansi"
)

type toolGroup struct {
	commands, reads, edits, failed int
	other                          map[string]int
	gate                           string  // "", "passes", "fails": the latest gate run
	running                        string  // the command still running, if any
	runningSince                   float64 // when it started (the call's event time)
	runningGoal                    string  // the goal it runs for: Now names it only while the agent is on that goal
}

func (t *toolGroup) empty() bool {
	return t.commands+t.reads+t.edits+t.failed == 0 && len(t.other) == 0 && t.gate == "" && t.running == ""
}

func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

// line is the group as one dim line, w cells at most.
func (t *toolGroup) line(w int) string {
	var parts []string
	if t.commands > 0 {
		parts = append(parts, "ran "+plural(t.commands, "command", "commands"))
	}
	if t.reads > 0 {
		parts = append(parts, "read "+plural(t.reads, "file", "files"))
	}
	if t.edits > 0 {
		parts = append(parts, "edited "+plural(t.edits, "file", "files"))
	}
	if len(t.other) > 0 {
		names := make([]string, 0, len(t.other))
		for n, c := range t.other {
			if c > 1 {
				n = fmt.Sprintf("%s ×%d", n, c)
			}
			names = append(names, n)
		}
		sort.Strings(names)
		parts = append(parts, "used "+strings.Join(names, ", "))
	}
	joined := strings.Join(parts, ", ")
	if joined != "" {
		joined = strings.ToUpper(joined[:1]) + joined[1:]
	} else {
		joined = "Working"
	}
	text := emberDim.Render("  " + joined)
	if t.failed > 0 {
		text += emberWarn.Render(" · " + plural(t.failed, "failed", "failed"))
	}
	switch t.gate {
	case "passes":
		text += emberCtxOK.Render(" · gate passes")
	case "fails":
		text += emberWarn.Render(" · gate fails")
	}
	if t.running != "" {
		text += emberDim.Render(" · running: ") + emberBusy.Render(t.running)
	}
	return ansi.Truncate(text, max(10, w), "…")
}

// fold takes an event into the group when it is tool work; false means it is drawn as its own block.
func (c *residentChat) fold(e residentEvent) bool {
	str := func(key string) string { s, _ := e.Data[key].(string); return s }
	g := &c.group
	if g.other == nil {
		g.other = map[string]int{}
	}
	switch e.Kind {
	case "assistant":
		calls, _ := e.Data["tool_calls"].([]any)
		for _, raw := range calls {
			call, _ := raw.(map[string]any)
			f, _ := call["function"].(map[string]any)
			name, _ := f["name"].(string)
			var args map[string]any
			if s, _ := f["arguments"].(string); s != "" {
				_ = json.Unmarshal([]byte(s), &args)
			}
			switch name {
			case "ask_user", "finish", "propose_plan":
				// drawn on their own (the question card, the finish, the plan card)
			case "terminal":
				if cmd, _ := args["command"].(string); strings.TrimSpace(cmd) != "" {
					g.commands++
					g.running, g.runningSince, g.runningGoal = oneLine(residentText(cmd)), e.Ts, e.Goal
				}
			case "read":
				g.reads++
			case "edit", "write":
				g.edits++
			default:
				g.other[name]++
			}
		}
		return false // its words are drawn (without the calls: compactAssistant)
	case "tool_result":
		name := str("name")
		if name == "ask_user" || name == "finish" {
			return false
		}
		result, _ := e.Data["result"].(map[string]any)
		if _, bad := result["error"]; bad {
			g.failed++
		}
		if name == "terminal" {
			if running, _ := result["running"].(bool); !running {
				g.running = ""
			}
		}
		return true
	case "gate":
		if ok, _ := e.Data["passed"].(bool); ok {
			g.gate = "passes"
		} else {
			g.gate = "fails"
		}
		return true
	}
	return false
}

// compactAssistant is an assistant event with only its words: the thinking line and the call cards go to the group.
func compactAssistant(e residentEvent) residentEvent {
	data := map[string]any{}
	for k, v := range e.Data {
		if k != "tool_calls" && k != "reasoning" {
			data[k] = v
		}
	}
	e.Data = data
	return e
}

// flushGroup fixes the group's line into the transcript and starts a new one.
func (c *residentChat) flushGroup(w int) {
	if !c.group.empty() {
		c.content += c.group.line(w) + "\n\n"
	}
	c.group = toolGroup{}
}
