// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// embergroup.go: the compact transcript's running line. Every action is its own card (emberactions.go); under them a
// dim line says the latest gate run and the command still running, fixed when the next reply or card arrives.
// Ctrl+O shows everything (each call, result and gate run as its own full card).

import (
	"encoding/json"
	"fmt"
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

// line is what the work in progress adds under the cards, w cells at most: the latest gate run and the command still
// running (every action is its own card: actionBlock).
func (t *toolGroup) line(w int) string {
	var segs []string
	switch t.gate {
	case "passes":
		segs = append(segs, emberCtxOK.Render(g.pass+" gate passes"))
	case "fails":
		segs = append(segs, emberWarn.Render(g.fail+" gate fails"))
	}
	if t.running != "" {
		segs = append(segs, emberDim.Render("running: ")+emberBusy.Render(t.running))
	}
	return ansi.Truncate("  "+strings.Join(segs, emberDim.Render(" · ")), max(10, w), "…")
}

// noteCalls keeps an assistant event's tool calls by id, so a result's card can name what was asked (both views).
func (c *residentChat) noteCalls(e residentEvent) []callInfo {
	var out []callInfo
	calls, _ := e.Data["tool_calls"].([]any)
	for _, raw := range calls {
		call, _ := raw.(map[string]any)
		f, _ := call["function"].(map[string]any)
		name, _ := f["name"].(string)
		var args map[string]any
		if s, _ := f["arguments"].(string); s != "" {
			_ = json.Unmarshal([]byte(s), &args)
		}
		if id, _ := call["id"].(string); id != "" {
			if c.calls == nil {
				c.calls = map[string]callInfo{}
			}
			c.calls[id] = callInfo{name: name, args: args}
		}
		out = append(out, callInfo{name: name, args: args})
	}
	return out
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
		for _, call := range c.noteCalls(e) {
			// each call becomes its own card when its result arrives (actionBlock); the line only says what still runs
			if call.name == "terminal" {
				if cmd, _ := call.args["command"].(string); strings.TrimSpace(cmd) != "" {
					g.running, g.runningSince, g.runningGoal = oneLine(residentText(cmd)), e.Ts, e.Goal
				}
			}
		}
		return false // its words are drawn (without the calls: compactAssistant)
	case "tool_result":
		name := str("name")
		if name == "ask_user" || name == "finish" {
			return false
		}
		result, _ := e.Data["result"].(map[string]any)
		if name == "terminal" {
			if running, _ := result["running"].(bool); !running {
				g.running = ""
			}
		}
		return true
	case "paused", "cancelled", "rewound", "turn_interrupted", "finished", "emergency_stop", "scheduled":
		g.running = "" // the call never ran or the goal stopped: nothing is running
		return false
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
	// only the gate's result stays in the history: "running: CMD" is the live tail's (fixed in, it stayed after the
	// command ended, was paused or cancelled, and repeated the command above its card)
	c.group.running = ""
	if !c.group.empty() {
		c.content += c.group.line(w) + "\n\n"
	}
	c.group = toolGroup{}
}
