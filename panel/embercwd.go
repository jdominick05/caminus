// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"path"
	"regexp"
	"strings"

	"github.com/charmbracelet/x/ansi"
)

var leadingCd = regexp.MustCompile(`^\s*cd\s+("[^"]+"|'[^']+'|[^\s;&|]+)\s*(&&|;)`)

// agentCwd is the folder the agent works in: the runtime keeps no shell between calls, each terminal call runs in its
// cwd (the session's home when none is given), so it is the newest terminal call's, with a leading "cd DIR &&" taken
// into account. The agent account's home reads "~". Empty when the status has not named the home yet.
func (m emberModel) agentCwd() string {
	home := m.resident.Home
	if home == "" {
		return ""
	}
	cwd := home
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Kind != "assistant" {
			continue
		}
		calls, _ := e.Data["tool_calls"].([]any)
		found := false
		for j := len(calls) - 1; j >= 0 && !found; j-- {
			c, _ := calls[j].(map[string]any)
			f, _ := c["function"].(map[string]any)
			if name, _ := f["name"].(string); name != "terminal" {
				continue
			}
			var args struct {
				Cwd     string `json:"cwd"`
				Command string `json:"command"`
			}
			raw, _ := f["arguments"].(string)
			if json.Unmarshal([]byte(raw), &args) != nil {
				continue
			}
			found = true
			cwd = resolveIn(home, home, args.Cwd)
			if g := leadingCd.FindStringSubmatch(args.Command); g != nil {
				cwd = resolveIn(home, cwd, strings.Trim(g[1], `"'`))
			}
		}
		if found {
			break
		}
	}
	// the agent account's home is the parent of homes/NAME
	if dir, base := path.Split(path.Dir(home)); base == "homes" {
		acct := strings.TrimSuffix(dir, "/")
		if cwd == acct {
			return "~"
		}
		if strings.HasPrefix(cwd, acct+"/") {
			return "~/" + strings.TrimPrefix(cwd, acct+"/")
		}
	}
	return cwd
}

// resolveIn resolves p against base, with ~ as the agent account's home (two levels above the session home).
func resolveIn(home, base, p string) string {
	switch {
	case p == "":
		return base
	case p == "~" || strings.HasPrefix(p, "~/"):
		return path.Clean(path.Join(path.Dir(path.Dir(home)), strings.TrimPrefix(p, "~")))
	case strings.HasPrefix(p, "/"):
		return path.Clean(p)
	}
	return path.Clean(path.Join(base, p))
}

// clipPath keeps a path's end, the part that says where: "…/inventory/tests".
func clipPath(p string, w int) string {
	if ansi.StringWidth(p) <= w {
		return p
	}
	r := []rune(p)
	return "…" + string(r[len(r)-max(1, w-1):])
}
