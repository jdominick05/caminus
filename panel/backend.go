// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// backend.go: the panel's only way to act. Every read and every action is one `caminus <command> --json` run (the Python
// CLI owns the logic: models, servers under the guard, the Qwen Code setup); the panel parses its JSON and renders it.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// camModel is one entry of `caminus models --json`.
type camModel struct {
	Name             string      `json:"name"`
	Title            string      `json:"title"`
	Arch             string      `json:"arch"`
	Cards            int         `json:"cards"`
	SizeGB           float64     `json:"size_gb"`
	Present          bool        `json:"present"`
	Path             string      `json:"path"`
	Speed            string      `json:"speed"`
	Notes            string      `json:"notes"`
	Modes            []serveMode `json:"serve_modes"`
	MissingDrafter   string      `json:"missing_drafter"`
	MissingDrafterGB float64     `json:"missing_drafter_gb"`
}

// serveMode is the architecture registry's supported card count and explicit drafter choices. Auto is the absence of
// --draft: the server then uses Arch.default_draft, including its measured defaults for this model and card count.
type serveMode struct {
	Cards  int      `json:"cards"`
	Drafts []string `json:"drafts"`
}

// serverInfo is one server of `caminus ps --json`.
type serverInfo struct {
	Name      string   `json:"name"`
	Model     string   `json:"model"`
	State     string   `json:"state"` // ready | loading | failed | stopped
	URL       string   `json:"url"`
	Port      int      `json:"port"`
	Cards     []string `json:"cards"`
	Pid       int      `json:"pid"`
	MaxCtx    int      `json:"max_ctx"`
	UptimeS   float64  `json:"uptime_s"`
	DecodeTPS *float64 `json:"decode_tps"`
	Busy      bool     `json:"busy"`
	Alias     string   `json:"alias"`
	Log       string   `json:"log"`
	Error     string   `json:"error"`
}

// cardInfo is one usable GPU of `caminus ps --json` (index: the position `--card` takes; pci: the card's identity).
type cardInfo struct {
	Index   int     `json:"index"`
	Name    string  `json:"name"`
	Pci     string  `json:"pci"`
	UsedMiB float64 `json:"used_mib"`
	LineMiB float64 `json:"line_mib"`
	Link    string  `json:"link"`
	LinkOK  bool    `json:"link_ok"`
	Display bool    `json:"display"`
	Status  string  `json:"status"` // supported | untested
}

// problemInfo is one line about the host's GPUs (caminus.device.host.problems): level error (Caminus cannot run anything:
// no supported card), warn (a card is not used, or a rule limits it: an unsupported card, a slot too slow for two-card
// models, an untested card) or info.
type problemInfo struct {
	Level string `json:"level"`
	Card  string `json:"card"`
	Text  string `json:"text"`
}

type psInfo struct {
	Servers  []serverInfo  `json:"servers"`
	Cards    []cardInfo    `json:"cards"`
	Problems []problemInfo `json:"problems"`
}

// harnessInfo is `caminus qwen-code status --json`.
type harnessInfo struct {
	Installed  bool   `json:"installed"`
	Binary     string `json:"binary"`
	Version    string `json:"version"`
	Pinned     string `json:"pinned"`
	Home       string `json:"home"`
	Configured bool   `json:"configured"`
	Npm        bool   `json:"npm"`
	Target     string `json:"target"` // the server it is set up for
}

// option is one flag of `caminus serve --options-json` (the advanced layout's list: every option the server accepts).
type option struct {
	Flag    string        `json:"flag"`
	Dest    string        `json:"dest"`
	Type    string        `json:"type"` // int | float | str | bool | ctx
	Default any           `json:"default"`
	Help    string        `json:"help"`
	Choices optionChoices `json:"choices"`
}

// argparse preserves numeric choices in --options-json. The editor holds command-line arguments as text; decode
// both numeric and string choices without rounding numbers or letting an unrelated option hide the whole form.
type optionChoices []string

func (c *optionChoices) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var choices optionChoices
	for _, value := range raw {
		value = bytes.TrimSpace(value)
		var text string
		if len(value) > 0 && value[0] == '"' {
			if err := json.Unmarshal(value, &text); err != nil {
				return err
			}
		} else {
			var number json.Number
			if err := json.Unmarshal(value, &number); err != nil || number == "" {
				return fmt.Errorf("option choice must be a string or number: %s", value)
			}
			text = number.String()
		}
		choices = append(choices, text)
	}
	*c = choices
	return nil
}

// actionResult is what every action command prints: {"ok": true, ...} or {"ok": false, "error": "..."}.
type actionResult struct {
	OK    bool   `json:"ok"`
	Error string `json:"error"`
	Note  string `json:"note"`
}

// ctlCLI runs the caminus CLI: argv is its command line (default `caminus`; CAMINUS_CLI or -caminus override it, e.g.
// "python -I -m caminus" from a checkout).
type ctlCLI struct {
	argv    []string
	timeout time.Duration
}

func newCtlCLI(cli string) *ctlCLI {
	if cli == "" {
		cli = os.Getenv("CAMINUS_CLI")
	}
	if cli == "" {
		cli = "caminus"
	}
	// a JSON array is taken as is (caminus panel passes its own interpreter and boot script, which holds spaces)
	var argv []string
	if strings.HasPrefix(strings.TrimSpace(cli), "[") && json.Unmarshal([]byte(cli), &argv) == nil && len(argv) > 0 {
		return &ctlCLI{argv: argv, timeout: 30 * time.Second}
	}
	return &ctlCLI{argv: strings.Fields(cli), timeout: 30 * time.Second}
}

// cmd is the exec.Cmd for `caminus args...` (also handed to tea.ExecProcess for the interactive commands).
func (b *ctlCLI) cmd(args ...string) *exec.Cmd {
	a := append(append([]string{}, b.argv[1:]...), args...)
	return exec.Command(b.argv[0], a...)
}

// run runs a non-interactive command and returns its stdout (stderr goes into the error).
func (b *ctlCLI) run(timeout time.Duration, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	a := append(append([]string{}, b.argv[1:]...), args...)
	c := exec.CommandContext(ctx, b.argv[0], a...)
	var out, errb bytes.Buffer
	c.Stdout, c.Stderr = &out, &errb
	err := c.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return out.Bytes(), fmt.Errorf("caminus %s: no answer after %s", strings.Join(args, " "), timeout)
	}
	if err != nil {
		msg := strings.TrimSpace(errb.String())
		if msg == "" {
			msg = err.Error()
		}
		return out.Bytes(), fmt.Errorf("caminus %s: %s", strings.Join(args, " "), lastLine(msg))
	}
	return out.Bytes(), nil
}

// call runs a command and decodes its JSON stdout into v.
func (b *ctlCLI) call(v any, timeout time.Duration, args ...string) error {
	out, err := b.run(timeout, args...)
	if err != nil && len(bytes.TrimSpace(out)) == 0 {
		return err
	}
	if jerr := json.Unmarshal(bytes.TrimSpace(out), v); jerr != nil {
		if err != nil {
			return err
		}
		return fmt.Errorf("caminus %s: not JSON (%v)", strings.Join(args, " "), jerr)
	}
	return nil
}

// act runs an action and returns its message (the error text when it failed).
func (b *ctlCLI) act(timeout time.Duration, args ...string) (string, error) {
	var r actionResult
	if err := b.call(&r, timeout, args...); err != nil {
		return "", err
	}
	if !r.OK {
		return "", fmt.Errorf("%s", r.Error)
	}
	return r.Note, nil
}

func lastLine(s string) string {
	l := strings.Split(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(l[len(l)-1])
}
