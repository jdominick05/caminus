// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Config: everything llm-top can't read from the machine or the server lives in llm-top.json.
//
// Lookup order: -config <path>, $LLMTOP_CONFIG, llm-top.json next to the executable (if it exists), then
// <user config dir>/llm-top/llm-top.json (~/.config on Linux). The file is written with defaults only when it does
// not exist yet; missing fields keep their defaults, and keys this version no longer reads (llm-top's backend,
// api_key, router_model, server_config, server_process, lmstudio_logs) are ignored, so older files still load.
// Relative paths inside it resolve against the file's own directory.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

type config struct {
	ServerURL   string  `json:"server_url"`
	SessionsDir string  `json:"sessions_dir"`
	HoldFile    string  `json:"hold_file"`
	VramGB      float64 `json:"vram_gb"`
	Context     int     `json:"context"`
	Help        string  `json:"_help"`
	OpenRGB     rgbCfg  `json:"openrgb"`
	Web         webCfg  `json:"web"`
	priceCfg
	path string // the file this came from
}

// defaultURL is caminus serve's default port on this machine.
const defaultURL = "http://127.0.0.1:8092"

var defaultConfig = config{
	HoldFile: "llm-watchdog.hold",
	Help: "server_url: the Caminus server's base URL (empty = " + defaultURL + ", caminus serve's default; when " +
		"nothing answers there, the ports the caminus process listens on are tried). sessions_dir: Qwen Code " +
		"projects dir (empty = ~/.qwen/projects). hold_file: first line shown as HOLD while it exists. vram_gb / " +
		"context: 0 = detect. openrgb: light RGB devices by agent state through OpenRGB's SDK server (enabled, host, " +
		"devices = name substrings, empty = all in per-LED mode; brightness 0-1; colors = phase -> #rrggbb; " +
		"switch_modes = opt into temporary Direct/Custom/Static mode changes, restored on exit). web: " +
		"serve the dashboard to browsers while the TUI runs (enabled, listen = host:port, token = required as " +
		"?token= when set; -web serves without the TUI).",
	OpenRGB:  rgbCfg{Host: "127.0.0.1:6742", Brightness: 1},
	Web:      webCfg{Listen: ":8765"},
	priceCfg: defaultPrices,
}

// configPath picks the config file; explicit is "" when no -config was given.
func configPath(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if p := os.Getenv("LLMTOP_CONFIG"); p != "" {
		return p
	}
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), "llm-top.json")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "caminus", "top.json") // (Caminus's own, never the standalone llm-top's llm-top.json)
	}
	return "llm-top.json"
}

// loadConfig reads path, or writes the defaults there when it does not exist. A broken file is not overwritten.
func loadConfig(path string) (config, error) {
	c := defaultConfig
	c.path = path
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		if b, err := json.MarshalIndent(defaultConfig, "", "  "); err == nil {
			_ = os.MkdirAll(filepath.Dir(path), 0o755)
			_ = os.WriteFile(path, b, 0o644)
		}
		return c, nil
	}
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(bytes.TrimPrefix(b, []byte("\xef\xbb\xbf")), &c); err != nil {
		var te *json.UnmarshalTypeError
		if errors.As(err, &te) { // the decoder read every other field: only that one keeps its default
			return c, fmt.Errorf("%s: %s should be %s, not %s (using its default)", path, te.Field, jsonKind(te.Type), te.Value)
		}
		c = defaultConfig
		c.path = path
		return c, fmt.Errorf("%s: %v (using defaults)", path, err)
	}
	return c, nil
}

// jsonKind names what a config field takes, in JSON's words.
func jsonKind(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "a string"
	case reflect.Bool:
		return "true or false"
	case reflect.Slice, reflect.Array:
		return "a list"
	case reflect.Struct, reflect.Map:
		return "an object"
	}
	return "a number"
}

// rel resolves a path from the config against the config file's directory; "~/" means the home dir.
func (c config) rel(p string) string {
	if p == "" {
		return ""
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		if h, err := os.UserHomeDir(); err == nil {
			return filepath.Join(h, p[1:])
		}
	}
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(filepath.Dir(c.path), p)
}

func (c config) sessionsGlob() string {
	d := c.rel(c.SessionsDir)
	if d == "" {
		h, _ := os.UserHomeDir()
		d = filepath.Join(h, ".qwen", "projects")
	}
	return filepath.Join(d, "*", "chats", "*.jsonl")
}

// baseURL is server_url (a bare host:port gets http://), else defaultURL.
func (c config) baseURL() string {
	base := strings.TrimRight(c.ServerURL, "/")
	switch {
	case base == "":
		return defaultURL
	case !strings.Contains(base, "://"):
		return "http://" + base
	}
	return base
}

// baseName is filepath.Base for both separators: session logs may hold either OS's paths (tool arguments).
func baseName(p string) string {
	p = strings.TrimRight(p, `/\`)
	if i := strings.LastIndexAny(p, `/\`); i >= 0 {
		p = p[i+1:]
	}
	return p
}
