// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestConfigCreatedWithDefaults(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "llm-top.json")
	c, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.InPerM != defaultPrices.InPerM || c.baseURL() != defaultURL || c.path != p {
		t.Errorf("defaults: %+v", c)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal("not written:", err)
	}
	for _, k := range []string{`"server_url"`, `"input_per_m"`, `"sessions_dir"`, `"vram_gb"`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("written file lacks %s", k)
		}
	}
	for _, k := range []string{`"backend"`, `"api_key"`, `"server_config"`, `"server_process"`, `"lmstudio_logs"`, `"router_model"`} {
		if strings.Contains(string(b), k) {
			t.Errorf("written file has %s", k)
		}
	}
	c2, err := loadConfig(p)
	if err != nil || c2.Help != c.Help || c2.priceCfg != c.priceCfg {
		t.Errorf("reload differs: %v", err)
	}
}

func TestConfigPartialWithBOM(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llm-top.json")
	// the 0.1.0-era file: prices only, written by PowerShell with a BOM
	write(t, p, "\xef\xbb\xbf{\"price_label\":\"Other\",\"input_per_m\":3,\"server_url\":\"10.0.0.5:8000\"}")
	c, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.Label != "Other" || c.InPerM != 3 || c.OutPerM != defaultPrices.OutPerM || c.HoldFile != "llm-watchdog.hold" {
		t.Errorf("merge: %+v", c)
	}
	if base := c.baseURL(); base != "http://10.0.0.5:8000" {
		t.Errorf("base %q", base)
	}
}

func TestConfigBrokenIsKept(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llm-top.json")
	write(t, p, `{"input_per_m": oops}`)
	c, err := loadConfig(p)
	if err == nil {
		t.Error("no error for broken JSON")
	}
	if c.InPerM != defaultPrices.InPerM {
		t.Error("broken file should fall back to defaults")
	}
	if b, _ := os.ReadFile(p); string(b) != `{"input_per_m": oops}` {
		t.Error("broken file was overwritten")
	}
}

// A value of the wrong type costs only that field: the rest of the file still counts, and the error names it.
func TestConfigWrongTypeKeepsTheRest(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llm-top.json")
	write(t, p, `{"server_url":"http://gpu-box:9000","vram_gb":"24","api_key":"k","openrgb":{"enabled":"yes","host":"h:1"}}`)
	c, err := loadConfig(p)
	if err == nil || !strings.Contains(err.Error(), "vram_gb should be a number, not string") {
		t.Errorf("error: %v", err)
	}
	if c.ServerURL != "http://gpu-box:9000" || c.VramGB != 0 || c.OpenRGB.Host != "h:1" || c.OpenRGB.Enabled {
		t.Errorf("fields lost or wrong: %+v", c)
	}
}

func TestConfigErrorInFooter(t *testing.T) {
	st := demoState(t)
	st.cfgErr = "llm-top.json: vram_gb should be a number, not string (using its default)"
	if out := frame(t, st, 120, 50); !strings.Contains(out, "config: llm-top.json: vram_gb should be a number") {
		t.Errorf("no config error in the footer:\n%s", out)
	}
}

// A file written for llm-top's other backends still loads: the keys this version no longer reads are ignored, without
// an error, and the rest counts.
func TestConfigOldKeysIgnored(t *testing.T) {
	p := filepath.Join(t.TempDir(), "llm-top.json")
	write(t, p, `{"server_url":"http://127.0.0.1:8093","backend":"lmstudio","api_key":"k","router_model":"m",
		"server_config":"llm-server.json","server_process":"llama-server","lmstudio_logs":"~/.lmstudio/server-logs",
		"sessions_dir":"~/s","openrgb":{"colors":{"asleep":"#ff0000"}}}`)
	c, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	h, _ := os.UserHomeDir()
	if c.baseURL() != "http://127.0.0.1:8093" || !strings.HasPrefix(c.sessionsGlob(), filepath.Join(h, "s")) {
		t.Errorf("fields lost: %+v", c)
	}
}

func TestBaseURL(t *testing.T) {
	c := defaultConfig
	for in, want := range map[string]string{"": defaultURL, "https://gpu-box:9000/": "https://gpu-box:9000",
		"10.0.0.5:8093": "http://10.0.0.5:8093", "http://127.0.0.1:8092": "http://127.0.0.1:8092"} {
		if c.ServerURL = in; c.baseURL() != want {
			t.Errorf("server_url %q: %q, want %q", in, c.baseURL(), want)
		}
	}
}

func TestConfigPaths(t *testing.T) {
	d := t.TempDir()
	c := defaultConfig
	c.path = filepath.Join(d, "llm-top.json")
	h, _ := os.UserHomeDir()
	for in, want := range map[string]string{"llm-watchdog.hold": filepath.Join(d, "llm-watchdog.hold"), "rel": filepath.Join(d, "rel"),
		"/abs/p": "/abs/p", "~": h, "~/x": filepath.Join(h, "x"), "": ""} {
		if got := c.rel(in); got != want {
			t.Errorf("rel(%q) = %q, want %q", in, got, want)
		}
	}
	if got := c.sessionsGlob(); got != filepath.Join(h, ".qwen", "projects", "*", "chats", "*.jsonl") {
		t.Errorf("default glob %q", got)
	}
	c.SessionsDir = "~/elsewhere"
	if got := c.sessionsGlob(); !strings.HasPrefix(got, filepath.Join(h, "elsewhere")) {
		t.Errorf("~ glob %q", got)
	}
	t.Setenv("LLMTOP_CONFIG", filepath.Join(d, "env.json"))
	if configPath("") != filepath.Join(d, "env.json") || configPath("x.json") != "x.json" {
		t.Error("configPath order")
	}
}

func TestBaseName(t *testing.T) {
	for in, want := range map[string]string{`C:\Users\me\src\Proj X`: "Proj X", "/work/me/src/p/": "p", "plain": "plain", `D:\m\a.gguf`: "a.gguf"} {
		if got := baseName(in); got != want {
			t.Errorf("baseName(%q) = %q", in, got)
		}
	}
}
