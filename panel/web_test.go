// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func get(t *testing.T, h http.Handler, method, target string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, target, nil)
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestWebFrame(t *testing.T) {
	st := demoState(t)
	evil := newSession("evil", `<script>alert(1)</script>&"x`)
	evil.model, evil.turns, evil.Mod = "qwen3-coder", 1, st.sessions[0].Mod
	st.sessions = append([]*session{evil}, st.sessions...)
	h := st.webHandler("")

	if w := get(t, h, "GET", "/", nil); w.Code != 200 || !strings.Contains(w.Body.String(), "<title>llm-top</title>") {
		t.Fatalf("page: %d %.200s", w.Code, w.Body.String())
	}
	if w := get(t, h, "GET", "/nope", nil); w.Code != 404 {
		t.Errorf("unknown path: %d", w.Code)
	}
	w := get(t, h, "GET", "/frame?w=120&h=40&sel="+url.QueryEscape("evil"), nil)
	body := w.Body.String()
	if w.Code != 200 || w.Header().Get("Content-Type") != "image/svg+xml" || !strings.HasPrefix(body, "<svg") {
		t.Fatalf("frame: %d %q %.200s", w.Code, w.Header().Get("Content-Type"), body)
	}
	if !strings.Contains(body, `data-cols="120"`) {
		t.Errorf("frame is not 120 columns: %.200s", body)
	}
	// session text is data: no markup gets through, only the escaped form
	if strings.Contains(body, "<script") || strings.Contains(body, `&"x`) || !strings.Contains(body, "&lt;script&gt;") {
		t.Errorf("session name not escaped")
	}
	if tags := regexp.MustCompile(`<(/?[a-z]+)`).FindAllStringSubmatch(body, -1); len(tags) > 0 {
		for _, m := range tags {
			switch strings.TrimPrefix(m[1], "/") {
			case "svg", "g", "rect", "text":
			default:
				t.Fatalf("unexpected element %q", m[1])
			}
		}
	}

	// tab moves the page's own selection, the TUI's stays where it was
	w = get(t, h, "GET", "/frame?d=1&sel=evil", nil)
	if sel, _ := url.QueryUnescape(w.Header().Get("X-Llmtop-Sel")); sel != st.sessions[1].Path {
		t.Errorf("tab picked %q, want %q", sel, st.sessions[1].Path)
	}

	// out-of-range sizes are clamped, gzip when asked for
	w = get(t, h, "GET", "/frame?w=5000&h=-3", map[string]string{"Accept-Encoding": "gzip"})
	if w.Header().Get("Content-Encoding") != "gzip" {
		t.Fatal("no gzip")
	}
	z, err := gzip.NewReader(w.Body)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(z)
	if !strings.Contains(string(b), `data-cols="200"`) {
		t.Errorf("width not clamped to 200: %.200s", b)
	}
}

// The web view is the TUI's frame: a Caminus server's panel is in it, and the page asks nothing of the server.
func TestWebCaminus(t *testing.T) {
	st := camState(t, "caminus_status.json")
	st.sessions = nil
	w := get(t, st.webHandler(""), "GET", "/frame?w=120&h=40", nil)
	body := w.Body.String()
	for _, want := range []string{"CAMINUS", "floor", "ms/token", "ttft", "queue", "prose", "reused", "XTX", "health", "tok/J", "parse"} {
		if !strings.Contains(body, ">"+want+"</text>") { // one <text> per word
			t.Errorf("frame lacks %q", want)
		}
	}
	if w.Code != 200 || strings.Contains(body, "<script") {
		t.Errorf("frame: %d", w.Code)
	}
}

func TestWebReadOnlyAndToken(t *testing.T) {
	st := demoState(t)
	open := st.webHandler("")
	for _, m := range []string{"POST", "PUT", "DELETE"} {
		if w := get(t, open, m, "/frame", nil); w.Code != 405 {
			t.Errorf("%s: %d", m, w.Code)
		}
	}
	h := st.webHandler("s3cret")
	for target, hdr := range map[string]map[string]string{
		"/":                    nil,
		"/frame":               nil,
		"/frame?token=wrong":   nil,
		"/frame?token=s3cret1": nil,
		"/frame?x=1":           {"Authorization": "Bearer nope"},
	} {
		if w := get(t, h, "GET", target, hdr); w.Code != 401 || strings.Contains(w.Body.String(), "<svg") {
			t.Errorf("%s %v: %d", target, hdr, w.Code)
		}
	}
	if w := get(t, h, "GET", "/?token=s3cret", nil); w.Code != 200 {
		t.Errorf("token in URL: %d", w.Code)
	}
	if w := get(t, h, "GET", "/frame", map[string]string{"Authorization": "Bearer s3cret"}); w.Code != 200 {
		t.Errorf("bearer: %d", w.Code)
	}
}

// TestFrameColors: parseANSI understands every SGR parameter render() emits, so no color is silently dropped.
func TestFrameColors(t *testing.T) {
	st := demoState(t)
	st.srv.Slot = slotJSON{NCtx: 131072, Processing: true, Task: 7, NPrompt: 61400, NProcessed: 9000, NCache: 52800}
	st.srv.Busy, st.srv.curRead = 1, 2400
	cam := camState(t, "caminus_status.json")
	cam.sessions = nil
	sgr := regexp.MustCompile("\x1b\\[([0-9;]*)m")
	for _, size := range [][3]int{{120, 40, 0}, {90, 60, 0}, {120, 40, 1}, {200, 60, 1}} {
		s := map[int]*state{0: st, 1: cam}[size[2]]
		for _, m := range sgr.FindAllStringSubmatch(model{st: s, w: size[0], h: size[1]}.render(), -1) {
			ps := strings.Split(m[1], ";")
			for k := 0; k < len(ps); k++ {
				switch ps[k] {
				case "", "0", "1", "22", "39", "49":
				case "38", "48":
					if k+4 >= len(ps) || ps[k+1] != "2" {
						t.Fatalf("color %q is not 24-bit", m[1])
					}
					k += 4
				default:
					t.Fatalf("SGR %q (in %q) not handled by parseANSI", ps[k], m[1])
				}
			}
		}
	}
}

// TestFrameWideGlyphs: emoji, CJK and combining marks take the columns render() gave them, so the right border
// stays put.
func TestFrameWideGlyphs(t *testing.T) {
	st := demoState(t)
	wide := newSession("wide", "cafe\u0301 ✅ 日本語 👍🏽")
	wide.model, wide.turns, wide.Mod = "qwen3-coder", 1, st.sessions[0].Mod
	st.sessions = append(st.sessions, wide)
	out := model{st: st, w: 120, h: 40}.render()
	if !strings.Contains(out, "日本語") {
		t.Fatal("wide session not shown")
	}
	lines := strings.Split(out, "\n")
	for i, row := range parseANSI(out) {
		if w := ansi.StringWidth(lines[i]); len(row) != w {
			t.Errorf("line %d: %d cells, render() made it %d wide: %q", i, len(row), w, ansi.Strip(lines[i]))
		}
	}
	if svg := frameSVG(out); !strings.Contains(svg, "日本語") || !strings.Contains(svg, "cafe\u0301") || strings.Contains(svg, "\x00") {
		t.Error("wide text lost in the SVG")
	}
}

// TestRenderWebSizes: /frame accepts up to 200 x 150, beyond what TestRenderAllSizes covers.
func TestRenderWebSizes(t *testing.T) {
	down := &state{cfg: defaultConfig, base: defaultURL, known: map[string]*session{}}
	for name, st := range map[string]*state{"down": down, "busy": demoState(t), "caminus": camState(t, "caminus_status.json")} {
		for w := 131; w <= 200; w++ {
			for _, h := range []int{10, 40, 70, 150} {
				frame(t, st, w, h)
			}
		}
		if t.Failed() {
			t.Fatalf("%s: see above", name)
		}
	}
}
