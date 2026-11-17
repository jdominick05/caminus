// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envstudioview.go: format readers for the studio. Free time makes images (contact sheets, renders) as well as text,
// and the studio showed everything as text: an image was a screen of replacement characters, and opening the file
// from the desktop asked for an admin password (the agent's home is not the owner's). Now:
//   - images (PNG, JPEG, GIF) draw in the panel as truecolor half blocks, scaled to fit, two pixels per cell;
//   - Markdown renders; JSON is indented; CSV / TSV is an aligned table; other text as it is;
//   - anything else (PDF, SVG, HTML, audio, video, an image at full size): `o` copies it into the owner's cache
//     (~/.cache/caminus/studio/AGENT/) over SSH and opens it in the desktop's viewer.

import (
	"bytes"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// studioKind is how the studio shows a file, from its name.
func studioKind(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".gif":
		return "image"
	case ".md", ".markdown":
		return "markdown"
	case ".json":
		return "json"
	case ".csv":
		return "csv"
	case ".tsv":
		return "tsv"
	case ".svg", ".html", ".htm", ".xhtml", ".js", ".sh", ".py":
		return "text" // shown as source here; they can run code in a desktop app (activeFormat)
	case ".pdf", ".webp", ".bmp", ".tif", ".tiff", ".mp3", ".wav", ".ogg", ".flac", ".mp4",
		".webm", ".mkv", ".zip", ".gz", ".tar", ".xz", ".npy", ".npz", ".sqlite3", ".db":
		return "external"
	}
	return "text"
}

// activeFormat is a file a desktop app may run code from (a browser runs an HTML or SVG file's scripts, a PDF reader
// its JavaScript, a desktop may execute a script): it runs as the owner, outside the agent's account, and the agent
// reads the web in free time. Opening one on the desktop takes a second press after a warning.
func activeFormat(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".html", ".htm", ".xhtml", ".svg", ".pdf", ".js", ".sh", ".py", ".desktop", ".jar", ".exe", "":
		return true
	}
	return false
}

type studioRawMsg struct {
	path string
	data []byte
	err  string
	open bool // fetched to open on the desktop, not to draw
}

func readStudioRaw(m emberModel, path string, open bool) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Data string `json:"data"`
		}
		if err := residentJSON(m, []string{"studio", m.agentName, "--read", path, "--raw", "--json"}, &out); err != nil {
			return studioRawMsg{path: path, err: err.Error(), open: open}
		}
		b, err := base64.StdEncoding.DecodeString(out.Data)
		if err != nil {
			return studioRawMsg{path: path, err: "unreadable bytes: " + err.Error(), open: open}
		}
		return studioRawMsg{path: path, data: b, open: open}
	}
}

// studioOpenDir is where opened files are copied: the owner's cache, one folder per agent.
func studioOpenDir(agent string) string {
	base := os.Getenv("XDG_CACHE_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".cache")
	}
	return filepath.Join(base, "caminus", "studio", filepath.Base(agent))
}

// openOnDesktop writes the bytes into the owner's cache and hands the copy to the desktop's viewer.
var openOnDesktop = func(agent, path string, data []byte) (string, error) {
	dir := studioOpenDir(agent)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	dst := filepath.Join(dir, filepath.Base(path))
	if err := os.WriteFile(dst, data, 0o600); err != nil {
		return "", err
	}
	if demoMode {
		return dst, nil
	}
	cmd := exec.Command("xdg-open", dst)
	if err := cmd.Start(); err != nil {
		return dst, err
	}
	go cmd.Wait()
	return dst, nil
}

// receiveStudioRaw takes fetched bytes: an image to draw, or a file to open on the desktop.
func (m emberModel) receiveStudioRaw(msg studioRawMsg) emberModel {
	st := &m.chat.studio
	if msg.err != "" {
		if msg.open {
			m.msg = "could not open it: " + msg.err
		} else if msg.path == st.reading {
			st.err = msg.err
		}
		return m
	}
	if msg.open {
		dst, err := openOnDesktop(m.agentName, msg.path, msg.data)
		if err != nil {
			m.msg = "copied to " + dst + " but the viewer did not start: " + err.Error()
		} else {
			m.msg = "opened in your desktop viewer (a copy in " + filepath.Dir(dst) + ")"
		}
		return m
	}
	if msg.path != st.reading {
		return m
	}
	img, format, err := image.Decode(bytes.NewReader(msg.data))
	if err != nil {
		st.err = "not an image this view can draw (" + err.Error() + "): o opens it on your desktop"
		return m
	}
	st.img, st.imgInfo = img, fmt.Sprintf("%s  %d×%d  %s", format, img.Bounds().Dx(), img.Bounds().Dy(), humanBytes(len(msg.data)))
	return m
}

func humanBytes(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

// halfBlocks draws an image in at most cols × rows cells: each cell is two pixels (▀ in the top colour on the bottom
// colour), each pixel the average of the source block it covers. Terminal cells are about twice as tall as wide, so
// two stacked pixels make a square.
func halfBlocks(img image.Image, cols, rows int) []string {
	b := img.Bounds()
	if b.Dx() == 0 || b.Dy() == 0 || cols < 1 || rows < 1 {
		return nil
	}
	w, h := cols, rows*2
	if float64(b.Dx())/float64(b.Dy()) > float64(w)/float64(h) {
		h = max(2, int(float64(w)*float64(b.Dy())/float64(b.Dx())))
	} else {
		w = max(1, int(float64(h)*float64(b.Dx())/float64(b.Dy())))
	}
	h += h % 2
	px := func(x, y int) color.RGBA {
		x0, x1 := b.Min.X+x*b.Dx()/w, b.Min.X+(x+1)*b.Dx()/w
		y0, y1 := b.Min.Y+y*b.Dy()/h, b.Min.Y+(y+1)*b.Dy()/h
		x1, y1 = max(x1, x0+1), max(y1, y0+1)
		var r, g, bl, n uint64
		step := max(1, (x1-x0)*(y1-y0)/64) // at most ~64 samples a pixel
		i := 0
		for yy := y0; yy < y1; yy++ {
			for xx := x0; xx < x1; xx++ {
				if i%step == 0 {
					cr, cg, cb, _ := img.At(xx, yy).RGBA()
					r, g, bl, n = r+uint64(cr>>8), g+uint64(cg>>8), bl+uint64(cb>>8), n+1
				}
				i++
			}
		}
		return color.RGBA{uint8(r / n), uint8(g / n), uint8(bl / n), 255}
	}
	out := make([]string, 0, h/2)
	for y := 0; y < h; y += 2 {
		var sb strings.Builder
		for x := 0; x < w; x++ {
			t, u := px(x, y), px(x, y+1)
			fmt.Fprintf(&sb, "\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm▀", t.R, t.G, t.B, u.R, u.G, u.B)
		}
		sb.WriteString("\x1b[0m")
		out = append(out, sb.String())
	}
	return out
}

// formatText renders a text file for the studio by its kind.
func formatText(kind, text string, width int) []string {
	switch kind {
	case "markdown":
		r, err := glamour.NewTermRenderer(glamour.WithStyles(emberMarkdownStyle()), glamour.WithWordWrap(max(20, min(width-2, readingWidth))))
		if err == nil {
			if s, err := r.Render(text); err == nil {
				return strings.Split(strings.TrimRight(s, "\n"), "\n")
			}
		}
	case "json":
		var v any
		if json.Unmarshal([]byte(text), &v) == nil {
			if b, err := json.MarshalIndent(v, "", "  "); err == nil {
				return strings.Split(string(b), "\n")
			}
		}
	case "csv", "tsv":
		r := csv.NewReader(strings.NewReader(text))
		if kind == "tsv" {
			r.Comma = '\t'
		}
		r.FieldsPerRecord, r.LazyQuotes = -1, true
		if recs, err := r.ReadAll(); err == nil && len(recs) > 0 {
			return csvTable(recs, width)
		}
	}
	return strings.Split(text, "\n")
}

// csvTable aligns records into columns (each at most 32 cells wide), the header bold.
func csvTable(recs [][]string, width int) []string {
	cols := 0
	for _, r := range recs {
		cols = max(cols, len(r))
	}
	wid := make([]int, cols)
	for _, r := range recs {
		for i, c := range r {
			wid[i] = min(32, max(wid[i], ansi.StringWidth(residentText(c))))
		}
	}
	var out []string
	for n, r := range recs {
		var parts []string
		for i := 0; i < cols; i++ {
			c := ""
			if i < len(r) {
				c = residentText(r[i])
			}
			c = ansi.Truncate(c, wid[i], "…")
			parts = append(parts, c+strings.Repeat(" ", max(0, wid[i]-ansi.StringWidth(c))))
		}
		line := strings.Join(parts, "  ")
		if n == 0 {
			line = lipgloss.NewStyle().Bold(true).Render(line)
		}
		out = append(out, line)
	}
	return out
}
