package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func pngBytes(t *testing.T, w, h int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(255 * x / w), 0, uint8(255 * y / h), 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestTheStudioDrawsAnImage(t *testing.T) {
	m := chatFixture(t)
	m.width, m.height = 120, 40
	m.chat.studio.open = true
	m.chat.studio.list = []studioEntry{{Path: "/h/studio/sheet.png", Name: "sheet.png", Size: 900, group: "studio"}}
	next, cmd := m.studioKey(keyPress("enter"))
	m = next.(emberModel)
	if cmd == nil || m.chat.studio.reading != "/h/studio/sheet.png" {
		t.Fatal("Enter on an image fetches its bytes")
	}
	m = m.receiveStudioRaw(studioRawMsg{path: "/h/studio/sheet.png", data: pngBytes(t, 64, 32)})
	if m.chat.studio.img == nil || !strings.Contains(m.chat.studio.imgInfo, "png  64×32") {
		t.Fatalf("decoded: %q", m.chat.studio.imgInfo)
	}
	view := m.studioOverlay(m.width, m.height-4)
	if !strings.Contains(view, "▀") || !strings.Contains(view, "\x1b[38;2;") {
		t.Fatal("the image is drawn as truecolor half blocks")
	}
	for _, row := range strings.Split(view, "\n") {
		if ansi.StringWidth(row) > m.width {
			t.Fatalf("a row is wider than the terminal: %d", ansi.StringWidth(row))
		}
	}
	if rows := halfBlocks(m.chat.studio.img, 40, 40); len(rows) != 10 || ansi.StringWidth(rows[0]) != 40 {
		t.Fatalf("2:1 image in 40 columns: %d rows of %d (aspect kept, two pixels a cell)", len(rows), ansi.StringWidth(rows[0]))
	}
}

func TestTheStudioFormatsTextByKind(t *testing.T) {
	csvRows := formatText("csv", "name,score\nalpha,1\nbeta-long-name,22\n", 80)
	if len(csvRows) != 3 || !strings.Contains(ansi.Strip(csvRows[1]), "alpha           1") {
		t.Fatalf("csv aligned:\n%s", strings.Join(csvRows, "\n"))
	}
	if js := formatText("json", `{"a":{"b":1}}`, 80); len(js) < 4 || !strings.Contains(js[2], `"b": 1`) {
		t.Fatalf("json indented: %q", js)
	}
	if md := ansi.Strip(strings.Join(formatText("markdown", "# Title\n\nSome **bold** words.", 80), "\n")); strings.Contains(md, "**") || !strings.Contains(md, "bold") {
		t.Fatalf("markdown rendered:\n%s", md)
	}
	if studioKind("x.html") != "text" || studioKind("x.pdf") != "external" || studioKind("x.PNG") != "image" {
		t.Fatal("kinds: html as source, pdf external, png an image")
	}
}

func TestOpeningAnActiveFormatTakesASecondPress(t *testing.T) {
	m := chatFixture(t)
	m.chat.studio.open = true
	m.chat.studio.list = []studioEntry{{Path: "/h/studio/page.html", Name: "page.html"}, {Path: "/h/studio/a.png", Name: "a.png"}}
	next, cmd := m.studioKey(keyPress("o"))
	m = next.(emberModel)
	if cmd != nil || !strings.Contains(m.msg, "Press o again") {
		t.Fatalf("an HTML file warns first: %q", m.msg)
	}
	next, cmd = m.studioKey(keyPress("o"))
	m = next.(emberModel)
	if cmd == nil {
		t.Fatal("the second press opens it")
	}
	m.chat.studio.sel = 1
	if _, cmd = m.studioKey(keyPress("o")); cmd == nil {
		t.Fatal("an image opens at once")
	}
	// another key between the presses disarms
	m.chat.studio.sel = 0
	next, _ = m.studioKey(keyPress("o"))
	m = next.(emberModel)
	next, _ = m.studioKey(keyPress("down"))
	m = next.(emberModel)
	m.chat.studio.sel = 0
	if _, cmd = m.studioKey(keyPress("o")); cmd != nil {
		t.Fatal("the second press must follow the warning directly")
	}
}

func keyPress(k string) tea.KeyPressMsg {
	switch k {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	}
	return tea.KeyPressMsg{Code: rune(k[0]), Text: k}
}
func TestMarkdownListsKeepTheirNumbersAndBullets(t *testing.T) {
	out := ansi.Strip(strings.Join(formatText("markdown", "1. **Headless** sway\n2. Bridge\n\n- a bullet", 80), "\n"))
	if !strings.Contains(out, "1. Headless") || !strings.Contains(out, "• a bullet") {
		t.Fatal("list markers")
	}
}
