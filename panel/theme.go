// Copyright (C) 2026 jdominick05

package main

import (
	"charm.land/glamour/v2/ansi"
	"charm.land/lipgloss/v2"
	"embed"
	"fmt"
	"image/color"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The panel's themes. Every view — monitor, control panel and the resident env
// views — draws from one palette of semantic roles, so a role (accent, healthy,
// needs-you, failing, chrome) always means the same job on screen.
//
// Theme files use btop's syntax: `key = value`, full-line `#` comments, colors
// as #rgb / #rrggbb / #rrggbbaa or `term_N` for the terminal's own palette.
// A few common btop keys (main_fg, title_fg, hi_fg) are accepted as aliases so
// simple community themes drop in. Unknown keys are ignored (btop files carry
// many), an unparsable value falls back to the built-in default for that role.
//
// The base is white text on the terminal's own background, grey for what is
// secondary, and color spent only on what the eye should find first: a title,
// the agent's voice, a running tool, a question, a failure.

//go:embed themes
var themeFS embed.FS

// defaultTheme is the caminus palette: the roles' resting values.
// A theme file only overrides what it names; the rest keeps these.
func defaultTheme() map[string]string {
	// Iron & Patina, Caminus's own palette (the furnace's metal): cast iron, verdigris, brass, copper. Every text and
	// accent colour reads at WCAG AA (4.5:1) or better on its background. Its background is in themes/caminus.theme, not
	// here: a theme without one keeps the terminal's own background.
	return map[string]string{
		"text":   "#ece8e1", // what you read: replies, prompts, values
		"muted":  "#a6aeb4", // resting text: hints, echoes, output
		"think":  "#cdd0cf", // thinking: dimmer than the reply, still comfortable to read
		"dim":    "#39424b", // rules, empty slots, chrome that recedes
		"accent": "#5fc4b0", // verdigris: titles, the active pane
		"ok":     "#8fd081", // healthy: ready, the agent's own voice
		"warn":   "#ecc95e", // needs you: questions, pauses, a nearing compaction
		"action": "#e0904f", // copper: tools running (Ember's flame)
		"hot":    "#ec6f60", // failure: errors, hot context
		"purple": "#d9ad62", // brass: the second stream of data
		"user":   "#7fb2e5", // the human: goals, steering, your words
		"teal":   "#5fc4b0", // quiet positive chrome, inline code
		"chrome": "#39424b", // selection background, borders

		"code_bg":      "#262c33",     // the plate under inline code
		"codeblock_bg": "#111417",     // the plate under fenced blocks
		"code_theme":   "github-dark", // the chroma theme for fenced code
	}
}

// optionalRoles are a theme's background, all optional: none of them keeps the terminal's own background (every theme
// before them). bg is the background (and makes light themes possible: the text roles are read against it), bg2 a
// second colour for a gradient, pattern the ink of a background pattern, bg_pattern the pattern itself.
var optionalRoles = map[string]bool{"bg": true, "bg2": true, "pattern": true, "bg_pattern": true, "overlay": true, "overlay_border": true}

// bgPatterns are the background patterns a theme may name (drawn in the empty cells behind the text, bgpaint.go).
var bgPatterns = map[string]bool{"none": true, "dots": true, "grid": true, "diagonal": true, "waves": true,
	"gradient": true, "stars": true}

// aliasRole maps a theme-file key onto a role. Our own names pass through;
// a few btop keys ride along so a stock theme at least gets the loud parts right.
func aliasRole(key string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "main_fg":
		return "text", true
	case "title_fg", "hi_fg":
		return "accent", true
	case "proc_misc_fg":
		return "teal", true
	}
	_, known := defaultTheme()[key]
	return key, known || optionalRoles[key]
}

// validThemeValue accepts the colors btop themes carry, or, for the non-color
// code_theme key, any non-empty word.
func validThemeValue(key, v string) bool {
	if v == "" || len(v) > 32 || strings.ContainsAny(v, " \t") {
		return false
	}
	if key == "code_theme" {
		return true
	}
	if key == "bg_pattern" {
		return bgPatterns[v]
	}
	if key == "bg" || key == "bg2" || key == "pattern" || key == "overlay" || key == "overlay_border" {
		// painted into every cell as 24-bit colour: only #rrggbb paints (a #rgb, #rrggbbaa or term_N loaded without an
		// error and then painted nothing, leaving a light theme's dark text on the terminal's dark background)
		_, _, _, ok := hexRGB(v)
		return ok
	}
	if strings.HasPrefix(v, "#") {
		switch len(v) {
		case 4, 7, 9:
			for _, r := range v[1:] {
				switch {
				case r >= '0' && r <= '9', r >= 'a' && r <= 'f', r >= 'A' && r <= 'F':
				default:
					return false
				}
			}
			return true
		}
		return false
	}
	if strings.HasPrefix(v, "term_") {
		return true
	}
	return false
}

const (
	hexText   = "#ffffff" // what you read: replies, prompts, values
	hexMuted  = "#a8adb6" // resting text: hints, echoes, output
	hexThink  = "#cfd4dc" // thinking: dimmer than the reply, still comfortable to read
	hexDim    = "#5c626d" // rules, empty slots, chrome that recedes
	hexAccent = "#4fd6ff" // bright cyan: titles, the active pane
	hexGreen  = "#4dff9e" // healthy: ready, the agent's own voice
	hexYellow = "#ffd75f" // needs you: questions, pauses, a nearing compaction
	hexOrange = "#ffa25f" // action: tools running
	hexRed    = "#ff5f6d" // failure: errors, hot context
	hexPurple = "#c98fff" // a distinct second stream of data
	hexBlue   = "#6fa8ff" // the human: goals, steering, your words
	hexTeal   = "#4fffd0" // quiet positive chrome, inline code
	hexGray   = "#4a4f5a" // selection background, borders
)

var (
	cDim    = lipgloss.Color(hexDim)
	cMuted  = lipgloss.Color(hexMuted)
	cThink  = lipgloss.Color(hexThink)
	cText   = lipgloss.Color(hexText)
	cAccent = lipgloss.Color(hexAccent)
	cGreen  = lipgloss.Color(hexGreen)
	cYellow = lipgloss.Color(hexYellow)
	cOrange = lipgloss.Color(hexOrange)
	cRed    = lipgloss.Color(hexRed)
	cPurple = lipgloss.Color(hexPurple)
	cBlue   = lipgloss.Color(hexBlue)
	cTeal   = lipgloss.Color(hexTeal)
	cGray   = lipgloss.Color(hexGray)
)

// themeCurrent holds the roles in force so views that build styles on the fly
// (the transcript headings, the markdown renderer) read the same values.
var themeCurrent = defaultTheme()

// themeRole answers the value a role currently holds: a theme's override, else
// the built-in default. Never ask for an unknown role.
func themeRole(role string) string {
	if v, ok := themeCurrent[role]; ok {
		return v
	}
	return defaultTheme()[role]
}

// themeColor is themeRole as a color.Color for places beyond the shared styles.
// overlayColors are a popup's surface and frame: the theme's overlay and overlay_border when it sets them, else its
// background (or code-block plate) tinted toward its accent, framed in the accent. A popup must not read as more of the
// transcript behind it.
func overlayColors() (surface, border string) {
	surface, border = themeCurrent["overlay"], themeCurrent["overlay_border"]
	if surface == "" {
		base := themeRole("bg")
		if base == "" {
			base = themeRole("codeblock_bg")
		}
		surface = blendHex(base, themeRole("accent"), 0.10)
	}
	if border == "" {
		border = themeRole("accent")
	}
	return surface, border
}

func themeColor(role string) color.Color {
	return lipgloss.Color(themeRole(role))
}

// ---------- the shared styles (rebuilt by applyThemeStyles) ----------

var (
	emberTitle  lipgloss.Style
	emberSel    lipgloss.Style
	emberHead   lipgloss.Style
	emberDim    lipgloss.Style
	emberThink  lipgloss.Style
	emberWarn   lipgloss.Style
	emberTool   lipgloss.Style
	emberCode   lipgloss.Style
	emberCtxHot lipgloss.Style
	emberCtxOK  lipgloss.Style
	emberBusy   lipgloss.Style
	emberShell  lipgloss.Style // shell mode's "!" prompt and border title
	emberTabOn  lipgloss.Style
	emberBorder lipgloss.Style
)

// catColors (top_main.go) bakes a slice copy at its own init; the apply pass
// re-points it at the live palette.

// applyThemeStyles rebuilds every shared style from the current roles. Styles
// capture their color at construction, so a switch without this pass would
// leave the old hues in place.
func applyThemeStyles() {
	cText = lipgloss.Color(themeRole("text"))
	cMuted = lipgloss.Color(themeRole("muted"))
	cThink = lipgloss.Color(themeRole("think"))
	cDim = lipgloss.Color(themeRole("dim"))
	cAccent = lipgloss.Color(themeRole("accent"))
	cGreen = lipgloss.Color(themeRole("ok"))
	cYellow = lipgloss.Color(themeRole("warn"))
	cOrange = lipgloss.Color(themeRole("action"))
	cRed = lipgloss.Color(themeRole("hot"))
	cPurple = lipgloss.Color(themeRole("purple"))
	cBlue = lipgloss.Color(themeRole("user"))
	cTeal = lipgloss.Color(themeRole("teal"))
	cGray = lipgloss.Color(themeRole("chrome"))

	// top_main.go bakes this slice at its own init; re-point it at the live palette.
	catColors = [nCat]color.Color{cGray, cBlue, cGreen, cOrange, cYellow, cPurple, cTeal}

	emberTitle = lipgloss.NewStyle().Bold(true).Foreground(cAccent)
	emberSel = lipgloss.NewStyle().Background(cGray).Foreground(cText)
	emberHead = lipgloss.NewStyle().Bold(true).Foreground(cGreen)
	emberDim = lipgloss.NewStyle().Foreground(cMuted)
	emberThink = lipgloss.NewStyle().Foreground(cThink)
	emberWarn = lipgloss.NewStyle().Foreground(cYellow)
	emberTool = lipgloss.NewStyle().Foreground(cOrange)
	emberCode = lipgloss.NewStyle().Foreground(cTeal)
	emberCtxHot = lipgloss.NewStyle().Foreground(cRed)
	emberCtxOK = lipgloss.NewStyle().Foreground(cGreen)
	emberBusy = lipgloss.NewStyle().Bold(true).Foreground(cBlue)
	emberShell = lipgloss.NewStyle().Bold(true).Foreground(cPurple)
	emberTabOn = lipgloss.NewStyle().Bold(true).Background(cGray).Foreground(cText)
	emberBorder = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cGray)
}

// themeRestore applies the panel's saved theme at startup; the package's own
// defaults are already live, and a bad or absent save changes nothing.
func init() {
	applyThemeStyles()
	// the saved choice, else the built-in caminus theme: its file carries the background the defaults leave out
	name := "caminus"
	if b, err := os.ReadFile(themeSavePath()); err == nil && strings.TrimSpace(string(b)) != "" {
		name = strings.TrimSpace(string(b))
	}
	if themeLoad(name) != nil && name != "caminus" {
		_ = themeLoad("caminus")
	}
}

// ---------- theme files ----------

// parseTheme extracts role overrides from btop-syntax text. It returns the
// parsed overrides and the lines it could not use as key = value.
func parseTheme(data []byte) (map[string]string, []string) {
	out, bad := map[string]string{}, []string{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			bad = append(bad, line)
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		role, known := aliasRole(k)
		// An unknown key is a btop key we never mapped: ignore it quietly, the
		// file is allowed to carry much more than we read. Only lines that are
		// not key = value at all are reported as bad.
		if !known {
			continue
		}
		if !validThemeValue(strings.ToLower(k), v) {
			bad = append(bad, k+" = "+v)
			continue
		}
		out[role] = v
	}
	return out, bad
}

// themeLoad parses a theme by name (a built-in, else ~/.config/caminus/themes/NAME.theme)
// and puts it on screen. A file that parses to nothing is an error, so a
// misnamed copy never blanks the panel.
func themeLoad(name string) error {
	if name == "" || strings.ContainsAny(name, "/\\:") {
		return fs.ErrNotExist
	}
	data, err := fs.ReadFile(themeFS, "themes/"+name+".theme")
	if err != nil {
		var userErr error
		if dir, derr := os.UserConfigDir(); derr == nil {
			data, userErr = os.ReadFile(filepath.Join(dir, "caminus", "themes", name+".theme"))
		} else {
			userErr = derr
		}
		if userErr != nil {
			return err
		}
	}
	overrides, bad := parseTheme(data)
	if len(overrides) == 0 {
		return fs.ErrInvalid
	}
	// a background role that was refused fails the load (with the reason): loaded without it, a light theme drew its
	// dark text on the terminal's own dark background
	for _, line := range bad {
		if k, _, ok := strings.Cut(line, " = "); ok && (k == "bg" || k == "bg2" || k == "pattern") {
			return fmt.Errorf("theme %s: %s must be #rrggbb (it paints every cell)", name, line)
		}
	}
	merged := defaultTheme()
	for k, v := range overrides {
		merged[k] = v
	}
	themeCurrent = merged
	applyThemeStyles()
	return nil
}

// themeSet loads a theme and records the choice; the record is one plain name
// so a broken file anywhere only costs the choice, never the panel.
func themeSet(name string) error {
	if err := themeLoad(name); err != nil {
		return err
	}
	if p := themeSavePath(); p != "" {
		_ = os.MkdirAll(filepath.Dir(p), 0o755)
		_ = os.WriteFile(p, []byte(name+"\n"), 0o644)
	}
	return nil
}

func themeSavePath() string {
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "caminus", "panel-theme")
	}
	return ""
}

// themeNames lists the built-ins and the user's own files, sorted.
func themeNames() []string {
	seen := map[string]bool{}
	var out []string
	if entries, err := fs.ReadDir(themeFS, "themes"); err == nil {
		for _, e := range entries {
			if n, ok := strings.CutSuffix(e.Name(), ".theme"); ok {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	if dir, err := os.UserConfigDir(); err == nil {
		if entries, err := os.ReadDir(filepath.Join(dir, "caminus", "themes")); err == nil {
			for _, e := range entries {
				if n, ok := strings.CutSuffix(e.Name(), ".theme"); ok && !seen[n] {
					seen[n] = true
					out = append(out, n)
				}
			}
		}
	}
	sort.Strings(out)
	return out
}

// ---------- markdown ----------

// emberMarkdownStyle is the transcript's markdown in the same palette as the views: body text in the text
// role, headings in the bright roles (accent, user, teal), inline code teal on a dark plate, links blue
// and underlined, and fenced blocks through a bright chroma theme (theme key code_theme), so code in a
// reply is colored rather than dimmed. Read per call: a switched theme restyles the transcript when the
// renderer is rebuilt.
func emberMarkdownStyle() ansi.StyleConfig {
	color := func(s string) *string { return &s }
	flag := func(b bool) *bool { return &b }
	indent := func(u uint) *uint { return &u }

	var st ansi.StyleConfig
	st.Document.Margin = indent(0)
	st.Document.Color = color(themeRole("text"))

	st.Heading.Color = color(themeRole("accent"))
	st.Heading.Bold = flag(true)
	st.H1.Color = color(themeRole("accent"))
	st.H1.Bold = flag(true)
	st.H2.Color = color(themeRole("user"))
	st.H2.Bold = flag(true)
	st.H3.Color = color(themeRole("teal"))
	st.H3.Bold = flag(true)
	st.H4.Color = color(themeRole("purple"))
	st.H5.Color = color(themeRole("ok"))
	st.H6.Color = color(themeRole("muted"))

	st.Emph.Italic = flag(true)
	st.Strong.Bold = flag(true)
	st.Strikethrough.CrossedOut = flag(true)
	st.Link.Color = color(themeRole("user"))
	st.LinkText.Color = color(themeRole("user"))
	st.LinkText.Underline = flag(true)

	st.Code.Color = color(themeRole("teal"))
	st.Code.BackgroundColor = color(themeRole("code_bg"))
	st.CodeBlock.Theme = themeRole("code_theme")
	st.CodeBlock.Color = color(themeRole("text"))
	st.CodeBlock.BackgroundColor = color(themeRole("codeblock_bg"))
	st.CodeBlock.Margin = indent(1)

	st.BlockQuote.Color = color(themeRole("muted"))
	st.BlockQuote.Indent = indent(1)
	st.HorizontalRule.Color = color(themeRole("dim"))
	st.Item.Color = color(themeRole("accent"))
	st.Enumeration.Color = color(themeRole("accent"))
	// a numbered item printed "1Headless compositor": the number needs its separator (and a bullet its mark)
	if st.Enumeration.BlockPrefix == "" {
		st.Enumeration.BlockPrefix = ". "
	}
	if st.Item.BlockPrefix == "" {
		st.Item.BlockPrefix = "• "
	}
	// a table's header and cells in the body's text colour: glamour draws the cells in this style (its borders in the
	// terminal's own colour), and the chrome role it had (a border grey) left the words barely readable on a dark
	// background while the rules around them were bright. glamour v2 has one style for header and cells.
	st.Table.Color = color(themeRole("text"))
	st.Table.CenterSeparator = color("┼")
	st.Table.ColumnSeparator = color("│")
	st.Table.RowSeparator = color("─")
	return st
}
