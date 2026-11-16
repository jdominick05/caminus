// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// envthemes.go: the theme screen (/theme with no name). Themes were a name typed blind (/theme NAME) out of a list
// of six. Now the screen lists every theme with a strip of its colours, and beside it every element the chat draws,
// in the theme under the cursor: moving previews it across the whole screen, Enter keeps it, Esc goes back to the
// one you had.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

type themeScreen struct {
	open  bool
	names []string
	sel   int
	was   string            // the saved theme's name (marked "yours")
	roles map[string]string // the colours in force when the screen opened: Esc restores exactly these

	describing bool   // n: typing a description for the brain to write a theme from
	draft      string // the description typed so far
	busy       string // the description being written (a spinner shows); "" when idle
	err        string // why the last one was refused
}

// themeGenMsg is the brain's answer for a described theme.
type themeGenMsg struct {
	desc, text, err string
}

// askTheme has the resident's brain write a theme from a description (the resident's theme op: one call, no tools).
func askTheme(m emberModel, desc string) tea.Cmd {
	return func() tea.Msg {
		var out struct {
			Text string `json:"text"`
		}
		if err := residentJSON(m, []string{"theme", m.agentName, "--stdin", "--json"}, &out, desc); err != nil {
			return themeGenMsg{desc: desc, err: err.Error()}
		}
		return themeGenMsg{desc: desc, text: out.Text}
	}
}

// genCodeThemes are the code styles a written theme may name (bundled chroma styles).
var genCodeThemes = map[string]bool{"dracula": true, "monokai": true, "nord": true, "onedark": true, "github-dark": true,
	"gruvbox": true, "catppuccin-mocha": true, "rose-pine": true, "solarized-dark": true, "tokyonight-night": true,
	"doom-one": true, "native": true, "vulcan": true, "paraiso-dark": true, "hrdark": true,
	// light backgrounds
	"github": true, "solarized-light": true, "catppuccin-latte": true, "rose-pine-dawn": true, "tokyonight-day": true}

// themeFromAnswer keeps what can be validated from the brain's answer (it is untrusted text): role = #rrggbb lines
// for the panel's known roles, a known code style, and a name made safe for a file. Every colour role must be there.
func themeFromAnswer(answer, desc string) (name, file string, err error) {
	vals := map[string]string{}
	for _, line := range strings.Split(answer, "\n") {
		k, v, ok := strings.Cut(strings.Trim(strings.TrimSpace(line), "`"), "=")
		if !ok {
			continue
		}
		k, v = strings.ToLower(strings.TrimSpace(k)), strings.TrimSpace(v)
		if i := strings.IndexAny(v, " \t#("); i > 0 && !strings.HasPrefix(v, "#") {
			v = v[:i]
		}
		if f := strings.Fields(v); len(f) > 0 {
			v = f[0]
		}
		switch {
		case k == "name":
			name = v
		case k == "code_theme":
			if genCodeThemes[v] {
				vals[k] = v
			}
		case k == "bg_pattern":
			if bgPatterns[v] {
				vals[k] = v
			}
		case k == "bg" || k == "bg2" || k == "pattern":
			if len(v) == 7 && validThemeValue(k, v) {
				vals[k] = strings.ToLower(v)
			}
		default:
			if _, known := defaultTheme()[k]; known && len(v) == 7 && validThemeValue(k, v) {
				vals[k] = strings.ToLower(v)
			}
		}
	}
	var missing []string
	for role := range defaultTheme() {
		if role != "code_theme" && vals[role] == "" {
			missing = append(missing, role)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return "", "", fmt.Errorf("the answer missed %s", strings.Join(missing, ", "))
	}
	if vals["code_theme"] == "" {
		vals["code_theme"] = "dracula"
	}
	name = safeThemeName(name, desc)
	var b strings.Builder
	fmt.Fprintf(&b, "# %s: written by the brain from the description: %s\n", name, strings.Join(strings.Fields(residentText(desc)), " "))
	b.WriteString("# btop-syntax theme: key = value, full-line # comments, #rrggbb colours.\n\n")
	roles := make([]string, 0, len(vals))
	for k := range vals {
		roles = append(roles, k)
	}
	sort.Strings(roles)
	for _, k := range roles {
		fmt.Fprintf(&b, "%s = %s\n", k, vals[k])
	}
	return name, b.String(), nil
}

// safeThemeName: lower-case letters, digits and dashes, at most 32; from the description when the answer gave none;
// never a built-in's or an existing file's name (a -2, -3 … suffix).
func safeThemeName(name, desc string) string {
	clean := func(s string) string {
		var b strings.Builder
		for _, r := range strings.ToLower(s) {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
				b.WriteRune(r)
			case r == ' ' || r == '-' || r == '_':
				if b.Len() > 0 && !strings.HasSuffix(b.String(), "-") {
					b.WriteByte('-')
				}
			}
		}
		return strings.Trim(b.String(), "-")
	}
	n := clean(name)
	if n == "" {
		n = clean(desc)
	}
	if len(n) > 32 {
		n = strings.Trim(n[:32], "-")
	}
	if n == "" {
		n = "my-theme"
	}
	taken := map[string]bool{}
	for _, t := range themeNames() {
		taken[t] = true
	}
	base := n
	for i := 2; taken[n]; i++ {
		n = fmt.Sprintf("%s-%d", base, i)
	}
	return n
}

// receiveThemeGen saves a written theme to the owner's themes folder and selects it for preview.
func (m emberModel) receiveThemeGen(msg themeGenMsg) emberModel {
	t := &m.chat.themes
	if !t.open || t.busy != msg.desc {
		return m
	}
	t.busy = ""
	if msg.err != "" {
		t.err = "could not write it: " + msg.err
		return m
	}
	name, file, err := themeFromAnswer(msg.text, msg.desc)
	if err != nil {
		t.err = "refused: " + err.Error()
		return m
	}
	dir, derr := os.UserConfigDir()
	if derr != nil {
		t.err = "no config folder: " + derr.Error()
		return m
	}
	dir = filepath.Join(dir, "caminus", "themes")
	if err := os.MkdirAll(dir, 0o755); err == nil {
		err = os.WriteFile(filepath.Join(dir, name+".theme"), []byte(file), 0o644)
	}
	if err != nil {
		t.err = "could not save it: " + err.Error()
		return m
	}
	t.names, t.err = themeNames(), ""
	for i, n := range t.names {
		if n == name {
			t.sel = i
		}
	}
	m.previewTheme(name)
	m.msg = "theme " + name + " written · Enter keeps it"
	return m
}

// themeInUse is the saved theme's name ("caminus" when none was chosen).
func themeInUse() string {
	if p := themeSavePath(); p != "" {
		if b, err := os.ReadFile(p); err == nil {
			if n := strings.TrimSpace(string(b)); n != "" {
				return n
			}
		}
	}
	return "caminus"
}

func (m emberModel) openThemes() emberModel {
	names := themeNames()
	was := themeInUse()
	sel := 0
	for i, n := range names {
		if n == was {
			sel = i
		}
	}
	roles := make(map[string]string, len(themeCurrent))
	for k, v := range themeCurrent {
		roles[k] = v
	}
	m.chat.themes = themeScreen{open: true, names: names, sel: sel, was: was, roles: roles}
	m.chat.compose.Blur()
	return m
}

// previewTheme puts a theme in force without saving it, and restyles what caches its colours.
func (m *emberModel) previewTheme(name string) {
	if themeLoad(name) != nil {
		return
	}
	m.chat.markdown = nil
	m.chat.content, m.chat.rendered = "", 0
	m.chatStyles()
	m.syncChat()
}

func (m emberModel) themesKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	t := &m.chat.themes
	n := len(t.names)
	if t.describing {
		switch k := msg.String(); k {
		case "esc":
			t.describing, t.draft = false, ""
		case "enter":
			if d := strings.TrimSpace(t.draft); d != "" && t.busy == "" {
				t.describing, t.draft, t.busy, t.err = false, "", d, ""
				return m, askTheme(m, d)
			}
		case "backspace":
			if r := []rune(t.draft); len(r) > 0 {
				t.draft = string(r[:len(r)-1])
			}
		case "ctrl+u":
			t.draft = ""
		case "ctrl+c":
			return m, tea.Quit
		default:
			if txt := msg.Key().Text; txt != "" && len([]rune(t.draft)) < 300 {
				t.draft += txt
			}
		}
		return m, nil
	}
	switch msg.String() {
	case "n", "/":
		t.describing, t.err = true, ""
		return m, nil
	case "up", "k", "shift+tab":
		if n > 0 {
			t.sel = (t.sel - 1 + n) % n
			m.previewTheme(t.names[t.sel])
		}
	case "down", "j", "tab":
		if n > 0 {
			t.sel = (t.sel + 1) % n
			m.previewTheme(t.names[t.sel])
		}
	case "enter":
		if n > 0 {
			name := t.names[t.sel]
			if err := themeSet(name); err != nil {
				m.msg = "could not load " + name + ": " + err.Error()
			} else {
				m.msg = "theme: " + name
			}
			m.previewTheme(name)
		}
		m.chat.themes = themeScreen{}
		return m, m.chat.compose.Focus()
	case "esc", "q":
		// back to exactly what was showing (the saved name can differ from it: a theme file edited since, a test)
		themeCurrent = t.roles
		applyThemeStyles()
		m.chat.markdown = nil
		m.chat.content, m.chat.rendered = "", 0
		m.chatStyles()
		m.syncChat()
		m.chat.themes = themeScreen{}
		return m, m.chat.compose.Focus()
	case "ctrl+c":
		return m, tea.Quit
	}
	return m, nil
}

func fgTheme(role string) lipgloss.Style { return lipgloss.NewStyle().Foreground(themeColor(role)) }

// themeSwatch is a strip of a theme's colours, read from its file without putting it in force.
func themeSwatch(name string) string {
	saved := themeCurrent
	defer func() { themeCurrent = saved }()
	if themeLoad(name) != nil {
		return ""
	}
	var b strings.Builder
	if bg := themeRole("bg"); bg != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(themeColor("bg")).Render("▐"))
	} else {
		b.WriteString(" ")
	}
	for _, role := range []string{"accent", "ok", "warn", "action", "hot", "purple", "user", "teal"} {
		b.WriteString(lipgloss.NewStyle().Foreground(themeColor(role)).Render("█"))
	}
	return b.String()
}

// themesOverlay: the list on the left, the elements in the theme under the cursor on the right.
func (m emberModel) themesOverlay(w, h int) string {
	t := m.chat.themes
	listW := 34
	// the swatches load each theme's file: put the previewed one back in force afterwards
	current := t.names[t.sel]
	var list []string
	for i, n := range t.names {
		mark := "  "
		name := lipgloss.NewStyle().Foreground(themeColor("muted")).Render(n)
		if i == t.sel {
			mark = lipgloss.NewStyle().Foreground(themeColor("accent")).Bold(true).Render("› ")
			name = lipgloss.NewStyle().Foreground(themeColor("text")).Bold(true).Render(n)
		}
		if n == t.was {
			name += emberDim.Render(" (yours)")
		}
		sw := themeSwatch(n)
		list = append(list, mark+name+strings.Repeat(" ", max(1, listW-10-ansi.StringWidth(ansi.Strip(mark+name))))+sw)
	}
	_ = themeLoad(current)
	applyThemeStyles()
	room := max(4, h-6)
	top := 0
	if t.sel >= room {
		top = t.sel - room + 1
	}
	list = list[top:min(len(list), top+room)]
	// the describe line: what you type, the brain writing it, or why the last one was refused
	switch {
	case t.describing:
		list = append(list, "", fgTheme("accent").Render("describe a theme"), fgTheme("text").Render(t.draft)+"█",
			emberDim.Render("Enter: the brain writes it · Esc"))
	case t.busy != "":
		list = append(list, "", emberWarn.Render(g.spinner[m.tickN%len(g.spinner)]+" writing a theme…"),
			emberDim.Render(ansi.Truncate(t.busy, listW-2, "…")))
	case t.err != "":
		list = append(list, "", emberCtxHot.Render(ansi.Truncate(t.err, listW-2, "…")), emberDim.Render("n  try again"))
	default:
		list = append(list, "", emberDim.Render("n  describe a new theme"))
	}
	pw := max(30, w-listW-8)
	preview := themePreview(current, pw)
	left := lipgloss.NewStyle().Width(listW).Render(strings.Join(list, "\n"))
	body := lipgloss.JoinHorizontal(lipgloss.Top, left, "  ", preview)
	return overlayBox("THEMES  "+current, body, "↑↓ preview · Enter keep · n describe a new one · Esc back", w, h)
}

// themePreview draws one of every element the chat uses, in the theme in force.
func themePreview(name string, w int) string {
	fgc := func(role string) lipgloss.Style { return lipgloss.NewStyle().Foreground(themeColor(role)) }
	var out []string
	add := func(s ...string) { out = append(out, s...) }
	add(shellHeader("resident", "working", "add tests for the parser and make them pass", 42, w))
	add("")
	add(fgc("user").Render(g.user+" ")+fgc("text").Render("add tests for the parser, then make them pass"), "")
	add(emberThink.Render("  I should read the parser first and see which cases have no test."), "")
	add(fgc("ok").Render(g.agent+" ")+fgc("text").Render("The parser has no test for empty input. I'll add one in ")+
		emberCode.Render("parser_test.go")+fgc("text").Render("."), "")
	add("  " + emberTool.Render(g.tool+" terminal") + "  " + fgc("muted").Render("go test ./parser/..."))
	add("  "+fgc("muted").Render("ok   parser  0.41s"), "")
	add(formatText("markdown", "```go\nfunc TestEmpty(t *testing.T) {\n\tif _, err := Parse(\"\"); err == nil {\n\t\tt.Fatal(\"want an error\")\n\t}\n}\n```", w)...)
	add("")
	add(emberWarn.Render("? the agent asks: may I install golangci-lint?"))
	add(emberCtxHot.Render("✗ the gate fails: 1 test failed"))
	add(todoLine([]todoItem{{"read the parser", "done"}, {"add the tests", "now"}, {"open a PR", "next"}}, w))
	add(emberSel.Render(" a selected row "), "")
	add(borderTitle(emberBorder.Render(fgc("muted").Render("a framed box: menus, the composer, Ember's thoughts")), "a title", emberDim))
	add(footBar([]string{g.model + " " + name, "120K/1M", "$0.42"}, []string{"esc browse", "/help"}, w))
	return strings.Join(out, "\n")
}

// overlayOpen: an overlay or the session picker covers the conversation (the background pattern is not drawn then).
func (m emberModel) overlayOpen() bool {
	c := m.chat
	return c.approvals || c.goals || c.models.open || c.identity.open || c.advisor.open || c.themes.open || c.studio.open || c.view.open || c.rewind.open ||
		c.page >= 0 || c.picker
}
