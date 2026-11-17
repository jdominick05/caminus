package main

// shell.go: the building blocks of the chat-first env shell (the redesign): glyph tiers, the header, tool cards, the
// inline approval card, the todo line, the overlay box, the status line, and the streaming renderer that keeps a live
// reply cheap to watch. Everything here is a pure function of its inputs and the live theme, so the mockups
// (shell_mockup_test.go) and the real views draw the same thing.

import (
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// ---------------------------------------------------------------- glyph tiers

// glyphs is one tier's symbols. unicode is the default and needs no special font; nerd needs a Nerd Font (e.g.
// MesloLGS NF from the powerlevel10k README) and adds icons and powerline segments; ascii is for terminals that
// render neither. A terminal cannot report its font, so the tier is the owner's setting (g on the theme screen cycles it).
type glyphs struct {
	name                                 string
	working, waiting, done, idle, failed string // goal states
	collapsed, expanded                  string // cards
	pass, fail                           string // gates
	todoDone, todoNow, todoNext          string
	user, agent, tool, edit, gate, root  string // card kinds
	model, clock, cost, speed, queue     string // status line
	compact, online, cache               string // status line: compactions, time online, cache hits
	segL, segR                           string // status segment edges (powerline in nerd)
	bar, barEmpty                        string
	spinner                              []string
}

var glyphTiers = map[string]glyphs{
	"unicode": {name: "unicode", working: "●", waiting: "◐", done: "✓", idle: "○", failed: "✗",
		collapsed: "▸", expanded: "▾", pass: "✓", fail: "✗", todoDone: "●", todoNow: "◐", todoNext: "○",
		user: "›", agent: "◆", tool: "$", edit: "±", gate: "✓", root: "⚿",
		model: "◇", clock: "◷", cost: "$", speed: "⇢", queue: "≡", compact: "⇊", online: "◉", cache: "↺", segL: "", segR: " · ",
		bar: "▓", barEmpty: "░", spinner: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}},
	"nerd": {name: "nerd", working: "", waiting: "", done: "", idle: "", failed: "",
		collapsed: "", expanded: "", pass: "", fail: "", todoDone: "", todoNow: "", todoNext: "",
		user: "", agent: "\U000f06a9", tool: "", edit: "", gate: "", root: "",
		model: "", clock: "", cost: "", speed: "", queue: "", compact: "\uf066", online: "\uf1e6", cache: "\uf0e7", segL: "", segR: "",
		bar: "█", barEmpty: "░", spinner: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}},
	"ascii": {name: "ascii", working: "*", waiting: "~", done: "+", idle: "o", failed: "x",
		collapsed: ">", expanded: "v", pass: "+", fail: "x", todoDone: "[x]", todoNow: "[~]", todoNext: "[ ]",
		user: ">", agent: "*", tool: ">", edit: "+-", gate: "+", root: "#",
		model: "", clock: "", cost: "$", speed: "", queue: "", compact: "", online: "", cache: "", segL: "", segR: " | ",
		bar: "#", barEmpty: ".", spinner: []string{"|", "/", "-", "\\"}},
}

// unicode is the shipped default: it needs no special font, so it works in any terminal and over SSH from a phone.
// nerd is a per-user choice saved from the theme screen (most terminals have no Nerd Font and would show empty boxes).
var g = glyphTiers["unicode"]

// init applies the user's saved tier (CAMINUS_GLYPHS wins).
func init() {
	name := strings.TrimSpace(os.Getenv("CAMINUS_GLYPHS"))
	if name == "" {
		if p := glyphsSavePath(); p != "" {
			if b, err := os.ReadFile(p); err == nil {
				name = strings.TrimSpace(string(b))
			}
		}
	}
	if t, ok := glyphTiers[name]; ok {
		g = t
	}
}

// glyphsSet switches the tier and saves it, like themeSet does for colours.
func glyphsSet(name string) error {
	t, ok := glyphTiers[name]
	if !ok {
		return fmt.Errorf("unknown glyphs %q (nerd, unicode, ascii)", name)
	}
	g = t
	if p := glyphsSavePath(); p != "" {
		_ = os.MkdirAll(filepath.Dir(p), 0o755)
		_ = os.WriteFile(p, []byte(name+"\n"), 0o644)
	}
	return nil
}

func glyphsSavePath() string {
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "caminus", "panel-glyphs")
	}
	return ""
}

// glyphsNext is the tier after the current one, for g on the theme screen.
func glyphsNext() string {
	order := []string{"nerd", "unicode", "ascii"}
	for i, n := range order {
		if n == g.name {
			return order[(i+1)%len(order)]
		}
	}
	return "unicode"
}

// ---------------------------------------------------------------- pieces

// stateStyle is the one accent per state: working blue, waiting amber, needs-you red, done green.
func stateStyle(state string) (string, lipgloss.Style) {
	switch state {
	case "working", "queued":
		return g.working, lipgloss.NewStyle().Foreground(cBlue)
	case "waiting_model", "waiting_root", "paused":
		return g.waiting, lipgloss.NewStyle().Foreground(cYellow)
	case "waiting_user", "error":
		return g.failed, lipgloss.NewStyle().Foreground(cRed)
	case "completed", "free", "dream":
		return g.done, lipgloss.NewStyle().Foreground(cGreen)
	}
	return g.idle, lipgloss.NewStyle().Foreground(cMuted)
}

// shellHeader: the agent, its state, what it is doing, and on the right the step count.
func shellHeader(agent, state, goal string, steps, w int) string {
	icon, st := stateStyle(state)
	detail := ""
	if goal != "" {
		detail = lipgloss.NewStyle().Foreground(cBlue).Render(goal)
	}
	return screenHeader(agent, icon+" "+strings.ReplaceAll(state, "_", " "), st, detail, emberDim.Render(plural(steps, "step", "steps")), w)
}

// screenHeader is every screen's first two lines (the env chat, caminus panel, caminus top): CAMINUS · what · its
// state in its one colour · dim details, something on the right, and a chrome rule under it.
func screenHeader(what, state string, st lipgloss.Style, detail, right string, w int) string {
	left := emberTitle.Render("CAMINUS") + emberDim.Render(" · ") + lipgloss.NewStyle().Foreground(cText).Render(what)
	if state != "" {
		left += " " + st.Render(state)
	}
	if detail != "" {
		left += emberDim.Render(" · ") + detail
	}
	gap := w - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		left = ansi.Truncate(left, max(1, w-lipgloss.Width(right)-2), "…")
		gap = max(1, w-lipgloss.Width(left)-lipgloss.Width(right))
	}
	return left + strings.Repeat(" ", gap) + right + "\n" + lipgloss.NewStyle().Foreground(cGray).Render(strings.Repeat("─", w))
}

// shellMessage: a turn of the conversation, with a thin rule instead of a box.
func shellMessage(who, text string, w int) string {
	text = terminalText(text, true)
	var mark, name string
	var st lipgloss.Style
	if who == "you" {
		mark, name, st = g.user, "you", lipgloss.NewStyle().Foreground(cBlue).Bold(true)
	} else {
		mark, name, st = g.agent, who, lipgloss.NewStyle().Foreground(cGreen).Bold(true)
	}
	label := st.Render(mark+" "+name) + "  "
	lw := lipgloss.Width(label)
	body := lipgloss.NewStyle().Foreground(cText).Render(ansi.Wordwrap(text, max(10, w-lw), ""))
	lines := strings.Split(body, "\n")
	out := []string{label + lines[0]}
	for _, l := range lines[1:] {
		out = append(out, strings.Repeat(" ", lw)+l)
	}
	return strings.Join(out, "\n")
}

// toolCard: one line per tool call (kind, what, result, time); expanded, its body under a dim rule.
func toolCard(kind, what, meta string, ok bool, expanded bool, body []string, w int) string {
	what, meta = terminalText(what, false), terminalText(meta, false)
	if len(body) > 0 {
		safe := make([]string, len(body))
		for i, b := range body {
			safe[i] = terminalText(b, true)
		}
		body = safe
	}
	icon, st := g.tool, emberTool
	switch kind {
	case "edit", "write":
		icon, st = g.edit, lipgloss.NewStyle().Foreground(cPurple)
	case "gate":
		icon, st = g.gate, emberCtxOK
		if !ok {
			icon, st = g.fail, emberCtxHot
		}
	case "error":
		icon, st = g.fail, emberCtxHot
	case "output", "result":
		icon, st = "", emberDim
		if !ok {
			icon, st = g.fail, emberWarn
		}
	case "note":
		icon, st = g.idle, emberDim
	case "read":
		icon, st = g.collapsed, emberDim
	case "free", "dream":
		icon, st = g.done, emberCtxOK
	}
	label := strings.TrimSpace(icon + " " + kind)
	fold := g.collapsed
	if expanded {
		fold = g.expanded
	}
	line := emberDim.Render(fold) + " " + st.Render(label) + "  " + lipgloss.NewStyle().Foreground(cText).Render(what)
	if meta != "" {
		line += emberDim.Render("  ·  " + meta)
	}
	line = ansi.Truncate(line, w, "…")
	if !expanded || len(body) == 0 {
		return line
	}
	rule := lipgloss.NewStyle().Foreground(cGray).Render("│ ")
	out := []string{line}
	for _, b := range body {
		st := emberDim
		switch {
		case strings.HasPrefix(b, "+"):
			st = emberCtxOK
		case strings.HasPrefix(b, "-"):
			st = emberCtxHot
		}
		// an open card shows everything: long lines wrap under the rule, nothing is cut
		for _, piece := range strings.Split(ansi.Hardwrap(b, max(4, w-6), true), "\n") {
			out = append(out, "   "+rule+st.Render(piece))
		}
	}
	return strings.Join(out, "\n")
}

// approvalCard: a root request or a question, inline where the human is already looking.
func approvalCard(kind, cmd, why string, w int) string {
	return approvalCardKeys(kind, cmd, why, true, w)
}

// approvalCardKeys draws the card; inline false says where to decide instead of offering keys the view does not
// handle yet (the chat decides root requests in /approvals until its own keys are wired).
func approvalCardKeys(kind, cmd, why string, inline bool, w int) string {
	// the card shows exactly what runs: control characters and escapes in the agent's text are shown as escapes, so a
	// \r or a cursor move cannot draw a different command over the one that will run (idempotent on safe text)
	cmd, why = terminalText(cmd, true), terminalText(why, true)
	title := g.root + " ROOT"
	if kind == "question" {
		title = "? QUESTION"
	}
	keys := emberWarn.Render("Ctrl+A") + emberDim.Render(" decides it (approve, grant for the goal, edit, deny)  ·  ") +
		emberWarn.Render("/deny TEXT") + emberDim.Render(" says what to do instead")
	if kind == "github" {
		// a hand-off runs once, as filed, with the owner's gh: no grant, no edit
		title = "⇡ GITHUB"
		keys = emberWarn.Render("Ctrl+A") + emberDim.Render(" decides it (run it as you, deny)  ·  ") +
			emberWarn.Render("/deny TEXT") + emberDim.Render(" says what to do instead")
	}
	if !inline {
		keys = emberDim.Render("decide in ") + emberWarn.Render("/approvals")
	}
	inner := lipgloss.NewStyle().Foreground(cText).Bold(true).Render(cmd) + "\n" +
		emberDim.Render("why: "+why) + "\n" + keys
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cYellow).Padding(0, 1).
		Width(min(w, 72)).Render(inner)
	// the title rides on the top border
	return borderTitle(box, title, emberWarn.Bold(true))
}

// borderTitle writes a title into a rendered box's top border, replacing the rule there: "⚿ ROOT" on the approval
// card, the composer's target. A box too narrow for the title keeps its plain border, so a narrow terminal loses the
// label and never the frame.
func borderTitle(box, title string, st lipgloss.Style) string {
	if title == "" {
		return box
	}
	label := st.Render(" " + title + " ")
	lines := strings.Split(box, "\n")
	if len(lines) == 0 || lipgloss.Width(lines[0]) <= lipgloss.Width(label)+4 {
		return box
	}
	top := lines[0]
	lines[0] = ansi.Truncate(top, 2, "") + label + ansi.TruncateLeft(top, 2+lipgloss.Width(label), "")
	return strings.Join(lines, "\n")
}

// borderNote puts a short note on the right of a box's top border (the composer's transient messages: "delivered to
// goal …"), clipped to leave the left title room; empty leaves the box as it is.
func borderNote(box, note string, st lipgloss.Style) string {
	if note == "" {
		return box
	}
	lines := strings.Split(box, "\n")
	w := lipgloss.Width(lines[0])
	room := w/2 - 4
	if room < 8 {
		return box
	}
	label := st.Render(" " + ansi.Truncate(note, room, "…") + " ")
	lw := lipgloss.Width(label)
	lines[0] = ansi.Truncate(lines[0], w-lw-2, "") + label + ansi.TruncateLeft(lines[0], w-2, "")
	return strings.Join(lines, "\n")
}

// todoLine: the agent's plan, pinned above the composer.
type todoItem struct {
	text  string
	state string // done, now, next
}

func todoLine(items []todoItem, w int) string {
	parts := []string{emberDim.Render("todo ")}
	for _, it := range items {
		switch it.state {
		case "done":
			parts = append(parts, emberCtxOK.Render(g.todoDone)+" "+emberDim.Render(it.text))
		case "now":
			parts = append(parts, emberBusy.Render(g.todoNow)+" "+lipgloss.NewStyle().Foreground(cText).Render(it.text))
		default:
			parts = append(parts, emberDim.Render(g.todoNext+" "+it.text))
		}
	}
	return ansi.Truncate(strings.Join(parts, "  "), w, "…")
}

// composer: where the human types. The target rides on the top border, the mark sits on the first line only, and the
// hint says what Enter does now.
func composer(draft, hint, title string, w int) string {
	text := emberDim.Render(hint)
	if draft != "" {
		text = lipgloss.NewStyle().Foreground(cText).Render(draft) + "█"
	}
	return composerBox(emberBusy.Render(g.user)+" "+text, title, w)
}

// composerBox frames the editor's rendered lines as the composer: a rounded box in the chrome colour, with the target
// on its top border. The caller draws the prompt mark (the editor's own, so a wrapped line keeps its indent).
func composerBox(input, title string, w int) string {
	return borderTitle(emberBorder.Padding(0, 1).Width(w).Render(input), title, emberDim)
}

// labBadge is a small mark for the lab that made the model, in that lab's colour (a symbol, not the lab's logo: the
// logos are trademarks and do not survive a terminal cell); an unknown model gets the plain model glyph.
func labBadge(model string) string {
	id := strings.ToLower(model)
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	if g.name == "ascii" {
		return g.model
	}
	mark := func(sym, hex string) string {
		st := lipgloss.NewStyle().Foreground(lipgloss.Color(hex)).Bold(true)
		if g.name == "nerd" {
			st = st.Background(cGray) // (inside the model's pill: without it the badge's cell broke the pill)
		}
		return st.Render(sym)
	}
	switch {
	case strings.HasPrefix(id, "claude"):
		return mark("✳", "#d97757") // Anthropic
	case strings.Contains(id, "mistral") || strings.Contains(id, "stral") || strings.HasPrefix(id, "ministral"):
		return mark("▞", "#fa520f") // Mistral (Codestral, Devstral, Magistral, Ministral)
	case strings.Contains(id, "glm"):
		return mark("Z", "#3b6cff") // Z.ai
	case strings.Contains(id, "deepseek"):
		return mark("◆", "#4d6bfe") // DeepSeek
	case strings.Contains(id, "qwen") || strings.HasPrefix(id, "k3-") || strings.Contains(id, "swift") ||
		strings.Contains(id, "thinkingcap") || strings.Contains(id, "arex"):
		return mark("◈", "#615ced") // Qwen (and the local Caminus models, Qwen-based)
	}
	return g.model
}

// statusLine: brain, context, money and speed, then the keys that matter now. In the nerd tier the segments are
// powerline pills; elsewhere they are joined by a dot.
type status struct {
	brain        string
	cwd          string // the folder the agent works in, left of the context pill; empty hides it
	used, window int
	compact      float64 // the compaction threshold as a fraction of the window; the bar warms toward it
	usd          float64 // this goal's spend; negative hides the segment (an unpriced brain)
	usdDay       float64 // today's spend across goals; shown with usd when positive
	tps          float64
	queue        int
	compactions  int     // times this goal's history was compacted; -1 hides the bubble (no goal in view)
	online       float64 // how long the agent has been online, seconds; 0 hides it
	cacheHit     float64 // today's input from the prompt cache, 0-1; negative hides it
	keys         []string
}

func statusLine(s status, w int) string {
	frac := 0.0
	if s.window > 0 {
		frac = float64(s.used) / float64(s.window)
	}
	cells := 8
	full := min(cells, int(frac*float64(cells)+0.5))
	trigger := s.compact
	if trigger <= 0 || trigger > 1 {
		trigger = .85
	}
	barSt := emberCtxOK
	if frac >= trigger*.85 {
		barSt = emberWarn
	}
	if frac >= trigger {
		barSt = emberCtxHot
	}
	bar := barSt.Render(strings.Repeat(g.bar, full)) + emberDim.Render(strings.Repeat(g.barEmpty, cells-full))
	segs := []string{
		strings.TrimSpace(labBadge(s.brain) + " " + s.brain),
	}
	if s.cwd != "" {
		segs = append(segs, emberCode.Render(clipPath(s.cwd, max(12, w/5))))
	}
	segs = append(segs,
		emberDim.Render(fmt.Sprintf("%s/%s ", short(s.used), short(s.window)))+bar,
	)
	if s.usd >= 0 {
		seg := strings.TrimSpace(fmt.Sprintf("%s%.2f", g.cost, s.usd))
		if s.usdDay > 0 {
			seg += emberDim.Render(fmt.Sprintf(" · %.2f today", s.usdDay))
		}
		segs = append(segs, seg)
	}
	if s.tps > 0 {
		segs = append(segs, strings.TrimSpace(fmt.Sprintf("%s %.0f tok/s", g.speed, s.tps)))
	}
	if s.queue > 0 {
		segs = append(segs, strings.TrimSpace(fmt.Sprintf("%s %d queued", g.queue, s.queue)))
	}
	// the newest bubbles last: on a narrow terminal they go first (footBar drops from the end)
	if s.compactions >= 0 {
		// beside the context gauge it belongs to
		word := "compactions"
		if s.compactions == 1 {
			word = "compaction"
		}
		segs = append(segs, strings.TrimSpace(fmt.Sprintf("%s %d %s", g.compact, s.compactions, word)))
	}
	if s.cacheHit >= 0 {
		segs = append(segs, strings.TrimSpace(fmt.Sprintf("%s %.0f%% cached", g.cache, s.cacheHit*100)))
	}
	if s.online > 0 {
		segs = append(segs, strings.TrimSpace(g.online+" online "+upFor(s.online)))
	}

	return footBar(segs, s.keys, w)
}

// footBar is the status line every screen ends with (the env chat, caminus panel): the segments as pills on the left
// (plain dim text with separators under the plain glyph set), the keys that matter now, dim, on the right.
func footBar(segs, keyList []string, w int) string {
	var left string
	if g.name == "nerd" {
		pill := lipgloss.NewStyle().Background(cGray).Foreground(cText)
		edge := lipgloss.NewStyle().Foreground(cGray)
		open := strings.SplitN(pill.Render("\x00"), "\x00", 2)[0] // the pill's colours, to put back after a reset
		for _, sg := range segs {
			// a styled piece inside a segment (the lab badge) ends in a full reset, which dropped the pill's background
			// for the rest of the segment: the colours are restored after each one
			sg = strings.ReplaceAll(strings.ReplaceAll(sg, "\x1b[0m", "\x1b[m"), "\x1b[m", "\x1b[m"+open)
			left += edge.Render(g.segL) + pill.Render(sg) + edge.Render(g.segR) + " "
		}
	} else {
		// each segment styled on its own: a segment with its own colours (the bar) would end the outer style early
		styled := make([]string, len(segs))
		for i, sg := range segs {
			if strings.Contains(sg, "\x1b[") {
				styled[i] = sg
			} else {
				styled[i] = emberDim.Render(sg)
			}
		}
		left = strings.Join(styled, emberDim.Render(g.segR))
	}
	keys := emberDim.Render(strings.Join(keyList, "  "))
	gap := w - lipgloss.Width(left) - lipgloss.Width(keys)
	if gap < 2 && len(segs) > 2 && len(keyList) > 0 {
		// short of room: the optional bubbles go, newest added first, before the keys do
		return footBar(segs[:len(segs)-1], keyList, w)
	}
	if gap < 2 {
		if len(keyList) == 0 {
			return ansi.Truncate(left, w, "…")
		}
		// the keys never vanish: past the room they take a line of their own under the segments (at 80-100 columns a
		// model's segments left no keys at all, and the Servers pane lost x / t / l)
		k := ansi.Truncate(keys, w, "…")
		return ansi.Truncate(left, w, "…") + "\n" + strings.Repeat(" ", max(0, w-lipgloss.Width(k))) + k
	}
	return left + strings.Repeat(" ", gap) + keys
}

func short(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%dK", n/1000)
	}
	return fmt.Sprint(n)
}

// onlySGR keeps colour and style sequences (ESC [ digits ; m) and removes every other escape sequence (cursor moves,
// screen clears, OSC titles / clipboard / hyperlinks, DCS) and control character except newline and tab.
func onlySGR(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == 0x1b && i+1 < len(s) && s[i+1] == '[':
			j := i + 2
			for j < len(s) && s[j] >= 0x30 && s[j] <= 0x3f {
				j++
			}
			for j < len(s) && s[j] >= 0x20 && s[j] <= 0x2f {
				j++
			}
			if j < len(s) && s[j] == 'm' && strings.Trim(s[i+2:j], "0123456789;:") == "" {
				b.WriteString(s[i : j+1])
			}
			if j < len(s) {
				i = j
			} else {
				i = len(s)
			}
		case c == 0x1b && i+1 < len(s) && (s[i+1] == ']' || s[i+1] == 'P' || s[i+1] == '_' || s[i+1] == '^' || s[i+1] == 'X'):
			// a string sequence: up to BEL or ST (ESC \)
			j := i + 2
			for j < len(s) && s[j] != 0x07 && !(s[j] == 0x1b && j+1 < len(s) && s[j+1] == '\\') {
				j++
			}
			if j < len(s) && s[j] == 0x1b {
				j++
			}
			i = j
		case c == 0x1b:
			i++ // a two-byte escape: dropped with its final byte
		case c == 0xc2 && i+1 < len(s) && s[i+1] >= 0x80 && s[i+1] <= 0x9f:
			i++ // a C1 control (U+0080-U+009F, CSI and OSC among them), as UTF-8 encodes it: both bytes dropped
		case (c < 0x20 && c != '\n' && c != '\t') || c == 0x7f:
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// lowerLabel lower-cases a title's leading all-caps label ("GOALS  1 open of 2" -> "goals  1 open of 2") and leaves
// the rest as it is: a model ID or a path in a title keeps its case.
func lowerLabel(title string) string {
	end := 0
	for i, r := range title {
		if !(r >= 'A' && r <= 'Z' || r == ' ' || r == '&') {
			break
		}
		end = i + 1
	}
	return strings.ToLower(title[:end]) + title[end:]
}

// sectionHead is how a reading surface starts (Ember's thoughts, tasks, now, the /btw answer): its label bold in its
// colour, a dim note, and a thin rule to the edge. The chat's rule for frames: a frame means you can type or act
// there (the composer, the / menu, overlays); what you only read gets this heading instead.
func sectionHead(label string, c color.Color, note string, w int) string {
	s := lipgloss.NewStyle().Foreground(c).Bold(true).Render(label)
	if note != "" {
		s += emberDim.Render("  " + note)
	}
	return s + " " + lipgloss.NewStyle().Foreground(cGray).Render(strings.Repeat("─", max(2, w-lipgloss.Width(s)-1)))
}

// sectionHeadRight is sectionHead with its note at the right end of the rule, out of the way of the label.
func sectionHeadRight(label string, c color.Color, note string, w int) string {
	s := lipgloss.NewStyle().Foreground(c).Bold(true).Render(label)
	n := emberDim.Render(note)
	return s + " " + lipgloss.NewStyle().Foreground(cGray).Render(strings.Repeat("─", max(2, w-lipgloss.Width(s)-lipgloss.Width(n)-2))) + " " + n
}

// overlayBox: every overlay is the same box with the same keys; Esc always closes.
func overlayBox(title, body, keys string, w, h int) string {
	// lower case and grey like every other box in the chat; the title stays the accent. The body passes onlySGR: a
	// screen that forgot to sanitise some agent text can at worst recolour it, never move the cursor, clear the screen,
	// set the title or the clipboard.
	inner := emberTitle.Render(lowerLabel(title)) + "\n" + lipgloss.NewStyle().Foreground(cGray).Render(strings.Repeat("─", max(4, w-4))) +
		"\n" + onlySGR(body)
	lines := strings.Split(inner, "\n")
	for len(lines) < h-4 {
		lines = append(lines, "")
	}
	if len(lines) > h-4 {
		lines = lines[:h-4]
	}
	lines = append(lines, emberDim.Render(ansi.Truncate(keys+"  ·  esc close", max(8, w-4), "…")))
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cGray).Padding(0, 1).
		Width(w).Render(strings.Join(lines, "\n"))
}

// ---------------------------------------------------------------- streaming renderer

// streamView keeps a live conversation cheap to watch. Finished blocks are rendered once (markdown and all) and
// kept as lines; only the live tail is formatted while tokens arrive, with a cheap incremental formatter (wrapping and
// a little inline styling, carrying fence state), and it gets its one full render when it completes. Deltas are
// buffered; the view is rebuilt at most once per frame, and only the visible lines are joined.
type streamView struct {
	width   int
	done    []string // rendered lines of finished blocks
	live    strings.Builder
	inFence bool
	speaker string                     // who is talking in the live block
	frame   int                        // spinner frame, advanced once per drawn frame by the caller
	render  func(string, int) []string // the full renderer for a finished block (glamour in the panel)
}

func newStreamView(width int, render func(string, int) []string) *streamView {
	return &streamView{width: width, render: render, speaker: "resident"}
}

// Append adds a delta to the live block (cheap: no rendering here).
func (v *streamView) Append(delta string) { v.live.WriteString(terminalText(delta, true)) }

// Finish renders the live block once and moves it to the cache.
func (v *streamView) Finish() {
	if v.live.Len() == 0 {
		return
	}
	v.done = append(v.done, v.render(v.live.String(), v.width)...)
	v.live.Reset()
	v.inFence = false
}

// liveLines formats the live tail cheaply: soft wrap, code fences dimmed, no markdown parse.
func (v *streamView) liveLines() []string {
	var out []string
	fence := v.inFence
	for _, raw := range strings.Split(v.live.String(), "\n") {
		if strings.HasPrefix(strings.TrimSpace(raw), "```") {
			fence = !fence
			out = append(out, emberDim.Render(raw))
			continue
		}
		st := lipgloss.NewStyle().Foreground(cText)
		if fence {
			st = emberCode
		}
		for _, l := range strings.Split(ansi.Wordwrap(raw, max(8, v.width-3), ""), "\n") {
			out = append(out, st.Render(l))
		}
	}
	return out
}

// View returns the last h lines: cached finished lines plus the formatted live tail, under its speaker's label.
func (v *streamView) View(h int) string {
	live := v.liveLines()
	if v.live.Len() > 0 {
		label := lipgloss.NewStyle().Foreground(cGreen).Bold(true).Render(g.agent+" "+v.speaker) + " " + emberBusy.Render(g.spinner[v.frame%len(g.spinner)])
		live = append([]string{label}, live...)
	}
	need := h - len(live)
	var head []string
	if need > 0 {
		start := max(0, len(v.done)-need)
		head = v.done[start:]
	} else {
		live = live[len(live)-h:]
	}
	return strings.Join(append(append([]string{}, head...), live...), "\n")
}
