// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// panelview.go: the panel's one screen, top to bottom.
//
//	CAMINUS 0.2.1                 XTX ▓▓▓▓▓▓▓░ 21.3/24G   XT ▓▓▓▓▓░░░ 15.1/20G     the cards' memory
//	╭ running ───────────────────────────────────────────────╮                      what runs, live
//	│ ● coder · Qwen3.8-Flash-Next · 2 cards · :8092 · 1M ctx │
//	│   104 tok/s decode · 2,800 tok/s prefill · 3 clients    │
//	╰─────────────────────────────────────────────────────────╯
//	 models                    coder                                                the list, the selected
//	 › coder        59 GB  ✓   Qwen3.8-Flash-Next Coder IQ1_M                       model's details and
//	   swift-1.5    20 GB  ✓   cards    [2] 1                                       choices beside it (below
//	 agents                                                                         it when narrow)
//	   Ember 3 sessions, 1 working · e     Qwen Code 0.26.9 · c
//	 ↑↓ model  ←→ choice  enter start  s stop  e Ember  c Qwen Code  ? help
//
// The options editor (o) and the key list (?) take the models' place; everything else stays.
package main

import (
	"fmt"
	"image/color"
	"strconv"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// twoColumnsAt is the narrowest width that keeps the details beside the list (below it otherwise).
const twoColumnsAt = 76

// maxLayoutW caps the screen's width: on a wider terminal it stays at the left, its lines short enough to read.
const maxLayoutW = 120

// View draws the panel w cells wide and at most h lines tall, as tall as its content (the host's size; the host
// paints the background and fills the rest).
func (m ctl) View(w, h int) string {
	if w <= 0 {
		w = 120
	}
	if h <= 0 {
		h = 40
	}
	W := min(max(w, 40), maxLayoutW) // (narrower than 40: drawn at 40 and cut to the terminal at the end)
	m.w, m.h = W, h

	top := []string{m.header(W)}
	top = append(top, m.hwLines(W)...)
	bottom := []string{m.noteLine(W)}
	bottom = append(bottom, strings.Split(m.keysLine(), "\n")...)

	card := m.runningRows(W - 4)
	agents := m.agentLines(W)
	room := h - len(top) - len(bottom) - len(agents)
	// the sparkline goes first when the screen is short; the models' section keeps at least its heading and 3 rows
	if room-(len(card)+2) < 4 && len(card) > 2 && m.live.url != "" {
		card = card[:len(card)-1]
	}
	top = append(top, strings.Split(panel("running", m.runningColor(), W, card), "\n")...)
	room -= len(card) + 2

	var body []string
	switch m.view {
	case viewHelp:
		var pages int
		body, pages = m.helpLines(W, room)
		if m.seen != nil {
			m.seen.helpPages = pages
		}
	case viewOptions:
		body = m.optionsLines(W, room)
	default:
		body = m.modelsLines(W, room)
	}
	body = append(body, agents...)
	out := append(append(top, body...), bottom...)
	if len(out) > h {
		out = append(out[:max(0, h-len(bottom))], bottom...)
		out = out[len(out)-min(len(out), h):]
	}
	if w < W {
		for i, l := range out {
			out[i] = ansi.Truncate(l, w, "")
		}
	}
	return strings.Join(out, "\n")
}

// render is the frame at the last View's size (or 120 x 40): the tests' and -once's whole screen.
func (m ctl) render() string { return m.View(m.w, m.h) }

// pg is the panel's symbols: the glyph tier Ember uses, except a Nerd Font's icons, whose cell width terminals and the
// layout's width count disagree on (the running card's border moved by a cell); the plain Unicode set takes their place.
func pg() glyphs {
	if g.name == "nerd" {
		return glyphTiers["unicode"]
	}
	return g
}

// ---------- header ----------

// header is CAMINUS and the version on the left, each card's memory on the right (as much of it as fits).
func (m ctl) header(W int) string {
	left := emberTitle.Render("CAMINUS") + " " + emberDim.Render(version)
	if m.busy != "" {
		left += "  " + fg(cYellow).Render(spin[m.frame%len(spin)]+" "+m.busy)
	}
	for _, bw := range []int{8, 5, 3, 0} {
		right := m.meters(bw)
		if gap := W - lipgloss.Width(left) - lipgloss.Width(right); gap >= 2 || right == "" {
			return left + strings.Repeat(" ", max(0, gap)) + right
		}
	}
	return ansi.Truncate(left, W, "…")
}

// meters is each card's memory: name, a bar of bw cells (none at 0), used / total in GB.
func (m ctl) meters(bw int) string {
	var p []string
	for _, c := range m.ps.Cards {
		tot := c.TotalMiB
		if tot <= 0 {
			tot = c.LineMiB
		}
		t := bold(cText).Render(c.Name)
		if bw > 0 {
			t += " " + bar(c.UsedMiB, tot, bw)
		}
		t += " " + fg(cMuted).Render(fmt.Sprintf("%.1f/%.0fG", c.UsedMiB/1024, tot/1024))
		p = append(p, t)
	}
	return strings.Join(p, "   ")
}

// hwError is the host's blocking hardware problem (no supported GPU), or "": the panel refuses to start anything then.
func (m ctl) hwError() string {
	for _, p := range m.ps.Problems {
		if p.Level == "error" {
			return p.Text
		}
	}
	return ""
}

// hwLines are the hardware errors and warnings under the header (notes: the options editor lists them).
func (m ctl) hwLines(W int) []string {
	var rows []string
	for _, p := range m.ps.Problems {
		switch p.Level {
		case "error":
			rows = append(rows, ansi.Truncate(bold(cRed).Render(pg().failed+" "+p.Text), W, "…"))
		case "warn":
			rows = append(rows, ansi.Truncate(fg(cYellow).Render("! "+p.Text), W, "…"))
		}
	}
	return rows
}

// ---------- the running card ----------

// liveServers are the servers that run (ready or loading), the registry's first, then those started outside it.
func (m ctl) liveServers() []serverInfo {
	var out []serverInfo
	for _, s := range m.ps.Servers {
		if s.State == "ready" || s.State == "loading" {
			out = append(out, s)
		}
	}
	return out
}

// primary is the server the running card shows and s / t / L / m / c act on: the selected model's when it runs, else
// the first ready one, else the first loading one. A record that failed or stopped is never it.
func (m ctl) primary() *serverInfo {
	live := m.liveServers()
	if len(live) == 0 {
		return nil
	}
	pick := -1
	if mi := m.selModel(); mi != nil {
		for i, s := range live {
			if s.Model == mi.Name && (pick < 0 || s.State == "ready" && live[pick].State != "ready") {
				pick = i
			}
		}
	}
	for i, s := range live {
		if pick < 0 && s.State == "ready" {
			pick = i
		}
	}
	if pick < 0 {
		pick = 0
	}
	return &live[pick]
}

// runningColor is the card's frame title colour: the accent while a server is ready, yellow while it loads.
func (m ctl) runningColor() color.Color {
	switch sv := m.primary(); {
	case sv == nil:
		return cGray
	case sv.State == "loading":
		return cYellow
	}
	return cAccent
}

// modelTitle is the catalog title of a model name ("" when the catalog has none).
func (m ctl) modelTitle(name string) string {
	for _, mi := range m.models {
		if mi.Name == name {
			return mi.Title
		}
	}
	return ""
}

// ctxText is a context size as "1M", "262k", "40k".
func ctxText(n int) string {
	return strings.Replace(kTok(n), ".0M", "M", 1)
}

// runningRows are the running card's lines, w cells wide: the server in use and its live numbers, a sparkline of its
// decode speed, then one line for each other server that runs; or the quiet line when none does.
func (m ctl) runningRows(w int) []string {
	if !m.loaded["ps"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " looking for servers")}
	}
	sv := m.primary()
	if sv == nil {
		return []string{fg(cMuted).Render("no server running · enter starts the selected model")}
	}
	dot := fg(cGreen).Render(pg().working)
	if sv.State == "loading" {
		dot = fg(cYellow).Render(spin[m.frame%len(spin)])
	}
	sep := fg(cMuted).Render(" · ")
	name := sv.Name
	if sv.External && sv.Model != "" { // (its record name is made up: observed-PORT)
		name = sv.Model
	}
	parts := []string{bold(cText).Render(name)}
	if name != sv.Model && sv.Model != "" {
		parts = append(parts, fg(cText).Render(sv.Model))
	}
	if t := m.modelTitle(sv.Model); t != "" {
		parts = append(parts, fg(cText).Render(t))
	} else if t := m.liveName(*sv); t != "" {
		parts = append(parts, fg(cText).Render(t))
	}
	switch n := len(sv.Cards); {
	case n == 1:
		parts = append(parts, fg(cMuted).Render(sv.Cards[0]))
	case n > 1:
		parts = append(parts, fg(cMuted).Render(strconv.Itoa(n)+" cards"))
	}
	parts = append(parts, fg(cAccent).Render(":"+strconv.Itoa(sv.Port)))
	if sv.MaxCtx > 0 {
		parts = append(parts, fg(cMuted).Render(ctxText(sv.MaxCtx)+" ctx"))
	}
	if sv.UptimeS > 0 {
		parts = append(parts, fg(cMuted).Render("up "+upFor(sv.UptimeS)))
	}
	if sv.State == "loading" {
		parts = append(parts, fg(cYellow).Render("loading"))
	}
	line := dot + " " + strings.Join(parts, sep)
	if lipgloss.Width(line) > w && len(parts) > 2 { // the model's title goes first when the line is long
		line = dot + " " + strings.Join(append(parts[:1:1], parts[2:]...), sep)
	}
	rows := []string{line}

	n := m.liveNumbers(*sv)
	var nums []string
	if n.decode > 0 {
		nums = append(nums, bold(cGreen).Render(fmt.Sprintf("%.0f", n.decode))+fg(cText).Render(" tok/s decode"))
	}
	if n.prefill > 0 {
		nums = append(nums, bold(cYellow).Render(humanInt(int(n.prefill+0.5)))+fg(cText).Render(" tok/s prefill"))
	}
	if m.live.url == sv.URL && m.live.srv.Up {
		nums = append(nums, fg(cText).Render(m.clientsText()))
	}
	if n.cached >= 0 {
		nums = append(nums, fg(cBlue).Render(fmt.Sprintf("%.0f%%", n.cached*100))+fg(cText).Render(" cached"))
	}
	if n.decode == 0 && n.prefill == 0 {
		nums = append([]string{fg(cMuted).Render("no requests yet")}, nums...)
	}
	if sv.External {
		nums = append(nums, fg(cMuted).Render("started outside the panel"))
	}
	rows = append(rows, "  "+strings.Join(nums, sep))
	for _, o := range m.liveServers() {
		if o.URL == sv.URL && o.Name == sv.Name {
			continue
		}
		on := o.Name
		if o.External && o.Model != "" {
			on = o.Model
		}
		t := fg(cGreen).Render(pg().working) + " " + bold(cText).Render(on) + sep + fg(cAccent).Render(":"+strconv.Itoa(o.Port))
		if o.State == "loading" {
			t += sep + fg(cYellow).Render("loading")
		}
		if o.External {
			t += sep + fg(cMuted).Render("started outside the panel")
		}
		rows = append(rows, t+fg(cMuted).Render("  (select its model to act on it)"))
	}
	if m.live.url == sv.URL && m.live.srv.Up {
		what := fg(cMuted).Render("idle")
		switch n.phase {
		case "writing":
			what = fg(cGreen).Render("writing")
		case "reading":
			what = fg(cYellow).Render("reading a prompt")
		}
		sw := max(8, w-2-7-1-lipgloss.Width(what))
		rows = append(rows, "  "+fg(cMuted).Render("decode ")+liveSpark(n.spark, sw)+" "+what)
	}
	return rows
}

// liveSpark is the decode speed of the last rounds, w cells (newest on the right): a dim floor where it was idle.
func liveSpark(v []float64, w int) string {
	if len(v) > w {
		v = v[len(v)-w:]
	}
	top := 0.0
	for _, x := range v {
		top = max(top, x)
	}
	var b strings.Builder
	b.WriteString(fg(cDim).Render(strings.Repeat("▁", w-len(v))))
	for _, x := range v {
		if x <= 0 || top <= 0 {
			b.WriteString(fg(cDim).Render("▁"))
			continue
		}
		b.WriteString(fg(cGreen).Render(string(blocks[max(1, min(8, int(x/top*8+0.5)))])))
	}
	return b.String()
}

// clientsText is the followed server's clients: its Qwen Code sessions and the Ember sessions that think with it,
// "3 clients", or "3 clients (Qwen Code 1, Ember 2)" when both kinds are there.
func (m ctl) clientsText() string {
	q, e := m.live.clients, m.emberClients()
	t := strconv.Itoa(q+e) + " clients"
	if q+e == 1 {
		t = "1 client"
	}
	if q > 0 && e > 0 {
		t += fmt.Sprintf(" (Qwen Code %d, Ember %d)", q, e)
	}
	return t
}

// ---------- models and the selected model's details ----------

// liveOf is the server that runs the model, nil when none does.
func (m ctl) liveOf(name string) *serverInfo {
	for _, s := range m.liveServers() {
		if s.Model == name {
			return &s
		}
	}
	return nil
}

// recordOf is the registry's record of the model's server that does not run (a failed or stopped start), nil if none.
func (m ctl) recordOf(name string) *serverInfo {
	for i := range m.ps.Servers {
		s := m.ps.Servers[i]
		if s.Model == name && !s.External && (s.State == "failed" || s.State == "stopped") {
			return &s
		}
	}
	return nil
}

func gbText(v float64) string { return strconv.FormatFloat(v, 'f', 0, 64) + " GB" }

// nameWidth is the list's name column: the longest name, 12 cells at least.
func (m ctl) nameWidth() int {
	nw := 12
	for _, mi := range m.models {
		nw = max(nw, lipgloss.Width(mi.Name))
	}
	return min(nw, 20)
}

// listRows are the model list's lines: cursor, name, download size, and a mark (● runs, ✓ on disk, a yellow ✓ when its
// drafter is missing, ↓ not downloaded).
func (m ctl) listRows() []string {
	if !m.loaded["models"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " reading the model list")}
	}
	if len(m.models) == 0 {
		return []string{fg(cMuted).Render("no models known (caminus models)")}
	}
	nw := m.nameWidth()
	var rows []string
	for i, mi := range m.models {
		cur, name := "  ", fg(cText).Render(mi.Name)
		if i == m.curM {
			cur, name = bold(cAccent).Render("› "), bold(cAccent).Render(mi.Name)
		}
		mark := fg(cGreen).Render(pg().done)
		switch {
		case m.liveOf(mi.Name) != nil:
			mark = fg(cGreen).Render(pg().working)
		case !mi.Present:
			mark = fg(cMuted).Render("↓")
		case mi.MissingDrafter != "":
			mark = fg(cYellow).Render(pg().done)
		}
		rows = append(rows, cur+fit(name, nw)+" "+fg(cMuted).Render(fmt.Sprintf("%6s", gbText(mi.SizeGB)))+"  "+mark)
	}
	return rows
}

// wrapped is s word-wrapped to w cells, at most n lines (the last one cut with …).
func wrapped(s string, w, n int) []string {
	if s == "" || w <= 0 {
		return nil
	}
	ls := strings.Split(ansi.Wordwrap(s, w, ""), "\n")
	if len(ls) > n {
		ls = ls[:n]
		ls[n-1] = ansi.Truncate(ls[n-1]+" …", w, "…")
	}
	for i := range ls {
		ls[i] = ansi.Truncate(ls[i], w, "…")
	}
	return ls
}

// choiceLine is one choice: its label, then every value with the chosen one bracketed and bright.
func (m ctl) choiceLine(label string, on bool, values []string, chosen string) string {
	lab := fg(cMuted).Render(fmt.Sprintf("%-9s", label))
	if on {
		lab = bold(cAccent).Render(fmt.Sprintf("%-9s", label))
	}
	var vs []string
	for _, v := range values {
		if v == chosen {
			vs = append(vs, bold(cAccent).Render("["+v+"]"))
		} else {
			vs = append(vs, fg(cMuted).Render(v))
		}
	}
	return lab + strings.Join(vs, " ")
}

// detailRows are the selected model's details, w cells wide: what it is, how fast, its choices, the context that fits,
// the edited options, what runs or failed, and what enter and p do.
func (m ctl) detailRows(w int) []string {
	mi := m.selModel()
	if mi == nil {
		return nil
	}
	var rows []string
	if mi.Title != "" {
		rows = append(rows, wrapped(fg(cText).Render(mi.Title), w, 2)...)
	}
	if mi.Speed != "" {
		rows = append(rows, wrapped(fg(cMuted).Render(mi.Speed), w, 2)...)
	}

	mode, draft := m.modelMode(*mi)
	kv, lc := m.modelExtras(*mi, mode)
	choices := m.choiceRows(*mi)
	cursorOn := func(c choiceRow) bool {
		return len(choices) > 0 && choices[min(m.choice, len(choices)-1)] == c
	}
	var cards []string
	for _, md := range mi.servingModes() {
		cards = append(cards, strconv.Itoa(md.Cards))
	}
	if len(cards) > 1 {
		rows = append(rows, m.choiceLine("cards", cursorOn(choiceCards), cards, strconv.Itoa(mode.Cards)))
	} else {
		rows = append(rows, fg(cMuted).Render(fmt.Sprintf("%-9s", "cards"))+fg(cText).Render(strconv.Itoa(mode.Cards)))
	}
	if len(mode.Drafts) > 0 {
		rows = append(rows, m.choiceLine("drafter", cursorOn(choiceDrafter), append([]string{"auto"}, mode.Drafts...), draft))
	}
	if len(mode.KV) > 1 {
		up := func(s string) string {
			if s == "auto" {
				return s
			}
			return strings.ToUpper(s)
		}
		vals := []string{"auto"}
		for _, f := range mode.KV {
			vals = append(vals, up(f))
		}
		rows = append(rows, m.choiceLine("KV cache", cursorOn(choiceKV), vals, up(kv)))
	}
	if mode.Long {
		chosen := "native"
		if lc {
			chosen = "long"
		}
		line := m.choiceLine("context", cursorOn(choiceLong), []string{"native", "long"}, chosen)
		if lc {
			if mode.Cards >= 2 {
				line += fg(cMuted).Render(" YaRN x4, up to 131,072")
			} else {
				line += fg(cMuted).Render(" YaRN x4, as far as the card holds")
			}
		}
		rows = append(rows, line)
	}
	if fr := m.fitRow(*mi); fr != "" && m.liveOf(mi.Name) == nil { // (a running model's own memory is in the cards' figure)
		rows = append(rows, wrapped(fr, w, 2)...)
	}
	if o := m.editedText(*mi); o != "" {
		rows = append(rows, wrapped(fg(cMuted).Render("options  ")+bold(cAccent).Render(o)+fg(cMuted).Render(" · o edits"), w, 2)...)
	}

	switch sv, rec := m.liveOf(mi.Name), m.recordOf(mi.Name); {
	case sv != nil && sv.External:
		rows = append(rows, fg(cGreen).Render("running on :"+strconv.Itoa(sv.Port))+fg(cMuted).Render(" · started outside the panel"))
	case sv != nil:
		rows = append(rows, fg(cGreen).Render("running on :"+strconv.Itoa(sv.Port))+fg(cMuted).Render(" · s stops it"))
	case rec != nil && rec.State == "failed" && rec.Error != "":
		rows = append(rows, wrapped(fg(cRed).Render("last start failed: "+rec.Error), w, 2)...)
	case rec != nil && rec.Note != "":
		rows = append(rows, wrapped(fg(cMuted).Render(rec.Note), w, 2)...)
	}
	switch {
	case !mi.Present:
		rows = append(rows, fg(cYellow).Render("not downloaded")+fg(cMuted).Render(" · p downloads it ("+gbText(mi.SizeGB)+")"))
	case mi.MissingDrafter != "":
		rows = append(rows, wrapped(bold(cAccent).Render("enter")+fg(cMuted).Render(" starts it · ")+
			fg(cYellow).Render(mi.MissingDrafter+" missing")+fg(cMuted).Render(fmt.Sprintf(": p downloads it (%.1f GB)", mi.MissingDrafterGB)), w, 2)...)
	case m.liveOf(mi.Name) == nil:
		rows = append(rows, bold(cAccent).Render("enter")+fg(cMuted).Render(" starts it"))
	}
	return rows
}

// editedText is the edited serve options and the card pin as they go to the next start ("" when none): nothing passed
// to the server stays out of sight.
func (m ctl) editedText(mi camModel) string {
	p := m.optionFlags("model", "config", "port")
	if mode, _ := m.modelMode(mi); mode.Cards == 1 && m.card != "" && m.card != "auto" {
		p = append(p, "--card", m.card)
	}
	return strings.Join(p, " ")
}

// modelsLines is the models' section in room lines at most: the list with the details beside it, or below it when
// the screen is narrow.
func (m ctl) modelsLines(W, room int) []string {
	if m.loaded["models"] && len(m.models) == 0 { // (no list to put details beside: the reason takes the whole width)
		why := fg(cMuted).Render("no models known (caminus models)")
		if err := m.readErr["models"]; err != nil {
			why = fg(cYellow).Render("couldn't read the model list: " + oneLine(err.Error()))
		}
		return append([]string{fg(cMuted).Render(" models")}, indent(wrapped(why, W-2, max(1, min(4, room-1))), " ")...)
	}
	list := m.listRows()
	listW := 2 + m.nameWidth() + 1 + 6 + 2 + 1 + 3
	head := fg(cMuted).Render(" models")
	name := ""
	if mi := m.selModel(); mi != nil {
		name = bold(cText).Render(mi.Name)
	}
	if W >= twoColumnsAt {
		dw := W - listW - 1
		det := m.detailRows(dw)
		n := max(len(list), len(det))
		n = min(n, max(3, room-1))
		lo := window(m.curM, len(list), n)
		out := []string{fit(head, listW) + " " + name}
		for i := 0; i < n; i++ {
			l, d := "", ""
			if lo+i < len(list) {
				l = list[lo+i]
			}
			if i < len(det) {
				d = det[i]
			}
			out = append(out, fit(" "+l, listW)+" "+ansi.Truncate(d, dw, "…"))
		}
		return out
	}
	det := m.detailRows(W - 3)
	nl := min(len(list), max(3, room-2-len(det)))
	lo := window(m.curM, len(list), nl)
	out := []string{head}
	for i := lo; i < lo+nl && i < len(list); i++ {
		out = append(out, " "+list[i])
	}
	nd := max(0, room-len(out)-1)
	out = append(out, "  "+name)
	for i := 0; i < min(nd, len(det)); i++ {
		out = append(out, "  "+det[i])
	}
	return out
}

// indent puts p before each line.
func indent(ls []string, p string) []string {
	for i := range ls {
		ls[i] = p + ls[i]
	}
	return ls
}

// window is the first row of an n-row view of a list of total rows that keeps cur in sight.
func window(cur, total, n int) int {
	if total <= n {
		return 0
	}
	return max(0, min(cur-n/2, total-n))
}

// ---------- agents ----------

// agentLines are Ember's line and Qwen Code's: side by side when they fit, else one under the other.
func (m ctl) agentLines(W int) []string {
	e, q := m.emberText(), m.qwenText()
	head := fg(cMuted).Render(" agents")
	if err := m.readErr["harness"]; err != nil && !m.hs.Installed { // (the reason on its own lines, wrapped, never cut)
		out := []string{head, ansi.Truncate("   "+e, W, "…"), ansi.Truncate("   "+q, W, "…")}
		return append(out, indent(wrapped(fg(cMuted).Render(oneLine(err.Error())), W-5, 2), "     ")...)
	}
	if lipgloss.Width(e)+lipgloss.Width(q)+8 <= W {
		return []string{head, "   " + e + "     " + q}
	}
	return []string{head, ansi.Truncate("   "+e, W, "…"), ansi.Truncate("   "+q, W, "…")}
}

func (m ctl) emberText() string {
	key := fg(cMuted).Render(" · ") + bold(cAccent).Render("e")
	name := bold(cText).Render("Ember")
	switch {
	case m.hosted:
		return name + fg(cMuted).Render(" (here)") + key + fg(cMuted).Render(" goes back")
	case !m.ember.known:
		return name + fg(cMuted).Render(" checking…")
	case !m.ember.ok:
		return name + fg(cYellow).Render(" not set up") + fg(cMuted).Render(" · run caminus ember setup")
	}
	c := m.live.ember
	if !c.known {
		return name + fg(cMuted).Render(" set up") + key
	}
	t := fmt.Sprintf(" %d session", c.sessions)
	if c.sessions != 1 {
		t += "s"
	}
	t += fmt.Sprintf(", %d working", c.active)
	w := fg(cText).Render(t)
	if c.active > 0 {
		w = fg(cText).Render(fmt.Sprintf(" %d session%s, ", c.sessions, map[bool]string{true: "s"}[c.sessions != 1])) +
			fg(cGreen).Render(fmt.Sprintf("%d working", c.active))
	}
	return name + w + key
}

func (m ctl) qwenText() string {
	name := bold(cText).Render("Qwen Code")
	h := m.hs
	switch {
	case !m.loaded["harness"]:
		return name + fg(cMuted).Render(" checking…")
	case m.readErr["harness"] != nil && !h.Installed: // (a failed read is not "not installed"; the reason goes under it)
		return name + fg(cYellow).Render(" couldn't read its status") + fg(cMuted).Render(" · ") + bold(cAccent).Render("R") +
			fg(cMuted).Render(" reads again")
	case !h.Installed && !h.Npm:
		return name + fg(cYellow).Render(" not installed") + fg(cMuted).Render(" · needs Node.js / npm")
	case !h.Installed:
		return name + fg(cYellow).Render(" not installed") + fg(cMuted).Render(" · ") + bold(cAccent).Render("i") + fg(cMuted).Render(" installs it")
	case !h.Configured:
		return name + " " + fg(cText).Render(h.Version) + fg(cYellow).Render(", not set up") + fg(cMuted).Render(" · ") +
			bold(cAccent).Render("u") + fg(cMuted).Render(" sets it up")
	}
	if !m.anyManaged() {
		return name + " " + fg(cText).Render(h.Version)
	}
	return name + " " + fg(cText).Render(h.Version) + fg(cMuted).Render(" · ") + bold(cAccent).Render("c")
}

// ---------- the last two lines ----------

func (m ctl) noteLine(W int) string {
	switch {
	case m.confirm != nil:
		return ansi.Truncate(bold(cYellow).Render(" ? "+m.confirm.q), W, "…")
	case m.note != "":
		c := cGreen
		if m.noteErr {
			c = cRed
		}
		return ansi.Truncate(fg(c).Render(" "+m.note), W, "…")
	}
	return ""
}

// keysLine is the key hints that fit the width: what the selected model and the running server allow, ? and q always.
// A key is never dropped silently: the labels shorten first, then go (the keys stay), and only then are keys left out,
// with "? more" at the end (the key list has every one).
func (m ctl) keysLine() string {
	w := m.w
	if w <= 0 {
		w = 120
	}
	type hint struct{ key, what, short string }
	var p []hint
	k := func(key, what string) hint { return hint{key, what, what} }
	switch {
	case m.confirm != nil:
		p = []hint{k("y", "yes"), k("any other key", "no")}
	case m.view == viewHelp:
		p = []hint{k("any key", "back")}
	case m.view == viewOptions && m.editing:
		p = []hint{k("enter", "save"), k("esc", "cancel"), k("ctrl+u", "clear")}
	case m.view == viewOptions:
		for _, x := range optionKeys() {
			p = append(p, k(x.keys, x.what))
		}
	default:
		p = append(p, k("↑↓", "model"))
		mi := m.selModel()
		if mi != nil {
			if n := len(m.choiceRows(*mi)); n > 1 {
				p = append(p, k("tab ←→", "choice"))
			} else if n == 1 {
				p = append(p, k("←→", "choice"))
			}
			if mi.Present && m.liveOf(mi.Name) == nil {
				p = append(p, k("enter", "start"))
			}
			if !mi.Present || mi.MissingDrafter != "" {
				p = append(p, k("p", "pull"))
			}
		}
		if sv := m.primary(); sv != nil && !sv.External {
			p = append(p, k("s", "stop"))
		}
		if m.hosted {
			p = append(p, hint{"e", "back to Ember", "back"})
		} else {
			p = append(p, k("e", "Ember"))
		}
		if m.anyManaged() { // (caminus qwen follows the servers caminus start records)
			p = append(p, hint{"c", "Qwen Code", "Qwen"})
		}
		p = append(p, hint{"o", "options", "opts"})
	}
	main := m.view == viewMain && m.confirm == nil
	quit := "quit"
	if m.hosted {
		quit = "close"
	}
	render := func(hs []hint, label func(hint) string, more bool) string {
		var parts []string
		for _, h := range hs {
			t := bold(cAccent).Render(h.key)
			if l := label(h); l != "" {
				t += emberDim.Render(" " + l)
			}
			parts = append(parts, t)
		}
		if main {
			help := "help"
			if more {
				help = "more"
			}
			parts = append(parts, bold(cAccent).Render("?")+emberDim.Render(" "+help), bold(cAccent).Render("q")+emberDim.Render(" "+quit))
		}
		return " " + strings.Join(parts, "  ")
	}
	if l := render(p, func(h hint) string { return h.what }, false); lipgloss.Width(l) <= w {
		return l
	}
	labelled := map[string]bool{} // (short labels, then the last keys' labels go one by one: the first keys keep theirs)
	for _, h := range p {
		labelled[h.key] = true
	}
	for i := len(p); i >= 0; i-- {
		if i < len(p) {
			labelled[p[i].key] = false
		}
		short := func(h hint) string {
			if labelled[h.key] {
				return h.short
			}
			return ""
		}
		if l := render(p, short, false); lipgloss.Width(l) <= w {
			return l
		}
	}
	for n := len(p) - 1; n > 0; n-- {
		if l := render(p[:n], func(hint) string { return "" }, true); lipgloss.Width(l) <= w {
			return l
		}
	}
	return ansi.Truncate(render(p[:min(1, len(p))], func(hint) string { return "" }, true), w, "…")
}

// ---------- the options editor and the key list ----------

// nextCard is the card choice after the current one: auto, then each usable card's position, then auto.
func (m ctl) nextCard() string {
	order := []string{"auto"}
	for _, c := range m.ps.Cards {
		order = append(order, strconv.Itoa(c.Index))
	}
	for i, k := range order {
		if k == m.card {
			return order[(i+1)%len(order)]
		}
	}
	return "auto"
}

// optionRows are the options editor's rows in height lines: the card pin, then every serve option (the ones in this
// colour go to the next start), and the help of the one under the cursor.
func (m ctl) optionRows(height int) []string {
	if !m.loaded["opts"] {
		return []string{fg(cMuted).Render(spin[m.frame%len(spin)] + " reading caminus serve's options")}
	}
	vo := m.visibleOpts()
	rows := []string{fg(cMuted).Render("for the next start: edited ones (") + bold(cAccent).Render("this colour") +
		fg(cMuted).Render(") go to it; every model's choices stay beside the model")}
	cur := func(i int) string {
		if i == m.curO {
			return bold(cAccent).Render("› ")
		}
		return "  "
	}
	card := fg(cMuted).Render("auto (a free card)")
	if m.card != "" && m.card != "auto" {
		card = bold(cAccent).Render(m.card)
	}
	all := []string{cur(0) + fit(fg(cText).Render("card for one-card models"), 26) + card}
	for i, o := range vo {
		v, edited := m.vals[o.Dest]
		val := fg(cMuted).Render(defaultText(o))
		if edited {
			val = bold(cAccent).Render(v)
		}
		if o.Type == "bool" {
			val = fg(cMuted).Render("off")
			if edited && v == "true" {
				val = bold(cAccent).Render("on")
			}
		}
		if m.editing && i+1 == m.curO {
			val = lipgloss.NewStyle().Foreground(cText).Background(cDim).Render(m.edit + "▏")
			if m.editFresh { // (selected: the first character typed replaces it)
				val = lipgloss.NewStyle().Foreground(cText).Background(cDim).Reverse(true).Render(m.edit) +
					lipgloss.NewStyle().Foreground(cText).Background(cDim).Render("▏")
			}
		}
		all = append(all, cur(i+1)+fit(fg(cText).Render(o.Flag), 26)+val)
	}
	if len(vo) == 0 {
		all = append(all, fg(cMuted).Render("  caminus serve --options-json gave no options"))
	}
	n := max(3, height-3)
	lo := window(m.curO, len(all), n)
	rows = append(rows, all[lo:min(len(all), lo+n)]...)
	help := "where a one-card model goes: auto picks a free card it fits on"
	if m.curO > 0 && m.curO-1 < len(vo) {
		help = vo[m.curO-1].Help
	}
	return append(rows, "", fg(cMuted).Render(oneLine(help)))
}

// cardRows are the cards as caminus ps reads them (memory against the guard's line, link, display) and the hardware
// notes.
func (m ctl) cardRows() []string {
	var rows []string
	for _, c := range m.ps.Cards {
		extra := ""
		if c.Link != "" {
			extra = " · " + c.Link
		}
		if c.Display {
			extra += " · display"
		}
		rows = append(rows, fit(bold(cText).Render(fmt.Sprintf("%d %s", c.Index, c.Name)), 8)+bar(c.UsedMiB, c.LineMiB, 20)+
			fg(cMuted).Render(fmt.Sprintf("  %s / %s MiB (the guard's line)%s", humanInt(int(c.UsedMiB)), humanInt(int(c.LineMiB)), extra)))
	}
	for _, p := range m.ps.Problems {
		switch p.Level {
		case "error":
			rows = append(rows, bold(cRed).Render(pg().failed+" "+p.Text))
		case "warn":
			rows = append(rows, fg(cYellow).Render("! "+p.Text))
		default:
			rows = append(rows, fg(cMuted).Render("· "+p.Text))
		}
	}
	if len(rows) == 0 {
		rows = append(rows, fg(cMuted).Render("no card readings"))
	}
	return rows
}

// optionsLines is the options editor in room lines: the options box, then the cards box when there is room.
func (m ctl) optionsLines(W, room int) []string {
	cards := m.cardRows()
	optH := room - 2
	if room >= 14+len(cards)+2 {
		optH = room - len(cards) - 4
	}
	out := strings.Split(panel("options", cAccent, W, m.optionRows(max(3, optH))), "\n")
	if len(out)+len(cards)+2 <= room {
		out = append(out, strings.Split(panel("cards", cGray, W, cards), "\n")...)
	}
	return out
}

// helpLines is every key of the main view and what it does: in two columns when both fit the width, and in pages when
// the keys do not fit the height (each page but the last ends with "… N more · ? next page"). It returns the lines and
// the number of pages.
func (m ctl) helpLines(W, room int) ([]string, int) {
	var items []string
	for _, x := range m.mainKeys() {
		items = append(items, fit(bold(cAccent).Render(x.keys), 7)+fg(cText).Render(x.what))
	}
	inner := W - 4
	cols, colW := 1, []int{inner}
	if half := (len(items) + 1) / 2; half > 0 {
		w0, w1 := 0, 0
		for i, it := range items {
			if i < half {
				w0 = max(w0, lipgloss.Width(it))
			} else {
				w1 = max(w1, lipgloss.Width(it))
			}
		}
		if inner >= 96 || w0+2+w1 <= inner { // (wide: the even split it always had; narrower: as wide as the keys need)
			cols, colW = 2, []int{inner / 2, inner - inner/2}
			if inner < 96 {
				colW = []int{w0 + 2, inner - w0 - 2}
			}
		}
	}
	grid := func(its []string) []string {
		per := (len(its) + cols - 1) / cols
		var rows []string
		for i := 0; i < per; i++ {
			var l string
			for c := 0; c < cols; c++ {
				if j := c*per + i; j < len(its) {
					l += fit(its[j], colW[c])
				}
			}
			rows = append(rows, l)
		}
		return rows
	}
	avail := max(2, room-2)
	rows := grid(items)
	foot := []string{"", fg(cMuted).Render("the full monitor with every engine number: m, or caminus top in a terminal")}
	switch {
	case len(rows)+len(foot) <= avail:
		return strings.Split(panel("keys", cAccent, W, append(rows, foot...)), "\n"), 1
	case len(rows) <= avail: // (the monitor hint goes before any key)
		return strings.Split(panel("keys", cAccent, W, rows), "\n"), 1
	}
	cols, colW = 1, []int{inner} // (pages are one column: a key never lands in a column too narrow for it)
	per := avail - 1             // (one row of each page says what is left)
	pages := (len(items) + per - 1) / per
	pg := m.helpPage % pages
	lo, hi := pg*per, min(len(items), (pg+1)*per)
	rows = grid(items[lo:hi])
	if hi < len(items) {
		rows = append(rows, fg(cMuted).Render(fmt.Sprintf("… %d more · ", len(items)-hi))+bold(cAccent).Render("?")+
			fg(cMuted).Render(" next page"))
	} else {
		rows = append(rows, fg(cMuted).Render(fmt.Sprintf("page %d of %d · any key back", pg+1, pages)))
	}
	return strings.Split(panel("keys", cAccent, W, rows), "\n"), pages
}
