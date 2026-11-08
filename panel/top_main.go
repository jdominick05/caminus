// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// llm-top: live TUI for the Caminus server and the Qwen Code agents using it.
//
// Sources (all read-only; locations and keys in llm-top.json, see config.go):
//   - the Caminus server: /health, /props, /metrics, /slots (server.go) and /caminus/status (caminus.go)
//   - Qwen Code session logs, ~/.qwen/projects/*/chats/*.jsonl (tailed incrementally)
//   - machine counters every 5 s from /proc and sysfs (sys.go)
//   - an optional hold file (GPU job in progress)
//
// Context composition is an estimate: characters per part type since the last compaction, scaled to the
// prompt size the server reported for that session's last request.
//
//	llm-top                   keys: tab = next agent session, q = quit
//	llm-top -config <path>    use that config file
//	llm-top -version          print the version and exit
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"image/color"
	"io"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

var version = "dev" // set at build time: -ldflags "-X main.version=..."

// ---------- palette ----------

var (
	cDim    = lipgloss.Color("#3b4252")
	cMuted  = lipgloss.Color("#6b7489")
	cText   = lipgloss.Color("#d8dee9")
	cAccent = lipgloss.Color("#88c0d0")
	cGreen  = lipgloss.Color("#a3be8c")
	cYellow = lipgloss.Color("#ebcb8b")
	cOrange = lipgloss.Color("#d08770")
	cRed    = lipgloss.Color("#bf616a")
	cPurple = lipgloss.Color("#b48ead")
	cBlue   = lipgloss.Color("#81a1c1")
	cTeal   = lipgloss.Color("#8fbcbb")
	cGray   = lipgloss.Color("#4c566a")
)

func fg(c color.Color) lipgloss.Style { return lipgloss.NewStyle().Foreground(c) }

// ---------- sessions ----------

const (
	catSys = iota
	catHist
	catUser
	catTool
	catCall
	catThink
	catReply
	nCat
)

var catNames = [nCat]string{"system + tools", "compacted history", "prompts + reminders", "tool output", "tool calls", "thinking (this turn)", "replies"}
var catColors = [nCat]color.Color{cGray, cBlue, cGreen, cOrange, cYellow, cPurple, cTeal}

type event struct {
	T      time.Time
	Kind   string // tool, user, reply, compact
	Name   string
	Detail string
	ID     string
	Done   bool
	DoneT  time.Time
	Err    bool
}

type usageJSON struct {
	Prompt   int `json:"promptTokenCount"`
	Cand     int `json:"candidatesTokenCount"`
	Thoughts int `json:"thoughtsTokenCount"`
	Cached   int `json:"cachedContentTokenCount"`
}

type session struct {
	Path, Project string
	named         bool // Project came from a "cwd" field
	Mod           time.Time
	off           int64
	rest          []byte
	cats          [nCat]int
	prompts       []int
	compactBefore map[int]bool
	events        []event
	pending       map[string]int
	model         string
	turns, tools  int
	compactions   int
	lastUsage     usageJSON
	lastKind      string
	lastT         time.Time
}

type partJSON struct {
	Text         string          `json:"text"`
	Thought      json.RawMessage `json:"thought"`
	FunctionCall *struct {
		ID   string                 `json:"id"`
		Name string                 `json:"name"`
		Args map[string]interface{} `json:"args"`
	} `json:"functionCall"`
	FunctionResponse json.RawMessage `json:"functionResponse"`
}

type recJSON struct {
	Type      string `json:"type"`
	Subtype   string `json:"subtype"`
	Timestamp string `json:"timestamp"`
	Message   *struct {
		Role  string     `json:"role"`
		Parts []partJSON `json:"parts"`
	} `json:"message"`
	Usage *usageJSON `json:"usageMetadata"`
	Exec  *struct {
		ModelID string `json:"modelId"`
	} `json:"executionContext"`
	SystemPayload json.RawMessage `json:"systemPayload"`
	Cwd           string          `json:"cwd"`
}

func (s *session) addEvent(e event) {
	s.events = append(s.events, e)
	if len(s.events) > 400 {
		drop := len(s.events) - 300
		s.events = s.events[drop:]
		for id, i := range s.pending {
			if i < drop {
				delete(s.pending, id)
			} else {
				s.pending[id] = i - drop
			}
		}
	}
}

func isThought(p partJSON) bool {
	t := strings.TrimSpace(string(p.Thought))
	return t != "" && t != "false" && t != "null"
}

func (s *session) apply(line []byte) {
	var r recJSON
	if json.Unmarshal(line, &r) != nil {
		return
	}
	t, _ := time.Parse(time.RFC3339Nano, r.Timestamp)
	if r.Cwd != "" && !s.named {
		s.Project, s.named = baseName(r.Cwd), true
	}
	if r.Exec != nil && r.Exec.ModelID != "" {
		s.model = r.Exec.ModelID
	}
	if r.Subtype == "chat_compression" {
		var sp struct {
			History []struct {
				Parts []json.RawMessage `json:"parts"`
			} `json:"compressedHistory"`
		}
		_ = json.Unmarshal(r.SystemPayload, &sp)
		s.cats = [nCat]int{}
		for _, h := range sp.History {
			for _, p := range h.Parts {
				s.cats[catHist] += len(p)
			}
		}
		s.compactions++
		s.compactBefore[len(s.prompts)] = true
		s.lastUsage = usageJSON{} // the prompt size before the compaction doesn't scale what is left
		s.addEvent(event{T: t, Kind: "compact", Name: "compact", Detail: "context compacted", Done: true, DoneT: t})
		s.lastKind, s.lastT = "compact", t
		return
	}
	if r.Message == nil {
		return
	}
	switch r.Type {
	case "user":
		txt := 0
		for _, p := range r.Message.Parts {
			txt += len(p.Text)
		}
		if txt > 0 {
			s.cats[catUser] += txt
			s.cats[catThink] = 0 // the chat template drops earlier turns' reasoning
			if r.Subtype == "" {
				s.turns++
				first := ""
				for _, p := range r.Message.Parts {
					if p.Text != "" && !strings.HasPrefix(strings.TrimSpace(p.Text), "<system-reminder>") {
						first = p.Text
						break
					}
				}
				if first != "" {
					s.addEvent(event{T: t, Kind: "user", Name: "you", Detail: first, Done: true, DoneT: t})
				}
			}
			s.lastKind, s.lastT = "user", t
		}
	case "tool_result":
		for _, p := range r.Message.Parts {
			if len(p.FunctionResponse) == 0 {
				continue
			}
			s.cats[catTool] += len(p.FunctionResponse)
			var fr struct {
				ID       string                     `json:"id"`
				Response map[string]json.RawMessage `json:"response"`
			}
			_ = json.Unmarshal(p.FunctionResponse, &fr)
			if i, ok := s.pending[fr.ID]; ok && i < len(s.events) {
				s.events[i].Done, s.events[i].DoneT = true, t
				_, s.events[i].Err = fr.Response["error"]
				delete(s.pending, fr.ID)
			}
		}
		s.lastKind, s.lastT = "tool_result", t
	case "assistant":
		if r.Usage != nil && r.Usage.Prompt > 0 {
			s.lastUsage = *r.Usage
			s.prompts = append(s.prompts, r.Usage.Prompt)
		}
		calls, reply := 0, ""
		for _, p := range r.Message.Parts {
			switch {
			case p.FunctionCall != nil:
				b, _ := json.Marshal(p.FunctionCall.Args)
				s.cats[catCall] += len(b) + len(p.FunctionCall.Name)
				s.tools++
				calls++
				s.addEvent(event{T: t, Kind: "tool", Name: p.FunctionCall.Name, Detail: toolDetail(p.FunctionCall.Name, p.FunctionCall.Args), ID: p.FunctionCall.ID})
				if p.FunctionCall.ID != "" {
					s.pending[p.FunctionCall.ID] = len(s.events) - 1
				}
			case isThought(p):
				s.cats[catThink] += len(p.Text)
			default:
				s.cats[catReply] += len(p.Text)
				if reply == "" {
					reply = p.Text
				}
			}
		}
		if calls == 0 && strings.TrimSpace(reply) != "" {
			s.addEvent(event{T: t, Kind: "reply", Name: "reply", Detail: reply, Done: true, DoneT: t})
			s.lastKind = "reply"
		} else {
			s.lastKind = "call"
		}
		s.lastT = t
	}
}

func toolDetail(name string, a map[string]interface{}) string {
	str := func(k string) string {
		if v, ok := a[k].(string); ok {
			return v
		}
		return ""
	}
	switch name {
	case "run_shell_command":
		if d := str("description"); d != "" {
			return d
		}
		return str("command")
	case "read_file", "write_file", "edit", "replace":
		p := str("file_path")
		if p == "" {
			p = str("absolute_path")
		}
		return baseName(p)
	case "grep_search", "glob", "search_file_content":
		return str("pattern")
	}
	keys := make([]string, 0, len(a))
	for k := range a {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if v, ok := a[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// tail reads whatever was appended since the last call.
func (s *session) tail() {
	f, err := os.Open(s.Path)
	if err != nil {
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return
	}
	if st.Size() < s.off { // rewritten: start over
		*s = *newSession(s.Path, s.Project)
	}
	if _, err := f.Seek(s.off, io.SeekStart); err != nil {
		return
	}
	rd := bufio.NewReaderSize(f, 1<<20)
	for {
		chunk, err := rd.ReadBytes('\n')
		s.off += int64(len(chunk))
		if len(chunk) > 0 {
			if chunk[len(chunk)-1] == '\n' {
				line := chunk
				if len(s.rest) > 0 {
					line = append(s.rest, chunk...)
					s.rest = nil
				}
				s.apply(bytes.TrimSpace(line))
			} else {
				s.rest = append(s.rest, chunk...)
			}
		}
		if err != nil {
			break
		}
	}
	s.Mod = st.ModTime()
}

func newSession(path, project string) *session {
	return &session{Path: path, Project: project, compactBefore: map[int]bool{}, pending: map[string]int{}}
}

// estimate returns tokens per category, scaled so they add up to total.
func (s *session) estimate(total int) [nCat]int {
	var out [nCat]int
	sum := 0.0
	var est [nCat]float64
	for i := 1; i < nCat; i++ {
		est[i] = float64(s.cats[i]) / 3.6
		sum += est[i]
	}
	if total <= 0 {
		for i := range est {
			out[i] = int(est[i])
		}
		return out
	}
	if sum > float64(total) {
		k := float64(total) / sum
		for i := 1; i < nCat; i++ {
			out[i] = int(est[i] * k)
		}
		return out
	}
	acc := 0
	for i := 1; i < nCat; i++ {
		out[i] = int(est[i])
		acc += out[i]
	}
	out[catSys] = total - acc
	return out
}

// ---------- shared state + pollers ----------

type state struct {
	mu       sync.Mutex
	cfg      config
	base     string // the configured server URL
	srv      serverState
	sys      sysState
	sysTried bool
	sessions []*session
	known    map[string]*session
	sav      savingsState
	lights   *rgbLights // OpenRGB, nil unless enabled; set before the TUI starts
	cfgErr   string     // what was wrong with the config file, shown in the footer
}

func (st *state) pollSessions(glob string) {
	home, _ := os.UserHomeDir()
	for {
		st.scanSessions(glob, home, time.Now())
		time.Sleep(1500 * time.Millisecond)
	}
}

// scanSessions tails the newest 6 session logs modified in the last 6 h.
func (st *state) scanSessions(glob, home string, now time.Time) {
	files, _ := filepath.Glob(glob)
	type cand struct {
		path, project string
		mod           time.Time
		size          int64
	}
	var cs []cand
	for _, f := range files {
		fi, err := os.Stat(f)
		if err != nil || now.Sub(fi.ModTime()) > 6*time.Hour || fi.Size() == 0 {
			continue
		}
		cs = append(cs, cand{f, projectName(filepath.Base(filepath.Dir(filepath.Dir(f))), home), fi.ModTime(), fi.Size()})
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].mod.After(cs[j].mod) })
	recent := map[string]bool{} // sessions of the last 6 h: the others are forgotten, not kept for weeks
	for _, c := range cs {
		recent[c.path] = true
	}
	if len(cs) > 6 {
		cs = cs[:6]
	}
	var list []*session
	for _, c := range cs {
		st.mu.Lock()
		s := st.known[c.path]
		st.mu.Unlock()
		if s == nil {
			s = newSession(c.path, c.project)
		}
		if !c.mod.Equal(s.Mod) || c.size != s.off { // size too: two appends can share one mtime tick
			ns := *s // parse outside the lock on a copy, then swap in
			ns.compactBefore = copyMapB(s.compactBefore)
			ns.pending = copyMapI(s.pending)
			ns.events = append([]event(nil), s.events...)
			ns.prompts = append([]int(nil), s.prompts...)
			ns.rest = append([]byte(nil), s.rest...)
			ns.tail()
			s = &ns
		}
		list = append(list, s)
	}
	st.mu.Lock()
	for _, s := range list {
		st.known[s.Path] = s
	}
	for p := range st.known {
		if !recent[p] {
			delete(st.known, p)
		}
	}
	st.sessions = mineOnly(list, st.srv.Props.Alias)
	st.mu.Unlock()
}

// mineOnly keeps the agent sessions that talk to this monitor's server: a session whose model id is known and is not
// one of the server's own ids (its alias, alias-nothink, the Coder's -low / -xhigh) belongs to the other server of a
// brain / hands pair, which has its own monitor. Unknown model or alias: kept.
func mineOnly(list []*session, alias string) []*session {
	if alias == "" {
		return list
	}
	out := list[:0:0]
	for _, s := range list {
		if s.model == "" || s.model == alias || strings.HasPrefix(s.model, alias+"-") {
			out = append(out, s)
		}
	}
	return out
}

// projectName turns Qwen Code's project dir (the cwd, lowercased, non-alphanumerics as "-") into a short name by
// dropping the home dir prefix. Used until a log line with "cwd" names the project properly.
func projectName(dir, home string) string {
	enc := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			return r
		}
		return '-'
	}, strings.ToLower(home))
	p := strings.ToLower(dir)
	if enc != "" && (p == enc || strings.HasPrefix(p, enc+"-")) {
		p = strings.TrimPrefix(strings.TrimPrefix(p[len(enc):], "-"), "src-")
	}
	if p == "" {
		return "home"
	}
	return p
}

func copyMapB(m map[int]bool) map[int]bool {
	o := make(map[int]bool, len(m))
	for k, v := range m {
		o[k] = v
	}
	return o
}

func copyMapI(m map[string]int) map[string]int {
	o := make(map[string]int, len(m))
	for k, v := range m {
		o[k] = v
	}
	return o
}

// ---------- drawing helpers ----------

func fit(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = ansi.Truncate(s, w, "…")
	if pad := w - lipgloss.Width(s); pad > 0 {
		s += strings.Repeat(" ", pad)
	}
	return s
}

func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	return s
}

func panel(title string, accent color.Color, w int, body []string) string {
	b := fg(cDim)
	inner := w - 4
	t := " " + ansi.Truncate(lipgloss.NewStyle().Foreground(accent).Bold(true).Render(title), max(0, w-5), "…") + " "
	top := b.Render("╭─") + t + b.Render(strings.Repeat("─", max(0, w-3-lipgloss.Width(t)))+"╮")
	lines := []string{top}
	for _, l := range body {
		lines = append(lines, b.Render("│ ")+fit(l, inner)+b.Render(" │"))
	}
	lines = append(lines, b.Render("╰"+strings.Repeat("─", max(0, w-2))+"╯"))
	return strings.Join(lines, "\n")
}

// box is a panel before layout: render() takes rows off the boxes until the frame fits the terminal's height.
type box struct {
	title string
	c     color.Color
	w     int
	rows  []string
	keep  int // rows at the end that go last (AGENTS' "now" line)
	min   int // rows that stay while other boxes can still give some up
}

func (b *box) height() int {
	if len(b.rows) == 0 {
		return 0
	}
	return len(b.rows) + 2
}

// cut takes up to k lines off the box and leaves at least floor rows: rows before the kept tail go first, a
// section (rows between blank lines) whole rather than half, then the tail; a box left without rows goes away
// with its border. It returns the lines taken, which can be more than k.
func (b *box) cut(k, floor int) int {
	h := b.height()
	if k <= 0 || h == 0 || len(b.rows) <= floor {
		return 0
	}
	if floor == 0 && k >= h-2 { // nothing worth a border would be left
		b.rows = nil
		return h
	}
	nk := min(b.keep, len(b.rows))
	head, tail := b.rows[:len(b.rows)-nk], b.rows[len(b.rows)-nk:]
	d := min(k, len(head), len(b.rows)-floor)
	if d > 0 && d < len(head) && head[len(head)-d] != "" { // the cut splits a section: take all of it
		for i := len(head) - d - 1; i >= floor; i-- {
			if head[i] == "" {
				d = len(head) - i
				break
			}
		}
	}
	head = head[:len(head)-d]
	tail = tail[min(len(tail), max(0, k-d), max(0, len(head)+len(tail)-floor)):]
	if len(tail) > 0 && tail[0] == "" && (len(head) == 0 || head[len(head)-1] == "") {
		tail = tail[1:] // no blank line at the top or twice where rows were taken out
	}
	b.rows = append(append([]string(nil), head...), tail...)
	for len(b.rows) > 0 && b.rows[len(b.rows)-1] == "" {
		b.rows = b.rows[:len(b.rows)-1]
	}
	return h - b.height()
}

func (b *box) render() string {
	if len(b.rows) == 0 {
		return ""
	}
	return panel(b.title, b.c, b.w, b.rows)
}

// shrink cuts the boxes, in order, until they stack into room lines: first down to each box's min, then further.
func shrink(room int, bs ...*box) {
	over := -room
	for _, b := range bs {
		over += b.height()
	}
	for pass := 0; pass < 2; pass++ {
		for _, b := range bs {
			if over <= 0 {
				return
			}
			over -= b.cut(over, b.min*(1-pass))
		}
	}
}

func joinBoxes(bs []*box) string {
	var out []string
	for _, b := range bs {
		if r := b.render(); r != "" {
			out = append(out, r)
		}
	}
	return strings.Join(out, "\n")
}

type seg struct {
	n  int
	c  color.Color
	ch string
}

// gauge draws segments proportionally into w cells; the rest is free space.
func gauge(segs []seg, total, w int) string {
	if total <= 0 || w <= 0 {
		return strings.Repeat(" ", max(w, 0))
	}
	var sb strings.Builder
	cum, cells := 0, 0
	for _, s := range segs {
		cum += s.n
		end := int(math.Round(float64(cum) / float64(total) * float64(w)))
		if end > w {
			end = w
		}
		if n := end - cells; n > 0 {
			sb.WriteString(fg(s.c).Render(strings.Repeat(s.ch, n)))
			cells = end
		}
	}
	if cells < w {
		sb.WriteString(fg(cDim).Render(strings.Repeat("·", w-cells)))
	}
	return sb.String()
}

var blocks = []rune(" ▁▂▃▄▅▆▇█")

// specRows are THROUGHPUT's speculative decoding rows: the drafter, acceptance over the last polls that drafted
// (sparkline per poll, bright) and since server start ("all"), accepted tokens per draft and, if there is room,
// the share of drafts accepted at each position. None when the server does not draft.
func specRows(srv *serverState, ri int) []string {
	mt := srv.Metrics
	tok, ok := mt["spec_decode_num_draft_tokens_total"]
	if !ok {
		return nil
	}
	name, known := "", false
	if srv.HaveSlots {
		name, known = drafters(srv.Slot.Params.SpecTypes)
	}
	if known && name == "" || !known && tok <= 0 {
		return nil
	}
	if name == "" {
		name = "draft"
	}
	if len(name) > 13 {
		name = name[:12] + "…"
	}
	pct := func(ok bool, x float64) string {
		if !ok {
			return "  —  "
		}
		return fmt.Sprintf("%4.1f%%", x*100)
	}
	acc, all := mt["spec_decode_num_accepted_tokens_total"], 0.0
	if tok > 0 {
		all = acc / tok
	}
	rate, per, rok := srv.Spec.recent()
	sw := max(4, ri-32)
	a := fmt.Sprintf("%-14s%s %s%s", name, sparkTo(srv.Spec.rates(), sw, 1, cPurple), fg(cPurple).Render(pct(rok, rate)),
		fg(cMuted).Render(" · "+pct(tok > 0, all)+" all"))
	perAll, pa := "—", "  — "
	if d := mt["spec_decode_num_drafts_total"]; d > 0 {
		perAll = fmt.Sprintf("%.1f", acc/d)
	}
	if rok && per > 0 {
		pa = fmt.Sprintf("%4.1f", per)
	}
	b := fmt.Sprintf("%-14s%s%s", "tok/draft", fg(cPurple).Render(pa), fg(cMuted).Render(" · "+perAll+" all"))
	if pos := specPos(mt); len(pos) > 0 {
		if room := ri - lipgloss.Width(b) - 9; room >= 4 {
			b += fg(cMuted).Render("  by pos ") + sparkTo(pos[:min(len(pos), room)], min(len(pos), room), 1, cPurple)
		}
	}
	return []string{a, b}
}

func sparkline(v []float64, w int, c color.Color) string { return sparkTo(v, w, 0, c) }

// sparkTo draws against a fixed top value mx (0: the largest value shown).
func sparkTo(v []float64, w int, mx float64, c color.Color) string {
	w = max(w, 0)
	if len(v) > w {
		v = v[len(v)-w:]
	}
	if mx <= 0 {
		for _, x := range v {
			mx = math.Max(mx, x)
		}
	}
	var sb strings.Builder
	sb.WriteString(strings.Repeat(" ", max(0, w-len(v))))
	for _, x := range v {
		i := 0
		if mx > 0 {
			i = int(math.Round(x / mx * 8))
		}
		if i == 0 && x > 0 {
			i = 1
		}
		sb.WriteRune(blocks[i])
	}
	return fg(c).Render(sb.String())
}

// columns draws a bar chart h rows tall; marked columns are drawn in the mark color.
func columns(v []float64, marks []bool, w, h int, mx float64, c, mc color.Color) []string {
	if w <= 0 || h <= 0 {
		return make([]string, max(h, 0))
	}
	if len(v) > w {
		marks = marks[len(v)-w:]
		v = v[len(v)-w:]
	}
	rows := make([]strings.Builder, h)
	off := w - len(v)
	for r := 0; r < h; r++ {
		rows[r].WriteString(strings.Repeat(" ", off))
	}
	for i, x := range v {
		eighths := 0
		if mx > 0 {
			eighths = int(math.Round(x / mx * float64(h*8)))
		}
		col := c
		if marks[i] {
			col = mc
		}
		for r := 0; r < h; r++ {
			level := (h - 1 - r) * 8
			n := eighths - level
			ch := ' '
			if n >= 8 {
				ch = '█'
			} else if n > 0 {
				ch = blocks[n]
			}
			rows[r].WriteString(fg(col).Render(string(ch)))
		}
	}
	out := make([]string, h)
	for r := range rows {
		out[r] = rows[r].String()
	}
	return out
}

func kTok(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 10_000:
		return fmt.Sprintf("%.0fk", float64(n)/1e3)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1e3)
	}
	return strconv.Itoa(n)
}

func ago(t time.Time) string {
	if t.IsZero() {
		return "—"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
}

func dur(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}

var toolIcons = map[string]string{
	"run_shell_command": "❯", "read_file": "◧", "read_many_files": "◧", "edit": "✎", "replace": "✎",
	"write_file": "✚", "grep_search": "⌕", "search_file_content": "⌕", "glob": "⌕", "list_directory": "▤",
	"web_fetch": "⇣", "google_web_search": "⇣", "ask_user_question": "?", "task": "⇶", "todo_write": "☑",
	"save_memory": "✦", "skill": "★",
}

func iconFor(e event) (string, color.Color) {
	switch e.Kind {
	case "user":
		return "▸", cGreen
	case "reply":
		return "◂", cTeal
	case "compact":
		return "✂", cRed
	}
	if strings.HasPrefix(e.Name, "mcp__") {
		return "◆", cPurple
	}
	if i, ok := toolIcons[e.Name]; ok {
		return i, cYellow
	}
	return "•", cYellow
}

func shortTool(n string) string {
	if strings.HasPrefix(n, "mcp__") {
		p := strings.Split(n, "__")
		if len(p) >= 3 {
			return p[1] + ":" + p[2]
		}
	}
	return strings.TrimSuffix(strings.TrimPrefix(n, "run_"), "_command")
}

// ---------- model ----------

type tickMsg time.Time

type model struct {
	st    *state
	w, h  int
	frame int
	sel   *string // Path of the session picked with tab; nil = the most recently active one
}

// listenBases turns the caminus process's listening sockets into base URLs (0.0.0.0 / :: = this machine).
func (st *state) listenBases() []string {
	st.mu.Lock()
	defer st.mu.Unlock()
	ss := st.sys
	if ss.Pid == 0 {
		return nil
	}
	var out []string
	for _, a := range ss.Listen {
		h, p, err := net.SplitHostPort(a)
		if err != nil {
			continue
		}
		if ip := net.ParseIP(h); ip == nil || ip.IsUnspecified() {
			h = "127.0.0.1"
		}
		out = append(out, "http://"+net.JoinHostPort(h, p))
	}
	return out
}

// agentFresh: without a tab pick, CONTEXT and ACTIVITY show the newest session only if it was active this
// recently; an older one is history, not what the server is doing (AGENTS still lists it, tab shows it).
const agentFresh = 10 * time.Minute

// shown is the index of the session CONTEXT and ACTIVITY show, -1 for none.
func (st *state) shown(sel *string) int {
	switch {
	case len(st.sessions) == 0:
		return -1
	case sel != nil && *sel != "":
		return st.selIndex(*sel)
	case time.Since(st.sessions[0].Mod) < agentFresh:
		return 0
	case st.srv.busy() && !st.sessions[0].Mod.Before(st.srv.busySince.Add(-2*time.Minute)):
		return 0 // a long request writes nothing to the log; but a session older than the request isn't its client
	}
	return -1
}

// modelRows describe the loaded model for CONTEXT when no agent session fills it.
func modelRows(srv *serverState, nctx int) []string {
	lab := func(s string) string { return fg(cMuted).Render(fmt.Sprintf("%-8s", s)) }
	var rows []string
	if srv.Props.Path == "" {
		return nil
	}
	rows = append(rows, lab("model")+fg(cText).Bold(true).Render(baseName(srv.Props.Path)))
	c := fmt.Sprintf("%s tokens", kTok(nctx))
	if srv.NSlots > 0 {
		c += fmt.Sprintf(" · %d slot%s", srv.NSlots, map[bool]string{true: "s"}[srv.NSlots > 1])
	}
	rows = append(rows, lab("context")+c)
	if srv.Props.Build != "" {
		rows = append(rows, lab("build")+fg(cMuted).Render(srv.Props.Build))
	}
	return rows
}

// reqRows are the finished requests for ACTIVITY when no agent session is shown, newest last.
func reqRows(done []reqEvent, n, w int) []string {
	if len(done) > n {
		done = done[len(done)-n:]
	}
	var rows []string
	for _, e := range done {
		in := kTok(e.Prompt-e.Cached) + " new"
		if e.Cached > 0 {
			in += " + " + kTok(e.Cached) + " cached"
		}
		parts := []string{fg(cGreen).Render("→ " + kTok(e.Out) + " written")}
		if e.ReadRate > 0 {
			parts = append(parts, fmt.Sprintf("read %.0f tok/s", e.ReadRate))
		}
		if e.GenRate > 0 {
			parts = append(parts, fmt.Sprintf("write %.1f tok/s", e.GenRate))
		}
		if e.Dur > 0 {
			parts = append(parts, dur(e.Dur))
		}
		if e.Ctx > 0 {
			parts = append(parts, fg(cMuted).Render("ctx "+kTok(e.Ctx)))
		}
		line := fmt.Sprintf("%s %s request %s  %s", fg(cMuted).Render(e.T.Local().Format("15:04:05")), fg(cAccent).Render("◆"),
			in, strings.Join(parts, fg(cMuted).Render(" · ")))
		rows = append(rows, ansi.Truncate(line, w, "…"))
	}
	return rows
}

// pick moves the selection d sessions on. The list is re-sorted by activity, so the choice is kept by path.
func (m model) pick(d int) {
	st := m.st
	st.mu.Lock()
	defer st.mu.Unlock()
	if n := len(st.sessions); n > 0 && m.sel != nil {
		cur := st.shown(m.sel)
		if cur < 0 && d > 0 { // nothing shown: tab starts at the newest
			cur = -1
		}
		*m.sel = st.sessions[((cur+d)%n+n)%n].Path
	}
}

// selIndex is the index of the session at path, 0 when it is gone.
func (st *state) selIndex(path string) int {
	for i, s := range st.sessions {
		if s.Path == path {
			return i
		}
	}
	return 0
}

func (m model) Init() tea.Cmd { return tick() }

func tick() tea.Cmd {
	return tea.Tick(150*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.w, m.h = msg.Width, msg.Height
	case tea.KeyPressMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case "tab":
			m.pick(1)
		case "shift+tab":
			m.pick(-1)
		}
	case tickMsg:
		m.frame++
		return m, tick()
	}
	return m, nil
}

// minW is the narrowest layout; smaller terminals see its left part.
const minW = 40

// spin uses quarter-circle glyphs: MesloLGS NF (the recommended font) has no braille, Cascadia Mono has both
var spin = []string{"◴", "◷", "◶", "◵"}

func (m model) View() tea.View {
	v := tea.NewView(m.render())
	v.AltScreen = true
	v.WindowTitle = "caminus top"
	return v
}

func (m model) render() string {
	st := m.st
	st.mu.Lock()
	defer st.mu.Unlock()
	W := m.w
	if W <= 0 {
		W = 120
	}
	// below minW the frame is laid out at minW and cut to the terminal at the end
	realW := W
	W = min(max(W, minW), 200)
	srv, sys := &st.srv, st.sys
	sl := srv.Slot
	nctx, ctxMark := sl.NCtx, ""
	if nctx == 0 {
		nctx = srv.Props.Gen.NCtx
	}
	if nctx == 0 {
		nctx = st.cfg.Context
	}
	if nctx == 0 {
		nctx, ctxMark = 131072, "?" // unknown: no /slots, no /props, no "context" in the config
	}

	// selected session: tab cycles, default = most recently active
	var sess *session
	if i := st.shown(m.sel); i >= 0 {
		sess = st.sessions[i]
	}

	// ----- header -----
	var pill string
	pillStyle := func(bg color.Color) lipgloss.Style {
		return lipgloss.NewStyle().Background(bg).Foreground(lipgloss.Color("#1b1f27")).Bold(true).Padding(0, 1)
	}
	sp := spin[m.frame%len(spin)]
	off := srv.offline() // why nothing is live, "" while a model is up
	if off == "no server running" && sys.Pid > 0 {
		off = procName + " is running but not answering" // starting up, stuck, or on another port
	}
	status := ""
	switch {
	case sys.Hold != "":
		pill = pillStyle(cPurple).Render("HOLD")
		status = fg(cPurple).Render("watchdog hold: " + sys.Hold)
		if off != "" {
			status += fg(cMuted).Render(" · " + off)
		}
	case !srv.Polled:
		pill = pillStyle(cMuted).Render(sp + " CONNECTING")
		status = fg(cMuted).Render("looking for a server at " + shortHost(st.base))
	case srv.Loading:
		pill = pillStyle(cYellow).Render(sp + " LOADING")
		status = fg(cYellow).Render("server is up and loading the model")
	case !srv.Up:
		pill = pillStyle(cRed).Render("DOWN")
		status = fg(cRed).Render(fmt.Sprintf("no answer from %s (stopped, or still starting)", shortHost(st.base)))
	case !srv.HaveSlots && srv.busy():
		pill = pillStyle(cGreen).Render(sp + " BUSY")
		status = fg(cMuted).Render(fmt.Sprintf("%.0f request(s) running · no /slots answer, so no live progress", srv.Metrics["requests_processing"]))
	case !srv.HaveSlots, !sl.Processing:
		pill = pillStyle(cMuted).Render("IDLE")
		status = fg(cMuted).Render("waiting for a request")
	case srv.Decoded == 0:
		pill = pillStyle(cYellow).Render(sp + " READING")
		done := sl.NCache + sl.NProcessed
		pct := 0.0
		if sl.NPrompt > 0 {
			pct = float64(done) / float64(sl.NPrompt)
		}
		eta := ""
		if srv.curRead > 1 {
			eta = " · eta " + dur(time.Duration(float64(max(0, sl.NPrompt-done))/srv.curRead*float64(time.Second)))
		}
		bw := 24
		status = fmt.Sprintf("%s %s %3.0f%%  %s new + %s reused  %s tok/s%s",
			gauge([]seg{{int(pct * 1000), cYellow, "█"}}, 1000, bw), "", pct*100,
			kTok(sl.NPrompt-sl.NCache), kTok(sl.NCache), fg(cYellow).Render(fmt.Sprintf("%.0f", srv.curRead)), eta)
	default:
		pill = pillStyle(cGreen).Render(sp + " WRITING")
		status = fmt.Sprintf("%s tokens  %s tok/s  %s",
			fg(cGreen).Render(kTok(srv.Decoded)), fg(cGreen).Render(fmt.Sprintf("%.1f", srv.curGen)),
			fg(cMuted).Render("for "+dur(time.Since(srv.busySince))))
	}
	if srv.Cam != nil && srv.busy() && off == "" && sys.Hold == "" {
		if l := camLoop(srv.Cam); l != "" { // the reasoning repeats itself: worth seeing before the request ends
			status += "  " + l
		}
	}
	if srv.compacting() && srv.Up && sys.Hold == "" {
		pill = pillStyle(cOrange).Render(sp + " ✂ COMPACTING")
		what := "✂ reading the history to summarize it · "
		if srv.phase() == "writing" {
			what = "✂ writing the summary · "
		}
		status = fg(cOrange).Render(what) + status
	}
	alias, mdl := srv.Props.Alias, "?"
	if alias == "" {
		alias = "caminus"
	}
	if m := strings.TrimSuffix(baseName(srv.Props.Path), ".gguf"); m != "" && m != alias {
		mdl = m
	}
	host := shortHost(srv.Base)
	if host == "" {
		host = shortHost(st.base)
	}
	slots := ""
	if srv.NSlots > 1 {
		slots = fmt.Sprintf(" · slot id %d · %d slots, %d busy", sl.ID, srv.NSlots, srv.Busy)
	}
	title := lipgloss.NewStyle().Foreground(cAccent).Bold(true).Render("◆ caminus top") + fg(cMuted).Render(fmt.Sprintf(" %s  %s · %s · %s%s", version, alias, mdl, host, slots))
	clock := fg(cMuted).Render(time.Now().Format("15:04:05"))
	room := W - lipgloss.Width(pill) - lipgloss.Width(clock) - 3
	title = ansi.Truncate(title, max(0, room), "…")
	header := title + strings.Repeat(" ", max(1, room+1-lipgloss.Width(title))) + pill + " " + clock
	statusLine := " " + fit(status, W-2)

	// ----- column widths -----
	twoCol := W >= 110
	lw, rw := W, W
	if twoCol {
		lw = W * 56 / 100
		rw = W - lw - 1
	}

	// ----- context panel -----
	inner := lw - 4
	var ctx []string
	used := sl.NPrompt + srv.Decoded
	segs := []seg{
		{sl.NCache, cBlue, "█"},
		{min(sl.NProcessed, max(0, sl.NPrompt-sl.NCache)), cYellow, "█"},
		{max(0, sl.NPrompt-sl.NCache-sl.NProcessed), cYellow, "░"},
		{srv.Decoded, cGreen, "█"},
	}
	if !sl.Processing {
		segs = []seg{{sl.NPrompt, cBlue, "█"}, {srv.Decoded, cGreen, "█"}}
	}
	if off != "" {
		ctx = append(ctx, gauge(nil, 1, inner))
		ctx = append(ctx, fg(cMuted).Render("no live context: "+off))
	} else {
		ctx = append(ctx, gauge(segs, nctx, inner))
		ctx = append(ctx, fmt.Sprintf("%s %s  %s %s  %s %s  %s",
			fg(cBlue).Render("█"), "reused", fg(cYellow).Render("█"), "read", fg(cGreen).Render("█"), "written",
			fg(cText).Bold(true).Render(fmt.Sprintf("%s / %s  (%.0f%%)", kTok(used), kTok(nctx)+ctxMark, 100*float64(used)/float64(nctx)))))
		if r := camContextRow(srv.Cam); r != "" {
			ctx = append(ctx, r)
		}
	}
	ctx = append(ctx, "")

	// waffle: what fills the context (estimated from the selected session)
	waffleEnd := -1 // rows up to the waffle's legend; the prompt-size chart after them goes first next to CAMINUS
	if sess != nil {
		total := sess.lastUsage.Prompt
		est := sess.estimate(total)
		cols := inner / 2
		rows := 8
		cells := cols * rows
		per := float64(nctx) / float64(cells)
		head := fmt.Sprintf("what fills it · %s · 1 cell ≈ %s tok · est.", sess.Project, kTok(int(per)))
		if off != "" || time.Since(sess.lastT) > 2*time.Minute { // the log's last request, not a live view
			head = fmt.Sprintf("what filled it · %s · %s ago · est.", sess.Project, ago(sess.lastT))
		}
		ctx = append(ctx, fg(cMuted).Render(head))
		var flat []color.Color
		for c := 0; c < nCat; c++ {
			n := int(math.Round(float64(est[c]) / per))
			for k := 0; k < n && len(flat) < cells; k++ {
				flat = append(flat, catColors[c])
			}
		}
		for r := 0; r < rows; r++ {
			var sb strings.Builder
			for c := 0; c < cols; c++ {
				i := r*cols + c
				if i < len(flat) {
					sb.WriteString(fg(flat[i]).Render("■ "))
				} else {
					sb.WriteString(fg(cDim).Render("· "))
				}
			}
			ctx = append(ctx, sb.String())
		}
		// legend, two per line
		var leg []string
		for c := 0; c < nCat; c++ {
			if est[c] <= 0 {
				continue
			}
			pct := 0.0
			if total > 0 {
				pct = 100 * float64(est[c]) / float64(total)
			}
			leg = append(leg, fmt.Sprintf("%s %-20s %6s %3.0f%%", fg(catColors[c]).Render("■"), catNames[c], kTok(est[c]), pct))
		}
		half := inner / 2
		for i := 0; i < len(leg); i += 2 {
			l := fit(leg[i], half)
			if i+1 < len(leg) {
				l += fit(leg[i+1], inner-half)
			}
			ctx = append(ctx, l)
		}
		waffleEnd = len(ctx)
		ctx = append(ctx, "")

		// prompt size per request, compactions in red
		ctx = append(ctx, fg(cMuted).Render(fmt.Sprintf("prompt size per request · last %d · %s compactions", min(len(sess.prompts), inner-6), fg(cRed).Render(strconv.Itoa(sess.compactions)))))
		vals := make([]float64, len(sess.prompts))
		marks := make([]bool, len(sess.prompts))
		for i, p := range sess.prompts {
			vals[i] = float64(p)
			marks[i] = sess.compactBefore[i]
		}
		bars := columns(vals, marks, inner-6, 5, float64(nctx), cAccent, cRed)
		for i, b := range bars {
			lab := "      "
			if i == 0 {
				lab = " " + fit(kTok(nctx), 5)
			} else if i == len(bars)-1 {
				lab = " 0    "
			}
			ctx = append(ctx, b+fg(cMuted).Render(lab))
		}
	} else {
		if off == "" {
			if rows := modelRows(srv, nctx); len(rows) > 0 {
				ctx = append(ctx, rows...)
				ctx = append(ctx, "")
			}
		}
		msg := "no Qwen Code session active in the last 6 h"
		if len(st.sessions) > 0 {
			msg = fmt.Sprintf("no agent active in the last %d min · tab shows earlier sessions", int(agentFresh.Minutes()))
		}
		ctx = append(ctx, fg(cMuted).Render(msg))
	}
	ctxBox := &box{title: "CONTEXT", c: cAccent, w: lw, rows: ctx, min: 2} // the gauge and its legend

	// ----- throughput panel -----
	ri := rw - 4
	sw := max(10, ri-20)
	var thr []string
	rv, gv := fg(cYellow).Render(fmt.Sprintf("%6.0f tok/s", srv.curRead)), fg(cGreen).Render(fmt.Sprintf("%6.1f tok/s", srv.curGen))
	if off != "" {
		rv, gv = fg(cMuted).Render("    — tok/s"), fg(cMuted).Render("    — tok/s")
	}
	thr = append(thr, fmt.Sprintf("%-6s%s %s", "read", sparkline(srv.ReadRate.v, sw, cYellow), rv))
	thr = append(thr, fmt.Sprintf("%-6s%s %s", "write", sparkline(srv.GenRate.v, sw, cGreen), gv))
	mt := srv.Metrics
	if mt != nil {
		cached, read := mt["prompt_tokens_cached_total"], mt["prompt_tokens_total"]
		reuse := 0.0
		if cached+read > 0 {
			reuse = cached / (cached + read)
		}
		thr = append(thr, fmt.Sprintf("%-14s%s %s", "cache reuse", gauge([]seg{{int(reuse * 1000), cBlue, "█"}}, 1000, max(6, ri-21)), fg(cBlue).Render(fmt.Sprintf("%4.1f%%", reuse*100))))
		thr = append(thr, specRows(srv, ri)...)
		thr = append(thr, fg(cMuted).Render(fmt.Sprintf("since start: %s read · %s reused · %s written",
			kTok(int(read)), kTok(int(cached)), kTok(int(mt["tokens_predicted_total"])))))
		avgR, avgW := 0.0, 0.0
		if t := mt["prompt_seconds_total"]; t > 0 {
			avgR = read / t
		}
		if t := mt["tokens_predicted_seconds_total"]; t > 0 {
			avgW = mt["tokens_predicted_total"] / t
		}
		thr = append(thr, fg(cMuted).Render(fmt.Sprintf("avg %.0f tok/s read · %.1f tok/s write · peak ctx %s",
			avgR, avgW, kTok(int(mt["n_tokens_max"])))))
	}
	vt := sys.VramTot
	if st.cfg.VramGB > 0 {
		vt = st.cfg.VramGB
	}
	if fb := fallbackRows(srv.Up && srv.busy(), off, sys, vt, runtime.NumCPU()); len(fb) > 0 {
		thr = append(thr, "")
		thr = append(thr, fb...)
	}
	thrBox := &box{title: "THROUGHPUT", c: cYellow, w: rw, rows: thr, min: 2}

	// ----- machine panel -----
	var mach []string
	if sys.OK {
		bw := max(6, ri-22)
		vr, vt := sys.VramGB, sys.VramTot
		if st.cfg.VramGB > 0 {
			vt = st.cfg.VramGB
		}
		var cards []gpuCard // discrete GPUs: one gauge each once there are two
		for _, c := range sys.GPUs {
			if c.Tot >= dgpuMinGB {
				cards = append(cards, c)
			}
		}
		switch {
		case len(cards) > 1:
			var names, on []string
			for i, c := range cards {
				mach = append(mach, fmt.Sprintf("%-6s%s %s", fmt.Sprintf("VRAM%d", i+1), gauge([]seg{{int(c.Used * 100), vramColor(c.Used, c.Tot), "█"}}, int(c.Tot*100), bw), fmt.Sprintf("%5.1f / %.0f GB", c.Used, c.Tot)))
				names = append(names, fmt.Sprintf("%d %s", i+1, cardName(c.Name)))
				if c.Model {
					on = append(on, strconv.Itoa(i+1))
				}
			}
			if len(on) > 0 {
				names = append(names, "model on "+strings.Join(on, "+"))
			}
			mach = append(mach, fg(cMuted).Render(strings.Join(names, " · ")))
		case vt > 0:
			mach = append(mach, fmt.Sprintf("%-6s%s %s", "VRAM", gauge([]seg{{int(vr * 100), vramColor(vr, vt), "█"}}, int(vt*100), bw), fmt.Sprintf("%5.1f / %.0f GB", vr, vt)))
		case vr > 0:
			mach = append(mach, fmt.Sprintf("%-6s%s", "VRAM", fmt.Sprintf("%.1f GB used (total unknown: vram_gb in the config)", vr)))
		default:
			mach = append(mach, fmt.Sprintf("%-6s%s", "VRAM", fg(cMuted).Render("no GPU memory counters")))
		}
		ru := sys.RamTot - sys.RamFree
		mach = append(mach, fmt.Sprintf("%-6s%s %s", "RAM", gauge([]seg{{int(ru * 100), ramColor(sys.RamFree), "█"}}, int(sys.RamTot*100), bw), fmt.Sprintf("%5.1f / %.0f GB", ru, sys.RamTot)))
		mach = append(mach, fg(cMuted).Render(fmt.Sprintf("%s memory %.1f GB · %.1f GB RAM free", procName, sys.SrvWS, sys.RamFree)))
		if off == "" {
			mach = append(mach, camMachineRows(srv.Cam)...)
		}
	} else if st.sysTried {
		mach = append(mach, fg(cMuted).Render("no machine counters on this system"))
	} else {
		mach = append(mach, fg(cMuted).Render("reading counters…"))
	}
	machBox := &box{title: "MACHINE", c: cOrange, w: rw, rows: mach}

	// ----- agents panel -----
	var ag []string
	for i, s := range st.sessions {
		mark := "  "
		if s == sess {
			mark = fg(cAccent).Render("▶ ")
		}
		live := fg(cDim).Render("●")
		if time.Since(s.Mod) < 90*time.Second {
			live = fg(cGreen).Render("●")
			if srv.busy() && i == 0 {
				live = fg(cYellow).Render(sp)
			}
		}
		ag = append(ag, fmt.Sprintf("%s%s %s %s", mark, live, fit(s.Project, 18),
			fg(cMuted).Render(fmt.Sprintf("%s %4s · %dt · %d calls · ✂%d", fit(s.model, 12), ago(s.Mod), s.turns, s.tools, s.compactions))))
	}
	if sess != nil {
		ag = append(ag, "")
		now := agentNow(sess, srv.busy(), srv.Decoded, sp)
		if srv.compacting() && len(st.sessions) > 0 && sess == st.sessions[0] {
			now = fg(cOrange).Render(sp + " ✂ compacting: the model summarizes the history")
		}
		ag = append(ag, "now  "+now)
	} else if len(st.sessions) > 0 {
		ag = append(ag, "", fg(cMuted).Render("now  no agent active · tab shows earlier ones"))
	}
	if len(ag) == 0 {
		ag = append(ag, fg(cMuted).Render("no agent sessions in the last 6 h"))
	}
	if st.lights != nil {
		ag = append(ag, fg(cMuted).Render("rgb  "+st.lights.Status()))
	}
	agBox := &box{title: "AGENTS  (tab = switch)", c: cGreen, w: rw, rows: ag, keep: len(ag) - len(st.sessions)} // the list goes first
	agBox.min = agBox.keep

	// ----- savings panel -----
	var sv []string
	if st.sav.Ready {
		rR, rW := 0.0, 0.0
		if mt != nil {
			if t := mt["prompt_seconds_total"]; t > 0 {
				rR = mt["prompt_tokens_total"] / t
			}
			if t := mt["tokens_predicted_seconds_total"]; t > 0 {
				rW = mt["tokens_predicted_total"] / t
			}
		}
		p := st.cfg.priceCfg
		all, today := st.sav.All, st.sav.Today
		api, pow, net := p.cost(all, rR, rW)
		_, _, netT := p.cost(today, rR, rW)
		big := lipgloss.NewStyle().Foreground(cGreen).Bold(true)
		since := "—"
		if !st.sav.First.IsZero() {
			since = st.sav.First.Local().Format("Jan 2")
		}
		perDay := 0.0
		if st.sav.Days > 0 {
			perDay = net / float64(st.sav.Days)
		}
		sv = append(sv, fmt.Sprintf("%s %s", big.Render("saved "+usd(net)), fg(cMuted).Render(fmt.Sprintf("since %s · %s/day · today %s", since, usd(perDay), usd(netT)))))
		cachedPct := 0.0
		if all.Prompt > 0 {
			cachedPct = 100 * float64(all.Cached) / float64(all.Prompt)
		}
		sv = append(sv, fmt.Sprintf("%-6s%s  %s  %s %s", "in", fg(cBlue).Render(kTok(int(all.Prompt))),
			fg(cMuted).Render(fmt.Sprintf("(%.0f%% cache hits)", cachedPct)), "out", fg(cGreen).Render(kTok(int(all.Out)))))
		sv = append(sv, fmt.Sprintf("%-6s%s  %s", "API", fg(cYellow).Render(usd(api)), fg(cMuted).Render(fmt.Sprintf("− power ≈ %s  · %d requests", usd(pow), all.Reqs))))
		sv = append(sv, fmt.Sprintf("%-6s%s  %s", "today", fmt.Sprintf("in %s · out %s", kTok(int(today.Prompt)), kTok(int(today.Out))), fg(cMuted).Render(fmt.Sprintf("%d requests", today.Reqs))))
		sv = append(sv, fg(cDim).Render(fmt.Sprintf("%s: $%g in · $%g write · $%g hit · $%g out /M · %gW · $%g/kWh", p.Label, p.InPerM, p.WritePerM, p.CachedPerM, p.OutPerM, p.BusyWatts, p.KwhPrice)))
	} else {
		sv = append(sv, fg(cMuted).Render("adding up session logs…"))
	}
	savBox := &box{title: "SAVINGS  (vs API · llm-top.json)", c: cGreen, w: rw, rows: sv}

	// ----- assemble -----
	// header, status line, ACTIVITY's border and the footer take 5 lines and ACTIVITY at least 3: the panels get
	// the rest, giving up rows in order of least use (the left column on its own in two-column mode)
	H := m.h
	if H <= 0 {
		H = 50
	}
	room = H - 5 - 3
	var body string
	if twoCol {
		cBox := camBox(srv, lw) // Caminus: under CONTEXT; each keeps its first rows longest
		if cBox.height() > 0 {  // the prompt-size chart, CAMINUS down to its speed rows, CONTEXT down to its gauge, ...
			if waffleEnd > 0 {
				ctxBox.cut(ctxBox.height()+cBox.height()-room, waffleEnd)
			}
			shrink(room, cBox, ctxBox)
		}
		ctxBox.cut(ctxBox.height()+cBox.height()-room, 0) // without Caminus, CONTEXT alone as before
		shrink(room, savBox, machBox, agBox, thrBox)
		left, right := joinBoxes([]*box{ctxBox, cBox}), joinBoxes([]*box{thrBox, savBox, machBox, agBox})
		switch {
		case left == "":
			body = right
		case right == "":
			body = left
		default:
			body = lipgloss.JoinHorizontal(lipgloss.Top, left, " ", right)
		}
	} else {
		cBox := camBox(srv, W)
		shrink(room, savBox, machBox, cBox, ctxBox, agBox, thrBox)
		body = joinBoxes([]*box{ctxBox, thrBox, cBox, savBox, machBox, agBox})
	}

	// ----- activity feed fills the rest -----
	bodyH := 0
	if body != "" {
		bodyH = lipgloss.Height(body)
	}
	n := max(3, H-5-bodyH)
	var feed []string
	if sess != nil {
		evs := sess.events
		if len(evs) > n {
			evs = evs[len(evs)-n:]
		}
		for _, e := range evs {
			ic, c := iconFor(e)
			name := e.Name
			if e.Kind == "tool" {
				name = shortTool(name)
			}
			tail := ""
			switch {
			case e.Kind != "tool":
			case !e.Done:
				tail = fg(cYellow).Render(fmt.Sprintf("%s %s", sp, dur(time.Since(e.T))))
			case e.Err:
				tail = fg(cRed).Render(fmt.Sprintf("✗ %s", dur(e.DoneT.Sub(e.T))))
			default:
				tail = fg(cGreen).Render(fmt.Sprintf("✓ %s", dur(e.DoneT.Sub(e.T))))
			}
			fw := W - 4 - 9 - 2 - 16 - 12
			feed = append(feed, fmt.Sprintf("%s %s %s %s %s", fg(cMuted).Render(e.T.Local().Format("15:04:05")),
				fg(c).Render(ic), fg(c).Render(fit(name, 16)), fit(oneLine(e.Detail), fw), tail))
		}
	}
	for len(feed) < n {
		feed = append([]string{""}, feed...)
	}
	feedTitle := "ACTIVITY"
	if sess == nil { // no agent session: what the server itself did since llm-top started
		feedTitle = "ACTIVITY  (server requests since llm-top started)"
		feed = reqRows(srv.Done, n, W-4)
		if len(feed) == 0 {
			feed = []string{fg(cMuted).Render("no requests yet: they show up here as they finish")}
			for len(feed) < n {
				feed = append(feed, "")
			}
		}
		for len(feed) < n {
			feed = append([]string{""}, feed...)
		}
	}
	feedPanel := panel(feedTitle, cTeal, W, feed)

	foot := fg(cDim).Render(ansi.Truncate(" tab switch agent · q quit · composition is estimated from session text since the last compaction", W, "…"))
	if st.cfgErr != "" { // stderr is gone behind the alternate screen
		foot = fg(cRed).Render(ansi.Truncate(" config: "+st.cfgErr, W, "…"))
	}
	parts := []string{header, statusLine, body, feedPanel, foot}
	if body == "" {
		parts = []string{header, statusLine, feedPanel, foot}
	}
	out := strings.Join(parts, "\n")
	if ls := strings.Split(out, "\n"); len(ls) > H { // a terminal too short for even 3 rows of ACTIVITY
		out = strings.Join(ls[:H], "\n")
	}
	if realW < W {
		ls := strings.Split(out, "\n")
		for i, l := range ls {
			ls[i] = ansi.Truncate(l, realW, "")
		}
		out = strings.Join(ls, "\n")
	}
	return out
}

// shortHost drops the scheme and a local host: "http://127.0.0.1:8092" -> ":8092".
func shortHost(u string) string {
	if i := strings.Index(u, "://"); i >= 0 {
		u = u[i+3:]
	}
	for _, lh := range []string{"127.0.0.1:", "localhost:"} {
		u = strings.Replace(u, lh, ":", 1)
	}
	return u
}

// agentPhase is what an agent session is doing; the AGENTS "now" line and the OpenRGB lights both use it.
// e is the running tool for phTool.
func agentPhase(s *session, processing bool, decoded int) (ph string, e *event) {
	for i := len(s.events) - 1; i >= 0 && i >= len(s.events)-6; i-- {
		if e := &s.events[i]; e.Kind == "tool" && !e.Done {
			return phTool, e
		}
	}
	switch {
	case time.Since(s.Mod) > 10*time.Minute && !processing:
		return phQuiet, nil
	case s.lastKind == "reply" && !processing:
		return phWaiting, nil
	case processing && decoded == 0:
		return phReading, nil
	case processing:
		return phWriting, nil
	}
	return phBetween, nil
}

func agentNow(s *session, processing bool, decoded int, sp string) string {
	ph, e := agentPhase(s, processing, decoded)
	switch ph {
	case phTool:
		return fg(cYellow).Render(fmt.Sprintf("%s running %s: %s (%s)", sp, shortTool(e.Name), oneLine(e.Detail), dur(time.Since(e.T))))
	case phQuiet:
		return fg(cMuted).Render("quiet for " + ago(s.Mod))
	case phWaiting:
		return fg(cTeal).Render("answered, waiting for you")
	case phReading:
		return fg(cYellow).Render(sp + " model reading the context")
	case phWriting:
		return fg(cGreen).Render(sp + " model thinking / writing")
	}
	return fg(cMuted).Render("between steps")
}

// cardName drops the vendor and brand from an amdgpu product name: "AMD Radeon RX 7900 XTX" is "RX 7900 XTX".
func cardName(n string) string {
	for _, p := range []string{"AMD ", "Radeon "} {
		n = strings.TrimPrefix(n, p)
	}
	if n == "" {
		return "GPU"
	}
	return n
}

func vramColor(gb, tot float64) color.Color {
	switch {
	case gb > tot-0.6:
		return cRed
	case gb > tot-2:
		return cOrange
	}
	return cGreen
}

func ramColor(free float64) color.Color {
	switch {
	case free < 2:
		return cRed
	case free < 5:
		return cOrange
	}
	return cGreen
}

const usage = `usage: llm-top [-config <path>] [-once [width height] | -web [host:port]] | -version
  -config <path>  config file (default: $LLMTOP_CONFIG, llm-top.json next to the exe, or the user config dir)
  -once [w h]     print one frame after ~7 s of polling and exit (no TTY needed)
  -web [addr]     no TUI: serve the dashboard to browsers on addr (default: web.listen, :8765)
  -version        print the version`

// topMain is llm-top's own main (the monitor, baked into Caminus 11-07): `caminus-panel top [llm-top's flags]`.
func topMain(args []string) {
	cfgPath, once, w, h := "", false, 140, 60
	web, webAddr := false, ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-version", "--version":
			fmt.Println("caminus top", version)
			return
		case "-config", "--config":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, usage)
				os.Exit(2)
			}
			i++
			cfgPath = args[i]
		case "-once", "--once":
			once = true
			if i+2 < len(args) {
				if a, err := strconv.Atoi(args[i+1]); err == nil {
					if b, err := strconv.Atoi(args[i+2]); err == nil {
						w, h, i = a, b, i+2
					}
				}
			}
		case "-web", "--web":
			web = true
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				i++
				webAddr = args[i]
			}
		case "-h", "-help", "--help":
			fmt.Println(usage)
			return
		default:
			fmt.Fprintf(os.Stderr, "llm-top: unknown argument %q\n%s\n", args[i], usage)
			os.Exit(2)
		}
	}
	cfg, err := loadConfig(configPath(cfgPath))
	if err != nil {
		fmt.Fprintln(os.Stderr, "llm-top:", err)
	}
	if v, err := strconv.ParseFloat(os.Getenv("LLMTOP_VRAM_GB"), 64); err == nil && v > 0 {
		cfg.VramGB = v
	}
	st := &state{cfg: cfg, known: map[string]*session{}}
	if err != nil {
		st.cfgErr = err.Error()
	}
	st.base = cfg.baseURL()
	if cfg.ServerURL != "" { // several Caminus servers (brain / hands): MACHINE follows the one on this port
		if u, err := url.Parse(st.base); err == nil {
			procPort = u.Port()
		}
	}
	glob := cfg.sessionsGlob()
	sc := newClient(st.base)
	if cfg.ServerURL == "" { // not configured: also try wherever the caminus process listens (caminus serve --port N)
		sc.more = st.listenBases
	}
	go st.pollServer(sc)
	go st.pollSessions(glob)
	go st.pollSys(cfg.rel(cfg.HoldFile))
	go st.pollSavings(glob)
	if once {
		time.Sleep(7 * time.Second)
		fmt.Println(model{st: st, w: w, h: h}.render())
		return
	}
	if web { // headless: no lights either, a TUI started next to it may drive them
		if webAddr == "" {
			webAddr = cfg.Web.Listen
		}
		webAddr = listenAddr(webAddr)
		fmt.Fprintf(os.Stderr, "llm-top: serving on %s\n", webAddr)
		fmt.Fprintln(os.Stderr, "llm-top:", st.serveWeb(webAddr, cfg.Web.Token))
		os.Exit(1)
	}
	if cfg.Web.Enabled {
		go func() {
			err := st.serveWeb(cfg.Web.Listen, cfg.Web.Token)
			st.mu.Lock()
			st.cfgErr = strings.TrimPrefix(st.cfgErr+"; web: "+err.Error(), "; ")
			st.mu.Unlock()
		}()
	}
	if cfg.OpenRGB.Enabled { // not in -once: a test frame shouldn't touch the lights
		st.lights = newRGB(cfg.OpenRGB, st)
		go st.lights.run()
	}
	_, err = tea.NewProgram(model{st: st, sel: new(string)}).Run()
	if st.lights != nil {
		st.lights.stop()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
