// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// panellive.go: the panel's live numbers, read the way the monitor (caminus top) reads them and with its readers: one
// round of the server's monitoring surface a second (server.go's srvClient and serverState: /health, /props, /metrics,
// /slots, /caminus/status), the Qwen Code sessions that talk to it (top_main.go's session scan), and Ember's sessions
// (`caminus ember agent peers / status --json`; a session whose model is one of the server's /v1/models ids is one of
// its clients). Every read is a command the panel's own ticks start, one at a time:
// nothing runs in the background once the panel is closed, and nothing here ever changes the server.
package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

// liveMon is the running server's live picture, shared by the component's copies (a pointer: one round in flight).
type liveMon struct {
	url      string // the server followed ("" = none)
	client   *srvClient
	srv      serverState
	inflight bool

	sess     *state // the session scan's state (top's reader): known logs and the sessions of this server
	scanning bool
	clients  int // Qwen Code sessions active in the last agentFresh that talk to this server
	rounds   int
	ids      []string  // the followed server's model ids (/v1/models): an Ember session thinking with one is its client
	idsAt    time.Time // when they were last asked

	extMissed map[string]int // outside servers the last ps reads missed while the live round still saw them, by URL

	emberBusy bool
	ember     emberCounts
}

// emberCounts is Ember's sessions: how many, and how many work on a goal now.
type emberCounts struct {
	known            bool
	sessions, active int
	brains           []string // the model id each running session thinks with (its status' model)
}

func newLiveMon() *liveMon {
	return &liveMon{sess: &state{known: map[string]*session{}}}
}

type (
	liveTick struct{ id, epoch int }
	liveMsg  struct {
		id  int
		url string
		p   probe
		ids []string // the server's model ids (/v1/models), asked when the round's idsDue was set; nil otherwise
	}
	sessionsMsg struct {
		id, n int
	}
	emberTick     struct{ id, epoch int }
	emberCountMsg struct {
		id int
		v  emberCounts
	}
)

// liveEvery is the live round's pace (the monitor asks every 500 ms; the panel shows less and asks half as often).
const liveEvery = time.Second

// emberEvery is how often Ember's sessions are counted: each one is a command over Ember's SSH login.
const emberEvery = time.Minute

func (m ctl) liveTickCmd(d time.Duration) tea.Cmd {
	id, ep := m.id, m.epoch
	if d <= 0 {
		return func() tea.Msg { return liveTick{id, ep} }
	}
	return tea.Tick(d, func(time.Time) tea.Msg { return liveTick{id, ep} })
}

func (m ctl) emberTickCmd(d time.Duration) tea.Cmd {
	id, ep := m.id, m.epoch
	if d <= 0 {
		return func() tea.Msg { return emberTick{id, ep} }
	}
	return tea.Tick(d, func(time.Time) tea.Msg { return emberTick{id, ep} })
}

// liveTarget is the URL of the server the running card shows ("" when none runs).
func (m ctl) liveTarget() string {
	if sv := m.primary(); sv != nil {
		return sv.URL
	}
	return ""
}

// liveRound is one round against the followed server, nil when none runs or a round is still out. It follows a new
// server from a clean picture.
func (m ctl) liveRound() tea.Cmd {
	lv := m.live
	target := m.liveTarget()
	if target != lv.url {
		lv.url, lv.srv, lv.inflight, lv.clients, lv.ids, lv.idsAt = target, serverState{}, false, 0, nil, time.Time{}
		lv.client = nil
		if target != "" {
			lv.client = newClient(target)
		}
	}
	if lv.client == nil || lv.inflight {
		return nil
	}
	lv.inflight = true
	c, id, url := lv.client, m.id, target
	askIDs := time.Since(lv.idsAt) >= idsEvery
	if askIDs {
		lv.idsAt = time.Now()
	}
	return func() tea.Msg {
		msg := liveMsg{id: id, url: url, p: c.round(time.Now())}
		if askIDs && msg.p.Health == http.StatusOK {
			msg.ids = modelIDs(c)
		}
		return msg
	}
}

// idsEvery is how often the server's model ids are asked again (they change only with a restart).
const idsEvery = 30 * time.Second

// modelIDs is the ids the server answers to (GET /v1/models), nil when it does not answer.
func modelIDs(c *srvClient) []string {
	code, b := c.get("/v1/models")
	if code != http.StatusOK {
		return nil
	}
	var v struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal(b, &v) != nil {
		return nil
	}
	ids := []string{}
	for _, d := range v.Data {
		if d.ID != "" {
			ids = append(ids, d.ID)
		}
	}
	return ids
}

// emberClients is how many running Ember sessions think with the followed server: their model is one of its ids, or
// its alias or an alias variant (alias-xhigh: the session logs' rule, session.talksTo). A paid brain never matches.
func (m ctl) emberClients() int {
	lv := m.live
	alias := lv.srv.Props.Alias
	n := 0
	for _, b := range lv.ember.brains {
		if b == "" {
			continue
		}
		if slices.Contains(lv.ids, b) || alias != "" && (b == alias || strings.HasPrefix(b, alias+"-")) {
			n++
		}
	}
	return n
}

// sessionScan counts the Qwen Code sessions of the followed server (top's scan: the newest logs of the last 6 h in
// Caminus's Qwen Code home, filtered by the server's alias); nil when there is nothing to scan or one still runs.
func (m ctl) sessionScan() tea.Cmd {
	lv := m.live
	if lv.url == "" || m.hs.Home == "" || lv.scanning {
		return nil
	}
	lv.scanning = true
	st, alias, id := lv.sess, lv.srv.Props.Alias, m.id
	glob := filepath.Join(m.hs.Home, "projects", "*", "chats", "*.jsonl")
	return func() tea.Msg {
		home, _ := os.UserHomeDir()
		st.mu.Lock()
		st.srv.Props.Alias = alias
		st.mu.Unlock()
		st.scanSessions(glob, home, time.Now())
		st.mu.Lock()
		n := 0
		for _, s := range st.sessions {
			if time.Since(s.Mod) < agentFresh {
				n++
			}
		}
		st.mu.Unlock()
		return sessionsMsg{id: id, n: n}
	}
}

// emberCount counts Ember's sessions and those working on a goal (the switcher's own reads).
func (m ctl) emberCount() tea.Cmd {
	be, id := m.be, m.id
	return func() tea.Msg {
		var peers []struct {
			Name string `json:"name"`
		}
		if err := be.call(&peers, 30*time.Second, "ember", "agent", "peers", "resident", "--json"); err != nil {
			return emberCountMsg{id: id}
		}
		v := emberCounts{known: true, sessions: len(peers)}
		for _, p := range peers { // (one after another: each is a command over Ember's one SSH connection)
			var st struct {
				Alive bool   `json:"daemon_alive"`
				Model string `json:"model"`
				Goals []struct {
					State string `json:"state"`
				} `json:"goals"`
			}
			if be.call(&st, 30*time.Second, "ember", "agent", "status", p.Name, "--json") != nil || !st.Alive {
				continue
			}
			v.brains = append(v.brains, st.Model)
			for _, g := range st.Goals {
				if g.State == "working" || g.State == "queued" {
					v.active++
					break
				}
			}
		}
		return emberCountMsg{id: id, v: v}
	}
}

func (m ctl) liveUpdate(msg tea.Msg) (ctl, tea.Cmd) {
	lv := m.live
	switch msg := msg.(type) {
	case liveTick:
		if !m.mine(msg.id, msg.epoch) {
			return m, nil
		}
		lv.rounds++
		cmds := []tea.Cmd{m.liveTickCmd(liveEvery), m.liveRound()}
		if lv.rounds%3 == 1 {
			cmds = append(cmds, m.sessionScan())
		}
		return m, tea.Batch(cmds...)
	case liveMsg:
		if msg.id != m.id {
			return m, nil
		}
		if msg.url == lv.url {
			lv.inflight = false
			lv.srv.apply(msg.p)
			if msg.ids != nil {
				lv.ids = msg.ids
			}
		}
	case sessionsMsg:
		if msg.id == m.id {
			lv.scanning, lv.clients = false, msg.n
		}
	case emberTick:
		if !m.mine(msg.id, msg.epoch) {
			return m, nil
		}
		next := m.emberTickCmd(emberEvery)
		if !m.ember.ok {
			next = m.emberTickCmd(5 * time.Second) // (asked again soon after the set-up read answers)
		}
		if m.ember.ok && !lv.emberBusy {
			lv.emberBusy = true
			return m, tea.Batch(next, m.emberCount())
		}
		return m, next
	case emberCountMsg:
		if msg.id == m.id {
			lv.emberBusy, lv.ember = false, msg.v
		}
	}
	return m, nil
}

// extGrace is how many ps reads in a row may miss an outside server the live round still sees: ps asks it /health
// with a 1 s limit, and a saturated server misses some (a running server must not blink out of the card).
const extGrace = 3

// keepExternal is servers with the outside servers the previous read listed but this one missed, kept while the live
// round still reaches them, for up to extGrace reads.
func (m ctl) keepExternal(servers []serverInfo) []serverInfo {
	lv := m.live
	if lv.extMissed == nil {
		lv.extMissed = map[string]int{}
	}
	listed := map[string]bool{}
	for _, s := range servers {
		if s.External {
			listed[s.URL] = true
			delete(lv.extMissed, s.URL)
		}
	}
	for _, s := range m.ps.Servers {
		if !s.External || listed[s.URL] || s.URL != lv.url || !lv.srv.Up || lv.extMissed[s.URL] >= extGrace {
			continue
		}
		lv.extMissed[s.URL]++
		servers = append(servers, s)
	}
	return servers
}

// liveNumbers is what the running card prints, from the live round where it has one and else from caminus ps.
type liveNumbers struct {
	decode, prefill float64 // tok/s, 0 = not known
	cached          float64 // the share of prompt tokens served from the cache, -1 = not known
	phase           string  // "reading", "writing" or "" (idle)
	spark           []float64
}

func (m ctl) liveNumbers(sv serverInfo) liveNumbers {
	n := liveNumbers{cached: -1}
	lv := m.live
	s := &lv.srv
	if lv.url != sv.URL || !s.Up {
		if sv.DecodeTPS != nil {
			n.decode = *sv.DecodeTPS
		}
		return n
	}
	n.phase = s.phase()
	n.spark = s.GenRate.v
	switch {
	case s.curGen > 0:
		n.decode = s.curGen
	case s.Cam != nil && s.Cam.Decode.MsPerTok != nil && *s.Cam.Decode.MsPerTok > 0:
		n.decode = 1000 / *s.Cam.Decode.MsPerTok
	case sv.DecodeTPS != nil:
		n.decode = *sv.DecodeTPS
	}
	switch {
	case s.curRead > 0:
		n.prefill = s.curRead
	case s.Cam != nil && s.Cam.Prefill.Rate.P50 != nil:
		n.prefill = *s.Cam.Prefill.Rate.P50
	}
	if mt := s.Metrics; mt != nil {
		if c, r := mt["prompt_tokens_cached_total"], mt["prompt_tokens_total"]; c+r > 0 {
			n.cached = c / (c + r)
		}
	}
	return n
}

// liveName is the model a live server serves, from its /props when the panel follows it ("" otherwise).
func (m ctl) liveName(sv serverInfo) string {
	if m.live.url == sv.URL && m.live.srv.Props.Path != "" {
		return strings.TrimSuffix(baseName(m.live.srv.Props.Path), ".gguf")
	}
	return ""
}
