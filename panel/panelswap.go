// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// panelswap.go: enter on a model while another server runs. One question ("stop alpha and start beta? enter / esc"),
// then caminus stop for each running server (it returns once the server's guard is gone), then caminus ps until the
// stopped servers are gone and their cards' memory is below the line caminus start itself refuses on, then caminus
// start. Each step shows in the header; a step that fails says which one and that the new model was not started.
// A server started outside the panel is never stopped here: caminus stop only knows the servers caminus start
// recorded, so the panel says what stops it (kill and its pid) instead.

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

// swapFreeMiB is caminus start's own refusal line (manage.py busy_elsewhere: a card with more than 2 GiB in use is
// held by another process): the start waits until each stopped server's card is under it.
const swapFreeMiB = 2048

// swapWait is how long the panel waits after the stops for the cards' memory; swapPoll is how often it reads ps.
var (
	swapWait = 60 * time.Second
	swapPoll = time.Second
)

// swapPlan is a confirmed swap, captured when the question was asked: the selection can move while it runs.
type swapPlan struct {
	seq   int      // this swap (a reply of an earlier one is dropped)
	stop  []string // the registry names to stop, in order
	cards []string // the cards their memory is on (empty: not known, every card is watched)
	model string   // the model to start
	args  []string // its caminus start arguments
	done  int      // servers stopped so far
	since time.Time
	held  string // the last wait's reason ("XTX 19.3 GB in use"), said if the wait runs out
}

type (
	swapStopMsg struct {
		id, seq int
		name    string
		err     error
	}
	swapPollMsg struct {
		id, seq int
		v       psInfo
		err     error
	}
)

var swapSeq int

// askSwap is enter's question when other servers run: stop them all, then start mi. A server started outside the
// panel can't be stopped by caminus stop, so the note says what stops it and nothing is asked.
func (m ctl) askSwap(mi camModel, live []serverInfo) (ctl, tea.Cmd) {
	for _, s := range live {
		if s.External {
			m.setNote(outsideStopNote(s, "first, then enter starts "+mi.Name), true)
			return m, nil
		}
	}
	swapSeq++
	p := &swapPlan{seq: swapSeq, model: mi.Name, args: m.startArgs(mi)}
	every := false
	for _, s := range live {
		p.stop = append(p.stop, s.Name)
		if len(s.Cards) == 0 {
			every = true
		}
		p.cards = append(p.cards, s.Cards...)
	}
	if every {
		p.cards = nil
	}
	names := strings.Join(p.stop, ", ")
	m.confirm = &pending{q: "stop " + names + " and start " + mi.Name + "? enter / esc", enter: true,
		yes: func(c *ctl) tea.Cmd {
			c.swap = p
			c.busy = "stopping " + p.stop[0] + " (then " + p.model + " starts)"
			return c.swapStop(p.stop[0])
		}}
	return m, nil
}

// swapStop runs caminus stop NAME (it waits until the server's guard has exited, up to its own 170 s).
func (m ctl) swapStop(name string) tea.Cmd {
	id, seq := m.id, m.swap.seq
	return func() tea.Msg {
		_, err := m.be.act(5*time.Minute, "stop", name, "--json")
		return swapStopMsg{id, seq, name, err}
	}
}

// swapRead reads caminus ps after a pause (none for the first read after the stops).
func (m ctl) swapRead(after time.Duration) tea.Cmd {
	id, seq := m.id, m.swap.seq
	return func() tea.Msg {
		time.Sleep(after)
		var v psInfo
		err := m.be.call(&v, 15*time.Second, "ps", "--json")
		return swapPollMsg{id, seq, v, err}
	}
}

// swapFail ends the swap with the step that failed: nothing more runs and the new model is not started.
func (m ctl) swapFail(what string) (ctl, tea.Cmd) {
	model := m.swap.model
	m.swap, m.busy = nil, ""
	m.refreshFits()
	m.setNote(model+" not started: "+what, true) // (the outcome first: the note line is cut at the screen's width)
	return m, tea.Batch(m.loadPS(), m.loadModels())
}

// swapUpdate runs one step's answer.
func (m ctl) swapUpdate(msg tea.Msg) (ctl, tea.Cmd) {
	p := m.swap
	switch msg := msg.(type) {
	case swapStopMsg:
		if p == nil || msg.id != m.id || msg.seq != p.seq {
			return m, nil
		}
		if msg.err != nil {
			return m.swapFail("couldn't stop " + msg.name + ": " + msg.err.Error())
		}
		p.done++
		if p.done < len(p.stop) {
			m.busy = "stopping " + p.stop[p.done] + " (then " + p.model + " starts)"
			return m, m.swapStop(p.stop[p.done])
		}
		p.since = time.Now()
		m.busy = "waiting for " + strings.Join(p.stop, ", ") + " to free the card"
		return m, m.swapRead(0)
	case swapPollMsg:
		if p == nil || msg.id != m.id || msg.seq != p.seq {
			return m, nil
		}
		stopped := strings.Join(p.stop, ", ") + " stopped"
		if msg.err != nil {
			p.held = "couldn't read the running servers: " + msg.err.Error()
		} else {
			m.readDone("ps", nil)
			msg.v.Servers = m.keepExternal(msg.v.Servers)
			m.ps = msg.v
			p.held = m.swapHeld()
			if p.held == "" {
				m.swap = nil
				m.busy = "starting " + p.model
				return m, m.action(stopped+" · start "+p.model, 15*time.Minute, p.args...)
			}
		}
		if time.Since(p.since) > swapWait {
			return m.swapFail(fmt.Sprintf("%s, but after %.0f s %s", stopped, swapWait.Seconds(), p.held))
		}
		m.busy = "waiting: " + p.held
		return m, m.swapRead(swapPoll)
	}
	return m, nil
}

// swapHeld is why the start must still wait ("" = it can go): a stopped server still listed, or a card it was on
// still holding more than caminus start accepts.
func (m ctl) swapHeld() string {
	for _, s := range m.liveServers() {
		for _, n := range m.swap.stop {
			if s.Name == n && !s.External {
				return n + " is still listed as " + s.State
			}
		}
	}
	for _, c := range m.ps.Cards {
		if !m.swapWatches(c.Name) || c.UsedMiB <= swapFreeMiB {
			continue
		}
		return fmt.Sprintf("%s still has %.1f GB in use", c.Name, c.UsedMiB/1024)
	}
	return ""
}

func (m ctl) swapWatches(card string) bool {
	if len(m.swap.cards) == 0 {
		return true
	}
	for _, c := range m.swap.cards {
		if c == card {
			return true
		}
	}
	return false
}

// ---------- servers started outside the panel ----------

// outsideStop is the command that stops a server caminus didn't start: a SIGTERM to the pid caminus ps found (caminus
// serve's clean shutdown: it drains, parks its conversation and exits; the packaged systemd unit does not restart it),
// else "" (no pid: kill 0 would signal a whole process group). The unit a process runs in is not named: it can be
// a terminal's or tmux's service rather than the server's own.
func outsideStop(sv serverInfo) string {
	if sv.Pid > 0 {
		return "kill " + strconv.Itoa(sv.Pid)
	}
	return ""
}

// shownName is a server's model where caminus ps knows it (an outside server's record name is made up: observed-PORT).
func shownName(sv serverInfo) string {
	if sv.External && sv.Model != "" {
		return sv.Model
	}
	return sv.Name
}

// outsideStopNote says what stops an outside server, the command first (the note line is cut at the screen's width).
// tail follows "stops NAME" (s: "", enter: "first, then enter starts beta").
func outsideStopNote(sv serverInfo, tail string) string {
	name := shownName(sv)
	cmd := outsideStop(sv)
	if cmd == "" {
		return name + " was started outside the panel: stop it where it was started" + strings.TrimPrefix(tail, "first")
	}
	t := cmd + " stops " + name
	if tail != "" {
		t += " " + tail
	}
	return t + " (started outside the panel, so caminus stop can't)"
}
