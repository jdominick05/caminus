// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Savings: what the local agent traffic would have cost at API prices, minus the GPU's electricity.
//
// Token counts come from every Qwen Code session log (usageMetadata of each model request), so they survive server
// restarts: promptTokenCount (of which cachedContentTokenCount was a cache hit) and candidates + thoughts as output.
// Prices and power are part of llm-top.json (config.go), written with reference defaults on first run.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"
)

type priceCfg struct {
	Label       string  `json:"price_label"`
	InPerM      float64 `json:"input_per_m"`
	CachedPerM  float64 `json:"cached_input_per_m"`
	WritePerM   float64 `json:"cache_write_per_m"`
	OutPerM     float64 `json:"output_per_m"`
	BusyWatts   float64 `json:"gpu_busy_watts"`
	KwhPrice    float64 `json:"kwh_price"`
	ReadTokSec  float64 `json:"fallback_read_tok_s"`
	WriteTokSec float64 `json:"fallback_write_tok_s"`
	Note        string  `json:"_note"`
}

var defaultPrices = priceCfg{
	Label:       "Claude Sonnet 5.5",
	InPerM:      2.00,
	CachedPerM:  0.10,
	WritePerM:   2.50,
	OutPerM:     10.00,
	BusyWatts:   380,
	KwhPrice:    0.17,
	ReadTokSec:  900,
	WriteTokSec: 27,
	Note: "USD per million tokens for the API you'd otherwise pay, cached input = prompt-cache hit price, cache_write = price of new prompt tokens when they are cached " +
		"(5-minute write; 0 = bill new tokens at input_per_m). Sonnet 5.5 rates from claude.com/pricing, 2026-10-17. " +
		"gpu_busy_watts = extra wall power while the GPU works; electricity is estimated from token counts and the " +
		"server's measured read/write speeds (fallback values when the server has no history yet).",
}

type tokTotals struct {
	Reqs                int
	Prompt, Cached, Out int64
}

func (t *tokTotals) add(o tokTotals) {
	t.Reqs += o.Reqs
	t.Prompt += o.Prompt
	t.Cached += o.Cached
	t.Out += o.Out
}

type usageFile struct {
	off  int64
	rest []byte
}

type ledger struct {
	files map[string]*usageFile
	days  map[string]*tokTotals // local date -> totals
	first time.Time
}

var usageKey = []byte(`"usageMetadata"`)

func (l *ledger) apply(line []byte) {
	if !bytes.Contains(line, usageKey) {
		return
	}
	var r struct {
		Type      string     `json:"type"`
		Timestamp string     `json:"timestamp"`
		Usage     *usageJSON `json:"usageMetadata"`
	}
	if json.Unmarshal(line, &r) != nil || r.Usage == nil || r.Usage.Prompt == 0 {
		return
	}
	t, err := time.Parse(time.RFC3339Nano, r.Timestamp)
	if err != nil {
		return
	}
	day := t.Local().Format("2006-01-02")
	d := l.days[day]
	if d == nil {
		d = &tokTotals{}
		l.days[day] = d
	}
	d.add(tokTotals{1, int64(r.Usage.Prompt), int64(r.Usage.Cached), int64(r.Usage.Cand + r.Usage.Thoughts)})
	if l.first.IsZero() || t.Before(l.first) {
		l.first = t
	}
}

func (l *ledger) scan(path string) {
	uf := l.files[path]
	if uf == nil {
		uf = &usageFile{}
		l.files[path] = uf
	}
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return
	}
	if st.Size() < uf.off { // truncated or rewritten: read it again; what was counted stays counted
		*uf = usageFile{}
	}
	if st.Size() <= uf.off {
		return
	}
	if _, err := f.Seek(uf.off, io.SeekStart); err != nil {
		return
	}
	rd := bufio.NewReaderSize(f, 1<<20)
	for {
		chunk, err := rd.ReadBytes('\n')
		uf.off += int64(len(chunk))
		if len(chunk) > 0 {
			if chunk[len(chunk)-1] == '\n' {
				line := chunk
				if len(uf.rest) > 0 {
					line = append(uf.rest, chunk...)
					uf.rest = nil
				}
				l.apply(line)
			} else {
				uf.rest = append(uf.rest, chunk...)
			}
		}
		if err != nil {
			break
		}
	}
}

type savingsState struct {
	Ready bool
	All   tokTotals
	Today tokTotals
	First time.Time
	Days  int
}

func (st *state) pollSavings(glob string) {
	l := &ledger{files: map[string]*usageFile{}, days: map[string]*tokTotals{}}
	for {
		s := l.update(glob, time.Now())
		st.mu.Lock()
		st.sav = s
		st.mu.Unlock()
		time.Sleep(10 * time.Second)
	}
}

// update scans every session log for new usage lines and sums them up.
func (l *ledger) update(glob string, now time.Time) savingsState {
	files, _ := filepath.Glob(glob)
	for _, f := range files {
		l.scan(f)
	}
	var s savingsState
	s.Ready, s.First = true, l.first
	if !l.first.IsZero() { // calendar days since the first one, days without use included
		y, m, d := l.first.Local().Date()
		s.Days = int(now.Local().Sub(time.Date(y, m, d, 0, 0, 0, 0, time.Local)).Hours()/24) + 1
	}
	today := now.Local().Format("2006-01-02")
	for day, t := range l.days {
		s.All.add(*t)
		if day == today {
			s.Today.add(*t)
		}
	}
	return s
}

// cost returns API cost, electricity cost and net saving in USD.
func (p priceCfg) cost(t tokTotals, readTokS, writeTokS float64) (api, power, net float64) {
	pos := func(x float64) float64 { return math.Max(0, x) } // a negative price in the config counts as 0
	fresh := pos(float64(t.Prompt - t.Cached))
	freshRate := pos(p.InPerM)
	if p.WritePerM > 0 { // agents cache their whole prompt: new tokens are cache writes
		freshRate = p.WritePerM
	}
	api = (fresh*freshRate + float64(t.Cached)*pos(p.CachedPerM) + float64(t.Out)*pos(p.OutPerM)) / 1e6
	if readTokS <= 0 {
		readTokS = p.ReadTokSec
	}
	if writeTokS <= 0 {
		writeTokS = p.WriteTokSec
	}
	secs := 0.0 // no usable speed at all: no electricity estimate rather than Inf
	if readTokS > 0 {
		secs += fresh / readTokS
	}
	if writeTokS > 0 {
		secs += float64(t.Out) / writeTokS
	}
	power = secs / 3600 * pos(p.BusyWatts) / 1000 * pos(p.KwhPrice)
	return api, power, api - power
}

func usd(x float64) string {
	switch {
	case x >= 1000:
		return fmt.Sprintf("$%.0f", x)
	case x >= 10:
		return fmt.Sprintf("$%.1f", x)
	}
	return fmt.Sprintf("$%.2f", x)
}
