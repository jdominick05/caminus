// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Caminus' own status: next to llama-server's monitoring surface (server.go) the server serves a versioned JSON
// status, GET /caminus/status, at the route /props names (caminus.status).
//
// The status gives what /metrics can't without label parsing: the engine against its bandwidth floor, the ms/token
// and TTFT splits, DFlash acceptance per content class and draft position, the prompt cache's tiers, the two-card
// split, context and QSA, reliability counters, power and tokens per joule, and what the agents do (tool-call parse
// failures, reasoning loops, reasoning vs answer tokens). Version 1 is read; any other version is shown as such.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"sort"
	"strings"

	"charm.land/lipgloss/v2"
)

const (
	camVersion = 1                 // the /caminus/status version this llm-top reads
	camPath    = "/caminus/status" // unless /props' caminus.status names another
)

// camWin is one of the status's windows: p50 / p95 / mean over the last values, max / count since start. A window
// with no values yet has nulls.
type camWin struct {
	P50   *float64 `json:"p50"`
	P95   *float64 `json:"p95"`
	Mean  *float64 `json:"mean"`
	Max   *float64 `json:"max"`
	Count float64  `json:"count"`
}

type camClass struct {
	Drafts     float64   `json:"drafts"`
	DraftTok   float64   `json:"draft_tokens"`
	Accepted   float64   `json:"accepted"`
	Acceptance *float64  `json:"acceptance"`
	PerPosRate []float64 `json:"per_pos_rate"`
}

type camCard struct {
	Index    int      `json:"index"`
	Name     string   `json:"name"`
	PowerW   *float64 `json:"power_w"`
	SclkMHz  *float64 `json:"sclk_mhz"`
	Perf     string   `json:"perf_level"`
	TempJ    *float64 `json:"temp_junction_c"`
	Margin   *float64 `json:"margin_mib"`
	PassMs   camWin   `json:"pass_ms"`
	WaitMs   camWin   `json:"exchange_wait_ms"`
	KVBytes  *float64 `json:"kv_bytes"`
	LinkWide string   `json:"link_width"`
}

// camStatus is the part of GET /caminus/status (version 1) llm-top shows. Numbers that can be null are pointers.
type camStatus struct {
	Version int    `json:"version"`
	Build   string `json:"build_info"`
	Stats   bool   `json:"stats_enabled"`
	Server  struct {
		Requests float64 `json:"requests"`
		Queue    float64 `json:"queue_depth"`
		Spec     string  `json:"speculative_types"`
	} `json:"server"`
	Physics struct {
		FloorMs  *float64 `json:"floor_ms"`
		PassPct  *float64 `json:"pass_pct"`
		TokenPct *float64 `json:"token_pct"`
		PassP50  *float64 `json:"k3_pass_p50_ms"`
	} `json:"physics"`
	Decode struct {
		Split    map[string]camWin `json:"split_ms"`
		PerPass  camWin            `json:"tokens_per_pass"`
		MsPerTok *float64          `json:"ms_per_token"`
	} `json:"decode"`
	TTFT struct {
		Split map[string]camWin `json:"split_ms"`
	} `json:"ttft"`
	Prefill struct {
		Rate camWin `json:"tokens_per_s"`
	} `json:"prefill"`
	DFlash struct {
		Drafts     float64             `json:"drafts"`
		Acceptance *float64            `json:"acceptance"`
		Classes    map[string]camClass `json:"classes"`
		PerPass    *float64            `json:"tokens_per_pass"`
		Plain      *float64            `json:"plain_ms_per_token"`
		Speedup    *float64            `json:"speedup_vs_plain"`
	} `json:"dflash"`
	Cache struct {
		Tokens  map[string]float64 `json:"tokens"`
		Sources map[string]float64 `json:"sources"`
		Restore camWin             `json:"restore_ms"`
		LCP     camWin             `json:"lcp"`
		VRAM    struct {
			Used    *float64 `json:"slots_used"`
			Slots   *float64 `json:"slots"`
			TierMiB *float64 `json:"tier_mib"`
		} `json:"vram"`
		Disk struct {
			GB    *float64 `json:"gb"`
			CapGB *float64 `json:"cap_gb"`
		} `json:"disk"`
		Events map[string]float64 `json:"events"`
		Saved  struct {
			PrefillS *float64 `json:"prefill_s"`
		} `json:"saved"`
	} `json:"cache"`
	Context struct {
		Position  *float64  `json:"position"`
		Max       *float64  `json:"max_ctx"`
		Remaining *float64  `json:"remaining"`
		Attention string    `json:"attention"`
		QSAFrom   *float64  `json:"qsa_from"`
		KVBytes   []float64 `json:"kv_bytes"`
	} `json:"context"`
	Reliability struct {
		Bails    map[string]float64 `json:"bails"`
		Retries  float64            `json:"retries"`
		Evicted  float64            `json:"kfd_evicted_ms"`
		Migrate  float64            `json:"pgmigrate"`
		Refusals map[string]float64 `json:"refusals"`
		Errors   map[string]float64 `json:"errors"`
	} `json:"reliability"`
	Cards []camCard `json:"cards"`
	Link  struct {
		PerS *float64 `json:"bytes_per_s"`
	} `json:"link"`
	Energy struct {
		TPJ struct {
			Decode  *float64 `json:"decode"`
			Prefill *float64 `json:"prefill"`
		} `json:"tokens_per_joule"`
	} `json:"energy"`
	Agent struct {
		Harness   map[string]float64 `json:"harness"`
		ToolCalls float64            `json:"tool_calls"`
		Failures  map[string]float64 `json:"tool_call_failures"`
		Loop      struct {
			Flagged   float64 `json:"requests_flagged"`
			Live      float64 `json:"live_max_repeat"`
			Threshold float64 `json:"repeat_threshold"`
		} `json:"loop"`
		Tokens struct {
			Reasoning float64 `json:"reasoning"`
			Answer    float64 `json:"answer"`
		} `json:"tokens"`
		MaxHits float64 `json:"max_tokens_hits"`
	} `json:"agent"`
}

// parseCam reads a status. Another version is an error (its fields may mean something else); a field of the wrong
// type only loses that field, as in the config.
func parseCam(b []byte) (*camStatus, error) {
	var v struct {
		Version *int `json:"version"`
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, fmt.Errorf("unreadable status: %v", err)
	}
	if v.Version == nil || *v.Version != camVersion {
		n := "none"
		if v.Version != nil {
			n = fmt.Sprint(*v.Version)
		}
		return nil, fmt.Errorf("status version %s, this llm-top reads %d", n, camVersion)
	}
	var cs camStatus
	if err := json.Unmarshal(b, &cs); err != nil {
		var te *json.UnmarshalTypeError
		if !errors.As(err, &te) {
			return nil, fmt.Errorf("unreadable status: %v", err)
		}
	}
	return &cs, nil
}

// ---------- number formats ----------

func val(p *float64) (float64, bool) {
	if p == nil {
		return 0, false
	}
	return *p, true
}

// msf prints a duration given in ms with a few significant digits: "0.67 ms", "17.8 ms", "260 ms", "3.60 s".
func msf(x float64) string {
	switch {
	case x >= 1000:
		return fmt.Sprintf("%.2f s", x/1000)
	case x >= 100:
		return fmt.Sprintf("%.0f ms", x)
	case x >= 10:
		return fmt.Sprintf("%.1f ms", x)
	}
	return fmt.Sprintf("%.2f ms", x)
}

// bytesf prints bytes in decimal units: "6.2 MB", "382 MB".
func bytesf(x float64) string {
	for _, u := range []struct {
		d float64
		n string
	}{{1e9, "GB"}, {1e6, "MB"}, {1e3, "kB"}} {
		if x >= u.d {
			if x/u.d >= 100 {
				return fmt.Sprintf("%.0f %s", x/u.d, u.n)
			}
			return fmt.Sprintf("%.1f %s", x/u.d, u.n)
		}
	}
	return fmt.Sprintf("%.0f B", x)
}

func pctp(p *float64) string {
	if x, ok := val(p); ok {
		return fmt.Sprintf("%.1f%%", x*100)
	}
	return "—"
}

func sum(m map[string]float64) float64 {
	t := 0.0
	for _, v := range m {
		t += v
	}
	return t
}

// nonzero lists a map's non-zero entries as "2 cut, 1 unclosed", in key order.
func nonzero(m map[string]float64) string {
	var ks []string
	for k, v := range m {
		if v != 0 {
			ks = append(ks, k)
		}
	}
	sort.Strings(ks)
	var out []string
	for _, k := range ks {
		out = append(out, fmt.Sprintf("%.0f %s", m[k], strings.ReplaceAll(k, "_", " ")))
	}
	return strings.Join(out, ", ")
}

// ---------- the CAMINUS panel ----------

type camPart struct {
	key, name string
	c         color.Color
}

var decodeParts = []camPart{{"k3", "k3", cAccent}, {"drafter", "drafter", cPurple}, {"sampler", "sampler", cYellow}, {"host", "host", cGray}}
var ttftParts = []camPart{{"queue", "queue", cMuted}, {"restore", "restore", cBlue}, {"batched", "batched", cYellow},
	{"handoff", "handoff", cOrange}, {"passes", "passes", cAccent}, {"host", "host", cGray}}

// camClasses are the content classes DFlash acceptance is counted in.
var camClasses = []string{"prose", "code", "shell", "tool"}

// msc is msf without the space, for legends: "8.29ms", "2.03s".
func msc(x float64) string { return strings.Replace(msf(x), " ", "", 1) }

// split draws the parts' window means as one bar (the means add up to the total's, unlike the medians) and
// returns it with a legend; scale divides every part (tokens per pass turns a pass into a token). unit false
// leaves the unit to the caller (every part in ms).
func split(win map[string]camWin, parts []camPart, scale float64, bw int, unit bool) (bar string, legend []string, total float64) {
	var segs []seg
	for _, p := range parts {
		m, ok := val(win[p.key].Mean)
		if !ok {
			continue
		}
		m /= scale
		total += m
		segs = append(segs, seg{int(m * 1000), p.c, "█"})
		v := msc(m)
		if !unit {
			v = strings.TrimSuffix(strings.TrimSuffix(msf(m), " ms"), " s")
		}
		legend = append(legend, fg(p.c).Render("■")+" "+p.name+" "+v)
	}
	return gauge(segs, int(total*1000), bw), legend, total
}

// camRows is the CAMINUS panel, w cells wide: none when there is no live model. Sections are cut from the end when
// the terminal is short: speed, TTFT, DFlash, the prompt cache, then the cards, reliability and the agents.
func camRows(srv *serverState, w int) []string {
	if srv.offline() != "" {
		return nil
	}
	cs := srv.Cam
	lab := func(s string) string { return fmt.Sprintf("%-8s", s) }
	pad := strings.Repeat(" ", 8)
	mut := fg(cMuted).Render
	switch {
	case srv.NotCam:
		return []string{fg(cOrange).Render(`not a Caminus server: its /props has no "engine": "caminus"`)}
	case cs == nil && srv.CamNote != "":
		return []string{mut(srv.CamNote)}
	case cs == nil && srv.CamCode != 0:
		return []string{mut(fmt.Sprintf("no %s (HTTP %d)", camPath, srv.CamCode))}
	case cs == nil:
		return []string{mut("reading " + camPath + "…")}
	}
	bw := max(6, w-8-35) // every bar the same width; the text right of it takes up to 34 cells
	var rows []string
	if l := camLoop(cs); l != "" {
		rows = append(rows, l)
	}
	if !cs.Stats {
		rows = append(rows, mut("the server's stats hook is off: no engine numbers"))
	}

	// speed against the bandwidth floor, and where a token's time goes
	if f, ok := val(cs.Physics.FloorMs); ok && f > 0 {
		tp, tok := val(cs.Physics.TokenPct)
		c := cOrange
		switch {
		case tp >= 60:
			c = cGreen
		case tp >= 30:
			c = cYellow
		}
		right := mut("no decode yet · floor " + msf(f))
		if tok {
			right = fg(c).Render(fmt.Sprintf("%.1f%%", tp)) + mut(" of "+msf(f)+" floor")
			if pp, ok := val(cs.Physics.PassPct); ok {
				right += mut(fmt.Sprintf(" · pass %.0f%%", pp))
			}
		}
		rows = append(rows, lab("speed")+gauge([]seg{{int(min(tp, 100) * 10), c, "█"}}, 1000, bw)+" "+right)
	}
	if per, ok := val(cs.Decode.PerPass.Mean); ok && per > 0 {
		bar, leg, tot := split(cs.Decode.Split, decodeParts, per, bw, false)
		if mt, ok := val(cs.Decode.MsPerTok); ok {
			tot = mt
		}
		rows = append(rows, lab("token")+bar+fmt.Sprintf(" %s/token · %.2f tok/pass", msf(tot), per),
			pad+strings.Join(leg, " ")+" ms")
	}
	rows = append(rows, "")

	// time to first token (queue included), and the prefill rate per request
	if tw := cs.TTFT.Split["total"]; tw.Count > 0 {
		bar, leg, tot := split(cs.TTFT.Split, ttftParts, 1, bw, true)
		right := msf(tot) + " mean"
		if p, ok := val(tw.P50); ok {
			right += " · p50 " + msf(p)
		}
		half := (len(leg) + 1) / 2
		b := pad + strings.Join(leg[half:], " ")
		if r, ok := val(cs.Prefill.Rate.P50); ok {
			b += mut(fmt.Sprintf(" · prefill %.0f tok/s", r))
		}
		rows = append(rows, lab("ttft")+bar+" "+right, pad+strings.Join(leg[:half], " "), b, "")
	}

	// DFlash: acceptance per content class, by draft position
	pur := fg(cPurple).Render
	if name, _ := drafters(cs.Server.Spec); name == "" {
		rows = append(rows, lab("dflash")+mut("off (speculative.types "+cs.Server.Spec+")"))
	} else if cs.DFlash.Drafts == 0 {
		rows = append(rows, lab("dflash")+mut(name+" · no drafts yet"))
	} else {
		a := lab("dflash") + pur(pctp(cs.DFlash.Acceptance)) + " accepted"
		if pp, ok := val(cs.DFlash.PerPass); ok {
			a += fmt.Sprintf(" · %.2f tok/pass", pp)
		}
		if sp, ok := val(cs.DFlash.Speedup); ok {
			a += fmt.Sprintf(" · %.2fx", sp)
			if pl, ok := val(cs.DFlash.Plain); ok {
				a += mut(" vs plain " + msf(pl))
			}
		}
		// per class "code 41% " and its share accepted at each draft position, in the cells left over
		var cl []string
		var active []int
		used := 0
		for _, k := range camClasses {
			c := cs.DFlash.Classes[k]
			t := mut(k + " —")
			if c.Drafts > 0 {
				acc := "—"
				if x, ok := val(c.Acceptance); ok {
					acc = fmt.Sprintf("%.0f%%", x*100)
				}
				t = k + " " + pur(acc) + " "
				active = append(active, len(cl))
			}
			used += lipgloss.Width(t) + 2
			cl = append(cl, t)
		}
		if len(active) > 0 {
			npos := max(1, min(8, (w-8-used+2)/len(active)))
			for _, k := range active {
				// a share, clamped to [0, 1] in a copy: a server's 1.5 or -0.2 must not index past sparkTo's blocks
				r := cs.DFlash.Classes[camClasses[k]].PerPosRate
				r = r[:min(len(r), npos)]
				c := make([]float64, len(r))
				for i, x := range r {
					c[i] = min(1, max(0, x))
				}
				cl[k] += sparkTo(c, len(c), 1, cPurple)
			}
		}
		rows = append(rows, a, pad+strings.Join(cl, "  "))
	}
	rows = append(rows, "")

	// the prompt cache: tokens by tier, the VRAM tier's slots and the disk tier against its cap, restores
	ct := cs.Cache.Tokens
	rows = append(rows, lab("cache")+fg(cBlue).Render(kTok(int(ct["vram"]+ct["disk"]+ct["exact"]))+" reused")+
		fmt.Sprintf(" · vram %s · disk %s · exact %s", kTok(int(ct["vram"])), kTok(int(ct["disk"])), kTok(int(ct["exact"]))))
	gw := max(4, (w-8)/2-20)
	var c2 []string
	if used, ok := val(cs.Cache.VRAM.Used); ok {
		if n, ok := val(cs.Cache.VRAM.Slots); ok && n > 0 {
			t := fmt.Sprintf("vram %s %.0f/%.0f", gauge([]seg{{int(used), cBlue, "█"}}, int(n), gw), used, n)
			if mib, ok := val(cs.Cache.VRAM.TierMiB); ok {
				t += fmt.Sprintf(" %.1f GB", mib/1024)
			}
			c2 = append(c2, t)
		}
	}
	if gb, ok := val(cs.Cache.Disk.GB); ok {
		if cp, ok := val(cs.Cache.Disk.CapGB); ok && cp > 0 {
			c2 = append(c2, fmt.Sprintf("disk %s %.1f / %.0f GB", gauge([]seg{{int(gb * 100), cBlue, "█"}}, int(cp*100), gw), gb, cp))
		}
	}
	if len(c2) > 0 {
		rows = append(rows, pad+strings.Join(c2, " · "))
	}
	var c3 []string
	if cs.Cache.Restore.Count > 0 {
		c3 = append(c3, "restore p50 "+msc(*nz(cs.Cache.Restore.P50))+" p95 "+msc(*nz(cs.Cache.Restore.P95)))
	}
	src, ev := cs.Cache.Sources, cs.Cache.Events
	c3 = append(c3, fmt.Sprintf("from vram %.0f · disk %.0f · cold %.0f · live %.0f · spill %.0f", src["vram"], src["disk"], src["cold"], src["live"], src["spill"]))
	if p, ok := val(cs.Cache.LCP.P50); ok {
		c3 = append(c3, "lcp p50 "+kTok(int(p)))
	}
	if s, ok := val(cs.Cache.Saved.PrefillS); ok && s > 0 {
		c3 = append(c3, fmt.Sprintf("saved %.1f s", s))
	}
	if ev["checkpoints"] > 0 {
		c3 = append(c3, fmt.Sprintf("%.0f checkpoints, %.0f spilled", ev["checkpoints"], ev["spilled"]))
	}
	rows = append(rows, pad+mut(strings.Join(c3, " · ")), "")

	// the two cards: each one's pass and its wait on the exchange; reliability; the agents as the server saw them
	var tp []string
	for _, c := range cs.Cards {
		if p, ok := val(c.PassMs.P50); ok {
			t := fmt.Sprintf("%s %s", c.Name, msf(p))
			if wt, ok := val(c.WaitMs.P50); ok {
				t += mut(" + " + msf(wt) + " wait")
			}
			tp = append(tp, t)
		}
	}
	if len(tp) > 0 {
		r := lab("cards") + strings.Join(tp, " · ")
		if l, ok := val(cs.Link.PerS); ok {
			r += mut(" · link " + bytesf(l) + "/s")
		}
		rows = append(rows, r)
	}
	rows = append(rows, lab("health")+camHealth(cs))
	ag := cs.Agent
	h := lab("agents") + fmt.Sprintf("%.0f requests · %.0f tool calls · ", cs.Server.Requests, ag.ToolCalls)
	if fails := sum(ag.Failures); fails > 0 {
		h += fg(cOrange).Render(fmt.Sprintf("%.0f parse failures (%s)", fails, nonzero(ag.Failures)))
	} else {
		h += "0 parse failures"
	}
	if hs := nonzero(ag.Harness); hs != "" {
		h += mut(" · " + hs)
	}
	loops := fmt.Sprintf("%.0f loops", ag.Loop.Flagged)
	if ag.Loop.Flagged > 0 {
		loops = fg(cOrange).Render(loops)
	}
	a2 := pad + loops + fmt.Sprintf(" · thinking %s / answer %s", kTok(int(ag.Tokens.Reasoning)), kTok(int(ag.Tokens.Answer)))
	a2 += fmt.Sprintf(" · %.0f at max_tokens", ag.MaxHits)
	return append(rows, h, a2)
}

// nz is p, or a zero for a null.
func nz(p *float64) *float64 {
	if p == nil {
		return new(float64)
	}
	return p
}

// camHealth is the reliability row: a green check while nothing went wrong.
func camHealth(cs *camStatus) string {
	r := cs.Reliability
	if sum(r.Bails) == 0 && r.Retries == 0 && r.Evicted == 0 && r.Migrate == 0 && sum(r.Refusals) == 0 && sum(r.Errors) == 0 {
		return fg(cGreen).Render("✓ no bails, retries, KFD evictions, refusals or errors")
	}
	var p []string // the worst first
	if n := sum(r.Errors); n > 0 {
		p = append(p, fg(cRed).Render(fmt.Sprintf("errors %.0f (%s)", n, nonzero(r.Errors))))
	}
	if n := sum(r.Bails); n > 0 {
		c := cOrange
		if r.Bails["failed"] > 0 {
			c = cRed
		}
		p = append(p, fg(c).Render(fmt.Sprintf("bails %.0f (%s)", n, nonzero(r.Bails))))
	}
	if n := sum(r.Refusals); n > 0 {
		p = append(p, fg(cOrange).Render(fmt.Sprintf("refusals %.0f (%s)", n, nonzero(r.Refusals))))
	}
	if r.Retries > 0 {
		p = append(p, fmt.Sprintf("retries %.0f", r.Retries))
	}
	if r.Evicted > 0 {
		p = append(p, "KFD evicted "+msf(r.Evicted))
	}
	if r.Migrate > 0 {
		p = append(p, fmt.Sprintf("%.0f page migrations", r.Migrate))
	}
	return strings.Join(p, fg(cMuted).Render(" · "))
}

// camLoop is the live loop alert: the running request's reasoning repeats one line at least the detector's
// threshold times.
func camLoop(cs *camStatus) string {
	l := cs.Agent.Loop
	if l.Threshold <= 0 || l.Live < l.Threshold {
		return ""
	}
	return fg(cRed).Render(fmt.Sprintf("▸ loop: the running request's reasoning repeats one line %.0fx", l.Live))
}

// camContextRow is CONTEXT's engine line: dense or QSA attention, KV per card, what the engine holds.
func camContextRow(cs *camStatus) string {
	if cs == nil {
		return ""
	}
	cx := cs.Context
	var p []string
	switch q, ok := val(cx.QSAFrom); {
	case cx.Attention == "qsa":
		p = append(p, fg(cPurple).Render("QSA attention"))
	case cx.Attention != "" && ok:
		p = append(p, fmt.Sprintf("%s attention (QSA from %s)", cx.Attention, kTok(int(q))))
	case cx.Attention != "":
		p = append(p, cx.Attention+" attention")
	}
	if len(cx.KVBytes) > 0 {
		p = append(p, fmt.Sprintf("KV %s per card", bytesf(cx.KVBytes[0])))
	}
	if pos, ok := val(cx.Position); ok {
		h := kTok(int(pos)) + " held"
		if rem, ok := val(cx.Remaining); ok {
			h += ", " + kTok(int(rem)) + " left"
		}
		p = append(p, h)
	}
	if len(p) == 0 {
		return ""
	}
	return fg(cMuted).Render(fmt.Sprintf("%-8s", "engine")) + strings.Join(p, " · ")
}

// camMachineRows are MACHINE's power lines: both cards' power and tokens per joule, then per card its power, clock,
// junction temperature, performance level and margin below the guard line. None without the server's hardware
// reader.
func camMachineRows(cs *camStatus) []string {
	if cs == nil {
		return nil
	}
	var rows []string
	tot, n := 0.0, 0
	for _, c := range cs.Cards {
		if p, ok := val(c.PowerW); ok {
			tot, n = tot+p, n+1
		}
	}
	if n == 0 {
		return nil
	}
	r := fmt.Sprintf("%-6s%.0f W", "power", tot)
	if d, ok := val(cs.Energy.TPJ.Decode); ok {
		r += fmt.Sprintf(" · %.3f tok/J decode", d)
	}
	if p, ok := val(cs.Energy.TPJ.Prefill); ok {
		r += fmt.Sprintf(" · %.3f prefill", p)
	}
	rows = append(rows, r)
	for _, c := range cs.Cards {
		p, ok := val(c.PowerW)
		if !ok {
			continue
		}
		l := fmt.Sprintf("%-6s%3.0f W", c.Name, p)
		if s, ok := val(c.SclkMHz); ok {
			l += fmt.Sprintf(" %4.0f MHz", s)
		}
		if t, ok := val(c.TempJ); ok {
			l += fmt.Sprintf(" %3.0fC", t)
		}
		if c.Perf != "" {
			perf := c.Perf
			if perf != "high" {
				perf = fg(cOrange).Render(perf)
			}
			l += " " + perf
		}
		if m, ok := val(c.Margin); ok {
			mc := cMuted
			if m < 512 {
				mc = cRed
			}
			l += fg(mc).Render(fmt.Sprintf(" · margin %.1f GB", m/1024))
		}
		rows = append(rows, l)
	}
	return rows
}

// camBox is the CAMINUS panel as a box: the speed rows stay longest.
func camBox(srv *serverState, w int) *box {
	return &box{title: "CAMINUS  (engine · " + camPath + ")", c: cPurple, w: w, rows: camRows(srv, w-4), min: 3}
}
