// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Caminus client: GETs only, never anything that changes the server.
//
// Caminus serves llama-server's monitoring surface under llama-server's names and shapes (/health, /props, /slots,
// /metrics with the llamacpp: series) plus its own versioned GET /caminus/status (caminus.go). Each 500 ms round asks
// /health, /metrics and /slots; every 2 s /props (n_ctx, alias, model path, build, the engine marker and the status
// route); the status every round while a request runs and the last one answered, else every 2 s (it is the largest
// answer: histograms and per-card data). A /props without the engine marker is another server: the status is not
// asked, and the CAMINUS panel says so.
package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
)

type slotJSON struct {
	ID         int       `json:"id"`
	NCtx       int       `json:"n_ctx"`
	Processing bool      `json:"is_processing"`
	Task       int       `json:"id_task"`
	NPrompt    int       `json:"n_prompt_tokens"`
	NProcessed int       `json:"n_prompt_tokens_processed"`
	NCache     int       `json:"n_prompt_tokens_cache"`
	Next       []nextTok `json:"next_token"`
	Params     struct {
		NPredict  int    `json:"n_predict"`
		SpecTypes string `json:"speculative.types"` // "draft-dflash", or "none" without a drafter
	} `json:"params"`
}

// Qwen Code asks for at most this many tokens when it compacts a session (COMPACT_MAX_OUTPUT_TOKENS in its
// chatCompressionService), less only when the history nearly fills the window (window - input - 1024). Its normal
// turns ask for more. A busy request with that budget is shown as COMPACTING.
const compactBudget = 20000

func isCompaction(nPredict, prompt, nCtx int) bool {
	switch {
	case nPredict == compactBudget:
		return true
	case nPredict > 0 && nPredict < compactBudget && nCtx > 0 && prompt > 0:
		return prompt+nPredict >= nCtx*9/10 // clamped to what is left of the window
	}
	return false
}

type nextTok struct {
	NRemain  int `json:"n_remain"`
	NDecoded int `json:"n_decoded"`
}

// next is the slot's token counters; a slot that never ran has none.
func (sl slotJSON) next() nextTok {
	if len(sl.Next) == 0 {
		return nextTok{}
	}
	return sl.Next[0]
}

// engineCaminus is /props' engine marker on a Caminus server.
const engineCaminus = "caminus"

type propsJSON struct {
	Gen struct {
		NCtx   int `json:"n_ctx"`
		Params struct {
			NPredict int `json:"n_predict"` // the server's default max tokens; -1 = unlimited
		} `json:"params"`
	} `json:"default_generation_settings"`
	Alias   string `json:"model_alias"`
	Path    string `json:"model_path"`
	Build   string `json:"build_info"`
	Engine  string `json:"engine"` // engineCaminus
	Caminus struct {
		Status string `json:"status"` // the status route
	} `json:"caminus"`
}

type ring struct{ v []float64 }

func (r *ring) push(x float64, max int) {
	r.v = append(r.v, x)
	if len(r.v) > max {
		r.v = r.v[len(r.v)-max:]
	}
}

type slotPrev struct {
	task, proc, dec int
	prompt, cache   int
	busy            bool
	t, start, decAt time.Time // start: request first seen busy; decAt: first token written
}

// reqEvent is one finished request, for ACTIVITY when no agent session is shown.
type reqEvent struct {
	T                        time.Time // started
	Dur                      time.Duration
	Prompt, Cached, Out, Ctx int
	ReadRate, GenRate        float64
	Slot                     int
}

const maxDone = 200

func (s *serverState) finished(e reqEvent) {
	s.Done = append(s.Done, e)
	if len(s.Done) > maxDone {
		s.Done = s.Done[len(s.Done)-maxDone:]
	}
}

type serverState struct {
	Polled          bool // at least one round applied: before that nothing is known
	Up, Loading     bool
	NotCam          bool // the last /props had no engine marker: not a Caminus server
	HaveSlots       bool
	Slot            slotJSON // the slot shown in CONTEXT
	NSlots, Busy    int
	Decoded, Remain int
	Metrics         map[string]float64
	Props           propsJSON
	ReadRate        ring
	GenRate         ring
	curRead, curGen float64
	prev            map[int]slotPrev
	busySince       time.Time
	Spec            specHist

	Done []reqEvent // finished requests since llm-top started, oldest first

	Cam     *camStatus // the last /caminus/status read, nil before the first or after a failed one
	CamCode int        // its HTTP code, 0 = not asked yet
	CamNote string     // why a 200 status could not be used (another version)

	Base string // URL that answered (or was tried last)
}

// offline says why there is no live model to show ("" when there is one): every panel says so instead of showing
// zeros or the last picture as if it were current.
func (s *serverState) offline() string {
	switch {
	case !s.Polled:
		return "looking for the server"
	case s.Loading:
		return "model loading"
	case !s.Up:
		return "no server running"
	}
	return ""
}

// compacting says whether the request shown is a Qwen Code compaction.
func (s *serverState) compacting() bool {
	if !s.Up {
		return false
	}
	sl := s.Slot
	// a server whose default is 20000 gives every request that budget: only a client-set one counts
	return s.HaveSlots && sl.Processing && sl.Params.NPredict != s.Props.Gen.Params.NPredict &&
		isCompaction(sl.Params.NPredict, sl.NPrompt, max(sl.NCtx, s.Props.Gen.NCtx))
}

// phase is "", "reading" or "writing" for the request shown.
func (s *serverState) phase() string {
	if !s.HaveSlots || !s.Slot.Processing {
		return ""
	}
	if s.Decoded == 0 {
		return "reading"
	}
	return "writing"
}

// busy says whether the server is working on something, from /slots or else /metrics.
func (s *serverState) busy() bool {
	if !s.Up {
		return false
	}
	if s.HaveSlots {
		return s.Busy > 0
	}
	return s.Metrics["requests_processing"] > 0
}

// ---------- client ----------

type srvClient struct {
	base       string
	cl         *http.Client
	bases      []string // candidates, tried in turn while nothing answers
	bi         int
	more       func() []string // extra candidate bases (the caminus process's listening ports), may be nil
	seen       bool            // base has answered at least once
	misses     int             // rounds in a row without an answer
	lastProps  time.Time
	caminus    bool      // the last /props that decoded had the engine marker
	lastStatus time.Time // the last /caminus/status question
	statusOK   bool      // ... and whether it answered 200
	statusPath string    // from /props' caminus.status
}

// rotateAfter is how many rounds without an answer move on to the next candidate once a server has answered: one
// slow answer (a 2 s timeout while the server is saturated) is not a server that went away.
const rotateAfter = 3

func newClient(base string) *srvClient {
	return &srvClient{base: base, bases: []string{base}, cl: &http.Client{Timeout: 2 * time.Second}}
}

func (c *srvClient) get(path string) (int, []byte) {
	resp, err := c.cl.Get(c.base + path)
	if err != nil {
		return 0, nil
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	return resp.StatusCode, b
}

// probe is one round of answers; codes are 0 when the endpoint was not asked (or did not answer), -1 when it
// answered 200 with something that did not decode.
type probe struct {
	T                                 time.Time
	Base                              string
	Health                            int
	PropsCode, MetricsCode, SlotsCode int
	Props                             propsJSON
	Metrics                           map[string]float64
	Slots                             []slotJSON
	CamCode                           int        // /caminus/status, 0 = not asked
	Cam                               *camStatus // nil unless CamCode is 200 and the status is version 1
	CamNote                           string     // why a 200 status was not used
	Miss                              bool       // no answer, but too soon to call the server gone: keep the last picture
}

func (c *srvClient) round(now time.Time) probe {
	p := c.ask(now)
	p.Base = c.base
	if p.Health != 0 {
		c.seen, c.misses = true, 0
		return p
	}
	c.misses++
	if c.more != nil {
		c.addBases(c.more())
	}
	if !c.seen || c.misses >= rotateAfter { // nothing there: try the next candidate
		c.bi = (c.bi + 1) % len(c.bases)
		c.base, c.seen, c.misses = c.bases[c.bi], false, 0
		return p
	}
	p.Miss = true
	return p
}

// addBases adds candidates not tried yet.
func (c *srvClient) addBases(bs []string) {
	for _, b := range bs {
		if !slices.Contains(c.bases, b) {
			c.bases = append(c.bases, b)
		}
	}
}

func (c *srvClient) ask(now time.Time) probe {
	p := probe{T: now}
	p.Health, _ = c.get("/health")
	if p.Health == 0 || p.Health == http.StatusServiceUnavailable {
		c.lastProps, c.lastStatus, c.caminus = time.Time{}, time.Time{}, false // look again once it is up
		return p
	}
	if now.Sub(c.lastProps) >= 2*time.Second {
		c.lastProps = now
		var b []byte
		if p.PropsCode, b = c.get("/props"); p.PropsCode == http.StatusOK {
			if json.Unmarshal(b, &p.Props) != nil {
				p.PropsCode = -1 // the last one that decoded stays
			} else {
				c.caminus, c.statusPath = p.Props.Engine == engineCaminus, p.Props.Caminus.Status
			}
		}
	}
	var b []byte
	if p.MetricsCode, b = c.get("/metrics"); p.MetricsCode == http.StatusOK {
		p.Metrics = parseMetrics(strings.NewReader(string(b)))
	}
	if p.SlotsCode, b = c.get("/slots"); p.SlotsCode == http.StatusOK && json.Unmarshal(b, &p.Slots) != nil {
		p.SlotsCode, p.Slots = -1, nil
	}
	busy := p.Metrics["requests_processing"]+p.Metrics["requests_deferred"] > 0
	for _, sl := range p.Slots {
		busy = busy || sl.Processing
	}
	// the status: every round while a request runs (and it answers), else at /props' pace
	if c.caminus && (busy && c.statusOK || now.Sub(c.lastStatus) >= 2*time.Second) {
		c.lastStatus = now
		path := c.statusPath
		if !strings.HasPrefix(path, "/") {
			path = camPath
		}
		if p.CamCode, b = c.get(path); p.CamCode == http.StatusOK {
			var err error
			if p.Cam, err = parseCam(b); err != nil {
				p.CamNote = err.Error()
			}
		}
		c.statusOK = p.CamCode == http.StatusOK
	}
	return p
}

// ---------- state update ----------

func (s *serverState) apply(p probe) {
	now := p.T
	if p.Miss {
		return
	}
	s.Base, s.Polled = p.Base, true
	s.Up = p.Health != 0 && p.Health != http.StatusServiceUnavailable
	s.Loading = p.Health == http.StatusServiceUnavailable
	if !s.Up {
		s.HaveSlots, s.Busy, s.NotCam = false, 0, false
		s.Metrics, s.prev, s.Slot.Processing = nil, nil, false
		s.Spec, s.Cam, s.CamCode, s.CamNote = specHist{}, nil, 0, ""
		s.ReadRate.push(0, 240)
		s.GenRate.push(0, 240)
		s.curRead, s.curGen = 0, 0
		return
	}
	if p.PropsCode == http.StatusOK {
		s.Props = p.Props
		if s.NotCam = p.Props.Engine != engineCaminus; s.NotCam { // no status is asked of it
			s.Cam, s.CamCode, s.CamNote = nil, 0, ""
		}
	}
	if p.MetricsCode == http.StatusOK {
		s.Metrics = p.Metrics
		s.Spec.add(p.Metrics)
	} else if p.MetricsCode != 0 {
		s.Metrics = nil
	}
	if p.CamCode != 0 {
		s.Cam, s.CamCode, s.CamNote = p.Cam, p.CamCode, p.CamNote
	}
	if p.SlotsCode != 0 {
		s.HaveSlots = p.SlotsCode == http.StatusOK && len(p.Slots) > 0
	}
	rr, gr := 0.0, 0.0
	reading, writing := false, false
	if p.SlotsCode == http.StatusOK && len(p.Slots) > 0 {
		if s.prev == nil {
			s.prev = map[int]slotPrev{}
		}
		s.NSlots, s.Busy = len(p.Slots), 0
		pick := -1 // keep the shown slot while it works, else the first busy one, else the shown one
		for i, sl := range p.Slots {
			if sl.Processing && sl.ID == s.Slot.ID && pick < 0 {
				pick = i
			}
		}
		for i, sl := range p.Slots {
			if sl.Processing && pick < 0 {
				pick = i
			}
		}
		for i, sl := range p.Slots {
			if sl.ID == s.Slot.ID && pick < 0 {
				pick = i
			}
		}
		if pick < 0 {
			pick = 0
		}
		for _, sl := range p.Slots {
			nt := sl.next()
			if sl.Processing {
				s.Busy++
				if nt.NDecoded == 0 {
					reading = true
				} else {
					writing = true
				}
			}
			start, decAt := time.Time{}, time.Time{}
			if sl.Processing {
				start = now
			}
			pv, ok := s.prev[sl.ID]
			if ok && sl.Processing && sl.Task == pv.task {
				if dt := now.Sub(pv.t).Seconds(); dt > 0 {
					rr += float64(max(0, sl.NProcessed-pv.proc)) / dt
					gr += float64(max(0, nt.NDecoded-pv.dec)) / dt
				}
				if !pv.start.IsZero() {
					start, decAt = pv.start, pv.decAt
				}
			}
			if sl.Processing && nt.NDecoded > 0 && decAt.IsZero() {
				decAt = now
			}
			if ok && pv.busy && (!sl.Processing || sl.Task != pv.task) { // pv's request is over
				e := reqEvent{T: pv.start, Prompt: pv.prompt, Cached: pv.cache, Out: pv.dec, Ctx: pv.prompt + pv.dec, Slot: sl.ID}
				end := pv.t
				if !sl.Processing && sl.Task == pv.task { // the slot still shows the request's final numbers
					e.Out, e.Ctx, end = nt.NDecoded, sl.NPrompt+nt.NDecoded, now
				}
				e.Dur = end.Sub(pv.start)
				if !pv.decAt.IsZero() {
					if d := pv.decAt.Sub(pv.start).Seconds(); d > 0 {
						e.ReadRate = float64(e.Prompt-e.Cached) / d
					}
					if d := end.Sub(pv.decAt).Seconds(); d > 0 {
						e.GenRate = float64(e.Out) / d
					}
				}
				s.finished(e)
			}
			s.prev[sl.ID] = slotPrev{task: sl.Task, proc: sl.NProcessed, dec: nt.NDecoded, prompt: sl.NPrompt, cache: sl.NCache,
				busy: sl.Processing, t: now, start: start, decAt: decAt}
		}
		sl := p.Slots[pick]
		if sl.Processing && (!s.Slot.Processing || sl.Task != s.Slot.Task || sl.ID != s.Slot.ID) {
			s.busySince = s.prev[sl.ID].start // switching to a slot that was already busy keeps its time
		}
		nt := sl.next()
		s.Slot, s.Decoded, s.Remain = sl, nt.NDecoded, nt.NRemain
	}
	s.ReadRate.push(rr, 240)
	s.GenRate.push(gr, 240)
	s.curRead = ema(s.curRead, rr, reading)
	s.curGen = ema(s.curGen, gr, writing)
}

func ema(prev, x float64, active bool) float64 {
	if !active {
		return 0
	}
	if prev == 0 {
		return x
	}
	return prev*0.6 + x*0.4
}

func parseMetrics(r io.Reader) map[string]float64 {
	m := map[string]float64{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		l := sc.Text()
		if strings.HasPrefix(l, "#") {
			continue
		}
		if i := strings.Index(l, "{"); i >= 0 { // only the per-position acceptance, kept as "name.N"
			pos, ok := strings.CutPrefix(l[i:], `{position="`)
			n, v, ok2 := strings.Cut(pos, `"}`)
			if f := strings.Fields(v); ok && ok2 && len(f) == 1 && strings.TrimPrefix(l[:i], "llamacpp:") == specPosKey {
				if x, err := strconv.ParseFloat(f[0], 64); err == nil {
					m[specPosKey+"."+n] = x
				}
			}
			continue
		}
		f := strings.Fields(l)
		if len(f) == 2 {
			if v, err := strconv.ParseFloat(f[1], 64); err == nil {
				m[strings.TrimPrefix(f[0], "llamacpp:")] = v
			}
		}
	}
	return m
}

func (st *state) pollServer(c *srvClient) {
	for {
		p := c.round(time.Now())
		st.mu.Lock()
		st.srv.apply(p)
		st.mu.Unlock()
		time.Sleep(500 * time.Millisecond)
	}
}

// ---------- speculative decoding ----------

const specPosKey = "spec_decode_num_accepted_tokens_per_pos_total"

// specWindow is how many polls that drafted count as "recent": ~60 s while drafting all the time.
const specWindow = 120

type specSample struct{ acc, tok, drafts float64 }

// specHist keeps the spec_decode counters' deltas between polls that drafted.
type specHist struct {
	prev    [3]float64 // accepted, draft tokens, drafts at the last scrape
	seen    bool
	Samples []specSample // oldest first, at most specWindow
}

func specCounters(mt map[string]float64) [3]float64 {
	return [3]float64{mt["spec_decode_num_accepted_tokens_total"], mt["spec_decode_num_draft_tokens_total"], mt["spec_decode_num_drafts_total"]}
}

func (h *specHist) add(mt map[string]float64) {
	if _, ok := mt["spec_decode_num_draft_tokens_total"]; !ok {
		*h = specHist{}
		return
	}
	c := specCounters(mt)
	if c[0] < h.prev[0] || c[1] < h.prev[1] || c[2] < h.prev[2] { // the server restarted: older samples are another run
		h.Samples, h.seen = nil, false
	}
	if h.seen {
		// counters print with 6 significant digits (%g): deltas of large ones are coarse, so keep them consistent
		d := specSample{acc: max(0, c[0]-h.prev[0]), tok: c[1] - h.prev[1], drafts: max(0, c[2]-h.prev[2])}
		if d.tok > 0 {
			d.acc = min(d.acc, d.tok)
			h.Samples = append(h.Samples, d)
			if len(h.Samples) > specWindow {
				h.Samples = h.Samples[len(h.Samples)-specWindow:]
			}
		}
	}
	h.prev, h.seen = c, true
}

// recent is the acceptance rate and accepted tokens per draft over the window (ok false without samples).
func (h *specHist) recent() (rate, perDraft float64, ok bool) {
	var a, t, d float64
	for _, x := range h.Samples {
		a, t, d = a+x.acc, t+x.tok, d+x.drafts
	}
	if t <= 0 {
		return 0, 0, false
	}
	if d > 0 {
		perDraft = a / d
	}
	return a / t, perDraft, true
}

// rates is the acceptance rate of each sample, for the sparkline.
func (h *specHist) rates() []float64 {
	v := make([]float64, len(h.Samples))
	for i, x := range h.Samples {
		v[i] = x.acc / x.tok
	}
	return v
}

// specPos is the share of drafts whose token at each position was accepted, since server start.
func specPos(mt map[string]float64) []float64 {
	d := mt["spec_decode_num_drafts_total"]
	if d <= 0 {
		return nil
	}
	var v []float64
	for i := 0; ; i++ {
		x, ok := mt[specPosKey+"."+strconv.Itoa(i)]
		if !ok {
			break
		}
		v = append(v, min(1, x/d))
	}
	for len(v) > 0 && v[len(v)-1] == 0 {
		v = v[:len(v)-1]
	}
	return v
}

// drafters names the speculative decoding from a slot's speculative.types: Caminus says "draft-dflash" ("dflash")
// with its drafter and "none" without. known is false for an empty string (a slot that never ran has no params);
// "" with known means none.
func drafters(types string) (name string, known bool) {
	switch types = strings.TrimSpace(types); types {
	case "":
		return "", false
	case "none":
		return "", true
	}
	return strings.TrimPrefix(types, "draft-"), true
}
