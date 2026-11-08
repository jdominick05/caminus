// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// OpenRGB: lights the machine's RGB devices by what the agent is doing, through OpenRGB's SDK server (TCP, default
// 127.0.0.1:6742, protocol 1-5). It is the one connection llm-top writes over, and it is off unless the config
// says "openrgb": {"enabled": true}. Only devices already in Direct mode are driven (no mode switches, which on
// some hardware write to flash), and the colors they had when first seen are put back on exit.
package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------- phases ----------

// Phases of the lights; agentPhase returns the agent ones, rgbPhase adds the server's.
const (
	phOffline = "offline"
	phLoading = "loading"
	phIdle    = "idle" // server up, no agent active
	phQuiet   = "quiet"
	phBetween = "between"
	phWaiting = "waiting"
	phReading = "reading"
	phWriting = "writing"
	phTool    = "tool"
	phCompact = "compacting"
)

// rgbPhase is the lights' phase: the server's state first, then the session the TUI shows without a tab pick.
// Callers hold st.mu.
func (st *state) rgbPhase() string {
	srv := &st.srv
	switch {
	case srv.Loading:
		return phLoading
	case srv.offline() != "":
		return phOffline
	case srv.compacting():
		return phCompact
	}
	if i := st.shown(nil); i >= 0 {
		ph, _ := agentPhase(st.sessions[i], srv.busy(), srv.Decoded)
		return ph
	}
	switch srv.phase() {
	case "reading":
		return phReading
	case "writing":
		return phWriting
	}
	return phIdle
}

// ---------- looks ----------

type ledRGB struct{ r, g, b uint8 }

// look is how a phase shows: fx "" is steady, "breathe" pulses every LED, "chase" runs a bright spot along each
// zone over a dim base. LEDs want saturated colors; the TUI's pastel palette would come out nearly white.
type look struct {
	c      ledRGB
	fx     string
	period time.Duration
}

var looks = map[string]look{
	phOffline: {ledRGB{60, 0, 0}, "", 0},
	phLoading: {ledRGB{255, 70, 0}, "breathe", 2 * time.Second},
	phIdle:    {ledRGB{0, 40, 60}, "", 0},
	phQuiet:   {ledRGB{0, 40, 60}, "", 0},
	phBetween: {ledRGB{0, 60, 255}, "", 0},
	phWaiting: {ledRGB{0, 220, 180}, "breathe", 3 * time.Second},
	phReading: {ledRGB{255, 170, 0}, "breathe", time.Second},
	phWriting: {ledRGB{0, 255, 40}, "chase", 1200 * time.Millisecond},
	phTool:    {ledRGB{255, 100, 0}, "", 0},
	phCompact: {ledRGB{255, 30, 0}, "chase", 700 * time.Millisecond},
}

func (c ledRGB) scale(k float64) ledRGB {
	f := func(v uint8) uint8 { return uint8(math.Round(float64(v) * min(max(k, 0), 1))) }
	return ledRGB{f(c.r), f(c.g), f(c.b)}
}

// parseHex reads "#rrggbb" (the # is optional).
func parseHex(s string) (ledRGB, bool) {
	s = strings.TrimPrefix(strings.TrimSpace(s), "#")
	v, err := strconv.ParseUint(s, 16, 32)
	if len(s) != 6 || err != nil {
		return ledRGB{}, false
	}
	return ledRGB{uint8(v >> 16), uint8(v >> 8), uint8(v)}, true
}

// colors gives n LEDs at time t. zones are the zones' LED counts; LEDs past them count as one more zone.
func (l look) colors(zones []int, n int, t time.Duration, br float64) []ledRGB {
	out := make([]ledRGB, n)
	ph := 0.0
	if l.period > 0 {
		ph = math.Mod(float64(t)/float64(l.period), 1)
	}
	switch l.fx {
	case "breathe":
		c := l.c.scale(br * (0.15 + 0.85*(0.5-0.5*math.Cos(2*math.Pi*ph))))
		for i := range out {
			out[i] = c
		}
	case "chase":
		i := 0
		for _, z := range append(append([]int(nil), zones...), n) {
			z = min(z, n-i)
			for j := 0; j < z; j++ {
				d := math.Abs(float64(j) + 0.5 - ph*float64(z))
				d = min(d, float64(z)-d) // the spot wraps around
				out[i+j] = l.c.scale(br * max(0.12, 1-d/2.5))
			}
			i += z
		}
	default:
		c := l.c.scale(br)
		for i := range out {
			out[i] = c
		}
	}
	return out
}

// ---------- config ----------

type rgbCfg struct {
	Enabled    bool              `json:"enabled"`
	Host       string            `json:"host"`
	Devices    []string          `json:"devices"`    // name substrings, case-insensitive; empty = every Direct-mode device
	Brightness float64           `json:"brightness"` // 0-1
	Colors     map[string]string `json:"colors"`     // phase -> #rrggbb
}

func (c rgbCfg) look(ph string) look {
	l := looks[ph]
	if v, ok := parseHex(c.Colors[ph]); ok {
		l.c = v
	}
	return l
}

func (c rgbCfg) wants(name string) bool {
	if len(c.Devices) == 0 {
		return true
	}
	for _, d := range c.Devices {
		if d != "" && strings.Contains(strings.ToLower(name), strings.ToLower(d)) {
			return true
		}
	}
	return false
}

// ---------- SDK protocol ----------

// Packet ids from OpenRGB's Documentation/OpenRGBSDK.md.
const (
	pktCount      = 0
	pktData       = 1
	pktProto      = 40
	pktClientName = 50
	pktListChange = 100
	pktUpdateLEDs = 1050
	orgbProto     = 5 // highest version parsed here; 6 changes device ids and the controller layout
)

var errListChanged = errors.New("OpenRGB device list changed")

type orgbConn struct {
	c     net.Conn
	proto uint32
}

func (o *orgbConn) send(dev, id uint32, data []byte) error {
	b := make([]byte, 16, 16+len(data))
	copy(b, "ORGB")
	binary.LittleEndian.PutUint32(b[4:], dev)
	binary.LittleEndian.PutUint32(b[8:], id)
	binary.LittleEndian.PutUint32(b[12:], uint32(len(data)))
	_, err := o.c.Write(append(b, data...))
	return err
}

func (o *orgbConn) read() (id uint32, data []byte, err error) {
	h := make([]byte, 16)
	if _, err = io.ReadFull(o.c, h); err != nil {
		return 0, nil, err
	}
	if string(h[:4]) != "ORGB" {
		return 0, nil, errors.New("not an OpenRGB SDK server")
	}
	n := binary.LittleEndian.Uint32(h[12:])
	if n > 16<<20 {
		return 0, nil, fmt.Errorf("OpenRGB packet of %d bytes", n)
	}
	data = make([]byte, n)
	_, err = io.ReadFull(o.c, data)
	return binary.LittleEndian.Uint32(h[8:]), data, err
}

// recv waits for a reply with id want; a device list change on the way aborts (indices may have moved).
func (o *orgbConn) recv(want uint32) ([]byte, error) {
	for {
		id, b, err := o.read()
		switch {
		case err != nil:
			return nil, err
		case id == pktListChange:
			return nil, errListChanged
		case id == want:
			return b, nil
		}
	}
}

func u32(v uint32) []byte { return binary.LittleEndian.AppendUint32(nil, v) }

// ledsPacket is UPDATELEDS' payload: data_size, num_colors, then R, G, B, 0 per LED.
func ledsPacket(cs []ledRGB) []byte {
	b := make([]byte, 6+4*len(cs))
	binary.LittleEndian.PutUint32(b, uint32(len(b)))
	binary.LittleEndian.PutUint16(b[4:], uint16(len(cs)))
	for i, c := range cs {
		b[6+4*i], b[7+4*i], b[8+4*i] = c.r, c.g, c.b
	}
	return b
}

// pkReader reads the SDK's little-endian fields; past the end it yields zeros and sets bad.
type pkReader struct {
	b   []byte
	bad bool
}

func (r *pkReader) take(n int) []byte {
	if n < 0 || n > len(r.b) {
		r.bad, r.b = true, nil
		return make([]byte, max(n, 0))
	}
	v := r.b[:n]
	r.b = r.b[n:]
	return v
}
func (r *pkReader) u16() int { return int(binary.LittleEndian.Uint16(r.take(2))) }
func (r *pkReader) u32() int { return int(binary.LittleEndian.Uint32(r.take(4))) }
func (r *pkReader) str() string {
	return strings.TrimRight(string(r.take(r.u16())), "\x00")
}

type ctrlInfo struct {
	name, loc string
	mode      string // active mode's name
	zones     []int  // LEDs per zone
	colors    []ledRGB
}

// parseController reads a REQUEST_CONTROLLER_DATA reply (data_size first) up to the colors.
func parseController(b []byte, proto uint32) (ctrlInfo, error) {
	var ci ctrlInfo
	r := &pkReader{b: b}
	r.u32() // data_size
	r.u32() // type
	ci.name = r.str()
	if proto >= 1 {
		r.str() // vendor
	}
	r.str() // description
	r.str() // version
	r.str() // serial
	ci.loc = r.str()
	nm, active := r.u16(), r.u32()
	for i := 0; i < nm && !r.bad; i++ {
		name := r.str()
		r.take(4 * 4) // value, flags, speed min/max
		if proto >= 3 {
			r.take(2 * 4) // brightness min/max
		}
		r.take(3 * 4) // colors min/max, speed
		if proto >= 3 {
			r.take(4) // brightness
		}
		r.take(2 * 4)       // direction, color mode
		r.take(4 * r.u16()) // mode colors
		if i == active {
			ci.mode = name
		}
	}
	nz := r.u16()
	for i := 0; i < nz && !r.bad; i++ {
		r.str()       // name
		r.take(3 * 4) // type, leds min/max
		ci.zones = append(ci.zones, r.u32())
		r.take(r.u16()) // matrix map
		if proto >= 4 {
			ns := r.u16()
			for j := 0; j < ns && !r.bad; j++ {
				r.str()
				r.take(3 * 4) // type, start, count
			}
		}
		if proto >= 5 {
			r.take(4) // zone flags
		}
	}
	nl := r.u16()
	for i := 0; i < nl && !r.bad; i++ {
		r.str()
		r.take(4) // led value
	}
	nc := r.u16()
	for i := 0; i < nc && !r.bad; i++ {
		c := r.take(4)
		ci.colors = append(ci.colors, ledRGB{c[0], c[1], c[2]})
	}
	if r.bad {
		return ci, fmt.Errorf("OpenRGB controller data cut short (%d bytes)", len(b))
	}
	return ci, nil
}

// ---------- driver ----------

type orgbDev struct {
	idx   uint32
	key   string // name + location, stable across reconnects
	zones []int
	sent  []ledRGB
}

// rgbLights runs the OpenRGB connection; status is shown in AGENTS.
type rgbLights struct {
	cfg  rgbCfg
	st   *state
	quit chan struct{}
	done chan struct{}

	dirty bool // a frame was sent: there is something to restore (run's goroutine only)

	mu     sync.Mutex
	status string
	orig   map[string][]ledRGB // colors when first seen, put back on exit
}

func newRGB(cfg rgbCfg, st *state) *rgbLights {
	if cfg.Host == "" {
		cfg.Host = defaultConfig.OpenRGB.Host
	}
	return &rgbLights{cfg: cfg, st: st, quit: make(chan struct{}), done: make(chan struct{}),
		status: "connecting to OpenRGB", orig: map[string][]ledRGB{}}
}

func (l *rgbLights) Status() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.status
}

func (l *rgbLights) set(s string) {
	l.mu.Lock()
	l.status = s
	l.mu.Unlock()
}

// stop puts the original colors back and closes the connection; it waits at most 4 s, so quitting never hangs on
// an OpenRGB server that stopped answering.
func (l *rgbLights) stop() {
	close(l.quit)
	select {
	case <-l.done:
	case <-time.After(4 * time.Second):
	}
}

func (l *rgbLights) run() {
	defer close(l.done)
	for {
		o, devs, err := l.connect()
		wait := 5 * time.Second
		switch {
		case errors.Is(err, errListChanged):
			wait = time.Second
		case err != nil:
			l.set(fmt.Sprintf("OpenRGB not answering on %s", shortHost(l.cfg.Host)))
		case len(devs) == 0:
			l.set("OpenRGB: no device in Direct mode")
			wait = 30 * time.Second // a mode change sends no list update; look again now and then
		default:
			if l.drive(o, devs) {
				return
			}
			wait = time.Second
		}
		if o != nil {
			o.c.Close()
		}
		select {
		case <-l.quit:
			if l.dirty { // quit between connections: the devices still show the last frame
				if o, devs, err := l.connect(); err == nil {
					changed, replies := listen(o)
					l.restore(o, devs, replies, changed)
				}
			}
			return
		case <-time.After(wait):
		}
	}
}

// connect does the handshake and lists the devices to drive.
func (l *rgbLights) connect() (*orgbConn, []orgbDev, error) {
	c, err := net.DialTimeout("tcp", l.cfg.Host, time.Second)
	if err != nil {
		return nil, nil, err
	}
	o := &orgbConn{c: c}
	_ = c.SetDeadline(time.Now().Add(2 * time.Second))
	if err := o.send(0, pktProto, u32(orgbProto)); err != nil {
		return o, nil, err
	}
	b, err := o.recv(pktProto) // a protocol 0 server never answers: the deadline ends it
	if err != nil || len(b) < 4 {
		return o, nil, fmt.Errorf("OpenRGB protocol: %w", err)
	}
	o.proto = min(binary.LittleEndian.Uint32(b), orgbProto)
	if err := o.send(0, pktClientName, []byte("llm-top\x00")); err != nil {
		return o, nil, err
	}
	if err := o.send(0, pktCount, nil); err != nil {
		return o, nil, err
	}
	if b, err = o.recv(pktCount); err != nil || len(b) < 4 {
		return o, nil, fmt.Errorf("OpenRGB count: %w", err)
	}
	var devs []orgbDev
	for i := uint32(0); i < binary.LittleEndian.Uint32(b); i++ {
		if err := o.send(i, pktData, u32(o.proto)); err != nil {
			return o, nil, err
		}
		d, err := o.recv(pktData)
		if err != nil {
			return o, nil, err
		}
		ci, err := parseController(d, o.proto)
		if err != nil || !strings.EqualFold(ci.mode, "Direct") || len(ci.colors) == 0 || !l.cfg.wants(ci.name) {
			continue
		}
		dv := orgbDev{idx: i, key: ci.name + "|" + ci.loc, zones: ci.zones}
		l.mu.Lock()
		if l.orig[dv.key] == nil {
			l.orig[dv.key] = ci.colors
		}
		l.mu.Unlock()
		devs = append(devs, dv)
	}
	_ = c.SetDeadline(time.Time{})
	return o, devs, nil
}

// drive sends frames until quit (true, after restoring the colors), a device list change or an error.
func (l *rgbLights) drive(o *orgbConn, devs []orgbDev) bool {
	changed, replies := listen(o)
	tk := time.NewTicker(80 * time.Millisecond) // ~12 fps: plenty for LEDs, gentle on SMBus RAM sticks
	defer tk.Stop()
	start, last := time.Now(), ""
	for {
		select {
		case <-l.quit:
			l.restore(o, devs, replies, changed)
			return true
		case <-changed:
			return false
		case now := <-tk.C:
			l.st.mu.Lock()
			ph := l.st.rgbPhase()
			l.st.mu.Unlock()
			if ph != last {
				l.set(fmt.Sprintf("%s on %d device%s", ph, len(devs), map[bool]string{true: "s"}[len(devs) != 1]))
				last = ph
			}
			lk := l.cfg.look(ph)
			for i := range devs {
				d := &devs[i]
				l.mu.Lock()
				n := len(l.orig[d.key])
				l.mu.Unlock()
				cs := lk.colors(d.zones, n, now.Sub(start), l.cfg.Brightness)
				if equalRGB(cs, d.sent) {
					continue
				}
				_ = o.c.SetWriteDeadline(now.Add(time.Second)) // a hung OpenRGB must not block quitting
				if o.send(d.idx, pktUpdateLEDs, ledsPacket(cs)) != nil {
					return false
				}
				d.sent, l.dirty = cs, true
			}
		}
	}
}

// listen reads everything the server sends from here on: changed closes on a device list change or a dead
// connection, replies carries the ids of other packets.
func listen(o *orgbConn) (changed chan struct{}, replies chan uint32) {
	changed, replies = make(chan struct{}), make(chan uint32, 4)
	go func() {
		defer close(changed)
		for {
			id, _, err := o.read()
			if err != nil || id == pktListChange {
				return
			}
			select {
			case replies <- id:
			default:
			}
		}
	}()
	return changed, replies
}

// restore puts each device's first-seen colors back, then waits for one round trip so OpenRGB has read them
// before the socket closes.
func (l *rgbLights) restore(o *orgbConn, devs []orgbDev, replies <-chan uint32, changed <-chan struct{}) {
	_ = o.c.SetWriteDeadline(time.Now().Add(time.Second))
	for _, d := range devs {
		l.mu.Lock()
		cs := l.orig[d.key]
		l.mu.Unlock()
		if o.send(d.idx, pktUpdateLEDs, ledsPacket(cs)) != nil {
			return
		}
	}
	if o.send(0, pktCount, nil) != nil {
		return
	}
	t := time.After(time.Second)
	for {
		select {
		case id := <-replies:
			if id == pktCount {
				o.c.Close()
				return
			}
		case <-changed:
			o.c.Close()
			return
		case <-t:
			o.c.Close()
			return
		}
	}
}

func equalRGB(a, b []ledRGB) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
