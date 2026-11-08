// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// OpenRGB: lights the machine's RGB devices by what the agent is doing, through OpenRGB's SDK server (TCP, default
// 127.0.0.1:6742, backwards-compatible SDK v3). It is off unless the config says "openrgb": {"enabled": true}.
// Active per-LED modes work without a mode change. switch_modes opts into temporary mode changes; the original
// colors and complete mode payload are restored on exit. No zone sizing, profile saves or firmware saves.
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

// look is how a phase shows. The renderer samples rings circularly and strips linearly. Saturated colors keep
// states recognizable on LEDs; a pastel palette would come out nearly white.
type look struct {
	c      ledRGB
	fx     string
	period time.Duration
}

var looks = map[string]look{
	phOffline: {ledRGB{60, 0, 0}, "", 0},
	phLoading: {ledRGB{255, 70, 0}, "orbit", 2 * time.Second},
	phIdle:    {ledRGB{0, 40, 60}, "", 0},
	phQuiet:   {ledRGB{0, 40, 60}, "", 0},
	phBetween: {ledRGB{0, 60, 255}, "", 0},
	phWaiting: {ledRGB{0, 220, 180}, "breathe", 3 * time.Second},
	phReading: {ledRGB{255, 170, 0}, "sweep", 1800 * time.Millisecond},
	phWriting: {ledRGB{0, 255, 40}, "chase", 1200 * time.Millisecond},
	phTool:    {ledRGB{255, 100, 0}, "gloss", 3 * time.Second},
	phCompact: {ledRGB{255, 30, 0}, "converge", 1600 * time.Millisecond},
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

// ---------- config ----------

type rgbCfg struct {
	Enabled     bool              `json:"enabled"`
	Host        string            `json:"host"`
	Devices     []string          `json:"devices"`         // name substrings, case-insensitive; empty = every usable per-LED device
	Brightness  float64           `json:"brightness"`      // 0-1
	Colors      map[string]string `json:"colors"`          // phase -> #rrggbb
	SwitchModes bool              `json:"switch_modes"`    // opt in: temporarily select a per-LED mode
	FPS         int               `json:"fps"`             // 1-60; 20 by default
	Speed       float64           `json:"speed"`           // 0.1-4; 1 by default
	FadeMS      int               `json:"fade_ms"`         // 0-2000; 0 disables crossfades
	Zones       map[string]string `json:"zones,omitempty"` // zone name -> ring, reverse-ring, strip
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
	pktUpdateMode = 1101
	orgbProto     = 3 // request this layout even from newer servers (v6 changes device IDs)
	modePerLED    = 1 << 5
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
	for b = append(b, data...); len(b) > 0; {
		n, err := o.c.Write(b)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		b = b[n:]
	}
	return nil
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
	for range 64 {
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
	return nil, errors.New("too many unrelated OpenRGB packets")
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
	name, loc, serial string
	kind              int
	mode              string // active mode's name
	active            int
	modes             []orgbMode
	zones             []int // LEDs per zone
	zoneNames         []string
	colors            []ledRGB
}

type orgbMode struct {
	name             string
	flags, colorMode uint32
	fieldsAt         int
	raw              []byte // the entire wire payload, including speed, brightness and mode colors
}

func (m orgbMode) perLED() bool { return m.flags&modePerLED != 0 && m.colorMode == 1 }

func (ci ctrlInfo) selectedMode(switchModes bool) int {
	if len(ci.modes) == 0 {
		return -1
	}
	if ci.modes[ci.active].perLED() {
		return ci.active
	}
	if switchModes {
		for _, name := range []string{"Direct", "Custom", "Static"} {
			for i, m := range ci.modes {
				if strings.EqualFold(m.name, name) && m.perLED() {
					return i
				}
			}
		}
	}
	return -1
}

// A rescan may change indices, replace a controller, or resize its zones. Only a matching identity AND layout
// can use an old snapshot; otherwise even a correctly re-discovered index could receive the wrong LED count.
func (ci ctrlInfo) key() string {
	k := fmt.Sprintf("%q|%q|%q|%d|%d", ci.name, ci.serial, ci.loc, ci.kind, len(ci.colors))
	for i, n := range ci.zones {
		k += fmt.Sprintf("|%q:%d", ci.zoneNames[i], n)
	}
	for _, m := range ci.modes {
		k += fmt.Sprintf("|%q:%d:%d", m.name, m.flags, m.colorMode)
	}
	return k
}

func modePacket(index int, m orgbMode, fullBrightness bool) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(8+len(m.raw)))
	b = binary.LittleEndian.AppendUint32(b, uint32(index))
	b = append(b, m.raw...)
	if fullBrightness && m.flags&16 != 0 {
		fields := b[8+m.fieldsAt:]
		binary.LittleEndian.PutUint32(fields[9*4:], binary.LittleEndian.Uint32(fields[5*4:]))
	}
	return b
}

// parseController reads a complete REQUEST_CONTROLLER_DATA reply. Never use partial descriptions to write hardware.
func parseController(b []byte, proto uint32) (ctrlInfo, error) {
	var ci ctrlInfo
	r := &pkReader{b: b}
	if r.u32() != len(b) || proto < 3 || proto > 5 {
		return ci, errors.New("invalid OpenRGB controller size or version")
	}
	ci.kind = r.u32()
	ci.name = r.str()
	if proto >= 1 {
		r.str() // vendor
	}
	r.str() // description
	r.str() // version
	ci.serial = r.str()
	ci.loc = r.str()
	nm, active := r.u16(), r.u32()
	ci.active = active
	for i := 0; i < nm && !r.bad; i++ {
		start := len(b) - len(r.b)
		name := r.str()
		fieldsAt := len(b) - len(r.b) - start
		r.u32() // value
		flags := uint32(r.u32())
		r.take(2 * 4) // speed min/max
		if proto >= 3 {
			r.take(2 * 4) // brightness min/max
		}
		r.take(3 * 4) // colors min/max, speed
		if proto >= 3 {
			r.take(4) // brightness
		}
		r.u32() // direction
		colorMode := uint32(r.u32())
		r.take(4 * r.u16()) // mode colors
		if r.bad {
			break
		}
		ci.modes = append(ci.modes, orgbMode{name, flags, colorMode, fieldsAt, append([]byte(nil), b[start:len(b)-len(r.b)]...)})
		if i == active {
			ci.mode = name
		}
	}
	nz := r.u16()
	for i := 0; i < nz && !r.bad; i++ {
		ci.zoneNames = append(ci.zoneNames, r.str())
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
	if proto >= 5 {
		for n := r.u16(); n > 0 && !r.bad; n-- {
			r.str() // alternate LED display names
		}
		r.u32() // controller flags
	}
	if r.bad || len(r.b) != 0 || (len(ci.modes) != 0 && active >= len(ci.modes)) || nl != nc {
		return ci, fmt.Errorf("invalid OpenRGB controller data (%d bytes)", len(b))
	}
	total := 0
	for _, n := range ci.zones {
		total += n
	}
	if total > nc {
		return ci, errors.New("OpenRGB zones exceed the device's LED count")
	}
	return ci, nil
}

// ---------- driver ----------

type orgbDev struct {
	idx  uint32
	key  string // physical identity and LED/mode layout, stable across index changes
	info ctrlInfo
	mode int
	sent []ledRGB
	anim rgbAnimator
}

type orgbSnapshot struct {
	info                 ctrlInfo
	touched, modeChanged bool
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
	orig   map[string]*orgbSnapshot // first-seen colors and mode; owned by run's goroutine
}

func newRGB(cfg rgbCfg, st *state) *rgbLights {
	cfg = cfg.normalized()
	if cfg.Host == "" {
		cfg.Host = defaultConfig.OpenRGB.Host
	}
	return &rgbLights{cfg: cfg, st: st, quit: make(chan struct{}), done: make(chan struct{}),
		status: "connecting to OpenRGB", orig: map[string]*orgbSnapshot{}}
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
			l.set(fmt.Sprintf("OpenRGB not answering on %s: %v", shortHost(l.cfg.Host), err))
		case len(devs) == 0:
			l.set("OpenRGB: no per-LED device ready (select Direct mode or enable switch_modes)")
			wait = 30 * time.Second // a mode change sends no list update; look again now and then
		default:
			if l.drive(o, devs) {
				o.c.Close()
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
				if o, devs, err := l.connect(); o != nil {
					if err == nil {
						changed, replies := listen(o)
						l.restore(o, devs, replies, changed)
					}
					o.c.Close()
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
	if err != nil {
		return o, nil, fmt.Errorf("OpenRGB protocol: %w", err)
	}
	if len(b) != 4 || binary.LittleEndian.Uint32(b) < orgbProto {
		return o, nil, errors.New("OpenRGB SDK v3 or newer is required")
	}
	o.proto = orgbProto
	if err := o.send(0, pktClientName, []byte("Caminus Top\x00")); err != nil {
		return o, nil, err
	}
	if err := o.send(0, pktCount, nil); err != nil {
		return o, nil, err
	}
	if b, err = o.recv(pktCount); err != nil {
		return o, nil, fmt.Errorf("OpenRGB count: %w", err)
	}
	if len(b) != 4 || binary.LittleEndian.Uint32(b) > 1024 {
		return o, nil, errors.New("invalid OpenRGB controller count")
	}
	var devs []orgbDev
	seen := map[string]bool{}
	for i := uint32(0); i < binary.LittleEndian.Uint32(b); i++ {
		if err := o.send(i, pktData, u32(o.proto)); err != nil {
			return o, nil, err
		}
		d, err := o.recv(pktData)
		if err != nil {
			return o, nil, err
		}
		ci, err := parseController(d, o.proto)
		if err != nil {
			return o, nil, err
		}
		mode := ci.selectedMode(l.cfg.SwitchModes)
		if mode < 0 || len(ci.colors) == 0 || !l.cfg.wants(ci.name) {
			continue
		}
		dv := orgbDev{idx: i, key: ci.key(), info: ci, mode: mode}
		if seen[dv.key] {
			return o, nil, errors.New("OpenRGB devices have duplicate identities; refusing ambiguous restoration")
		}
		seen[dv.key] = true
		if l.orig[dv.key] == nil {
			l.orig[dv.key] = &orgbSnapshot{info: ci}
		}
		devs = append(devs, dv)
	}
	_ = c.SetDeadline(time.Time{})
	return o, devs, nil
}

// drive sends frames until quit (true, after restoring the colors), a device list change or an error.
func (l *rgbLights) drive(o *orgbConn, devs []orgbDev) bool {
	changed, replies := listen(o)
	for _, d := range devs {
		select {
		case <-changed:
			return false
		case <-l.quit:
			l.restore(o, devs, replies, changed)
			return true
		default:
		}
		if l.cfg.SwitchModes {
			s := l.orig[d.key]
			s.touched, s.modeChanged, l.dirty = true, true, true
			_ = o.c.SetWriteDeadline(time.Now().Add(time.Second))
			if o.send(d.idx, pktUpdateMode, modePacket(d.mode, d.info.modes[d.mode], true)) != nil {
				return false
			}
		}
	}
	tk := time.NewTicker(time.Second / time.Duration(l.cfg.FPS))
	defer tk.Stop()
	last := ""
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
			for i := range devs {
				d := &devs[i]
				cs := d.anim.frame(l.cfg, d.info, ph, now)
				if equalRGB(cs, d.sent) {
					continue
				}
				_ = o.c.SetWriteDeadline(now.Add(time.Second)) // a hung OpenRGB must not block quitting
				select {
				case <-changed:
					return false
				default:
				}
				l.orig[d.key].touched, l.dirty = true, true
				if o.send(d.idx, pktUpdateLEDs, ledsPacket(cs)) != nil {
					return false
				}
				d.sent = cs
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
	defer o.c.Close()
	_ = o.c.SetWriteDeadline(time.Now().Add(time.Second))
	for _, d := range devs {
		select {
		case <-changed:
			return // old indices are no longer safe, even for restoring
		default:
		}
		s := l.orig[d.key]
		if !s.touched {
			continue
		}
		if o.send(d.idx, pktUpdateLEDs, ledsPacket(s.info.colors)) != nil {
			return
		}
		if s.modeChanged {
			if o.send(d.idx, pktUpdateMode, modePacket(s.info.active, s.info.modes[s.info.active], false)) != nil {
				return
			}
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
