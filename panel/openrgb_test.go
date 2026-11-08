// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/binary"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---------- mock OpenRGB SDK server ----------

type mockDev struct {
	name, mode string
	zones      []int
	colors     []ledRGB
}

// ctrlData encodes a REQUEST_CONTROLLER_DATA reply at protocol 5, shaped like the real devices: a Direct mode and
// a hardware one, zones with a matrix map and segments.
func ctrlData(d mockDev) []byte {
	var b []byte
	u16 := func(v int) { b = binary.LittleEndian.AppendUint16(b, uint16(v)) }
	u32 := func(v int) { b = binary.LittleEndian.AppendUint32(b, uint32(v)) }
	str := func(s string) { u16(len(s) + 1); b = append(append(b, s...), 0) }
	u32(1) // type
	str(d.name)
	str("Vendor")
	str("desc")
	str("1.0")
	str("serial")
	str("HID: " + d.name)
	modes := []string{"Direct", "Static"}
	u16(len(modes))
	for i, m := range modes {
		if m == d.mode {
			u32(i)
		}
	}
	for _, m := range modes {
		str(m)
		for j := range 12 { // value, flags, speed min/max, brightness min/max, colors min/max, speed, brightness, direction, color mode
			v := 0
			if m == "Direct" {
				switch j {
				case 1:
					v = modePerLED
				case 11:
					v = 1
				}
			}
			u32(v)
		}
		u16(1) // one mode color
		u32(0xff)
	}
	u16(len(d.zones))
	for i, n := range d.zones {
		str("zone")
		u32(1)
		u32(n)
		u32(n)
		u32(n)
		if i == 0 { // a 1x2 matrix map
			u16(16)
			u32(1)
			u32(2)
			u32(0)
			u32(1)
		} else {
			u16(0)
		}
		u16(1) // one segment
		str("seg")
		u32(0)
		u32(0)
		u32(n)
		u32(0) // zone flags
	}
	u16(len(d.colors))
	for range d.colors {
		str("led")
		u32(0)
	}
	u16(len(d.colors))
	for _, c := range d.colors {
		b = append(b, c.r, c.g, c.b, 0)
	}
	u16(0) // led display names
	u32(0) // flags
	return append(binary.LittleEndian.AppendUint32(nil, uint32(len(b)+4)), b...)
}

// A separately encoded SDK v3 response: a v6 server serializes this older layout when asked for v3.
func ctrlData3(d mockDev) []byte {
	var b []byte
	u16 := func(v int) { b = binary.LittleEndian.AppendUint16(b, uint16(v)) }
	u32 := func(v int) { b = binary.LittleEndian.AppendUint32(b, uint32(v)) }
	str := func(s string) { u16(len(s) + 1); b = append(append(b, s...), 0) }
	u32(0)
	for _, s := range []string{d.name, "Vendor", "desc", "1.0", "serial", "HID: " + d.name} {
		str(s)
	}
	u16(2)
	if d.mode == "Direct" {
		u32(0)
	} else {
		u32(1)
	}
	for _, name := range []string{"Direct", "Static"} {
		str(name)
		flags, colorMode := 0, 3
		if name == "Direct" {
			flags, colorMode = modePerLED|16, 1
		}
		// The nondefault speed, brightness and mode color must survive restoration byte for byte.
		for _, v := range []int{7, flags, 1, 100, 0, 100, 0, 2, 37, 63, 1, colorMode} {
			u32(v)
		}
		u16(1)
		u32(0x123456)
	}
	u16(len(d.zones))
	for i, n := range d.zones {
		str("JARGB " + string(rune('1'+i)))
		for _, v := range []int{1, 0, 240, n} {
			u32(v)
		}
		u16(0)
	}
	u16(len(d.colors))
	for range d.colors {
		str("LED")
		u32(0)
	}
	u16(len(d.colors))
	for _, c := range d.colors {
		b = append(b, c.r, c.g, c.b, 0)
	}
	return append(binary.LittleEndian.AppendUint32(nil, uint32(len(b)+4)), b...)
}

type sdkPkt struct {
	dev, id uint32
	data    []byte
}

type mockSDK struct {
	ln    net.Listener
	data  [][]byte
	mu    sync.Mutex
	got   []sdkPkt
	conns []net.Conn
}

func newMockSDK(t *testing.T, devs ...mockDev) *mockSDK {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	m := &mockSDK{ln: ln}
	for _, d := range devs {
		m.data = append(m.data, ctrlData3(d))
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			m.mu.Lock()
			m.conns = append(m.conns, c)
			m.mu.Unlock()
			go m.serve(c)
		}
	}()
	return m
}

func (m *mockSDK) serve(c net.Conn) {
	defer c.Close()
	o := &orgbConn{c: c}
	for {
		h := make([]byte, 16)
		if _, err := io.ReadFull(c, h); err != nil {
			return
		}
		p := sdkPkt{binary.LittleEndian.Uint32(h[4:]), binary.LittleEndian.Uint32(h[8:]), make([]byte, binary.LittleEndian.Uint32(h[12:]))}
		if _, err := io.ReadFull(c, p.data); err != nil {
			return
		}
		m.mu.Lock()
		m.got = append(m.got, p)
		count := len(m.data)
		var description []byte
		if p.id == pktData && int(p.dev) < count {
			description = append([]byte(nil), m.data[p.dev]...)
		}
		m.mu.Unlock()
		switch p.id {
		case pktProto:
			o.send(0, pktProto, u32(6))
		case pktCount:
			o.send(0, pktCount, u32(uint32(count)))
		case pktData:
			o.send(p.dev, pktData, description)
		}
	}
}

func (m *mockSDK) packets(id uint32) []sdkPkt {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []sdkPkt
	for _, p := range m.got {
		if p.id == id {
			out = append(out, p)
		}
	}
	return out
}

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	for end := time.Now().Add(3 * time.Second); time.Now().Before(end); time.Sleep(20 * time.Millisecond) {
		if ok() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

// ---------- tests ----------

func TestParseController(t *testing.T) {
	d := mockDev{"Corsair Vengeance RGB DDR5", "Direct", []int{2, 3}, []ledRGB{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}}}
	ci, err := parseController(ctrlData(d), 5)
	if err != nil {
		t.Fatal(err)
	}
	if ci.name != d.name || ci.loc != "HID: "+d.name || ci.mode != "Direct" || len(ci.zones) != 2 || ci.zones[1] != 3 ||
		!equalRGB(ci.colors, d.colors) {
		t.Errorf("parsed %+v", ci)
	}
	d.mode = "Static"
	if ci, _ := parseController(ctrlData(d), 5); ci.mode != "Static" {
		t.Errorf("active mode %q, want Static", ci.mode)
	}
	b := ctrlData(d)
	if _, err := parseController(b[:len(b)-30], 5); err == nil {
		t.Errorf("cut-short data parsed without an error")
	}
}

func TestLedsPacket(t *testing.T) {
	got := ledsPacket([]ledRGB{{0x11, 0x22, 0x33}, {0xaa, 0xbb, 0xcc}})
	want := []byte{14, 0, 0, 0, 2, 0, 0x11, 0x22, 0x33, 0, 0xaa, 0xbb, 0xcc, 0}
	if string(got) != string(want) {
		t.Errorf("UPDATELEDS payload % x, want % x", got, want)
	}
}

func TestLookColors(t *testing.T) {
	steady := look{ledRGB{200, 100, 0}, "", 0}
	if c := steady.colors(nil, 3, 0, 0.5); len(c) != 3 || c[2] != (ledRGB{100, 50, 0}) {
		t.Errorf("steady at half brightness: %v", c)
	}
	br := look{ledRGB{200, 0, 0}, "breathe", time.Second}
	lo, hi := br.colors(nil, 1, 0, 1)[0], br.colors(nil, 1, 500*time.Millisecond, 1)[0]
	if lo.r != 30 || hi.r != 200 {
		t.Errorf("breathe from %v to %v, want 30 to 200", lo, hi)
	}
	// a chase runs in each zone on its own: the spot is at the start of both zones at t=0
	ch := look{ledRGB{0, 250, 0}, "chase", time.Second}
	c := ch.colors([]int{4, 4}, 8, 0, 1)
	if c[0].g <= c[2].g || c[4].g <= c[6].g || c[0] != c[4] {
		t.Errorf("chase frame %v", c)
	}
	if c := ch.colors([]int{4}, 6, 0, 1); len(c) != 6 || c[4] == (ledRGB{}) {
		t.Errorf("LEDs past the zones left dark: %v", c)
	}
	cfg := rgbCfg{Colors: map[string]string{phWaiting: "#ff00ff", phWriting: "bogus"}}
	if cfg.look(phWaiting).c != (ledRGB{255, 0, 255}) || cfg.look(phWriting).c != looks[phWriting].c {
		t.Errorf("color overrides: %v %v", cfg.look(phWaiting).c, cfg.look(phWriting).c)
	}
}

func TestRGBPhase(t *testing.T) {
	up := serverState{Polled: true, Up: true}
	sess := func(kind string, ev ...event) []*session {
		return []*session{{Mod: time.Now(), lastKind: kind, events: ev}}
	}
	busy := up
	busy.HaveSlots, busy.Busy, busy.Slot.Processing = true, 1, true
	writing := busy
	writing.Decoded = 40
	cases := []struct {
		name string
		srv  serverState
		ss   []*session
		want string
	}{
		{"no server", serverState{Polled: true}, nil, phOffline},
		{"loading", serverState{Polled: true, Loading: true}, nil, phLoading},
		{"up, no agent", up, nil, phIdle},
		{"up, no agent, a request", writing, nil, phWriting},
		{"answered", up, sess("reply"), phWaiting},
		{"tool running", up, sess("call", event{T: time.Now(), Kind: "tool", Name: "shell"}), phTool},
		{"reading", busy, sess("tool_result"), phReading},
		{"writing", writing, sess("tool_result"), phWriting},
		{"between", up, sess("tool_result"), phBetween},
	}
	for _, c := range cases {
		st := &state{srv: c.srv, sessions: c.ss}
		if got := st.rgbPhase(); got != c.want {
			t.Errorf("%s: %s, want %s", c.name, got, c.want)
		}
	}
}

// The client drives only Direct-mode devices, re-reads the list when it changes and restores the colors on stop.
func TestOpenRGBMock(t *testing.T) {
	orig := []ledRGB{{9, 9, 9}, {8, 8, 8}, {7, 7, 7}}
	m := newMockSDK(t,
		mockDev{"MSI MYSTIC LIGHT", "Direct", []int{1, 2}, orig},
		mockDev{"Razer Naga", "Static", []int{3}, []ledRGB{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}})
	st := &state{srv: serverState{Polled: true, Up: true}}
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1}, st)
	go l.run()

	waitFor(t, "a frame", func() bool { return len(m.packets(pktUpdateLEDs)) > 0 })
	if p := m.packets(pktClientName); len(p) != 1 || string(p[0].data) != "Caminus Top\x00" {
		t.Errorf("client name %v", p)
	}
	if v := m.packets(pktData)[0].data; binary.LittleEndian.Uint32(v) != 3 {
		t.Errorf("controller data asked at protocol %d", binary.LittleEndian.Uint32(v))
	}
	idle := ledsPacket([]ledRGB{looks[phIdle].c, looks[phIdle].c, looks[phIdle].c})
	for _, p := range m.packets(pktUpdateLEDs) {
		if p.dev != 0 || string(p.data) != string(idle) {
			t.Fatalf("UPDATELEDS to device %d: % x, want device 0: % x", p.dev, p.data, idle)
		}
	}
	waitFor(t, "status", func() bool { return l.Status() == "idle on 1 device" })

	// steady colors are sent once, not every frame
	time.Sleep(300 * time.Millisecond)
	if n := len(m.packets(pktUpdateLEDs)); n != 1 {
		t.Errorf("%d UPDATELEDS for a steady look, want 1", n)
	}

	// the phase moves the lights
	st.mu.Lock()
	st.sessions = []*session{{Mod: time.Now(), lastKind: "reply"}}
	st.mu.Unlock()
	waitFor(t, "waiting", func() bool { return strings.HasPrefix(l.Status(), "waiting") })

	// a device list change makes it reconnect and list the devices again
	m.mu.Lock()
	c := m.conns[0]
	m.mu.Unlock()
	(&orgbConn{c: c}).send(0, pktListChange, nil)
	waitFor(t, "a reconnect", func() bool { return len(m.packets(pktProto)) == 2 && len(m.packets(pktData)) == 4 })
	time.Sleep(200 * time.Millisecond)

	l.stop()
	ups := m.packets(pktUpdateLEDs)
	if last := ups[len(ups)-1]; string(last.data) != string(ledsPacket(orig)) {
		t.Errorf("not restored: last UPDATELEDS % x, want the first-seen colors % x", last.data, ledsPacket(orig))
	}
}

func TestOpenRGBDown(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := ln.Addr().String()
	ln.Close()
	l := newRGB(rgbCfg{Host: addr}, &state{})
	go l.run()
	waitFor(t, "status", func() bool { return strings.HasPrefix(l.Status(), "OpenRGB not answering") })
	l.stop()
}

func TestRenderRGBStatus(t *testing.T) {
	st := demoState(t)
	st.lights = newRGB(rgbCfg{}, st)
	st.lights.set("writing on 6 devices")
	if out := frame(t, st, 120, 50); !strings.Contains(out, "rgb  writing on 6 devices") {
		t.Errorf("no rgb line in AGENTS:\n%s", out)
	}
	frame(t, st, 90, 40)
}

// Quitting while the client waits to reconnect (after a device list change) still puts the colors back, over a
// fresh connection.
func TestOpenRGBQuitBetweenConnections(t *testing.T) {
	orig := []ledRGB{{9, 9, 9}, {8, 8, 8}}
	m := newMockSDK(t, mockDev{"MSI MYSTIC LIGHT", "Direct", []int{2}, orig})
	st := &state{srv: serverState{Polled: true, Up: true}}
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1}, st)
	go l.run()
	waitFor(t, "a frame", func() bool { return len(m.packets(pktUpdateLEDs)) > 0 })
	m.mu.Lock()
	c := m.conns[0]
	m.mu.Unlock()
	(&orgbConn{c: c}).send(0, pktListChange, nil)
	time.Sleep(300 * time.Millisecond) // the client is in its 1 s wait now
	if n := len(m.packets(pktProto)); n != 1 {
		t.Fatalf("%d handshakes before quit, want 1", n)
	}
	l.stop()
	ups := m.packets(pktUpdateLEDs)
	if last := ups[len(ups)-1]; string(last.data) != string(ledsPacket(orig)) || len(m.packets(pktProto)) != 2 {
		t.Errorf("not restored over a new connection: last UPDATELEDS % x, %d handshakes", last.data, len(m.packets(pktProto)))
	}
}
