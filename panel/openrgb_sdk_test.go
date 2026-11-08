// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"
)

func TestControllerValidation(t *testing.T) {
	d := mockDev{"board", "Direct", []int{0, 2}, []ledRGB{{1, 2, 3}, {4, 5, 6}}}
	for _, protocol := range []uint32{3, 5} {
		b := ctrlData3(d)
		if protocol == 5 {
			b = ctrlData(d)
		}
		for cut := 0; cut < len(b); cut++ {
			short := append([]byte(nil), b[:cut]...)
			if cut >= 4 {
				binary.LittleEndian.PutUint32(short, uint32(cut))
			}
			if _, err := parseController(short, protocol); err == nil {
				t.Fatalf("protocol %d accepted truncation at %d", protocol, cut)
			}
		}
		if _, err := parseController(append(b, 0), protocol); err == nil {
			t.Fatal("accepted trailing byte")
		}
	}
	ci, err := parseController(ctrlData3(d), 3)
	if err != nil || ci.kind != 0 || ci.zoneNames[1] != "JARGB 2" || ci.selectedMode(false) != 0 {
		t.Fatalf("lost device metadata: %+v, %v", ci, err)
	}
	// Same physical identity, different LED allocation: never reuse the old snapshot.
	old := ci.key()
	d.zones, d.colors = []int{4}, make([]ledRGB, 4)
	ci, _ = parseController(ctrlData3(d), 3)
	if ci.key() == old {
		t.Fatal("resized device reused an old snapshot key")
	}
	d.zones = []int{5}
	if _, err := parseController(ctrlData3(d), 3); err == nil {
		t.Fatal("zones exceed LED allocation")
	}
	bad := ctrlData3(d)
	// The active mode follows six SDK strings and num_modes.
	r := pkReader{b: bad}
	r.take(8)
	for range 6 {
		r.str()
	}
	r.u16()
	binary.LittleEndian.PutUint32(r.b, 99)
	if _, err := parseController(bad, 3); err == nil {
		t.Fatal("accepted invalid active mode")
	}
}

func TestOpenRGBModeRestore(t *testing.T) {
	orig := []ledRGB{{17, 23, 31}, {41, 53, 67}}
	m := newMockSDK(t, mockDev{"board", "Static", []int{2}, orig})
	ci, err := parseController(m.data[0], 3)
	if err != nil {
		t.Fatal(err)
	}
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1, SwitchModes: true},
		&state{srv: serverState{Polled: true, Up: true}})
	go l.run()
	waitFor(t, "temporary per-LED mode", func() bool { return len(m.packets(pktUpdateLEDs)) > 0 })
	l.stop()
	modes := m.packets(pktUpdateMode)
	if len(modes) != 2 || binary.LittleEndian.Uint32(modes[0].data[4:]) != 0 {
		t.Fatalf("mode packets: %+v", modes)
	}
	// Brightness is applied to a copy: the stored Direct payload still says 63.
	if got := binary.LittleEndian.Uint32(modes[0].data[8+ci.modes[0].fieldsAt+36:]); got != 100 {
		t.Fatalf("temporary brightness %d, want 100", got)
	}
	want := append(binary.LittleEndian.AppendUint32(nil, uint32(8+len(ci.modes[1].raw))), u32(1)...)
	want = append(want, ci.modes[1].raw...)
	if !bytes.Equal(modes[1].data, want) {
		t.Fatal("original speed, brightness, mode colors or active mode lost")
	}
	ups := m.packets(pktUpdateLEDs)
	if !bytes.Equal(ups[len(ups)-1].data, ledsPacket(orig)) {
		t.Fatal("original per-LED colors lost")
	}
	// Mode changes do not issue SAVEMODE, profile-save or zone-resize commands.
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.got {
		switch p.id {
		case pktProto, pktClientName, pktCount, pktData, pktUpdateLEDs, pktUpdateMode:
		default:
			t.Fatalf("unexpected write command %d", p.id)
		}
	}
}

func TestOpenRGBActiveCustomMode(t *testing.T) {
	m := newMockSDK(t, mockDev{"board", "Direct", []int{2}, make([]ledRGB, 2)})
	m.data[0] = bytes.ReplaceAll(m.data[0], []byte("Direct"), []byte("Custom"))
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1}, &state{srv: serverState{Polled: true, Up: true}})
	go l.run()
	waitFor(t, "Custom-mode frame", func() bool { return len(m.packets(pktUpdateLEDs)) > 0 })
	l.stop()
	if len(m.packets(pktUpdateMode)) != 0 {
		t.Fatal("default configuration changed a device mode")
	}
}

func TestOpenRGBDuplicateIdentityWritesNothing(t *testing.T) {
	d := mockDev{"board", "Direct", []int{2}, make([]ledRGB, 2)}
	m := newMockSDK(t, d, d)
	l := newRGB(rgbCfg{Host: m.ln.Addr().String()}, &state{})
	o, _, err := l.connect()
	if o != nil {
		o.c.Close()
	}
	if err == nil || len(m.packets(pktUpdateLEDs)) != 0 || len(m.packets(pktUpdateMode)) != 0 {
		t.Fatal("ambiguous devices must be refused before writing")
	}
}

type partialSDKConn struct {
	net.Conn
	data  []byte
	limit int
}

func (c *partialSDKConn) Write(b []byte) (int, error) {
	n := min(c.limit, len(b))
	c.data = append(c.data, b[:n]...)
	return n, nil
}

func TestSDKShortWrites(t *testing.T) {
	c := &partialSDKConn{limit: 3}
	if err := (&orgbConn{c: c}).send(2, pktCount, u32(7)); err != nil {
		t.Fatal(err)
	}
	want := append([]byte("ORGB"), u32(2)...)
	want = append(append(append(want, u32(pktCount)...), u32(4)...), u32(7)...)
	if !bytes.Equal(c.data, want) {
		t.Fatal("partial writes lost bytes")
	}
	c.limit = 0
	if err := (&orgbConn{c: c}).send(0, pktCount, nil); err != io.ErrShortWrite {
		t.Fatal("zero write must fail")
	}
}

func TestOpenRGBRestoreAfterReorder(t *testing.T) {
	a, b := []ledRGB{{1, 2, 3}}, []ledRGB{{4, 5, 6}, {7, 8, 9}}
	m := newMockSDK(t, mockDev{"board", "Direct", []int{1}, a}, mockDev{"ram", "Direct", []int{2}, b})
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1}, &state{srv: serverState{Polled: true, Up: true}})
	go l.run()
	waitFor(t, "initial frames", func() bool { return len(m.packets(pktUpdateLEDs)) == 2 })
	m.mu.Lock()
	m.data[0], m.data[1] = m.data[1], m.data[0]
	c := m.conns[0]
	m.mu.Unlock()
	if err := (&orgbConn{c: c}).send(0, pktListChange, nil); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "reordered frames", func() bool { return len(m.packets(pktUpdateLEDs)) >= 4 })
	l.stop()
	ups := m.packets(pktUpdateLEDs)
	last := ups[len(ups)-2:]
	if last[0].dev != 0 || !bytes.Equal(last[0].data, ledsPacket(b)) || last[1].dev != 1 || !bytes.Equal(last[1].data, ledsPacket(a)) {
		t.Fatal("original colors restored through stale device indices")
	}
}

func TestOpenRGBNoRestoreThroughChangedList(t *testing.T) {
	a, b := net.Pipe()
	defer b.Close()
	changed := make(chan struct{})
	close(changed)
	l := newRGB(rgbCfg{}, &state{})
	l.restore(&orgbConn{c: a}, []orgbDev{{key: "old"}}, nil, changed)
	_ = b.SetReadDeadline(time.Now().Add(time.Second))
	var buf [16]byte
	if n, err := b.Read(buf[:]); n != 0 || err != io.EOF {
		t.Fatal("wrote a stale device after topology change")
	}
}
