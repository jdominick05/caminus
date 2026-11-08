// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

func TestRGBRingRotation(t *testing.T) {
	c := defaultConfig.OpenRGB
	ci := ctrlInfo{kind: 0, zones: []int{0, 8, 8, 1}, zoneNames: []string{"empty", "JARGB 2", "JARGB_V2_3", "single"}, colors: make([]ledRGB, 17)}
	period := looks[phWriting].period
	first, next := c.frame(ci, phWriting, 0), c.frame(ci, phWriting, period/8)
	for i := range 8 {
		if next[i] != first[(i+7)%8] {
			t.Fatal("J2 did not rotate by one distinct LED")
		}
		if next[8+i] != first[8+(i+1)%8] {
			t.Fatal("J3 did not rotate in the opposite direction")
		}
	}
	if first[0] == first[7] {
		t.Fatal("ring repeats its endpoint like a strip")
	}
	for phase, l := range looks {
		if l.period == 0 {
			continue
		}
		before, after := c.frame(ci, phase, l.period-time.Microsecond), c.frame(ci, phase, 0)
		for i := range before {
			for _, d := range []int{int(before[i].r) - int(after[i].r), int(before[i].g) - int(after[i].g), int(before[i].b) - int(after[i].b)} {
				if d < -1 || d > 1 {
					t.Fatalf("%s has a jump at the loop seam", phase)
				}
			}
		}
	}
}

func TestRGBRingSelection(t *testing.T) {
	c := defaultConfig.OpenRGB
	ci := ctrlInfo{kind: 0, zones: []int{8}, zoneNames: []string{"JARGB 2"}, colors: make([]ledRGB, 8)}
	if !c.layout(ci, 0).ring {
		t.Fatal("sized motherboard fan header not recognized")
	}
	ci.kind = 1 // RAM: eight LEDs do not imply a fan.
	if c.layout(ci, 0).ring {
		t.Fatal("DIMM misidentified as a ring")
	}
	ci.kind, ci.zones[0] = 0, 24
	if c.layout(ci, 0).ring {
		t.Fatal("automatic ring assumed a different fan size")
	}
	ci.zoneNames[0] = "Pump ring"
	c.Zones = map[string]string{"pump ring": "reverse-ring"}
	if l := c.layout(ci, 0); !l.ring || !l.reverse {
		t.Fatal("explicit ring layout ignored")
	}
	ci.zoneNames[0], ci.zones[0] = "JARGB 2", 8
	c.Zones = map[string]string{"JARGB 2": "strip"}
	if c.layout(ci, 0).ring {
		t.Fatal("strip override did not disable automatic ring")
	}
}

func TestRGBFramesAndBrightness(t *testing.T) {
	c := defaultConfig.OpenRGB
	c.Brightness = .4
	for _, n := range []int{0, 1, 2, 8, 24, 120} {
		ci := ctrlInfo{kind: 0, zones: []int{0, n}, zoneNames: []string{"empty", "JARGB 2"}, colors: make([]ledRGB, n)}
		for phase := range looks {
			for step := range 81 {
				frame := c.frame(ci, phase, time.Duration(step)*50*time.Millisecond)
				if len(frame) != n {
					t.Fatal("wrong LED count")
				}
				for _, v := range frame {
					if v.r > 102 || v.g > 102 || v.b > 102 {
						t.Fatal("brightness limit exceeded")
					}
				}
			}
		}
	}
	ci := ctrlInfo{colors: make([]ledRGB, 1)}
	if equalRGB(c.frame(ci, phReading, 0), c.frame(ci, phReading, looks[phReading].period/4)) {
		t.Fatal("a single LED did not show activity")
	}
	ci = ctrlInfo{kind: 0, zones: []int{8}, zoneNames: []string{"JARGB 2"}, colors: make([]ledRGB, 8)}
	want := c.frame(ci, phWriting, 400*time.Millisecond)
	c.Speed = 2
	if !equalRGB(want, c.frame(ci, phWriting, 200*time.Millisecond)) {
		t.Fatal("speed did not preserve phase")
	}
	c.Colors = map[string]string{phWriting: "#000000"}
	for _, v := range c.frame(ci, phWriting, 300*time.Millisecond) {
		if v != (ledRGB{}) {
			t.Fatal("black override emitted light")
		}
	}
}

func TestRGBCrossfadeInterrupt(t *testing.T) {
	c := defaultConfig.OpenRGB
	ci := ctrlInfo{colors: []ledRGB{{200, 150, 100}}}
	now := time.Unix(0, 0)
	var a rgbAnimator
	if got := a.frame(c, ci, phIdle, now); !equalRGB(got, ci.colors) {
		t.Fatal("startup fade did not start at the reported colors")
	}
	now = now.Add(300 * time.Millisecond)
	idle := a.frame(c, ci, phIdle, now)
	if !equalRGB(idle, c.frame(ci, phIdle, 300*time.Millisecond)) {
		t.Fatal("startup fade did not finish")
	}
	if got := a.frame(c, ci, phWriting, now); !equalRGB(got, idle) {
		t.Fatal("phase change jumped")
	}
	now = now.Add(150 * time.Millisecond)
	mid := a.frame(c, ci, phWriting, now)
	if equalRGB(mid, idle) || equalRGB(mid, c.frame(ci, phWriting, 150*time.Millisecond)) {
		t.Fatal("crossfade has no intermediate frame")
	}
	if got := a.frame(c, ci, phReading, now); !equalRGB(got, mid) {
		t.Fatal("rapid transition did not start at the displayed frame")
	}
	now = now.Add(300 * time.Millisecond)
	if got := a.frame(c, ci, phReading, now); !equalRGB(got, c.frame(ci, phReading, 300*time.Millisecond)) {
		t.Fatal("interrupted fade did not finish")
	}
	// Even startup fades respect the user's brightness limit.
	c.Brightness = 0
	a = rgbAnimator{}
	for step := range 10 {
		for _, v := range a.frame(c, ci, phWriting, now.Add(time.Duration(step)*50*time.Millisecond)) {
			if v != (ledRGB{}) {
				t.Fatal("zero brightness flashed during startup")
			}
		}
	}
}

func TestRGBConfigDefaults(t *testing.T) {
	p := filepath.Join(t.TempDir(), "top.json")
	write(t, p, `{"openrgb":{"enabled":true}}`)
	c, err := loadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.OpenRGB.FPS != 20 || c.OpenRGB.Speed != 1 || c.OpenRGB.FadeMS != 300 {
		t.Fatal("old config lost new defaults")
	}
	bad := rgbCfg{FPS: -100, Speed: math.NaN(), FadeMS: 9999, Brightness: 9}.normalized()
	if bad.FPS != 20 || bad.Speed != 1 || bad.FadeMS != 300 || bad.Brightness != 1 {
		t.Fatal("invalid options not bounded")
	}
	good := rgbCfg{FPS: 10, Speed: .5, FadeMS: 0, Brightness: 0}.normalized()
	if good.FPS != 10 || good.Speed != .5 || good.FadeMS != 0 || good.Brightness != 0 {
		t.Fatal("valid options changed")
	}
}

func TestOpenRGBDrivesCrossfade(t *testing.T) {
	m := newMockSDK(t, mockDev{"board", "Direct", []int{2}, make([]ledRGB, 2)})
	st := &state{srv: serverState{Polled: true, Up: true}}
	l := newRGB(rgbCfg{Host: m.ln.Addr().String(), Brightness: 1, FadeMS: 200, FPS: 20}, st)
	go l.run()
	waitFor(t, "steady idle", func() bool {
		p := m.packets(pktUpdateLEDs)
		return len(p) > 0 && string(p[len(p)-1].data) == string(ledsPacket([]ledRGB{looks[phIdle].c, looks[phIdle].c}))
	})
	n := len(m.packets(pktUpdateLEDs))
	st.mu.Lock()
	st.sessions = []*session{{Mod: time.Now(), lastKind: "call", events: []event{{T: time.Now(), Kind: "tool", Name: "shell"}}}}
	st.mu.Unlock()
	waitFor(t, "a crossfaded frame", func() bool {
		for _, p := range m.packets(pktUpdateLEDs)[n:] {
			if p.data[6] > 0 && p.data[8] > 0 && p.data[8] < 60 {
				return true
			}
		}
		return false
	})
	l.stop()
}
