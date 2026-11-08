// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"math"
	"strings"
	"time"
	"unicode"
)

type rgbLayout struct {
	ring, reverse bool
	offset        float64
}

func (c rgbCfg) normalized() rgbCfg {
	if c.FPS < 1 || c.FPS > 60 {
		c.FPS = 20
	}
	if math.IsNaN(c.Speed) || math.IsInf(c.Speed, 0) || c.Speed < .1 || c.Speed > 4 {
		c.Speed = 1
	}
	if c.FadeMS < 0 || c.FadeMS > 2000 {
		c.FadeMS = 300
	}
	if math.IsNaN(c.Brightness) || math.IsInf(c.Brightness, 0) {
		c.Brightness = 1
	}
	c.Brightness = min(1, max(0, c.Brightness))
	return c
}

func (c rgbCfg) layout(ci ctrlInfo, zone int) rgbLayout {
	name := ci.zoneNames[zone]
	style, explicit := c.Zones[name]
	if !explicit {
		for key, value := range c.Zones {
			if strings.EqualFold(key, name) {
				style, explicit = value, true
				break
			}
		}
	}
	if explicit {
		switch style {
		case "strip":
			return rgbLayout{}
		case "ring":
			return rgbLayout{ring: true}
		case "reverse-ring":
			return rgbLayout{ring: true, reverse: true}
		}
	}
	// Eight-pixel ARGB fan headers need one sample per angle, not a duplicated strip endpoint.
	// A DIMM with eight LEDs is still a strip. Other fan layouts are selected explicitly by zone name.
	if ci.kind == 0 && ci.zones[zone] == 8 {
		key := strings.Map(func(r rune) rune {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				return unicode.ToLower(r)
			}
			return -1
		}, name)
		switch key {
		case "jargb2", "jargbv22":
			return rgbLayout{ring: true}
		case "jargb3", "jargbv23":
			return rgbLayout{ring: true, reverse: true, offset: .5}
		}
	}
	return rgbLayout{}
}

func rgbSmooth(x float64) float64 {
	x = min(1, max(0, x))
	return x * x * (3 - 2*x)
}

func rgbMix(a, b ledRGB, k float64) ledRGB {
	k = min(1, max(0, k))
	f := func(x, y uint8) uint8 { return uint8(math.Round(float64(x) + (float64(y)-float64(x))*k)) }
	return ledRGB{f(a.r, b.r), f(a.g, b.g), f(a.b, b.b)}
}

func (l look) zoneColors(n int, t time.Duration, br float64, layout rgbLayout) []ledRGB {
	out := make([]ledRGB, n)
	phase := 0.0
	if l.period > 0 {
		phase = float64(t%l.period) / float64(l.period)
	}
	for i := range out {
		x := .5 // A single LED shows the effect, rather than a permanently dark endpoint.
		if layout.ring {
			x = float64(i) / float64(n)
		} else if n > 1 {
			x = float64(i) / float64(n-1)
		}
		if layout.reverse {
			x = -x
		}
		x += layout.offset
		head := phase
		if !layout.ring {
			head = .5 - .5*math.Cos(2*math.Pi*phase)
		}
		glow := func(at float64) float64 {
			if layout.ring {
				return math.Exp(8 * (math.Cos(2*math.Pi*(x-at)) - 1))
			}
			return math.Exp(-30 * (x - at) * (x - at))
		}
		energy := 1.0
		switch l.fx {
		case "breathe":
			energy = .15 + .85*(.5-.5*math.Cos(2*math.Pi*phase))
			if layout.ring {
				energy *= .75 + .25*glow(head)
			}
		case "orbit":
			energy = .18 + .82*max(glow(head), glow(head+.5))
		case "sweep":
			energy = .18 + .82*math.Sqrt(glow(head))
		case "chase":
			energy = .12 + .88*min(1, glow(head)+.3*glow(head-.14))
		case "gloss":
			energy = .7 + .3*glow(head)
		case "converge":
			energy = .12 + .88*max(glow(head), glow(1-head))
		}
		out[i] = l.c.scale(br * energy)
	}
	return out
}

func (c rgbCfg) frame(ci ctrlInfo, phase string, t time.Duration) []ledRGB {
	l := c.look(phase)
	t = time.Duration(float64(t) * c.Speed)
	out := make([]ledRGB, 0, len(ci.colors))
	for z, n := range ci.zones {
		out = append(out, l.zoneColors(n, t, c.Brightness, c.layout(ci, z))...)
	}
	// Controllers can expose LEDs outside their named zones. They still get the same agent state.
	out = append(out, l.zoneColors(len(ci.colors)-len(out), t, c.Brightness, rgbLayout{})...)
	return out
}

// Each device keeps the last displayed frame. Rapid phase changes start from that frame, not from the old
// phase's nominal color; a tool/reading/writing burst therefore never restarts an unfinished fade with a jump.
type rgbAnimator struct {
	phase      string
	since      time.Time
	from, last []ledRGB
}

func (a *rgbAnimator) frame(c rgbCfg, ci ctrlInfo, phase string, now time.Time) []ledRGB {
	if a.phase != phase {
		a.from = append(a.from[:0], a.last...)
		if a.last == nil {
			a.from = append(a.from[:0], ci.colors...)
			for i := range a.from {
				a.from[i] = a.from[i].scale(c.Brightness)
			}
		}
		a.phase, a.since = phase, now
	}
	t := max(time.Duration(0), now.Sub(a.since))
	cs := c.frame(ci, phase, t)
	fade := time.Duration(c.FadeMS) * time.Millisecond
	if t < fade && len(a.from) == len(cs) {
		k := rgbSmooth(float64(t) / float64(fade))
		for i := range cs {
			cs[i] = rgbMix(a.from[i], cs[i], k)
		}
	}
	a.last = cs
	return cs
}
