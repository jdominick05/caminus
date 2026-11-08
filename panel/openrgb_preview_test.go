// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// CAMINUS_RGB_PREVIEW=/absolute/path.svg go test -run TestRGBPreview .
// Render actual effect frames into an animated SVG; no OpenRGB server or lighting hardware is involved.
func TestRGBPreview(t *testing.T) {
	path := os.Getenv("CAMINUS_RGB_PREVIEW")
	if path == "" {
		t.Skip("set CAMINUS_RGB_PREVIEW to export the effects")
	}
	c := defaultConfig.OpenRGB
	c.Brightness = .85
	ci := ctrlInfo{kind: 0, zones: []int{8, 8, 24}, zoneNames: []string{"JARGB 2", "JARGB 3", "Strip"}, colors: make([]ledRGB, 40)}
	var out strings.Builder
	fmt.Fprint(&out, `<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="1020" viewBox="0 0 1040 1020"><rect width="1040" height="1020" fill="#202630"/><g font-family="monospace" fill="#d8dee9"><text x="32" y="42" font-size="24" font-weight="bold">CAMINUS / AGENT LIGHTS</text><text x="32" y="70" font-size="13" fill="#a8b5c9">Actual renderer frames · fan rings + linear RAM / GPU zones</text><text x="265" y="112" font-size="12">JARGB 2</text><text x="405" y="112" font-size="12">JARGB 3</text><text x="555" y="112" font-size="12">RAM / GPU STRIP</text>`)
	for row, phase := range []string{phOffline, phLoading, phIdle, phBetween, phWaiting, phReading, phWriting, phTool, phCompact} {
		y := 165 + row*96
		fmt.Fprintf(&out, `<path d="M32 %dH1008" stroke="#354050"/><text x="32" y="%d" font-size="17">%s</text>`, y+46, y+5, phase)
		period := looks[phase].period
		if period == 0 {
			period = 2 * time.Second
		}
		var frames [][]ledRGB
		for step := range 41 {
			frames = append(frames, c.frame(ci, phase, period*time.Duration(step)/40))
		}
		for led := range 40 {
			first := frames[0][led]
			color := fmt.Sprintf("#%02x%02x%02x", first.r, first.g, first.b)
			if led < 16 {
				x := 300
				if led >= 8 {
					x = 440
				}
				a := 2*math.Pi*float64(led%8)/8 - math.Pi/2
				fmt.Fprintf(&out, `<circle cx="%.2f" cy="%.2f" r="8" stroke="#4c566a" fill="%s">`, float64(x)+30*math.Cos(a), float64(y)+30*math.Sin(a), color)
			} else {
				fmt.Fprintf(&out, `<rect x="%d" y="%d" width="15" height="26" rx="3" fill="%s">`, 555+(led-16)*18, y-13, color)
			}
			var values []string
			for _, frame := range frames {
				v := frame[led]
				values = append(values, fmt.Sprintf("#%02x%02x%02x", v.r, v.g, v.b))
			}
			fmt.Fprintf(&out, `<animate attributeName="fill" values="%s" dur="%.3fs" repeatCount="indefinite"/>`, strings.Join(values, ";"), period.Seconds())
			if led < 16 {
				fmt.Fprint(&out, "</circle>")
			} else {
				fmt.Fprint(&out, "</rect>")
			}
		}
	}
	fmt.Fprint(&out, `<text x="32" y="1000" font-size="12" fill="#a8b5c9">Agent-state effects only. Phase changes crossfade over 300 ms in the monitor.</text></g></svg>`)
	if err := os.WriteFile(path, []byte(out.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
