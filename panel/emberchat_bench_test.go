// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

var residentFrameSink string

// benchChat is the chat fixture on a wide terminal with events of history. The history grows by events (a reply and a
// tool card each), so the transcript really is longer: one big output was cut to its last lines by the card and every
// size drew the same screen.
func benchChat(b *testing.B, events int) emberModel {
	m := chatFixture(b)
	m.width, m.height = 200, 60
	evs := make([]residentEvent, 0, events)
	for i := 0; i < events; i++ {
		ev := residentEvent{Seq: 2 + i, Goal: "original", Kind: "assistant", Data: map[string]any{
			"content": fmt.Sprintf("Reply %d with a Unicode café, **some markdown** and `code`.", i)}}
		if i%2 == 1 {
			ev = residentEvent{Seq: 2 + i, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "terminal",
				"result": map[string]any{"exit_code": float64(0), "output": strings.Repeat("Observed tool output line.\n", 5)}}}
		}
		evs = append(evs, ev)
	}
	m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: evs})
	if n := m.chat.transcript.TotalLineCount(); n < events {
		b.Fatalf("the history has %d lines for %d events", n, events)
	}
	return m
}

// Exercise the actual key-update/view path with an unchanged transcript. The
// cost of typing should follow the visible screen, not the whole conversation.
func BenchmarkResidentTyping(b *testing.B) {
	for _, events := range []int{10, 500, 2000} {
		b.Run(fmt.Sprintf("history_%d_events", events), func(b *testing.B) {
			m := benchChat(b, events)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.chat.compose.SetValue("short steering draft")
				next, _ := m.Update(tea.KeyPressMsg{Code: 'a', Text: "a"})
				m = next.(emberModel)
				residentFrameSink = m.View().Content
			}
		})
	}
}

// BenchmarkResidentLiveWide: a wide terminal, where the reply streams into the transcript, at history sizes. snapshot
// is one live update drawn (it sets the transcript's content again); frame is one tick drawn between live updates
// (the spinner moves, the text does not), which must cost what typing does: with the spinner in the streamed block,
// every frame reset the whole transcript (5.7 against 3.2 ms at 2000 events).
func BenchmarkResidentLiveWide(b *testing.B) {
	for _, events := range []int{10, 500, 2000} {
		for _, kind := range []string{"snapshot", "frame"} {
			b.Run(fmt.Sprintf("history_%d_events/%s", events, kind), func(b *testing.B) {
				m := benchChat(b, events)
				m.chat.live = residentLive{active: true, turn: "t1", wrapText: &wrapCache{}, wrapThink: &wrapCache{}}
				next, _ := m.Update(residentLiveMsg{turn: "t1", text: "tok "})
				m = next.(emberModel)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if kind == "snapshot" {
						next, _ = m.Update(residentLiveMsg{turn: "t1", text: "tok "})
						m = next.(emberModel)
					} else {
						m.tickN++
					}
					residentFrameSink = m.View().Content
				}
			})
		}
	}
}
