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

// Exercise the actual key-update/view path with an unchanged transcript. The
// cost of typing should follow the visible screen, not the whole conversation.
func BenchmarkResidentTyping(b *testing.B) {
	for _, lines := range []int{20, 2000, 10000} {
		b.Run(fmt.Sprintf("history_%d_lines", lines), func(b *testing.B) {
			m := chatFixture(b)
			m.width, m.height = 200, 60
			m.receiveResident(residentMsg{agent: "resident", status: m.resident, events: []residentEvent{{Seq: 2, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"output": strings.Repeat("Observed tool output with a Unicode café and a stable, unchanged transcript.\n", lines)}}}}})
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
