package main

import (
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"strings"
	"testing"
)

// BenchmarkLiveBody: one snapshot of a growing reply (a delta appended, the body drawn), at reply lengths.
func BenchmarkLiveBody(b *testing.B) {
	for _, n := range []int{50, 1000, 5000} {
		b.Run(fmt.Sprintf("%d_lines", n), func(b *testing.B) {
			l := residentLive{active: true, wrapText: &wrapCache{}, wrapThink: &wrapCache{}, text: strings.Repeat("a line of the reply with `code` in it and some words\n", n)}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				l = l.appendText("tok ")
				_ = l.body(96)
			}
		})
	}
}

// The incremental wrap must draw exactly what wrapping the whole text would, at every step of a growing reply,
// across a width change, and when a new turn starts shorter.
func TestWrapCacheMatchesAFullWrap(t *testing.T) {
	full := func(s string, w int) []string {
		var out []string
		for _, ln := range strings.Split(ansi.Wrap(s, w, ""), "\n") {
			out = append(out, highlightLive(ln))
		}
		return out
	}
	c := &wrapCache{}
	text := ""
	deltas := []string{"Hello `wor", "ld` and a long line " + strings.Repeat("abc ", 40), "\nsecond\n", "", "\n\nthird 界🙂 line", " ends\n", "x"}
	for step, d := range deltas {
		text += d
		for _, w := range []int{30, 30, 57} { // the repeat hits the cache; 57 forces a rebuild
			got, want := c.wrap(text, w, highlightLive), full(text, w)
			if strings.Join(got, "\n") != strings.Join(want, "\n") {
				t.Fatalf("step %d width %d:\n got %q\nwant %q", step, w, got, want)
			}
		}
	}
	if got := c.wrap("new turn", 57, highlightLive); strings.Join(got, "\n") != strings.Join(full("new turn", 57), "\n") {
		t.Fatalf("a shorter text must start over: %q", got)
	}
}
