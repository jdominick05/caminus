package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
)

// mockHome draws the chat-first home screen from the shell's pieces, with a reply still streaming.
func mockHome(w int) string {
	sv := newStreamView(w, func(text string, width int) []string {
		return strings.Split(shellMessage("resident", text, width), "\n")
	})
	sv.Append("The bounds check reads one past the end when n == len. I changed `>` to `>=` and ")
	sv.Append("I'm adding a parser test for the empty input next.\n```c\nassert(parse(\"\", 0) == 0);\n```")
	header, _ := shellHeader("resident", "working", "Make the parser tests pass", 14, w)
	parts := []string{
		header,
		"",
		emberDim.Render("  while you were away (2 stretches): read about tide pools; built a haiku machine  ·  /recap"),
		"",
		shellMessage("you", "Make the parser tests pass and add one for empty input.", w),
		"",
		shellMessage("resident", "I'll run the failing target first, then look at the bounds check.", w),
		toolCard("terminal", "make test", "exit 2 · 3.1s", false, false, nil, w),
		toolCard("edit", "src/parse.c", "+1 −1", true, true, []string{"-    if (n > len) return -1;", "+    if (n >= len) return -1;"}, w),
		toolCard("gate", "make test", "exit 0 · 2.8s · 41 passed", true, false, nil, w),
		"",
		approvalCard("root", "apt-get install -y libcmocka-dev", "the new test needs the cmocka headers", w),
		"",
		sv.View(7),
		"",
		todoLine([]todoItem{{"find the failing test", "done"}, {"fix the bounds check", "done"}, {"add the empty-input test", "now"}, {"run the gate", "next"}}, w),
		composer("", "message the agent · Enter nudges at the next step · /steer redirects", "to goal original", w-2),
		statusLine(status{brain: "k3-coder", used: 48_000, window: 1_048_574, usd: 0, speed: replySpeed{write: 31, read: 1240, source: "engine"}, queue: 1,
			keys: []string{"esc stop", "^A approve", "/ cmds"}}, w),
	}
	return strings.Join(parts, "\n")
}

func mockApprovals(w, h int) string {
	row := func(sel bool, kind, text, age string) string {
		l := lipgloss.NewStyle().Foreground(cText).Render(kind+"  "+text) + emberDim.Render("  ·  "+age)
		if sel {
			return emberSel.Render(" " + g.collapsed + " " + kind + "  " + text + "  ·  " + age + " ")
		}
		return "   " + l
	}
	body := strings.Join([]string{
		row(true, g.root+" root", "apt-get install -y libcmocka-dev", "2 min"),
		row(false, "? ask ", "Want to see the tide-pool sketch? (free time)", "40 min · AFK, withdrawn"),
		"",
		emberDim.Render("Enter shows the full request  ·  a approve  ·  e edit  ·  d deny with an instruction"),
	}, "\n")
	return overlayBox("APPROVALS  1 pending", body, "↑↓ move  ·  Enter open", w, h)
}

func mockStudio(w, h int) string {
	body := strings.Join([]string{
		emberHead.Render("for you"),
		"   " + g.collapsed + " haiku-machine/   a 5-7-5 generator; " + emberDim.Render("\"strange but pleasant\""),
		"",
		emberHead.Render("projects") + emberDim.Render("  2 of 3 active"),
		"   " + emberCtxOK.Render(g.working) + " haiku machine   " + emberDim.Render("better syllable source next  ·  last 11-15"),
		"   " + emberCtxOK.Render(g.working) + " tide notes      " + emberDim.Render("reading on intertidal zones  ·  last 11-15"),
		"   " + emberDim.Render(g.idle+" roguelike       paused"),
		"",
		emberHead.Render("journal"),
		"   11-15 11:45  " + emberDim.Render("settled in; disk at 92% worth watching; built the haiku machine"),
		"   11-15 03:10  " + emberDim.Render("dream: merged notes on the terminal cursor into the tools page"),
		"",
		emberDim.Render("free time: on  ·  $0.01 of $0.50 today  ·  next stretch in 52 min  ·  /heartbeat off"),
	}, "\n")
	return overlayBox("STUDIO", body, "↑↓ move  ·  Enter open  ·  r recap", w, h)
}

// TestShellMockups writes the mockups as ANSI files when MOCKUP_DIR is set (cat them in a terminal). Without it, it
// only checks that every piece renders inside the width in every tier.
func TestShellMockups(t *testing.T) {
	dir := os.Getenv("MOCKUP_DIR")
	saved := g
	defer func() { g = saved; _ = themeLoad("caminus") }()
	for _, theme := range []string{"caminus", "tokyo-night"} {
		if err := themeLoad(theme); err != nil {
			t.Fatal(err)
		}
		for _, tier := range []string{"unicode", "nerd", "ascii"} {
			g = glyphTiers[tier]
			const w, h = 100, 22
			screens := map[string]string{"home": mockHome(w), "approvals": mockApprovals(w-4, h), "studio": mockStudio(w-4, h)}
			for name, s := range screens {
				for i, line := range strings.Split(s, "\n") {
					if lipgloss.Width(line) > w {
						t.Fatalf("%s/%s/%s line %d is %d wide: %q", theme, tier, name, i, lipgloss.Width(line), line)
					}
				}
				if dir != "" && tier != "ascii" {
					p := filepath.Join(dir, theme+"-"+tier+"-"+name+".ans")
					if err := os.WriteFile(p, []byte(s+"\n"), 0o644); err != nil {
						t.Fatal(err)
					}
				}
			}
		}
	}
}

func BenchmarkStreamView(b *testing.B) {
	for _, n := range []int{20, 2000, 10000} {
		sv := newStreamView(100, func(text string, width int) []string { return strings.Split(text, "\n") })
		for i := 0; i < n; i++ {
			sv.done = append(sv.done, "finished line of an earlier reply, already rendered")
		}
		b.Run(fmt.Sprint(n)+"_lines", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sv.Append("tok ")
				_ = sv.View(40)
				if i%200 == 199 {
					sv.Finish()
				}
			}
		})
	}
}
