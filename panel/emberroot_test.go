// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
)

// fakeSudo puts a stand-in sudo first on PATH: it runs the command as the caller with an empty environment (as
// sudo's env_reset does), so only what the panel sets inside the sudo boundary reaches the command.
func fakeSudo(t *testing.T) {
	t.Helper()
	bin := t.TempDir()
	script := "#!/bin/sh\nexec env -i PATH=\"$PATH\" \"$@\"\n"
	if err := os.WriteFile(filepath.Join(bin, "sudo"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+":"+os.Getenv("PATH"))
}

// killPidFile ends a process the test command left behind (its pid written to path), by pid.
func killPidFile(t *testing.T, path string) {
	t.Helper()
	t.Cleanup(func() {
		raw, err := os.ReadFile(path)
		if err != nil {
			return
		}
		if pid, err := strconv.Atoi(strings.TrimSpace(string(raw))); err == nil && pid > 1 {
			syscall.Kill(pid, syscall.SIGKILL)
		}
	})
}

// runRoot runs execRoot(cmd) the way the program does and fails the test if it does not come back within limit.
func runRoot(t *testing.T, cmd string, wait, limit time.Duration) (error, string, time.Duration) {
	t.Helper()
	r, tmp, err := execRoot(cmd)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp)
	r.wait = wait
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- r.Run() }()
	select {
	case err = <-done:
	case <-time.After(limit):
		out, _ := os.ReadFile(tmp)
		t.Fatalf("the root run did not come back within %s (output so far %q)", limit, out)
	}
	out, _ := os.ReadFile(tmp)
	return err, string(out), time.Since(start)
}

func TestExecRootArgv(t *testing.T) {
	r, tmp, err := execRoot("rm -rf /tmp/x && echo 'hi there'")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp)
	want := append(append([]string{"sudo", "env"}, rootEnv...), "sh", "-c", "rm -rf /tmp/x && echo 'hi there'")
	if strings.Join(r.argv, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("argv: want %q, got %q", want, r.argv)
	}
	if r.out != tmp || r.wait != rootWait {
		t.Fatalf("output file %q (want %q), wait %s", r.out, tmp, r.wait)
	}
}

func TestDemoRootRunRecordsAndRunsNothing(t *testing.T) {
	demoMode = true
	t.Cleanup(func() { demoMode = false })
	err, out, _ := runRoot(t, "rm -rf /nothing", time.Minute, 10*time.Second)
	if err != nil || !strings.Contains(out, "demo: approved, not run as root: rm -rf /nothing") {
		t.Fatalf("demo run: %v %q", err, out)
	}
}

// The owner's report: apt finished ("processing triggers for desktop-file-utils") and the panel never came back.
// A background child that keeps the command's output open must not hold the panel once the command has exited.
func TestRootRunReturnsWhenABackgroundChildKeepsTheOutputOpen(t *testing.T) {
	fakeSudo(t)
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	killPidFile(t, pidFile)
	err, out, took := runRoot(t, "echo done; sleep 30 & echo $! > "+shQuote(pidFile), time.Minute, 10*time.Second)
	if err != nil || rootExitCode(err) != 0 {
		t.Fatalf("want exit 0, got %v", err)
	}
	if !strings.Contains(out, "done") {
		t.Fatalf("output not recorded: %q", out)
	}
	if took > 5*time.Second {
		t.Fatalf("waited %s for a command that exited at once", took)
	}
	if _, err := os.Stat(pidFile); err != nil {
		t.Fatal("the background child never started; the test proves nothing")
	}
}

func TestRootRunIsNonInteractive(t *testing.T) {
	fakeSudo(t)
	cmd := `printf '%s|%s|%s|%s\n' "$DEBIAN_FRONTEND" "$NEEDRESTART_MODE" "$PAGER" "$SYSTEMD_PAGER"; ` +
		`if read answer; then echo "read: $answer"; else echo no-input; fi`
	err, out, _ := runRoot(t, cmd, time.Minute, 10*time.Second)
	if err != nil {
		t.Fatalf("want exit 0, got %v (%q)", err, out)
	}
	if !strings.Contains(out, "noninteractive|l|cat|cat") {
		t.Fatalf("the environment did not reach the command through sudo: %q", out)
	}
	if !strings.Contains(out, "no-input") {
		t.Fatalf("a prompt read something other than end of input: %q", out)
	}
}

func TestRootRunTimesOutWithThePidStillRunning(t *testing.T) {
	fakeSudo(t)
	pidFile := filepath.Join(t.TempDir(), "cmd.pid")
	killPidFile(t, pidFile)
	err, out, took := runRoot(t, "echo started; echo $$ > "+shQuote(pidFile)+"; exec sleep 30", 400*time.Millisecond, 10*time.Second)
	to, ok := err.(rootTimeout)
	if !ok || rootExitCode(err) != rootRC {
		t.Fatalf("want a timeout (rc %d), got %v", rootRC, err)
	}
	if took > 5*time.Second {
		t.Fatalf("the bounded wait took %s", took)
	}
	raw, _ := os.ReadFile(pidFile)
	pid, _ := strconv.Atoi(strings.TrimSpace(string(raw)))
	if pid != to.pid {
		t.Fatalf("recorded pid %d, the command runs as %d", to.pid, pid)
	}
	if syscall.Kill(pid, 0) != nil {
		t.Fatal("the panel killed the command; it must only stop waiting")
	}
	want := "timed out after " + (400 * time.Millisecond).String() + ", still running as pid " + strconv.Itoa(pid)
	if !strings.Contains(out, "started") || !strings.Contains(out, want) {
		t.Fatalf("the record does not say what happened: %q", out)
	}
	if got := (rootTimeout{pid: 7, after: 30 * time.Minute}).Error(); got != "timed out after 30 min, still running as pid 7" {
		t.Fatal(got)
	}
}

// The whole approval: the a key claims the request, runs it, and the result reaches the bridge (the agent's answer)
// with the request no longer executing, also when the wait runs out.
func TestApprovedRootRequestAlwaysReachesTheBridge(t *testing.T) {
	fakeSudo(t)
	saved, savedWait := execRootCmd, rootWait
	t.Cleanup(func() { execRootCmd, rootWait = saved, savedWait })
	execRootCmd = func(r *rootRun, done tea.ExecCallback) tea.Cmd {
		return func() tea.Msg { return done(r.Run()) }
	}
	dir := t.TempDir()
	cases := []struct {
		id, cmd string
		wait    time.Duration
		rc      int
		out     string
	}{
		{"child", "echo installed; sleep 30 & echo $! > " + shQuote(filepath.Join(dir, "child.pid")), time.Minute, 0, "installed"},
		{"hangs", "echo $$ > " + shQuote(filepath.Join(dir, "hangs.pid")) + "; echo waiting; exec sleep 30", 300 * time.Millisecond, rootRC, "still running as pid"},
	}
	for _, c := range cases {
		killPidFile(t, filepath.Join(dir, c.id+".pid"))
		rootWait = c.wait
		bridge := t.TempDir()
		writeReq(t, bridge, c.id, c.cmd, 1)
		m := emberModel{user: "agent", bridge: bridge, auditPath: filepath.Join(t.TempDir(), "a.jsonl"), key: "k",
			seen: map[string]bool{}, mode: "normal", width: 120, height: 40, tab: 6}
		m.refreshLocal()
		mm, cmd := m.reqKeys("a") // what the chat's approval card does on Yes
		if cmd == nil {
			t.Fatalf("%s: approve ran nothing (%s)", c.id, mm.(emberModel).msg)
		}
		got := make(chan tea.Msg, 1)
		go func() { got <- cmd() }()
		var msg tea.Msg
		select {
		case msg = <-got:
		case <-time.After(10 * time.Second):
			t.Fatalf("%s: the approved run never came back to the panel", c.id)
		}
		mm, _ = mm.(emberModel).Update(msg)
		m = mm.(emberModel)
		raw, _ := os.ReadFile(filepath.Join(bridge, "res", c.id+".json"))
		var res map[string]any
		if err := json.Unmarshal(raw, &res); err != nil {
			t.Fatalf("%s: %v %s", c.id, err, raw)
		}
		out, _ := res["out"].(string)
		if res["decision"] != "approved" || res["rc"] != float64(c.rc) || !strings.Contains(out, c.out) {
			t.Fatalf("%s: the agent's answer is %v", c.id, res)
		}
		if m.tab != 6 || m.mode != "normal" {
			t.Fatalf("%s: the panel did not come back to the chat (tab %d, mode %q)", c.id, m.tab, m.mode)
		}
	}
}
