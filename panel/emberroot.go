// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	tea "charm.land/bubbletea/v2"
)

// rootEnv is set inside the sudo boundary (sudo resets the caller's environment, so it goes through `env`) for an
// approved root command: package tools then ask nothing (debconf, apt-listchanges, needrestart lists instead of
// asking) and nothing opens a pager. The approved string stays one untouched `sh -c` word. Matches ROOT_ENV in
// python/caminus/ember/bridge.py (tests/ember/test_env.py compares them).
var rootEnv = []string{"DEBIAN_FRONTEND=noninteractive", "APT_LISTCHANGES_FRONTEND=none", "NEEDRESTART_MODE=l",
	"PAGER=cat", "SYSTEMD_PAGER=cat", "GIT_PAGER=cat"}

// rootWait bounds the wait for an approved root command. Past it the panel stops waiting, records rc 124 with the
// pid still running and returns to the chat; it never kills a root process. Longer than sudo's own password wait,
// so sudo is no longer reading the terminal when the panel takes it back. Matches ROOT_WAIT in bridge.py.
var rootWait = 30 * time.Minute

// rootRC is the exit code recorded when the panel stopped waiting (timeout(1)'s code).
const rootRC = 124

// rootArgv is the argv that runs an approved string as root.
func rootArgv(cmd string) []string {
	argv := append([]string{"sudo", "env"}, rootEnv...)
	return append(argv, "sh", "-c", cmd)
}

// rootRun runs an approved root command for tea.Exec (the terminal is released, so sudo can ask for the password
// on it). The command's output goes to a regular file, never a pipe, and the wait is for the process itself: a
// background child that keeps the output open (a daemon a package's triggers started) cannot hold the panel, as it
// did while the panel waited for a tee pipe's end of file. Its stdin is /dev/null, so nothing waits on a prompt;
// sudo reads the password from the terminal itself. The file is copied to the terminal while it runs.
type rootRun struct {
	argv []string
	out  string // the file the output goes to (read back for the bridge answer)
	term io.Writer
	wait time.Duration
	poll time.Duration
}

func (r *rootRun) SetStdin(io.Reader)    {}
func (r *rootRun) SetStdout(w io.Writer) { r.term = w }
func (r *rootRun) SetStderr(io.Writer)   {}

// rootTimeout is Run's error when the wait ran out with the command still running.
type rootTimeout struct {
	pid   int
	after time.Duration
}

func (e rootTimeout) Error() string {
	return fmt.Sprintf("timed out after %s, still running as pid %d", waitText(e.after), e.pid)
}

func waitText(d time.Duration) string {
	if d >= time.Minute {
		return fmt.Sprintf("%d min", int(d.Round(time.Minute)/time.Minute))
	}
	return d.String()
}

func (r *rootRun) Run() error {
	term := r.term
	if term == nil {
		term = io.Discard
	}
	out, err := os.OpenFile(r.out, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return err
	}
	defer out.Close()
	null, err := os.Open(os.DevNull)
	if err != nil {
		return err
	}
	defer null.Close()
	tail, err := os.Open(r.out)
	if err != nil {
		return err
	}
	defer tail.Close()
	// *os.File for every stream: exec then starts no copier, and Wait returns when the process exits.
	c := exec.Command(r.argv[0], r.argv[1:]...)
	c.Stdin, c.Stdout, c.Stderr = null, out, out
	if err = c.Start(); err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- c.Wait() }() // also reaps a command that outlives the wait
	poll := r.poll
	if poll <= 0 {
		poll = 100 * time.Millisecond
	}
	wait := r.wait
	if wait <= 0 {
		wait = rootWait
	}
	tick := time.NewTicker(poll)
	defer tick.Stop()
	limit := time.NewTimer(wait)
	defer limit.Stop()
	for {
		select {
		case err = <-done:
			io.Copy(term, tail)
			return err
		case <-tick.C:
			io.Copy(term, tail)
		case <-limit.C:
			io.Copy(term, tail)
			e := rootTimeout{pid: c.Process.Pid, after: wait}
			note := "\ncaminus: " + e.Error() + "; stopped waiting (its later output is not recorded)\n"
			out.WriteString(note)
			io.WriteString(term, note)
			return e
		}
	}
}

// execRoot builds the run of cmd under sudo and the file its output lands in (the caller reads and removes it).
func execRoot(cmd string) (*rootRun, string, error) {
	f, err := os.CreateTemp("", "caminus-env-root-*.out")
	if err != nil {
		return nil, "", err
	}
	tmp := f.Name()
	f.Close()
	argv := rootArgv(cmd)
	if demoMode { // the demo records the approval and runs nothing as root
		argv = []string{"echo", "demo: approved, not run as root: " + cmd}
	}
	return &rootRun{argv: argv, out: tmp, wait: rootWait}, tmp, nil
}

// rootExitCode is the exit code a finished rootRun reports to the bridge.
func rootExitCode(err error) int {
	if err == nil {
		return 0
	}
	if _, ok := err.(rootTimeout); ok {
		return rootRC
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	return 1
}

// execRootCmd hands a rootRun to the program (a variable so a test can run it in place of a terminal).
var execRootCmd = func(r *rootRun, done tea.ExecCallback) tea.Cmd { return tea.Exec(r, done) }
