// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The workers row (#342, #382 review): the "last" child is the finished one with the newest Started - the status's
// own sort is not the card's fact; and the cap counts money and the Go plan's allowance, so the row shows their sum.
func TestWorkersRowNamesTheNewestFinishedChildAndCountsAllowance(t *testing.T) {
	if got := workersRow(nil); got != "" {
		t.Fatalf("no children: %q", got)
	}
	ws := []residentWorker{
		{Child: "b", SpendUSD: 0.01, AllowanceUSD: 0.02, CapUSD: 0.5, Outcome: "completed", Started: 200},
		{Child: "a", SpendUSD: 0.03, AllowanceUSD: 0, CapUSD: 0.5, Outcome: "capped", Started: 100},
	}
	if got, want := workersRow(ws), "no children running · last completed $0.0300"; got != want {
		t.Fatalf("last row: %q, want %q", got, want)
	}
	// The list's order is the query's, not a fact: the older child first still names the newer one (#382 review).
	ws = []residentWorker{
		{Child: "old", SpendUSD: 0.01, CapUSD: 0.5, Outcome: "capped", Started: 100},
		{Child: "new", SpendUSD: 0.07, CapUSD: 0.5, Outcome: "completed", Started: 300},
		{Child: "mid", SpendUSD: 0.02, CapUSD: 0.5, Outcome: "completed", Started: 200},
	}
	if got, want := workersRow(ws), "no children running · last completed $0.0700"; got != want {
		t.Fatalf("newest by Started: %q, want %q", got, want)
	}
	ws = []residentWorker{
		{Child: "r1", SpendUSD: 0, AllowanceUSD: 0.01, CapUSD: 0.5, Running: true},
		{Child: "r2", SpendUSD: 0.02, AllowanceUSD: 0.02, CapUSD: 0.25, Running: true},
	}
	if got, want := workersRow(ws), "2 children · $0.0500 of $0.75"; got != want {
		t.Fatalf("running row: %q, want %q", got, want)
	}
}

func TestResidentGoalCommandUsesStdin(t *testing.T) {
	m := emberModel{cli: []string{"caminus"}, agentName: "resident"}
	c := residentCommand(m, []string{"send", "resident", "--stdin", "--json"}, "goal with $(shell) and secrets")
	if strings.Contains(strings.Join(c.Args, " "), "secrets") || c.Stdin == nil {
		t.Fatalf("goal leaked into argv: %v", c.Args)
	}
	if strings.Join(c.Args, " ") != "caminus ember agent send resident --stdin --json" {
		t.Fatal(c.Args)
	}
}

// The chat's /steer is the one steering path: it redirects the targeted goal, and the text travels on stdin so
// shell syntax in it stays literal. (This replaces the tests of the removed agent page's steering editor.)
func TestChatSteerTargetsGoalAndUsesStdin(t *testing.T) {
	dir := t.TempDir()
	argsPath, inputPath := filepath.Join(dir, "args"), filepath.Join(dir, "input")
	cli := filepath.Join(dir, "fake-cli")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$STEER_ARGS\"\ncat > \"$STEER_INPUT\"\nprintf '{\"queued\":true}\\n'\n"
	if err := os.WriteFile(cli, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("STEER_ARGS", argsPath)
	t.Setenv("STEER_INPUT", inputPath)
	m := emberModel{cli: []string{cli}, agentName: "resident", resident: residentStatus{Goals: []residentGoal{{ID: "original", State: "working"}}}}
	m.initChat()
	m.chat.target = "original"
	message := "Try another approach; literal $(text) stays in stdin."
	_, cmd := m.chatSlash("/steer " + message)
	if cmd == nil {
		t.Fatal("/steer sent nothing")
	}
	if result := cmd().(residentActionMsg); strings.Contains(result.text, "error") {
		t.Fatal(result.text)
	}
	args, _ := os.ReadFile(argsPath)
	input, _ := os.ReadFile(inputPath)
	if string(args) != "ember\nagent\nsend\nresident\n--goal-id\noriginal\n--steer\n--stdin\n--json\n" || string(input) != message {
		t.Fatalf("wrong steering destination or input: %q / %q", args, input)
	}
	m.chat.target = ""
	if _, cmd := m.chatSlash("/steer something"); cmd != nil {
		t.Fatal("/steer without a target goal must not send")
	}
	if strings.ContainsRune(residentText("observed\x1b[2Joutput"), '\x1b') {
		t.Fatal("agent terminal escape retained")
	}
}

func TestBridgeRejectsPathIDsAndIgnoresPredictableTempSymlink(t *testing.T) {
	root := t.TempDir()
	bridge := filepath.Join(root, "bridge")
	os.MkdirAll(filepath.Join(bridge, "res"), 0o770)
	target := filepath.Join(root, "protected")
	os.WriteFile(target, []byte("unchanged"), 0o600)
	writeReq(t, bridge, "ok", "id -u", 1)
	os.Symlink(target, filepath.Join(bridge, "res", "ok.json.tmp"))
	if err := writeAnswer(bridge, "human", emberReq{ID: "../protected"}, "denied", "", nil, "", "", ""); err == nil {
		t.Fatal("path ID accepted")
	}
	if err := writeAnswer(bridge, "human", emberReq{ID: "ok"}, "denied", "", nil, "", "", ""); err != nil {
		t.Fatal(err)
	}
	value, _ := os.ReadFile(target)
	if string(value) != "unchanged" {
		t.Fatal("symlink target overwritten")
	}
	info, _ := os.Stat(filepath.Join(bridge, "res", "ok.json"))
	if info.Mode().Perm() != 0o660 {
		t.Fatalf("bridge mode: %o", info.Mode().Perm())
	}
}

func TestSSHWarningDoesNotBecomeProtocolOutput(t *testing.T) {
	dir := t.TempDir()
	script := "#!/bin/sh\nprintf 'ControlSocket already exists; disabling multiplexing\\n' >&2\nprintf 'agent\\n'\n"
	if err := os.WriteFile(filepath.Join(dir, "ssh"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	m := emberModel{user: "agent", key: "fixture"}
	msg := probeCmd(m)().(probeMsg)
	if !msg.ok || msg.err != "" {
		t.Fatalf("warning broke successful probe: %+v", msg)
	}
	output, err := sshOutput(sshCmd(m, "jobs", "fixture"))
	if err != nil || output != "agent\n" {
		t.Fatalf("diagnostic entered protocol output: %q %v", output, err)
	}
}

func TestPanelCallsAreBoundedSoAStallCannotHangTheComposer(t *testing.T) {
	m := chatFixture(t)
	m.cli = []string{"/bin/true"}
	if cmd := residentCommand(m, []string{"status", "resident", "--json"}, ""); cmd == nil || cmd.Cancel == nil {
		t.Fatal("a resident CLI call must carry a deadline")
	}
	m.user, m.key = "resident", "/nonexistent-key"
	if cmd := sshCmd(m, "jobs", "true"); cmd.Cancel == nil {
		t.Fatal("an SSH poll must carry a deadline")
	}
}

func TestACrashShowsItsExceptionNotTheTracebackHeader(t *testing.T) {
	out := []byte("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    import y\nModuleNotFoundError: No module named 'y'\n")
	if got := cliError(out); got != "agent runtime error: ModuleNotFoundError: No module named 'y'" {
		t.Fatalf("the exception, not the header: %q", got)
	}
	if got := cliError([]byte("error: no agent named x\n")); got != "error: no agent named x" {
		t.Fatalf("plain errors pass as they are: %q", got)
	}
}

// The SSH transport prints notices on stderr ("ControlSocket … already exists, disabling multiplexing" when the
// shared connection refuses a session): the status, the sessions and the switcher's cards must still read stdout's
// JSON. Mixed in, every parse failed: one session in the switcher and "stopped" in the header.
func TestResidentJSONIgnoresTransportNoticesOnStderr(t *testing.T) {
	script := filepath.Join(t.TempDir(), "fake-caminus")
	body := "#!/bin/sh\n" +
		"echo 'mux_client_request_session: session request failed: Session open refused by peer' >&2\n" +
		"echo 'ControlSocket /x already exists, disabling multiplexing' >&2\n" +
		"case \"$3\" in\n" +
		"  peers) echo '[{\"name\":\"resident\"},{\"name\":\"research\"}]' ;;\n" +
		"  memory) echo '{\"memory\":\"m\"}' ;;\n" +
		"  *) echo '{\"name\":\"resident\",\"model\":\"claude-haiku-5-5\",\"daemon_alive\":true,\"goals\":[]}' ;;\n" +
		"esac\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	m := emberModel{agentName: "resident", cli: []string{script}}
	out, err := runResident(residentCommand(m, []string{"status", "resident", "--json"}, ""))
	var st residentStatus
	if err != nil || json.Unmarshal(out, &st) != nil || st.Model != "claude-haiku-5-5" || !st.DaemonAlive {
		t.Fatalf("the status reads stdout only: %v %q", err, out)
	}
	msg := residentSessions(m)().(residentSessionsMsg)
	if msg.err != "" || len(msg.names) != 2 {
		t.Fatalf("every session is listed despite the notices: %+v", msg)
	}
	if _, err := runResident(nil); err == nil {
		t.Fatal("no CLI is an error, not a panic")
	}
}

// A runtime from before `aims` (the one the owner had installed): its status still makes a full card, with the
// goal's own prompt.
func TestASessionCardFromAnOlderRuntimeWithoutAims(t *testing.T) {
	script := filepath.Join(t.TempDir(), "fake-caminus")
	body := "#!/bin/sh\necho 'ControlSocket /x already exists, disabling multiplexing' >&2\ncase \"$3\" in\n" +
		"  github) echo '{\"linked\":false}' ;;\n" +
		"  *) echo '{\"name\":\"research\",\"model\":\"claude-haiku-5-5\",\"daemon_alive\":true,\"spend_today\":4.12,\"context\":1000000,\"context_used\":52335,\"goals\":[{\"id\":\"g1\",\"prompt\":\"design a game\",\"state\":\"completed\",\"created\":1,\"updated\":181}]}' ;;\n" +
		"esac\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	c := loadSessionCard(emberModel{agentName: "resident", cli: []string{script}}, "research")
	if c.status != "listening" || c.goal != "design a game" || c.aimed || c.model != "claude-haiku-5-5" || c.spendToday != 4.12 || c.ctxUsed != 52335 {
		t.Fatalf("a full card from a status without aims: %+v", c)
	}
}

// Destroy cancels every live goal, not only the one the card shows: a second goal waiting on root left its request
// in the bridge for a session that no longer existed.
func TestDestroyCancelsEveryLiveGoalThenRemoves(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	script := filepath.Join(dir, "fake-caminus")
	body := "#!/bin/sh\nop=$3\nshift 2\necho \"$*\" >> " + log + "\ncase \"$op\" in\n" +
		"  status) echo '{\"name\":\"research\",\"daemon_alive\":true,\"goals\":[{\"id\":\"a\",\"prompt\":\"x\",\"state\":\"working\"},{\"id\":\"b\",\"prompt\":\"y\",\"state\":\"waiting_root\"},{\"id\":\"c\",\"prompt\":\"z\",\"state\":\"completed\"}]}' ;;\n" +
		"  remove) echo '{\"name\":\"research\",\"removed\":true,\"kept\":\"/k\"}' ;;\n" +
		"  *) echo '{}' ;;\nesac\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	m := emberModel{agentName: "resident", cli: []string{script}}
	c := loadSessionCard(m, "research")
	if strings.Join(c.active, ",") != "a,b" {
		t.Fatalf("the live goals are a and b: %v", c.active)
	}
	_ = os.Remove(log)
	// a card still loading knows no goals: destroy reads them itself
	msg := destroyResidentSession(m, "research", nil)().(residentDestroyedMsg)
	calls, _ := os.ReadFile(log)
	want := "remove research --dry-run --json\nstatus research --json\ngithub research status --json\ncancel research a --json\ncancel research b --json\nremove research --json\n"
	if msg.err != "" || msg.kept != "/k" || !strings.HasSuffix(string(calls), want) {
		t.Fatalf("dry run, cancel each live goal, then remove: %+v\n%s", msg, calls)
	}
}

// Restore lists the kept sessions, then restores the exact kept folder picked (not just the newest of that name), and
// says when the service did not start; a runtime's refusal comes back as its reason, an old runtime's as the fix.
func TestRestoreListsThenRestoresThePickedFolder(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	script := filepath.Join(dir, "fake-caminus")
	body := "#!/bin/sh\nshift 2\necho \"$*\" >> " + log + "\ncase \"$2\" in\n" +
		"  --json) echo '{\"closed\":[{\"name\":\"research\",\"kept\":\"research-20261118-091500\",\"closed_at\":1795000000,\"size\":2048,\"open\":false}]}' ;;\n" +
		"  research) echo '{\"name\":\"research\",\"restored\":true,\"started\":false,\"start_error\":\"no user manager\"}' ;;\n" +
		"  open) echo '{\"error\":\"a session named open is open: close it first\"}'; exit 1 ;;\n" +
		"  *) echo 'usage: caminus ember agent: invalid choice: restore' >&2; exit 2 ;;\nesac\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	m := emberModel{agentName: "resident", cli: []string{script}}
	got := loadClosedSessions(m)().(closedSessionsMsg)
	if got.err != "" || len(got.closed) != 1 || got.closed[0].Kept != "research-20261118-091500" || got.closed[0].Size != 2048 {
		t.Fatalf("the list: %+v", got)
	}
	msg := restoreResidentSession(m, got.closed[0])().(residentRestoredMsg)
	calls, _ := os.ReadFile(log)
	if !strings.HasSuffix(string(calls), "restore research --kept research-20261118-091500 --json\n") {
		t.Fatalf("restore names the kept folder picked:\n%s", calls)
	}
	if !strings.Contains(msg.text, "restored research") || !strings.Contains(msg.text, "did not start") || !strings.Contains(msg.text, "no user manager") {
		t.Fatalf("a restore whose service did not start says so: %q", msg.text)
	}
	if msg := restoreResidentSession(m, closedSession{Name: "open", Kept: "open-1"})().(residentRestoredMsg); !strings.Contains(msg.text, "is open: close it first") || strings.Contains(msg.text, "{") {
		t.Fatalf("a refusal shows the runtime's reason, not its JSON: %q", msg.text)
	}
	if msg := restoreResidentSession(m, closedSession{Name: "old", Kept: "old-1"})().(residentRestoredMsg); !strings.Contains(msg.text, "agent install") {
		t.Fatalf("a runtime without restore names the fix: %q", msg.text)
	}
}
