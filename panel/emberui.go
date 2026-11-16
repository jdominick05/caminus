// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// caminus-panel env: the human side of the agent environment as a TUI (`caminus ember ui`,
// docs/specs/spec-ember.md section 7). The Python CLI runs it with the paths as flags:
//
//	caminus-panel env --user NAME --bridge DIR --audit FILE --key FILE
//
// Tabs (tab / shift-tab or 1-6), all live:
//
//	1 requests  the bridge queue: ROOT (a approve — sudo runs attached to the terminal via
//	            tea.ExecProcess so the password prompt works, output teed to a temp file;
//	            e edit, d deny, s skip) and ASK (a answer; a number picks a choice)
//	2 jobs      detached processes in the agent user (k kill, l tail the log, n new job)
//	3 procs     the agent user's busiest processes (ps over ssh)
//	5 ctx       running Caminus servers and the context window each reports in /props
//	6 audit     the tail of the audit log
//
// Answer files and audit lines use the same JSON schema as python/caminus/ember/{bridge,audit}.py,
// so `caminus ember watch` / `audit` and this TUI are interchangeable.
package main

import (
	"bytes"
	"charm.land/lipgloss/v2"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/charmbracelet/x/ansi"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"

	tea "charm.land/bubbletea/v2"
)

type emberReq struct {
	Agent   string   `json:"agent"`
	Goal    string   `json:"goal"`
	Digest  string   `json:"-"`
	ID      string   `json:"id"`
	Pid     int      `json:"pid"`
	User    string   `json:"user"`
	CWD     string   `json:"cwd"`
	Kind    string   `json:"kind"` // "root" (Cmd) or "question" (Text, Choices); empty = root
	Cmd     string   `json:"cmd"`
	Text    string   `json:"text"`
	Choices []string `json:"choices"`
	Reason  string   `json:"reason"`
	Ts      float64  `json:"ts"`
	Granted bool     `json:"granted"` // the resident says this exact command was granted for its goal (checked, not trusted)
	// a GitHub hand-off (kind "github"): what the agent's token may not do upstream, as checked fields; the owner's
	// side builds the gh command from them (caminus ember answer) and runs it as the owner
	Summary string         `json:"summary"`
	Github  map[string]any `json:"github"`
}

func (r emberReq) kind() string {
	if r.Kind == "" {
		return "root"
	}
	return r.Kind
}

type job struct {
	Name, Pid, State, Log string
}

type server struct {
	Port  int
	Model string
	Nctx  string
}

var tabs = []string{"requests", "jobs", "procs", "notes & memory", "context", "audit", "agent", "sessions", "settings"}

var emberNavigation = []int{7, 5, 8, 3, 4}

type emberModel struct {
	notes          notifier // desktop notifications already shown (notify.go)
	user           string
	bridge         string
	auditPath      string
	key            string
	cli            []string
	agentName      string
	defaultAgent   string
	pendingLast    string
	statusAt       time.Time // when the last status arrived (the online clock ticks from it)
	resident       residentStatus
	residentEvents []residentEvent
	goalSel        int
	chat           residentChat
	residentEpoch  uint64
	sessionNames   []string
	sessionSel     int
	sessionMemory  string
	pageScroll     int

	tab     int
	reqs    []emberReq
	sel     int
	seen    map[string]bool
	jobs    []job
	jobSel  int
	procs   []string
	servers []server
	events  []string
	logName string
	logView string

	mode        string // "normal", "edit", "answer", "sessionnew", "newjob1", "newjob2"
	edit        string
	editRequest *emberReq
	newJob      string

	msg    string
	msgOf  string // the message msgAt dates: a message shows for a few seconds, then clears
	msgAt  time.Time
	sshOK  bool
	sshN   int
	tickN  int
	width  int
	height int
}

type emberTickMsg time.Time
type probeMsg struct {
	ok  bool
	n   int
	err string
}
type jobsMsg []job
type procsMsg []string
type serversMsg []server
type execDoneMsg struct {
	goal     string
	granted  bool
	claim    string
	id       string
	decision string
	cmd      string
	rc       int
	out      string
}
type sshOutMsg struct {
	what string
	out  string
	err  error
}

func emberMain(args []string) {
	fs := flag.NewFlagSet("ember", flag.ExitOnError)
	user := fs.String("user", "", "the agent user")
	bridge := fs.String("bridge", "", "the bridge directory")
	audit := fs.String("audit", "", "the audit log path")
	key := fs.String("key", "", "the harness ssh key")
	cli := fs.String("cli", "", "JSON argv of the caminus CLI (for starting server jobs)")
	agent := fs.String("agent", "resident", "resident agent name")
	draft := fs.String("draft", "", "text the composer starts with (a new goal, sent on Enter)")
	demo := fs.Bool("demo", false, "the demo (tools/ember_demo.py): no SSH, and an approved root command is recorded, never run")
	fs.Parse(args)
	if *demo {
		demoMode = true
		if *user == "" {
			*user = "demo"
		}
		if *key == "" {
			*key = "none"
		}
	}
	if *user == "" || *bridge == "" || *audit == "" || *key == "" {
		fmt.Fprintln(os.Stderr, "caminus-panel env: --user --bridge --audit --key are all required")
		os.Exit(2)
	}
	var cliArgv []string
	if *cli != "" {
		if err := json.Unmarshal([]byte(*cli), &cliArgv); err != nil {
			fmt.Fprintln(os.Stderr, "caminus-panel env: bad --cli JSON:", err)
			os.Exit(2)
		}
	}
	m := emberModel{user: *user, bridge: *bridge, auditPath: *audit, key: *key, cli: cliArgv,
		agentName: *agent, defaultAgent: *agent, tab: 6, seen: map[string]bool{}, mode: "normal",
		width: 110, height: 32}
	m.initChat()
	// Open directly into the last conversation; the first session list decides
	// whether that session still exists (residentSessionsMsg below).
	if demoMode { // straight into the demo agent; the real last session is neither read nor replaced
		m.openResident(*agent)
		m.pendingLast = *agent
	} else {
		openLastSession(&m)
	}
	if *draft != "" {
		m.chat.compose.SetValue(*draft)
		m.chat.newGoal = true
	}
	if m.pendingLast == "" {
		m.chat.picker = true // no last conversation: start in the chat with the session picker open
	}
	p := tea.NewProgram(m)
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "caminus-panel env:", err)
		os.Exit(1)
	}
}

func (m emberModel) Init() tea.Cmd {
	cmds := append([]tea.Cmd{tickCmd(), probeCmd(m)}, sshRefresh(m)...)
	cmds = append(cmds, residentRefresh(m), residentSessions(m))
	return tea.Batch(cmds...)
}

func tickCmd() tea.Cmd {
	return tea.Tick(500*time.Millisecond, func(t time.Time) tea.Msg { return emberTickMsg(t) })
}

func sshCmd(m emberModel, what, script string) *exec.Cmd {
	cmd, _ := sshCmdTimeout(m, what, script)
	return cmd
}

// sshCmdTimeout bounds a remote call: a stalled SSH session must not leave a child running for the life
// of the panel, and the caller releases the timer as soon as the call returns.
func sshCmdTimeout(m emberModel, what, script string) (*exec.Cmd, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	if demoMode { // no agent account in the demo: the SSH views answer empty, the probe answers as the agent user
		if what == "probe" {
			return exec.CommandContext(ctx, "echo", m.user), cancel
		}
		return exec.CommandContext(ctx, "true"), cancel
	}
	c := exec.CommandContext(ctx, "ssh", "-i", m.key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
		"-o", "ControlMaster=auto", "-o", "ControlPath="+filepath.Join(os.TempDir(), "caminus-envui-%r@%h:%p"),
		m.user+"@127.0.0.1", "bash", "-s")
	c.Stdin = strings.NewReader(script)
	return c, cancel
}

func sshRun(m emberModel, what, script string) tea.Cmd {
	return func() tea.Msg {
		cmd, cancel := sshCmdTimeout(m, what, script)
		defer cancel()
		out, err := sshOutput(cmd)
		return sshOutMsg{what: what, out: out, err: err}
	}
}

func sshOutput(cmd *exec.Cmd) (string, error) {
	var diagnostics bytes.Buffer
	cmd.Stderr = &diagnostics
	out, err := cmd.Output()
	if err != nil {
		return string(out), fmt.Errorf("%w: %s", err, strings.TrimSpace(diagnostics.String()))
	}
	return string(out), nil
}

func probeCmd(m emberModel) tea.Cmd {
	return func() tea.Msg {
		cmd, cancel := sshCmdTimeout(m, "probe", "id -un")
		defer cancel()
		out, err := sshOutput(cmd)
		ok := err == nil && strings.TrimSpace(out) == m.user
		detail := ""
		if err != nil {
			detail = err.Error()
		} else if !ok {
			detail = "unexpected user: " + strings.TrimSpace(out)
		}
		return probeMsg{ok: ok, n: m.sshN + 1, err: detail}
	}
}

// the per-tab ssh scripts (run as the agent user; read-only except kill/new-job which act on its own jobs)
func jobsScript() string {
	return `for p in ~/env-jobs/*.pid; do [ -e "$p" ] || continue; n=$(basename "$p" .pid); pid=$(cat "$p"); if kill -0 "$pid" 2>/dev/null; then s=alive; else s=dead; fi; echo "$n|$pid|$s|~/env-logs/$n.log"; done`
}

func procsScript(user string) string {
	return fmt.Sprintf("ps -u %s -o pid=,etimes=,pcpu=,pmem=,args= --sort=-pcpu | head -12", shQuote(user))
}

func serversScript() string {
	return `for p in $(seq 8092 8101); do printf 'PORT %d\n' "$p"; curl -s --max-time 1 "http://127.0.0.1:$p/props"; echo; done`
}

// sshRefresh fetches only what the visible page shows. Every round used to run all four scripts every 2 s on every
// page, pages nobody could open included, and the servers script alone can take ten seconds.
func sshRefresh(m emberModel) []tea.Cmd {
	switch m.tab {
	case 1:
		return []tea.Cmd{sshRun(m, "jobs", jobsScript())}
	case 2:
		return []tea.Cmd{sshRun(m, "procs", procsScript(m.user))}
	case 4:
		return []tea.Cmd{sshRun(m, "servers", serversScript())}
	case 9:
		return []tea.Cmd{sshRun(m, "jobs", jobsScript()), sshRun(m, "procs", procsScript(m.user))}
	}
	return nil
}

// ---------------------------------------------------------------- bridge and audit (pure, tested)

func loadRequests(bridge string) ([]emberReq, error) {
	names, err := filepath.Glob(filepath.Join(bridge, "req", "*.json"))
	if err != nil {
		return nil, err
	}
	var out []emberReq
	for _, n := range names {
		id := strings.TrimSuffix(filepath.Base(n), ".json")
		if _, err := os.Stat(filepath.Join(bridge, "res", id+".json")); err == nil {
			continue
		}
		data, err := os.ReadFile(n)
		if err != nil {
			continue
		}
		var r emberReq
		if err := json.Unmarshal(data, &r); err != nil {
			continue
		}
		if r.ID == "" {
			r.ID = id
		}
		if !validBridgeID(id) || r.ID != id {
			continue
		}
		r.Digest = requestDigest(data)
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Ts < out[j].Ts })
	return out, nil
}

func writeAnswer(bridge, actor string, r emberReq, decision, cmd string, rc *int, out, errText, answer string) error {
	if !validBridgeID(r.ID) {
		return fmt.Errorf("invalid bridge request id")
	}
	res := map[string]any{"id": r.ID, "decision": decision, "cmd": cmd, "approved_by": actor,
		"rc": rc, "out": out, "err": errText, "answer": answer, "ts": float64(time.Now().UnixMilli()) / 1000}
	data, _ := json.Marshal(res)
	dir := filepath.Join(bridge, "res")
	if err := os.MkdirAll(dir, 0o770); err != nil {
		return err
	}
	return withBridgeLock(bridge, r.ID, func() error {
		if _, err := checkPendingRequest(bridge, r); err != nil {
			return err
		}
		return writeBridgeJSON(filepath.Join(dir, r.ID+".json"), data)
	})
}

func validBridgeID(s string) bool {
	if len(s) < 1 || len(s) > 128 {
		return false
	}
	for i, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			continue
		}
		if i > 0 && (r == '_' || r == '-') {
			continue
		}
		return false
	}
	return true
}

func writeBridgeJSON(path string, data []byte) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".bridge-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if err = f.Chmod(0o660); err != nil {
		return err
	}
	if stat, ok := info.Sys().(*syscall.Stat_t); ok {
		if err = f.Chown(-1, int(stat.Gid)); err != nil {
			return err
		}
	}
	if _, err = f.Write(data); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func appendAudit(path string, event map[string]any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	event["ts"] = float64(time.Now().UnixMilli()) / 1000
	data, _ := json.Marshal(event)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	return err
}

// tailLines returns the last n lines, reading at most the file's last 256 KiB: the audit log only grows, and
// reading all of it twice a second grew with it.
func tailLines(path string, n int) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	const window = 256 << 10
	var data []byte
	if st, err := f.Stat(); err == nil && st.Size() > window {
		data = make([]byte, window)
		if _, err := f.ReadAt(data, st.Size()-window); err != nil {
			return nil
		}
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			data = data[i+1:] // drop the partial first line
		}
	} else if data, err = io.ReadAll(f); err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if lines[0] == "" {
		lines = nil
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

// shQuote quotes s for a POSIX shell word.
func shQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// demoMode is set by `env -demo`: nothing reaches SSH or sudo.
var demoMode bool

// execRoot builds the command that runs cmd under sudo attached to the terminal, teeing the
// output to a temp file (the sudo password prompt goes to the tty, not the pipe).
func execRoot(cmd string) (*exec.Cmd, string, error) {
	f, err := os.CreateTemp("", "caminus-env-root-*.out")
	if err != nil {
		return nil, "", err
	}
	tmp := f.Name()
	f.Close()
	script := "set -o pipefail; sudo sh -c " + shQuote(cmd) + " 2>&1 | tee " + shQuote(tmp)
	if demoMode { // the demo records the approval and runs nothing as root
		script = "echo " + shQuote("demo: approved, not run as root: "+cmd) + " | tee " + shQuote(tmp)
	}
	return exec.Command("bash", "-c", script), tmp, nil
}

func parseJobs(out string) []job {
	var js []job
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		parts := strings.Split(line, "|")
		if len(parts) != 4 {
			continue
		}
		js = append(js, job{Name: parts[0], Pid: parts[1], State: parts[2], Log: parts[3]})
	}
	return js
}

func parseServers(out string) []server {
	var ss []server
	port := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "PORT ") {
			port, _ = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "PORT ")))
			continue
		}
		line = strings.TrimSpace(line)
		if port == 0 || !strings.HasPrefix(line, "{") {
			continue
		}
		var p struct {
			ModelAlias string `json:"model_alias"`
			Model      string `json:"model"`
			Nctx       any    `json:"n_ctx"`
			DGS        struct {
				Nctx any `json:"n_ctx"`
			} `json:"default_generation_settings"`
		}
		if json.Unmarshal([]byte(line), &p) == nil {
			nctx := p.Nctx
			if nctx == nil {
				nctx = p.DGS.Nctx
			}
			ss = append(ss, server{Port: port, Model: p.ModelAlias, Nctx: fmt.Sprintf("%v", nctx)})
		}
		port = 0
	}
	return ss
}

// ---------------------------------------------------------------- model

func (m emberModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.syncChat()
		return m, nil
	case emberTickMsg:
		m.tickN++
		if m.msg != m.msgOf {
			m.msgOf, m.msgAt = m.msg, time.Now()
		} else if m.msg != "" && time.Since(m.msgAt) > 6*time.Second {
			m.msg, m.msgOf = "", ""
		}
		m.refreshLocal()
		cmds := []tea.Cmd{tickCmd()}
		// The SSH probe is a liveness check, not data: every 10 s is enough (it ran every second).
		if m.tickN%20 == 0 {
			cmds = append(cmds, probeCmd(m))
		}
		if m.tickN%4 == 0 {
			cmds = append(cmds, sshRefresh(m)...)
		}
		// With the live reader open the conversation receives its events from the stream; the status poll
		// only catches goal-list changes, so it slows to every 5 s there.
		residentEvery := 4
		if m.tab == 6 && m.chat.stream != nil {
			residentEvery = 10
		}
		if m.tickN%residentEvery == 0 && (m.tab == 6 || m.tab == 7 || m.tab == 8 || m.tab == 3) {
			cmds = append(cmds, residentRefresh(m))
		}
		// The conversation page keeps one live reader open; it is started here so a page that was
		// entered without a key press (a reopened session) streams too.
		if m.tab == 6 && m.chat.stream == nil {
			cmds = append(cmds, startStream(m))
		}
		return m, tea.Batch(cmds...)
	case probeMsg:
		m.sshOK, m.sshN = msg.ok, msg.n
		if msg.err != "" {
			m.msg = msg.err
		}
		return m, nil
	case jobsMsg:
		m.jobs = msg
		if m.jobSel >= len(m.jobs) {
			m.jobSel = max(0, len(m.jobs)-1)
		}
		return m, nil
	case procsMsg:
		m.procs = msg
		return m, nil
	case serversMsg:
		m.servers = msg
		return m, nil
	case residentMsg:
		if msg.epoch != m.residentEpoch || (msg.agent != "" && msg.agent != m.agentName) {
			return m, nil
		}
		if msg.err != "" {
			m.msg = msg.err
		} else {
			m.receiveResident(msg)
		}
		if msg.err == "" && m.chat.recapAsked == recapNotAsked && validSessionName(m.agentName) {
			// once per opened session: what free time did since you last looked (nothing when you never looked away)
			m.chat.recapAsked = recapAtOpen
			if seen := loadSeen(m.agentName); seen > 0 {
				return m, fetchRecap(m, seen)
			}
			return m, fetchRecap(m, 0)
		}
		return m, nil
	case residentComposeResult:
		return m.finishCompose(msg)
	case residentSessionsMsg:
		if msg.agent == m.agentName {
			m.sessionNames, m.sessionMemory = msg.names, msg.memory
			if msg.err != "" {
				m.msg = msg.err
			}
			if m.pendingLast != "" {
				last := m.pendingLast
				m.pendingLast = ""
				gone := msg.err != "" || (len(m.sessionNames) > 0 && !slices.Contains(m.sessionNames, last))
				if last != m.defaultAgent && gone {
					m.agentName = m.defaultAgent
					m.residentEpoch++
					m.resident, m.residentEvents, m.chat = residentStatus{}, nil, residentChat{}
					m.goalSel, m.sessionMemory = 0, ""
					m.tab = 6
					m.initChat()
					m.chat.picker = true // pick another session in the chat's own picker
					m.syncChat()
					m.msg = "session " + terminalText(last, true) + " no longer exists; pick another"
					return m, tea.Batch(residentRefresh(m), residentSessions(m))
				}
			}
		}
		return m, nil
	case residentCreatedMsg:
		if msg.err != "" {
			m.msg = msg.err
			return m, nil
		}
		m.openResident(msg.name)
		return m, tea.Batch(residentRefresh(m), residentSessions(m))
	case modelsLoadedMsg:
		return m.modelsLoaded(msg), nil
	case identityLoadedMsg:
		return m.identityLoaded(msg), nil
	case advisorLoadedMsg:
		return m.advisorLoaded(msg), nil
	case editorDoneMsg:
		if msg.err != "" {
			m.msg = msg.err
		} else {
			m.chat.compose.SetValue(msg.text)
			m.msg = "draft back from the editor"
		}
		return m, m.chat.compose.Focus()
	case bangDoneMsg:
		return m.bangDone(msg), m.chat.compose.Focus()
	case textViewMsg:
		if msg.err != "" {
			m.msg = msg.err
			return m, nil
		}
		m.chat.view = textView{open: true, title: msg.title, lines: msg.lines, colourDiff: msg.diff}
		m.msg = ""
		return m, nil
	case rewindListMsg:
		if msg.goal == m.chat.rewind.goal {
			m.chat.rewind.steps, m.chat.rewind.err = msg.steps, msg.err
			m.chat.rewind.sel = max(0, len(msg.steps)-1)
		}
		return m, nil
	case themeGenMsg:
		return m.receiveThemeGen(msg), nil
	case btwMsg:
		b := &m.chat.btw
		if !b.open || b.question != msg.question {
			return m, nil // closed, or a newer question replaced it
		}
		b.loading, b.err, b.top = false, msg.err, 0
		if msg.err == "" {
			// the agent's words, sanitised, then rendered as Markdown at the section's width
			b.lines = formatText("markdown", residentText(msg.answer), max(30, min(readingWidth+2, m.width-20)))
		}
		m.syncChat()
		return m, nil
	case unsendMsg:
		switch {
		case msg.err != "":
			m.msg = "pull back: " + msg.err
		case msg.text == nil:
			// nothing waiting: ↑ was a request for history, so the last message you sent comes back to edit
			if strings.TrimSpace(m.chat.compose.Value()) == "" {
				if next, done := m.recallStep("up", true); done {
					return next, nil
				}
			}
			m.msg = "nothing to pull back: the agent has read your messages"
		default:
			m.chat.compose.SetValue(*msg.text)
			m.msg = "pulled back: edit it and Enter to send again"
		}
		return m, nil
	case recapMsg:
		return m.receiveRecap(msg), nil
	case studioMsg:
		m.chat.studio.loading, m.chat.studio.err = false, msg.err
		if msg.err == "" {
			m.chat.studio.list, m.chat.studio.projects = msg.entries, msg.projects
		}
		return m, nil
	case studioTextMsg:
		if msg.path == m.chat.studio.reading {
			switch {
			case msg.err != "":
				m.chat.studio.err = msg.err
			case msg.binary:
				m.chat.studio.text = []string{"A binary file: the panel cannot show it as text. Press o to open it in your desktop's viewer."}
			default:
				m.chat.studio.text = formatText(studioKind(msg.path), msg.text, max(20, m.width-6))
			}
		}
		return m, nil
	case studioRawMsg:
		return m.receiveStudioRaw(msg), nil
	case residentActionMsg:
		m.msg = msg.text
		return m, residentRefresh(m)
	case residentStreamStartMsg:
		if msg.err != "" || msg.epoch != m.residentEpoch || msg.agent != m.agentName {
			return m, nil
		}
		m.chat.stopStream()
		m.chat.stream = msg.state
		return m, streamLine(msg.agent, msg.epoch, msg.state.sc)
	case residentLiveMsg:
		if msg.epoch != m.residentEpoch || (msg.agent != "" && msg.agent != m.agentName) {
			m.chat.stopStream()
			return m, nil
		}
		if msg.err != "" || msg.done {
			m.chat.stopStream()
			m.syncChat()
			return m, nil
		}
		if msg.reset || msg.turn != m.chat.live.turn {
			m.chat.live = residentLive{turn: msg.turn, text: msg.text, reasoning: msg.reasoning,
				tools: msg.tools, active: true, wrapText: &wrapCache{}, wrapThink: &wrapCache{}}
		} else {
			m.chat.live.text += msg.text
			m.chat.live.reasoning += msg.reasoning
			if len(msg.tools) > 0 {
				m.chat.live.tools = msg.tools
			}
		}
		wasBottom := m.chat.transcript.AtBottom()
		m.syncChat()
		if wasBottom {
			m.chat.transcript.GotoBottom()
		}
		if m.chat.stream != nil {
			return m, streamLine(msg.agent, msg.epoch, m.chat.stream.sc)
		}
		return m, nil
	case sshOutMsg:
		switch msg.what {
		case "jobs":
			m.jobs = parseJobs(msg.out)
		case "procs":
			m.procs = strings.Split(strings.TrimSpace(msg.out), "\n")
		case "servers":
			m.servers = parseServers(msg.out)
		case "kill":
			m.msg = "kill: " + strings.TrimSpace(msg.out)
			_ = appendAudit(m.auditPath, map[string]any{"event": "kill_job", "out": strings.TrimSpace(msg.out)})
		case "newjob":
			m.msg = "spawned pid " + strings.TrimSpace(msg.out)
			_ = appendAudit(m.auditPath, map[string]any{"event": "spawn", "pid": strings.TrimSpace(msg.out), "cmd": m.newJob})
		case "log":
			m.logView = msg.out
		case "serve":
			m.msg = strings.TrimSpace(msg.out)
			_ = appendAudit(m.auditPath, map[string]any{"event": "serve_ui", "out": strings.TrimSpace(msg.out)})
		}
		return m, nil
	case execDoneMsg:
		err := finishRoot(m.bridge, msg.id, msg.claim, msg.rc, msg.out)
		ev := map[string]any{"event": "root_result", "id": msg.id, "decision": msg.decision, "cmd": msg.cmd,
			"rc": msg.rc, "goal": msg.goal}
		if msg.granted {
			ev["granted"] = true // what a later grantRecorded looks for (the Python CLI writes the same)
		}
		_ = appendAudit(m.auditPath, ev)
		m.seen[msg.id] = true
		m.msg = fmt.Sprintf("%s: exit %d", msg.decision, msg.rc)
		if err != nil {
			m.msg += "; " + err.Error()
		}
		m.refreshLocal()
		return m, nil // the running tick chain continues; a second one would double every poll
	case tea.KeyPressMsg:
		if m.tab != 6 {
			// Leaving the conversation page ends its live reader; the next visit starts a fresh one.
			m.chat.stopStream()
		}
		if m.tab == 6 && (m.mode == "" || m.mode == "normal") {
			return m.chatKey(msg)
		}
		return m.onKey(msg.String())
	case tea.MouseWheelMsg:
		if m.tab != 6 && (m.mode == "normal" || m.mode == "") {
			view := m.detailViewport()
			view, _ = view.Update(msg)
			m.pageScroll = view.YOffset()
			return m, nil
		}
	}
	if m.tab == 6 && m.chat.ready && (m.mode == "" || m.mode == "normal") {
		return m.chatUpdate(msg)
	}
	return m, nil
}

func (m *emberModel) refreshLocal() {
	reqs, err := loadRequests(m.bridge)
	if err == nil {
		var vis []emberReq
		for _, r := range reqs {
			if !m.seen[r.ID] {
				vis = append(vis, r)
			}
		}
		m.reqs = vis
		m.notifyRequests()
	}
	if m.sel >= len(m.reqs) {
		m.sel = max(0, len(m.reqs)-1)
	}
	if m.tab == 5 {
		m.events = tailLines(m.auditPath, 200)
	}
	m.syncQuestion()
}

func (m emberModel) onKey(k string) (tea.Model, tea.Cmd) {
	// text input modes
	if m.mode != "" && m.mode != "normal" {
		switch k {
		case "enter":
			return m.submitInput()
		case "esc":
			m.mode, m.edit, m.newJob = "normal", "", ""
			m.editRequest = nil
			return m, nil
		case "backspace":
			if len(m.edit) > 0 {
				// Remove the last character, not the last byte: cutting a byte split multi-byte text.
				_, size := utf8.DecodeLastRuneInString(m.edit)
				m.edit = m.edit[:len(m.edit)-size]
			}
			return m, nil
		default:
			if k == "space" {
				m.edit += " "
			} else if r := msgRunes(k); r != nil {
				m.edit += string(r)
			}
			return m, nil
		}
	}
	if m.tab != 6 {
		switch k {
		case "pgup", "pgdown", "home", "end":
			view := m.detailViewport()
			switch k {
			case "pgup":
				view.PageUp()
			case "pgdown":
				view.PageDown()
			case "home":
				view.GotoTop()
			case "end":
				view.GotoBottom()
			}
			m.pageScroll = view.YOffset()
			return m, nil
		}
	}
	switch k {
	case "q", "ctrl+c":
		return m, tea.Quit
	case "r":
		m.refreshLocal()
		return m, tea.Batch(append(sshRefresh(m), residentRefresh(m))...)
	}
	switch m.tab {
	case 0:
		return m.reqKeys(k)
	case 1:
		return m.jobKeys(k)
	case 7:
		return m.sessionKeys(k)
	}
	return m, nil
}

func (m emberModel) submitInput() (tea.Model, tea.Cmd) {
	switch m.mode {
	case "denywhy":
		instead := strings.TrimSpace(m.edit)
		m.mode, m.edit = "normal", ""
		if m.editRequest == nil || instead == "" {
			m.editRequest = nil
			return m, nil
		}
		r := *m.editRequest
		m.editRequest = nil
		if err := writeAnswer(m.bridge, currentUser(), r, "denied", r.Cmd, nil, "", "", instead); err != nil {
			m.msg = err.Error()
			m.refreshLocal()
			return m, nil
		}
		_ = appendAudit(m.auditPath, map[string]any{"event": "decision", "id": r.ID,
			"decision": "denied", "by": currentUser(), "instead": instead})
		m.seen[r.ID] = true
		m.msg = "denied " + r.ID + " with an instruction (sent as a steer)"
		m.refreshLocal()
		return m, nil
	case "sessionnew":
		name := strings.TrimSpace(m.edit)
		m.mode, m.edit = "normal", ""
		if name == "" {
			return m, nil
		}
		return m, createResidentSession(m, name)
	case "answer":
		if m.editRequest == nil {
			m.mode = "normal"
			return m, nil
		}
		r := *m.editRequest
		m.editRequest = nil
		ans := m.edit
		if len(r.Choices) > 0 {
			if n, err := strconv.Atoi(strings.TrimSpace(ans)); err == nil && n >= 1 && n <= len(r.Choices) {
				ans = r.Choices[n-1]
			}
		}
		m.mode, m.edit = "normal", ""
		if err := writeAnswer(m.bridge, currentUser(), r, "answered", "", nil, "", "", ans); err != nil {
			m.msg = err.Error()
			m.refreshLocal()
			return m, nil
		}
		_ = appendAudit(m.auditPath, map[string]any{"event": "decision", "id": r.ID,
			"decision": "answered", "by": currentUser(), "answer": ans})
		m.seen[r.ID] = true
		m.msg = "answered " + r.ID
		m.refreshLocal()
		// The answer reaches the agent through the bridge and stays in the goal's transcript; nothing is written to
		// its memory automatically (the agent decides what to remember).
		return m, nil
	case "edit":
		cmd := m.edit
		m.mode, m.edit = "normal", ""
		if cmd == "" || m.editRequest == nil {
			return m, nil
		}
		r := *m.editRequest
		m.editRequest = nil
		c, tmp, err := execRoot(cmd)
		if err != nil {
			m.msg = err.Error()
			return m, nil
		}
		token, err := claimRoot(m.bridge, currentUser(), r, "edited", cmd)
		if err != nil {
			os.Remove(tmp)
			m.msg = err.Error()
			m.refreshLocal()
			return m, nil
		}
		m.msg = "running under sudo (approve with your password)..."
		return m, tea.ExecProcess(c, execCallback(r.ID, "edited", cmd, tmp, token))
	case "newjob1":
		name := strings.TrimSpace(m.edit)
		m.mode, m.edit = "newjob2", ""
		m.newJob = name
		return m, nil
	case "newjob2":
		name := m.newJob
		cmd := m.edit
		m.mode, m.edit, m.newJob = "normal", "", ""
		if name == "" || cmd == "" {
			return m, nil
		}
		script := fmt.Sprintf("mkdir -p ~/env-jobs ~/env-logs; nohup setsid bash -c %s > ~/env-logs/%s.log 2>&1 & echo $! > ~/env-jobs/%s.pid; cat ~/env-jobs/%s.pid",
			shQuote(cmd), shQuote(name), shQuote(name), shQuote(name))
		return m, sshRun(m, "newjob", script)
	case "serve1":
		model := strings.TrimSpace(m.edit)
		m.mode, m.edit, m.newJob = "serve2", "", model
		return m, nil
	case "serve2":
		model := m.newJob
		placement := strings.TrimSpace(m.edit)
		m.mode, m.edit, m.newJob = "normal", "", ""
		if model == "" {
			return m, nil
		}
		if placement == "" {
			placement = "auto"
		}
		name := model
		args := append(append([]string{}, m.cli...), "ember", "serve", name, model, "--placement", placement)
		m.msg = "caminus ember serve " + model + " --placement " + placement + " ..."
		return m, func() tea.Msg {
			c := exec.Command(args[0], args[1:]...)
			out, err := c.CombinedOutput()
			return sshOutMsg{what: "serve", out: string(out), err: err}
		}
	}
	return m, nil
}

func execCallback(id, decision, cmd, tmp, claim string) func(error) tea.Msg {
	return func(err error) tea.Msg {
		rc := 0
		if ee, ok := err.(*exec.ExitError); ok {
			rc = ee.ExitCode()
		} else if err != nil {
			rc = 1
		}
		out, _ := os.ReadFile(tmp)
		os.Remove(tmp)
		return execDoneMsg{id: id, decision: decision, cmd: cmd, rc: rc, out: string(out), claim: claim}
	}
}

func (m emberModel) reqKeys(k string) (tea.Model, tea.Cmd) {
	switch k {
	case "up", "k":
		if m.sel > 0 {
			m.sel--
		}
	case "down", "j":
		if m.sel < len(m.reqs)-1 {
			m.sel++
		}
	case "D":
		// deny with an instruction: "no, do this instead" goes to the agent as a steer
		if m.sel < len(m.reqs) {
			r := m.reqs[m.sel]
			m.mode, m.edit, m.editRequest = "denywhy", "", &r
		}
	case "d":
		if m.sel < len(m.reqs) {
			r := m.reqs[m.sel]
			if err := writeAnswer(m.bridge, currentUser(), r, "denied", r.Cmd, nil, "", "", ""); err != nil {
				m.msg = err.Error()
				m.refreshLocal()
				return m, nil
			}
			_ = appendAudit(m.auditPath, map[string]any{"event": "decision", "id": r.ID,
				"decision": "denied", "by": currentUser()})
			m.seen[r.ID] = true
			m.msg = "denied " + r.ID
			m.refreshLocal()
		}
	case "s":
		if m.sel < len(m.reqs) {
			m.seen[m.reqs[m.sel].ID] = true
			m.refreshLocal()
		}
	case "a", "g":
		// g approves and grants this exact command for the rest of its goal (repeat uses then need one a)
		if m.sel < len(m.reqs) {
			r := m.reqs[m.sel]
			if r.kind() == "question" {
				if k == "g" {
					return m, nil
				}
				m.mode, m.edit = "answer", ""
				m.editRequest = &r
				return m, nil
			}
			if r.kind() == "github" {
				if k == "g" {
					return m, nil // a hand-off is one action, never granted for the goal
				}
				m.seen[r.ID] = true
				m.msg = "running on GitHub as you: " + terminalText(r.Summary, false)
				return m, approveHandoff(m, r)
			}
			c, tmp, err := execRoot(r.Cmd)
			if err != nil {
				m.msg = err.Error()
				return m, nil
			}
			grant := k == "g"
			token, err := claimRootGrant(m.bridge, currentUser(), r, "approved", r.Cmd, grant)
			if err != nil {
				os.Remove(tmp)
				m.msg = err.Error()
				m.refreshLocal()
				return m, nil
			}
			m.msg = "running under sudo (approve with your password)..."
			if grant {
				m.msg = "granted for this goal; running under sudo (approve with your password)..."
			}
			done := execCallback(r.ID, "approved", r.Cmd, tmp, token)
			goal := r.Goal
			return m, tea.ExecProcess(c, func(err error) tea.Msg {
				msg := done(err).(execDoneMsg)
				msg.goal, msg.granted = goal, grant
				return msg
			})
		}
	case "e":
		if m.sel < len(m.reqs) && m.reqs[m.sel].kind() == "root" { // (a GitHub hand-off is approved as filed)
			m.mode, m.edit = "edit", m.reqs[m.sel].Cmd
			r := m.reqs[m.sel]
			m.editRequest = &r
		}
	}
	return m, nil
}

func (m emberModel) jobKeys(k string) (tea.Model, tea.Cmd) {
	switch k {
	case "up", "k":
		if m.jobSel > 0 {
			m.jobSel--
		}
	case "down", "j":
		if m.jobSel < len(m.jobs)-1 {
			m.jobSel++
		}
	case "K":
		if m.jobSel < len(m.jobs) {
			j := m.jobs[m.jobSel]
			return m, sshRun(m, "kill", fmt.Sprintf("pid=$(cat ~/env-jobs/%s.pid 2>/dev/null) && kill \"$pid\" 2>/dev/null; rm -f ~/env-jobs/%s.pid; echo killed %s", shQuote(j.Name), shQuote(j.Name), j.Name))
		}
	case "l":
		if m.jobSel < len(m.jobs) {
			m.logName = m.jobs[m.jobSel].Name
			return m, sshRun(m, "log", fmt.Sprintf("tail -c 4000 ~/env-logs/%s.log 2>/dev/null || echo '(no log yet)'", shQuote(m.jobs[m.jobSel].Name)))
		}
	case "n":
		m.mode, m.edit = "newjob1", ""
	case "S":
		// (an agent-owned server ran outside the GPU guard, unregistered and blind to GPU-window leases: off for
		// this release; the guarded, registered way is caminus start)
		m.msg = "agent-owned servers are off in this release: start the model with `caminus start MODEL`, then /model"

	}
	return m, nil
}

// msgRunes returns the text of a key press (single runes), empty for special keys.
func msgRunes(k string) []rune {
	if len(k) == 1 {
		return []rune(k)
	}
	if strings.HasPrefix(k, "ctrl+") && len(k) == 6 {
		return []rune{rune(k[5] - 'a' + 1)}
	}
	return nil
}

func currentUser() string {
	if u := os.Getenv("USER"); u != "" {
		return u
	}
	return strconv.Itoa(os.Getuid())
}

// ---------------------------------------------------------------- view
// The named styles (emberTitle, emberSel, ...) live in theme.go, built from the
// theme's roles; a theme switch rebuilds them there.

func (m emberModel) render() string {
	// The chat is the only screen; the old pages' content opens as overlays (envpages.go). renderPages is kept only
	// for the page tests and is never reached from a key.
	m.initChat()
	return m.renderChat()
}

func (m emberModel) renderPages() string {
	var b strings.Builder
	b.WriteString(emberTitle.Render("CAMINUS ENV") + "  " + emberDim.Render("the agent's home: "+m.user+" on this machine") + "\n")
	for _, i := range emberNavigation {
		t := tabs[i]
		if i == m.tab {
			b.WriteString(" " + emberTabOn.Render(t) + " ")
		} else {
			b.WriteString(" " + emberDim.Render(t) + " ")
		}
	}
	b.WriteString("\n\n")
	if m.tab == 7 || m.tab == 5 || m.tab == 8 || m.tab == 3 || m.tab == 4 {
		view := m.detailViewport()
		b.WriteString(emberBorder.Width(view.Width()).Render(view.View()))
	} else {
		b.WriteString(emberBorder.Render(m.pane()))
	}
	switch m.mode {
	case "sessionnew":
		b.WriteString("\n\n" + emberWarn.Render("new session name (Enter creates; Esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "edit":
		b.WriteString("\n\n" + emberWarn.Render("edit root command (enter runs it under sudo, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "answer":
		b.WriteString("\n\n" + emberWarn.Render("answer the question (enter sends, esc cancels; a number picks a choice):") + "\n" + terminalText(m.edit, false) + "█")
	case "note":
		b.WriteString("\n\n" + emberWarn.Render("note for the agent (enter leaves it, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "newjob1":
		b.WriteString("\n\n" + emberWarn.Render("new detached job — name (enter, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "serve1":
		b.WriteString("\n\n" + emberWarn.Render("new server job — model name (enter, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "serve2":
		b.WriteString("\n\n" + emberWarn.Render("server "+m.newJob+" — placement auto|0|1|2 (enter=auto, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	case "newjob2":
		b.WriteString("\n\n" + emberWarn.Render("new job "+m.newJob+" — command (enter spawns it detached, esc cancels):") + "\n" + terminalText(m.edit, false) + "█")
	}
	status := fmt.Sprintf("ssh: %s  requests: %d  jobs: %d  msg: %s", map[bool]string{true: "OK", false: "FAIL"}[m.sshOK], len(m.reqs), len(m.jobs), terminalText(m.msg, true))
	b.WriteString("\n\n" + emberDim.Render(emberClip(status, m.width)))
	b.WriteString("\n" + emberDim.Render(emberClip("Tab/1–5 pages · Enter open · n new · PgUp/PgDn scroll · q quit · r refresh", m.width)))
	return b.String()
}

func (m emberModel) pane() string {
	switch m.tab {
	case 0:
		return m.reqPane()
	case 1:
		return m.jobsPane()
	case 2:
		return emberHead.Render("PROCESSES") + "\n" + sanitizedLines(m.procs)
	case 3:
		// Read-only: steering goes through /steer, so the old bridge notes are not shown or written from here.
		return emberHead.Render("SESSION MEMORY · "+residentText(m.agentName)) + "\n" +
			strings.Join(formatText("markdown", residentText(m.sessionMemory), max(40, m.width-8)), "\n")
	case 4:
		return m.ctxPane()
	case 7:
		return m.sessionsPane()
	case 8:
		return m.residentSettingsPane()
	case 9:
		return m.jobsPane() + "\n\n" + emberHead.Render("PROCESSES") + "\n" + sanitizedLines(m.procs)
	case 10:
		return m.contextPane()
	default:
		return m.auditPane()
	}
}

func (m emberModel) reqPane() string {
	lines := []string{emberHead.Render("REQUESTS (root: a approve, g approve + grant for the goal, e edit, d deny, D deny + what instead, s skip · ask: a answer)")}
	if len(m.reqs) == 0 {
		lines = append(lines, emberDim.Render("  none waiting"))
	}
	for i, r := range m.reqs {
		body, tag := terminalText(r.Cmd, false), "ROOT"
		if r.kind() == "question" {
			tag, body = "ASK ", terminalText(r.Text, false)
			if len(r.Choices) > 0 {
				body += "  [" + terminalText(strings.Join(r.Choices, " | "), false) + "]"
			}
		}
		line := fmt.Sprintf("  %s %s from %s (pid %d, %s)\n    %s", tag, terminalText(r.ID, false), terminalText(r.User, false), r.Pid, terminalText(r.CWD, false), body)
		if r.Reason != "" {
			line += "\n    reason: " + terminalText(r.Reason, false)
		}
		if i == m.sel {
			line = emberSel.Render(line)
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func (m emberModel) jobsPane() string {
	lines := []string{emberHead.Render("JOBS in the agent user (K kill, l tail log, n new job)")}
	if len(m.jobs) == 0 {
		lines = append(lines, emberDim.Render("  none"))
	}
	for i, j := range m.jobs {
		line := fmt.Sprintf("  %-16s pid=%-8s %s  log=%s", residentText(j.Name), residentText(j.Pid), residentText(j.State), residentText(j.Log))
		if i == m.jobSel {
			line = emberSel.Render(line)
		}
		lines = append(lines, line)
	}
	if m.logName != "" {
		lines = append(lines, "", emberHead.Render("log tail: "+residentText(m.logName)), emberDim.Render(residentText(m.logView)))
	}
	return strings.Join(lines, "\n")
}

func (m emberModel) ctxPane() string {
	lines := []string{emberHead.Render("SERVERS on this machine and their context windows (/props n_ctx)")}
	if len(m.servers) == 0 {
		lines = append(lines, emberDim.Render("  none running on 127.0.0.1:8092-8101"))
	}
	for _, s := range m.servers {
		lines = append(lines, fmt.Sprintf("  127.0.0.1:%d  %s  context %s", s.Port, residentText(s.Model), residentText(s.Nctx)))
	}
	return strings.Join(lines, "\n")
}

func sanitizedLines(ls []string) string {
	out := make([]string, len(ls))
	for i, l := range ls {
		out[i] = residentText(l)
	}
	return strings.Join(out, "\n")
}

// auditPane: the owner's audit log as a table, newest last: when, what happened, and its detail on one line (the
// command, the decision, the target), instead of each raw JSON record.
func (m emberModel) auditPane() string {
	lines := []string{emberHead.Render("AUDIT (tail of " + m.auditPath + ")")}
	if len(m.events) == 0 {
		lines = append(lines, emberDim.Render("  empty"))
	}
	w := max(40, m.width-8)
	for _, e := range m.events {
		var rec map[string]any
		if json.Unmarshal([]byte(e), &rec) != nil {
			lines = append(lines, emberDim.Render(ansi.Truncate(residentText(e), w, "…")))
			continue
		}
		when := ""
		if ts, ok := rec["ts"].(float64); ok {
			when = time.Unix(int64(ts), 0).Format("01-02 15:04")
		}
		event, _ := rec["event"].(string)
		var detail []string
		for _, k := range []string{"decision", "agent", "target", "user", "rc"} {
			if v, ok := rec[k]; ok && fmt.Sprint(v) != "" {
				detail = append(detail, k+" "+fmt.Sprint(v))
			}
		}
		for _, k := range []string{"cmd", "answer", "reason"} {
			if v, ok := rec[k].(string); ok && v != "" {
				detail = append(detail, strings.Join(strings.Fields(v), " "))
			}
		}
		evSt := lipgloss.NewStyle().Foreground(cText)
		switch {
		case strings.Contains(event, "root") || strings.Contains(event, "decision"):
			evSt = lipgloss.NewStyle().Foreground(cYellow)
		case strings.Contains(event, "stop") || strings.Contains(event, "cancel") || strings.Contains(event, "halt"):
			evSt = lipgloss.NewStyle().Foreground(cRed)
		}
		row := emberDim.Render(fmt.Sprintf("%-11s ", when)) + evSt.Render(fmt.Sprintf("%-16s ", residentText(event))) +
			emberDim.Render(residentText(strings.Join(detail, " · ")))
		lines = append(lines, ansi.Truncate(row, w, "…"))
	}
	return strings.Join(lines, "\n")
}

func (m emberModel) View() tea.View {
	frame := m.render()
	bg := themeBackground()
	if bg != nil {
		// the theme's background in every cell (a terminal may ignore the background request below); its pattern
		// behind the conversation only (not over the composer and status line, nor under an overlay)
		w := max(20, m.width)
		protect := 0
		if m.tab == 6 {
			protect = max(0, m.height-lipgloss.Height(m.chatBottom())-lipgloss.Height(m.chatFoot(w)))
		}
		frame = paintBackground(frame, protect, m.width, m.height, m.tab == 6 && !m.overlayOpen())
	}
	v := tea.NewView(frame)
	// (no tea.View BackgroundColor / ForegroundColor: GNOME Terminal ignores the request, and the renderer then
	// treated cells in that colour as the default and dropped it, so most rows stayed the terminal's dark; the
	// painter writes both into every cell instead)
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	v.WindowTitle = m.windowTitle()
	return v
}
