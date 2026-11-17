// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func writeQuestion(t *testing.T, bridge, id, text string, choices []string, rec string, ts float64) {
	t.Helper()
	for _, d := range []string{"req", "res"} {
		if err := os.MkdirAll(filepath.Join(bridge, d), 0o770); err != nil {
			t.Fatal(err)
		}
	}
	data, _ := json.Marshal(emberReq{ID: id, Kind: "question", Agent: "resident", Text: text, Choices: choices, Recommended: rec, Ts: ts})
	if err := os.WriteFile(filepath.Join(bridge, "req", id+".json"), data, 0o660); err != nil {
		t.Fatal(err)
	}
}

func promptFixture(t *testing.T) (emberModel, string) {
	m := chatFixture(t)
	m.width, m.height = 120, 40
	bridge := t.TempDir()
	m.bridge, m.auditPath, m.seen = bridge, filepath.Join(t.TempDir(), "audit.jsonl"), map[string]bool{}
	saved := promptGrace
	promptGrace = 0 // (the tests press at once; TestARequestNeverTakesTheKeyboardMidSentence checks the grace)
	t.Cleanup(func() { promptGrace = saved })
	return m, bridge
}

func answerFile(t *testing.T, bridge, id string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(bridge, "res", id+".json"))
	if err != nil {
		t.Fatalf("no answer written for %s: %v", id, err)
	}
	var v map[string]any
	_ = json.Unmarshal(data, &v)
	return v
}

func press(t *testing.T, m emberModel, keys ...tea.KeyPressMsg) emberModel {
	t.Helper()
	for _, k := range keys {
		next, _ := m.chatKey(k)
		m = next.(emberModel)
	}
	return m
}

var (
	kDown  = tea.KeyPressMsg{Code: tea.KeyDown}
	kEnter = tea.KeyPressMsg{Code: tea.KeyEnter}
	kEsc   = tea.KeyPressMsg{Code: tea.KeyEscape}
	kTab   = tea.KeyPressMsg{Code: tea.KeyTab}
)

func TestARootRequestAsksWhereYouTypeAndNoDenies(t *testing.T) {
	m, bridge := promptFixture(t)
	writeReq(t, bridge, "r1", "apt-get install -y python3-pytest", float64(time.Now().Unix()))
	m.refreshLocal()
	m.syncQuestion()
	box := ansi.Strip(m.chatComposer())
	for _, want := range []string{"root · 1 of 1", "left", "apt-get install -y python3-pytest", "1. Yes", "› 2. No", "3. Tell it something else", "Esc later"} {
		if !strings.Contains(box, want) {
			t.Fatalf("the prompt takes the composer's place with %q:\n%s", want, box)
		}
	}
	m = press(t, m, kEnter)
	if v := answerFile(t, bridge, "r1"); v["decision"] != "denied" {
		t.Fatalf("No is preselected: Enter alone denies: %v", v)
	}
	if m.chat.question != nil {
		t.Fatal("the prompt closes once decided")
	}
}

func TestTellItSomethingElseDeniesWithTheText(t *testing.T) {
	m, bridge := promptFixture(t)
	writeReq(t, bridge, "r1", "rm -rf /var/cache/foo", float64(time.Now().Unix()))
	m.refreshLocal()
	m.syncQuestion()
	m = press(t, m, tea.KeyPressMsg{Code: '3', Text: "3"})
	if !m.chat.questionCustom || !strings.Contains(ansi.Strip(m.chatComposer()), "Enter sends it") {
		t.Fatal("the last option opens a text field")
	}
	for _, r := range "use the cache dir in /tmp" {
		m = press(t, m, tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	m = press(t, m, kEnter)
	v := answerFile(t, bridge, "r1")
	if v["decision"] != "denied" || !strings.Contains(string(mustJSON(v)), "use the cache dir in /tmp") {
		t.Fatalf("a denial that carries what to do instead: %v", v)
	}
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

func TestAQuestionsChoicesAnswerWithArrowsEnterOrADigit(t *testing.T) {
	m, bridge := promptFixture(t)
	now := float64(time.Now().Unix())
	writeQuestion(t, bridge, "q1", "Which Python?", []string{"3.12", "3.14", "both"}, "3.14", now)
	m.refreshLocal()
	m.syncQuestion()
	box := ansi.Strip(m.chatComposer())
	if !strings.Contains(box, "question · 1 of 1") || !strings.Contains(box, "› 2. 3.14 (Recommended)") || !strings.Contains(box, "4. Answer in your own words") {
		t.Fatalf("the recommended choice is marked and preselected:\n%s", box)
	}
	m = press(t, m, kDown, kEnter) // from the recommended (2) down to 3
	if v := answerFile(t, bridge, "q1"); v["answer"] != "both" {
		t.Fatalf("↓ Enter answers the next choice: %v", v)
	}
	m2, bridge2 := promptFixture(t)
	writeQuestion(t, bridge2, "q2", "Which Python?", []string{"3.12", "3.14"}, "", now)
	m2.refreshLocal()
	m2.syncQuestion()
	press(t, m2, tea.KeyPressMsg{Code: '2', Text: "2"})
	if v := answerFile(t, bridge2, "q2"); v["answer"] != "3.14" {
		t.Fatalf("2 answers the second: %v", v)
	}
}

func TestSomethingElseAnswersInYourOwnWords(t *testing.T) {
	m, bridge := promptFixture(t)
	writeQuestion(t, bridge, "q1", "Which Python?", []string{"3.12"}, "", float64(time.Now().Unix()))
	m.refreshLocal()
	m.syncQuestion()
	m = press(t, m, kDown, kEnter)
	for _, r := range "whatever the CI uses" {
		m = press(t, m, tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	press(t, m, kEnter)
	if v := answerFile(t, bridge, "q1"); v["answer"] != "whatever the CI uses" {
		t.Fatalf("free text answers: %v", v)
	}
}

func TestEscLeavesItWaitingAndTabBringsItBack(t *testing.T) {
	m, bridge := promptFixture(t)
	writeReq(t, bridge, "r1", "systemctl restart foo", float64(time.Now().Unix()))
	m.chat.compose.SetValue("")
	m.refreshLocal()
	m.syncQuestion()
	m = press(t, m, kEsc)
	if m.chat.question != nil || !m.chat.compose.Focused() {
		t.Fatal("Esc gives the composer back")
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "r1.json")); err == nil {
		t.Fatal("Esc decides nothing")
	}
	if !strings.Contains(ansi.Strip(m.chatBottom()), "1 waiting · Tab answers") {
		t.Fatalf("a line above the composer says it waits:\n%s", ansi.Strip(m.chatBottom()))
	}
	m.syncQuestion()
	if m.chat.question != nil {
		t.Fatal("a refresh does not push a request put off back up")
	}
	m = press(t, m, kTab)
	if m.chat.question == nil || m.chat.question.ID != "r1" {
		t.Fatal("Tab brings it back")
	}
}

func TestTwoWaitingSayOneOfTwoAndOneEscHidesBoth(t *testing.T) {
	m, bridge := promptFixture(t)
	now := float64(time.Now().Unix())
	writeReq(t, bridge, "r1", "apt-get install -y jq", now-10)
	writeQuestion(t, bridge, "q1", "Which Python?", []string{"3.12", "both"}, "", now)
	m.refreshLocal()
	m.syncQuestion()
	if box := ansi.Strip(m.chatComposer()); !strings.Contains(box, "root · 1 of 2") || !strings.Contains(box, "apt-get install -y jq") {
		t.Fatalf("the oldest first, with how many wait:\n%s", box)
	}
	m = press(t, m, kEsc)
	m.syncQuestion()
	if m.chat.question != nil || !m.chat.compose.Focused() {
		t.Fatal("one Esc gives the composer back, whatever else waits (keys meant for one never reach another)")
	}
	if !strings.Contains(ansi.Strip(m.chatBottom()), "2 waiting · Tab answers") {
		t.Fatalf("both still wait:\n%s", ansi.Strip(m.chatBottom()))
	}
	writeQuestion(t, bridge, "q2", "Run the slow tests too?", []string{"yes", "no"}, "", now+5)
	m.refreshLocal()
	m.syncQuestion()
	if m.chat.question == nil || m.chat.question.ID != "q2" {
		t.Fatal("a new request comes up where you type")
	}
}

func TestARequestsRecordIsOneLineWithItsStateAndTheModelsNoteNeverShows(t *testing.T) {
	m, bridge := promptFixture(t)
	writeReq(t, bridge, "r1", "apt-get install -y jq", float64(time.Now().Unix()))
	m.refreshLocal()
	call := `{"command":"apt-get install -y jq","reason":"tests need it"}`
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "c1", "type": "function", "function": map[string]any{"name": "ask_root", "arguments": call}}}}},
		{Seq: 2, Goal: "original", Kind: "approval_requested", Data: map[string]any{"id": "r1", "cmd": "apt-get install -y jq", "reason": "tests need it"}},
		{Seq: 3, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "ask_root", "call_id": "c1", "result": map[string]any{
			"id": "r1", "pending": true, "note": "your human has not answered yet. Work on what does not need the answer"}}},
		{Seq: 4, Goal: "original", Kind: "question", Data: map[string]any{"id": "q9", "text": "Which Python?", "choices": []any{"3.12", "3.14"}}},
		{Seq: 5, Goal: "original", Kind: "tool_result", Data: map[string]any{"name": "ask_user", "call_id": "c2", "result": map[string]any{
			"id": "q9", "decision": "answered", "by": "timeout", "answer": "(no answer) Silence is not approval"}}},
	}
	for _, expanded := range []bool{false, true} {
		m.chat.expanded = expanded
		m.chat.rendered, m.chat.content = 0, ""
		m.syncChat()
		v := ansi.Strip(m.chat.content)
		if !strings.Contains(v, "root: apt-get install -y jq · waiting") || !strings.Contains(v, "? Which Python? · recommended (timed out)") || strings.Contains(v, "(3.12 / 3.14)") {
			t.Fatalf("expanded=%v: one line each, with the state:\n%s", expanded, v)
		}
		if strings.Contains(v, "has not answered yet") || strings.Contains(v, "Silence is not approval") || regexp.MustCompile(`(?m)^\s*\? · `).MatchString(v) {
			t.Fatalf("expanded=%v: no model-facing note and no orphan status line:\n%s", expanded, v)
		}
	}
}

func TestTheActivityLineDoesNotSayTypeWhileThePromptAsks(t *testing.T) {
	m, bridge := promptFixture(t)
	writeQuestion(t, bridge, "q1", "Which Python?", []string{"3.12"}, "", float64(time.Now().Unix()))
	m.refreshLocal()
	m.syncQuestion()
	m.resident.Goals = []residentGoal{{ID: "original", State: "working", Prompt: "p"}}
	m.chat.target = "original"
	m.residentEvents = []residentEvent{{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
		map[string]any{"id": "w", "type": "function", "function": map[string]any{"name": "wait", "arguments": `{"until":"message"}`}}}}}}
	if v := ansi.Strip(m.activityLine(120)); strings.Contains(v, "type below") {
		t.Fatalf("the prompt asks; the activity line must not say type below: %q", v)
	}
}

func TestAnOlderRequestFileStillGetsItsRecommendedChoice(t *testing.T) {
	m, bridge := promptFixture(t)
	writeQuestion(t, bridge, "q1", "Which Python?", []string{"3.12", "3.14"}, "", float64(time.Now().Unix()))
	m.residentEvents = []residentEvent{{Seq: 1, Goal: "original", Kind: "assistant", Data: map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
		map[string]any{"id": "c", "type": "function", "function": map[string]any{"name": "ask_user",
			"arguments": `{"question":"Which Python?","choices":["3.12","3.14"],"recommended":"3.14"}`}}}}}}
	m.refreshLocal()
	m.syncQuestion()
	if box := ansi.Strip(m.chatComposer()); !strings.Contains(box, "› 2. 3.14 (Recommended)") {
		t.Fatalf("the call's recommended choice, when the file lacks it:\n%s", box)
	}
}

func TestHelpHasNoApprovalsScreen(t *testing.T) {
	help := ansi.Strip(strings.Join(helpLines(120), "\n"))
	for _, gone := range []string{"/approvals", "/deny", "Ctrl+A"} {
		if strings.Contains(help, gone) {
			t.Fatalf("%s is gone: requests ask where you type", gone)
		}
	}
	if !strings.Contains(help, "Tab") {
		t.Fatal("/help says Tab answers a waiting request")
	}
}

func TestARecordFollowsTheBridgeOnceAnswered(t *testing.T) {
	m, bridge := promptFixture(t)
	now := float64(time.Now().Unix())
	writeReq(t, bridge, "r1", "apt-get install -y jq", now-20)
	writeQuestion(t, bridge, "q1", "Which Python version should the project target, given the CI and the servers?",
		[]string{"3.12 (the system Python)", "3.14 (newest)", "both"}, "", now-10)
	writeReq(t, bridge, "r2", "systemctl restart foo", now)
	m.refreshLocal()
	m.residentEvents = []residentEvent{
		{Seq: 1, Goal: "original", Kind: "approval_requested", Data: map[string]any{"id": "r1", "cmd": "apt-get install -y jq"}},
		{Seq: 2, Goal: "original", Kind: "question", Data: map[string]any{"id": "q1", "text": "Which Python version should the project target, given the CI and the servers?",
			"choices": []any{"3.12 (the system Python)", "3.14 (newest)", "both"}}},
		{Seq: 3, Goal: "original", Kind: "approval_requested", Data: map[string]any{"id": "r2", "cmd": "systemctl restart foo"}},
	}
	m.syncQuestion()
	m = press(t, m, kEnter) // r1: No (preselected)
	m.syncQuestion()
	m = press(t, m, tea.KeyPressMsg{Code: '1', Text: "1"}) // q1: the first choice
	// r2 was decided by the resident's watch elsewhere: only its event says so (its files are gone)
	_ = os.Remove(filepath.Join(bridge, "req", "r2.json"))
	m.residentEvents = append(m.residentEvents, residentEvent{Seq: 4, Goal: "original", Kind: "request_answered", Data: map[string]any{"request": "r2", "decision": "approved"}})
	m.refreshLocal()
	m.width = 90
	m.syncChat()
	v := ansi.Strip(m.chat.content)
	for _, want := range []string{"root: apt-get install -y jq · no", "· 3.12 (the system Python) (you)", "root: systemctl restart foo · yes"} {
		if !strings.Contains(v, want) {
			t.Fatalf("the record follows the bridge: %q missing:\n%s", want, v)
		}
	}
	if strings.Contains(v, "waiting") || strings.Contains(v, "3.14 (newest)") {
		t.Fatalf("answered: no waiting, no choices:\n%s", v)
	}
}

func TestARequestNeverTakesTheKeyboardMidSentence(t *testing.T) {
	m, bridge := promptFixture(t)
	m.chat.compose.SetValue("then use 2 workers")
	writeReq(t, bridge, "r1", "systemctl restart foo", float64(time.Now().Unix()))
	m.refreshLocal()
	m.syncQuestion()
	if m.chat.question != nil || !m.chat.compose.Focused() {
		t.Fatal("a request arriving while you type waits; the keyboard stays yours")
	}
	if !strings.Contains(ansi.Strip(m.chatBottom()), "1 waiting · Tab answers") {
		t.Fatalf("it says it waits:\n%s", ansi.Strip(m.chatBottom()))
	}
	m = press(t, m, kTab)
	if m.chat.question == nil || m.chat.compose.Value() != "then use 2 workers" {
		t.Fatal("Tab brings it up and the draft is kept")
	}
	promptGrace = time.Hour // a prompt that just opened ignores Enter and digits
	m.chat.questionAt = time.Now()
	m = press(t, m, kEnter, tea.KeyPressMsg{Code: '1', Text: "1"})
	if _, err := os.Stat(filepath.Join(bridge, "res", "r1.json")); err == nil || m.chat.question == nil {
		t.Fatal("a key already on its way decided a request nobody had read")
	}
}

// The countdown follows the resident's away rule (approval.wait_left; test_the_away_rule_matches_the_panels_countdown
// holds the same cases): question_timeout while the human is at the screen, and while away only the cap runs.
func TestQuestionCountdownFollowsPresence(t *testing.T) {
	r := emberReq{Kind: "question", Ts: 1000, Expires: 1900, AwayMax: 28800, AwayAfter: 300}
	at := func(s float64) time.Time { return time.UnixMilli(int64(s * 1000)) }
	for _, c := range []struct {
		now  float64
		p    presence
		left float64
		away bool
		text string
	}{
		{1899, presence{}, 1, false, "0:01 left"},
		{1900, presence{}, 0, false, "timing out"},
		{1500, presence{Seen: 1490}, 400, false, "6:40 left"},                // at the screen since before it was asked
		{1500, presence{Seen: 1199}, 28300, true, "waits while you're away"}, // no input for 301 s
		{5200, presence{Seen: 5100, Back: 5000}, 700, false, "11:40 left"},   // back at 5000: question_timeout from there
		{29750, presence{Seen: 29700, Back: 29500}, 50, false, "0:50 left"},  // the cap still holds
		{29900, presence{Seen: 20000, Back: 20000}, -100, true, "timing out"},
	} {
		left, away := questionLeft(r, c.p, at(c.now))
		if left != time.Duration(c.left*float64(time.Second)) || away != c.away {
			t.Errorf("at %v with %+v: %v away=%v, want %vs away=%v", c.now, c.p, left, away, c.left, c.away)
		}
		if got := timeLeft(r, c.p, at(c.now)); got != c.text {
			t.Errorf("at %v with %+v: %q, want %q", c.now, c.p, got, c.text)
		}
	}
	long := r
	long.Expires = 1000 + 7200
	if got := timeLeft(long, presence{Seen: 1000}, at(1000)); got != "2:00:00 left" {
		t.Errorf("hours: %q", got)
	}
	root := emberReq{Kind: "root", Ts: 1000, Expires: 1900}
	if got := timeLeft(root, presence{Seen: 1}, at(1300)); got != "10:00 left" {
		t.Errorf("a root request keeps its plain 15 minutes whether or not you are there: %q", got)
	}
}

// A key in the chat stamps the bridge for the resident, and the prompt says the question waits while you are away.
func TestPresenceStampReachesTheBridgeAndThePrompt(t *testing.T) {
	m, bridge := promptFixture(t)
	now := float64(time.Now().Unix())
	writeQuestion(t, bridge, "q1", "Delete or archive?", []string{"delete", "archive"}, "archive", now)
	path := filepath.Join(bridge, "req", "q1.json")
	var r map[string]any
	data, _ := os.ReadFile(path)
	_ = json.Unmarshal(data, &r)
	r["expires"], r["away_max"], r["away_after"] = now+900, 28800, 300
	data, _ = json.Marshal(r)
	if err := os.WriteFile(path, data, 0o660); err != nil {
		t.Fatal(err)
	}
	stale, _ := json.Marshal(presence{Seen: now - 3600, Back: now - 7200})
	if err := os.WriteFile(filepath.Join(bridge, "presence.json"), stale, 0o660); err != nil {
		t.Fatal(err)
	}
	m.refreshLocal()
	m.syncQuestion()
	if v := ansi.Strip(m.questionBox()); !strings.Contains(v, "waits while you're away") {
		t.Fatalf("away an hour: the prompt says it waits:\n%s", v)
	}
	next, _ := m.Update(kDown) // (Update: the stamp is taken there, before any page handles the key)
	m = next.(emberModel)
	m.refreshLocal()
	var p presence
	data, _ = os.ReadFile(filepath.Join(bridge, "presence.json"))
	if json.Unmarshal(data, &p) != nil || p.Seen < now || p.Back != p.Seen {
		t.Fatalf("a key after an hour away stamps the return at once: %s", data)
	}
	if v := ansi.Strip(m.questionBox()); !regexp.MustCompile(`1[45]:\d\d left`).MatchString(v) {
		t.Fatalf("back: the full question_timeout from now:\n%s", v)
	}
	written := p.Seen
	next, _ = m.Update(tea.KeyPressMsg{Code: tea.KeyUp})
	m = next.(emberModel)
	m.refreshLocal()
	data, _ = os.ReadFile(filepath.Join(bridge, "presence.json"))
	_ = json.Unmarshal(data, &p)
	if p.Seen != written {
		t.Fatal("while you keep typing the stamp is rewritten at most every 30 s")
	}
}
