// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func questionFixture(t *testing.T, choices []string) emberModel {
	t.Helper()
	m := chatFixture(t)
	m.bridge = t.TempDir()
	m.auditPath = filepath.Join(t.TempDir(), "audit.jsonl")
	m.resident.Goals[0].State = "waiting_user"
	writeQuestion(t, m.bridge, emberReq{ID: "question-a", Agent: "resident", Goal: "original", User: "agent", Kind: "question", Text: "Which approach should I use?", Choices: choices})
	m.chat.compose.SetValue("keep this steering draft")
	m.refreshLocal()
	m.syncChat()
	return m
}

func writeQuestion(t *testing.T, bridge string, r emberReq) {
	t.Helper()
	os.MkdirAll(filepath.Join(bridge, "req"), 0770)
	os.MkdirAll(filepath.Join(bridge, "res"), 0770)
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if err = writeBridgeJSON(filepath.Join(bridge, "req", r.ID+".json"), b); err != nil {
		t.Fatal(err)
	}
}

func questionResponse(t *testing.T, m emberModel, id string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(m.bridge, "res", id+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var r map[string]any
	if err = json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestInlineQuestionSelectorAnswersOnlyAfterEnter(t *testing.T) {
	m := questionFixture(t, []string{"Inspect read-only", "Ask for the required root command"})
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyF2})
	m = next.(emberModel)
	view := ansi.Strip(m.render())
	for _, want := range []string{"YOUR ANSWER", "↑/↓ select · Enter answer", "Inspect read-only", "Write your own answer"} {
		if !strings.Contains(view, want) {
			t.Fatalf("missing %q", want)
		}
	}
	if _, err := os.Stat(filepath.Join(m.bridge, "res", "question-a.json")); !os.IsNotExist(err) {
		t.Fatal("rendering submitted a response")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyDown})
	m = next.(emberModel)
	if m.chat.questionChoice != 1 {
		t.Fatal("Down did not select the next choice")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	r := questionResponse(t, m, "question-a")
	if r["decision"] != "answered" || r["answer"] != "Ask for the required root command" {
		t.Fatal(r)
	}
	if m.chat.compose.Value() != "keep this steering draft" || m.chat.question != nil {
		t.Fatal("answer erased steering draft or retained question")
	}
}

func TestInlineCustomAnswerPreservesUnicodeAndRequiresText(t *testing.T) {
	m := questionFixture(t, []string{"First", "Second"})
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyF2})
	m = next.(emberModel)
	next, _ = m.questionKey(tea.KeyPressMsg{Code: 'e', Text: "e"})
	m = next.(emberModel)
	next, cmd := m.questionKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if cmd != nil || m.chat.question == nil {
		t.Fatal("blank custom answer submitted")
	}
	text := "My own café answer\nwith another line"
	next, _ = m.chatUpdate(tea.PasteMsg{Content: text})
	m = next.(emberModel)
	next, _ = m.questionKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if questionResponse(t, m, "question-a")["answer"] != text {
		t.Fatal("custom answer changed")
	}
}

func TestInlineQuestionRefreshCannotRetargetEnter(t *testing.T) {
	m := questionFixture(t, []string{"Original answer"})
	writeBridgeJSON(filepath.Join(m.bridge, "res", "question-a.json"), []byte(`{"id":"question-a","decision":"withdrawn"}`))
	writeQuestion(t, m.bridge, emberReq{ID: "question-b", Agent: "resident", Goal: "original", Kind: "question", Text: "A different question", Choices: []string{"Different answer"}})
	m.refreshLocal()
	m.chat.questionArmed = true
	next, _ := m.questionKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if _, err := os.Stat(filepath.Join(m.bridge, "res", "question-b.json")); !os.IsNotExist(err) {
		t.Fatal("Enter answered a different request after refresh")
	}
	if questionResponse(t, m, "question-a")["decision"] != "withdrawn" {
		t.Fatal("withdrawal overwritten")
	}
	if m.chat.questionError == "" {
		t.Fatal("stale question was not reported")
	}
}

func TestInlineQuestionsRespectAgentGoalAndRootKind(t *testing.T) {
	m := questionFixture(t, []string{"One"})
	m.chat.question = nil
	m.reqs = []emberReq{{ID: "wrong-agent", Agent: "other", Goal: "original", Kind: "question"}, {ID: "wrong-goal", Agent: "resident", Goal: "other", Kind: "question"}, {ID: "root", Agent: "resident", Goal: "original", Kind: "root", Cmd: "root command"}}
	m.syncQuestion()
	if m.chat.question != nil {
		t.Fatal("another session, goal or root request entered question selector")
	}
	r := m.reqs[2]
	m.chat.question = &r
	next, _ := m.answerQuestion("approved")
	m = next.(emberModel)
	if m.chat.questionError == "" {
		t.Fatal("question answer allowed root approval")
	}
}

func TestInlineQuestionEscapeDefersWithoutLosingDraft(t *testing.T) {
	m := questionFixture(t, []string{"One"})
	next, _ := m.questionKey(tea.KeyPressMsg{Code: tea.KeyEscape})
	m = next.(emberModel)
	m.syncQuestion()
	if m.chat.question != nil || m.chat.compose.Value() != "keep this steering draft" {
		t.Fatal("deferral lost a draft or reopened the same question")
	}
	if _, err := os.Stat(filepath.Join(m.bridge, "res", "question-a.json")); !os.IsNotExist(err) {
		t.Fatal("Escape submitted an answer")
	}
}

func TestQuestionFrameFitsLongChoicesAndCustomInput(t *testing.T) {
	for _, size := range [][2]int{{40, 18}, {80, 24}, {140, 40}} {
		m := questionFixture(t, []string{strings.Repeat("Long first choice ", 8), strings.Repeat("Second choice ", 10)})
		m.width, m.height = size[0], size[1]
		m.chat.questionArmed = true
		for _, custom := range []bool{false, true} {
			m.chat.questionCustom = custom
			frame := m.render()
			if len(strings.Split(frame, "\n")) > m.height {
				t.Fatalf("%v custom=%v frame too tall", size, custom)
			}
			for _, line := range strings.Split(frame, "\n") {
				if ansi.StringWidth(line) > m.width {
					t.Fatalf("%v custom=%v frame too wide", size, custom)
				}
			}
		}
	}
}

func TestQuestionArrivalDoesNotTurnDraftSubmissionIntoAnAnswer(t *testing.T) {
	m := questionFixture(t, []string{"First", "Second"})
	for _, r := range "jk2e" {
		next, _ := m.chatKey(tea.KeyPressMsg{Code: r, Text: string(r)})
		m = next.(emberModel)
	}
	next, _ := m.chatUpdate(tea.PasteMsg{Content: " pasted"})
	m = next.(emberModel)
	if m.chat.compose.Value() != "keep this steering draftjk2e pasted" || m.chat.questionArmed {
		t.Fatal("ordinary draft input activated or entered the picker")
	}
	next, cmd := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if cmd != nil || m.chat.question == nil {
		t.Fatal("question arrival captured a draft Enter as consent")
	}
	if _, err := os.Stat(filepath.Join(m.bridge, "res", "question-a.json")); !os.IsNotExist(err) {
		t.Fatal("draft Enter submitted an answer")
	}
	next, _ = m.chatKey(tea.KeyPressMsg{Code: tea.KeyF2})
	m = next.(emberModel)
	if !m.chat.questionArmed {
		t.Fatal("F2 did not explicitly activate the picker")
	}
}

func TestQuestionWithoutDraftOpensPickerDirectly(t *testing.T) {
	m := questionFixture(t, []string{"First", "Second"})
	m.chat.question = nil
	m.chat.compose.Reset()
	m.syncQuestion()
	if !m.chat.questionArmed || !strings.Contains(ansi.Strip(m.render()), "YOUR ANSWER") {
		t.Fatal("empty composer should show the inline picker immediately")
	}
}

func TestQuestionToolJSONIsReplacedByHumanQuestionAndAnswer(t *testing.T) {
	m := chatFixture(t)
	call := residentEvent{Kind: "assistant", Data: map[string]any{"tool_calls": []any{map[string]any{"function": map[string]any{"name": "ask_user", "arguments": `{"question":"What next?","choices":["A","B"]}`}}}}}
	if m.chat.eventBlock(call) != "" {
		t.Fatal("raw ask_user JSON still displayed")
	}
	answer := residentEvent{Kind: "tool_result", Data: map[string]any{"name": "ask_user", "result": map[string]any{"decision": "answered", "answer": "Choice B"}}}
	if !strings.Contains(ansi.Strip(m.chat.eventBlock(answer)), "Choice B") {
		t.Fatal("human answer missing from history")
	}
	output := residentEvent{Kind: "tool_result", Data: map[string]any{"name": "terminal", "result": map[string]any{"output": "line one\r\nline two\r\n"}}}
	if strings.Contains(ansi.Strip(m.chat.eventBlock(output)), `\r`) {
		t.Fatal("normal PTY newlines rendered as literal escapes")
	}
}

func TestQuestionArrivingWhileBrowsingRequiresF2(t *testing.T) {
	m := chatFixture(t)
	m.bridge = t.TempDir()
	m.auditPath = filepath.Join(t.TempDir(), "audit.jsonl")
	m.resident.Goals[0].State = "waiting_user"
	writeQuestion(t, m.bridge, emberReq{ID: "question-browse", Agent: "resident", Goal: "original", User: "agent",
		Kind: "question", Text: "Which approach should I use?", Choices: []string{"one", "two"}})
	m.chat.compose.Blur() // Esc: browsing history, empty draft
	m.refreshLocal()
	m.syncChat()
	if m.chat.questionArmed {
		t.Fatal("a question that arrives while browsing must not arm itself")
	}
	next, _ := m.chatKey(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(emberModel)
	if !m.chat.compose.Focused() {
		t.Fatal("Enter while browsing should return to typing")
	}
	if m.chat.questionArmed {
		t.Fatal("Enter while browsing must not answer the question")
	}
}
