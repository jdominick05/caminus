// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

// emberquestion.go: the inline prompt. A pending request of the session in view (a root command, a GitHub hand-off
// or a question) takes the composer's place: the command or the question, numbered options to pick with the arrows,
// Enter or a digit, and a last option to say something else in your own words. Esc leaves it waiting (a line above
// the composer says so; Tab brings it back, and so does the next request). There is no separate approvals screen.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// defaultRequestWait is how long a request waits when its file does not say (the resident's question_timeout).
const defaultRequestWait = 15 * time.Minute

// sessionRequests are the pending requests of the session in view, oldest first.
func (m emberModel) sessionRequests() []emberReq {
	var out []emberReq
	for _, r := range m.reqs {
		if r.Agent == "" || r.Agent == m.agentName {
			out = append(out, r)
		}
	}
	return out
}

// syncQuestion opens the prompt on the oldest pending request not put off with Esc; a request answered elsewhere
// closes it. A new request opens it even when an older one was put off.
func (m *emberModel) syncQuestion() {
	if !m.chat.ready || m.chat.sending {
		return
	}
	pending := m.sessionRequests()
	if q := m.chat.question; q != nil {
		for _, r := range pending {
			if r.ID == q.ID && r.Digest == q.Digest {
				return
			}
		}
		m.closePrompt() // answered, withdrawn or timed out elsewhere: the next one, if any, below
	}
	for _, r := range pending {
		if m.chat.later[r.ID] {
			continue
		}
		if strings.TrimSpace(m.chat.compose.Value()) != "" && m.chat.compose.Focused() {
			// you are typing: a request never takes the keyboard mid-sentence (the next Enter would decide it);
			// it waits on the line above the composer, and Tab brings it up
			if m.chat.later == nil {
				m.chat.later = map[string]bool{}
			}
			m.chat.later[r.ID] = true
			continue
		}
		m.openPrompt(r)
		return
	}
}

// promptGrace is how long a freshly opened prompt ignores Enter and digits: a key already on its way (the end of a
// sentence, a double Enter) must not decide a request the human has not read.
var promptGrace = 600 * time.Millisecond

func (m *emberModel) openPrompt(r emberReq) {
	copy := r
	m.chat.question = &copy
	m.chat.questionAt = time.Now()
	m.chat.questionArmed = true
	m.chat.questionCustom = false
	m.chat.questionError = ""
	m.chat.questionChoice = 0
	if r.Recommended == "" && r.kind() == "question" {
		r.Recommended = m.askedRecommended(r.Text) // an older resident's request file does not carry it; its call does
		m.chat.question.Recommended = r.Recommended
	}
	if r.kind() != "question" {
		// No is preselected for a root command and a GitHub hand-off (the owner's choice): one Enter on a hand-off
		// posted it under their name, with no second check
		m.chat.questionChoice = 1
	}
	opts := promptOptions(r)
	for i, o := range opts {
		if r.Recommended != "" && o == r.Recommended {
			m.chat.questionChoice = i // the recommended choice is preselected
		}
	}
	m.chat.questionInput = textarea.New()
	m.chat.questionInput.Prompt = "› "
	m.chat.questionInput.ShowLineNumbers = false
	m.chat.questionInput.CharLimit = 0
	m.chat.questionInput.MaxHeight = 0
	m.chat.questionInput.Placeholder = map[bool]string{true: "what it should do instead", false: "your answer"}[r.kind() != "question"]
	m.chat.questionInput.SetWidth(max(16, m.width-6))
	m.chat.questionInput.SetHeight(1)
	m.chat.compose.Blur() // (the draft stays; it comes back when the prompt closes)
}

func (m *emberModel) closePrompt() {
	m.chat.question = nil
	m.chat.questionArmed = false
	m.chat.questionCustom = false
	m.chat.questionError = ""
	m.chat.compose.Focus()
}

// promptOptions: a question's choices, or Yes / No for a root command and a hand-off. The free-text option is not
// among them (it is drawn apart, under a rule).
func promptOptions(r emberReq) []string {
	if r.kind() == "question" {
		return r.Choices
	}
	return []string{"Yes", "No"}
}

func freeTextLabel(r emberReq) string {
	if r.kind() == "question" {
		return "Answer in your own words"
	}
	return "Tell it something else"
}

// questionLeft is how long the request still waits before its timeout answers it (a question: the advisor's pick or
// its recommended option; an approval or root: no), and whether it waits because the human is away. It mirrors the
// resident's approval.wait_left: a work question waits its question_timeout (Expires - Ts) from Ts or from the human's
// return, whichever is later, and nothing but the r.AwayMax cap runs out while they are away. No stamp yet (p.Seen 0)
// waits from Ts.
func questionLeft(r emberReq, p presence, now time.Time) (time.Duration, bool) {
	end := r.Ts + defaultRequestWait.Seconds()
	if r.Expires > 0 {
		end = r.Expires
	}
	t := float64(now.UnixMilli()) / 1000
	if r.AwayMax > 0 {
		capAt, after := r.Ts+r.AwayMax, r.AwayAfter
		if after <= 0 {
			after = presenceAwayAfter
		}
		switch {
		case p.Seen == 0:
			end = min(end, capAt)
		case t-p.Seen > after:
			return time.Duration((capAt - t) * float64(time.Second)), true
		default:
			end = min(max(r.Ts, p.Back)+end-r.Ts, capAt)
		}
	}
	return time.Duration((end - t) * float64(time.Second)), false
}

// timeLeft is the prompt's countdown text.
func timeLeft(r emberReq, p presence, now time.Time) string {
	left, away := questionLeft(r, p, now)
	switch {
	case left <= 0:
		return "timing out"
	case away:
		return "waits while you're away"
	case left >= time.Hour:
		return fmt.Sprintf("%d:%02d:%02d left", int(left.Hours()), int(left.Minutes())%60, int(left.Seconds())%60)
	}
	return fmt.Sprintf("%d:%02d left", int(left.Minutes()), int(left.Seconds())%60)
}

// questionBox draws the prompt where the composer was, full width under a rule (as Claude Code asks): a chip naming
// what it is, "1 of N" and the time left on the same line, the command or question in bold (the reason dim under a
// command), the numbered options with the highlighted one in the accent, the free-text option under a thin rule,
// and one hint line.
func (m emberModel) questionBox() string {
	r := m.chat.question
	if r == nil {
		return ""
	}
	w := max(18, m.width-2)
	inner := w - 2
	pending := m.sessionRequests()
	pos := 1
	for i, p := range pending {
		if p.ID == r.ID {
			pos = i + 1
		}
	}
	chip := map[string]string{"root": g.root + " root", "github": "⇡ github", "question": "? question"}[r.kind()]
	head := emberWarn.Bold(true).Render(chip) + emberDim.Render(fmt.Sprintf(" · %d of %d", pos, max(1, len(pending))))
	left := emberDim.Render(timeLeft(*r, m.presence, time.Now()))
	head += strings.Repeat(" ", max(1, inner-ansi.StringWidth(ansi.Strip(head))-ansi.StringWidth(ansi.Strip(left)))) + left
	wrap := func(s string, st lipgloss.Style) []string {
		var out []string
		for _, l := range strings.Split(ansi.Wordwrap(residentText(s), inner, " "), "\n") {
			out = append(out, st.Render(l))
		}
		return out
	}
	lines := []string{emberDim.Render(strings.Repeat("─", w)), " " + head}
	if n := m.noticeShown(); n != "" { // the last answer's flash ("root request: no"): the next prompt took its frame
		lines = append(lines, " "+emberDim.Render(emberClip(terminalText(n, false), inner)))
	}
	switch r.kind() {
	case "question":
		lines = append(lines, prefixAll(wrap(r.Text, lipgloss.NewStyle().Foreground(cText).Bold(true)), " ")...)
	case "github":
		summary := prefixAll(wrap(r.Summary, lipgloss.NewStyle().Foreground(cText).Bold(true)), " ")
		var why []string
		if strings.TrimSpace(r.Reason) != "" {
			why = prefixAll(wrap("why: "+r.Reason, emberDim), " ")
		}
		lines = append(lines, summary...)
		if _, body := handoffText(r.Github); strings.TrimSpace(body) != "" { // what it will post as you
			all := strings.Split(strings.TrimRight(residentText(body), "\n"), "\n")
			// the body takes what the summary, the reason and the options leave (the reason was cut off before, and
			// the hidden lines were counted one short)
			room := max(1, m.promptBudget(*r)-len(summary)-len(why)-1)
			shown := all[:min(len(all), 10, room)]
			for _, l := range shown {
				lines = append(lines, " "+emberDim.Render(emberClip("│ "+l, inner-1)))
			}
			if len(all) > len(shown) {
				lines = append(lines, " "+emberDim.Render(fmt.Sprintf("│ … %d more lines", len(all)-len(shown))))
			}
		}
		lines = append(lines, why...)
	default:
		lines = append(lines, prefixAll(wrap(r.Cmd, lipgloss.NewStyle().Foreground(cText).Bold(true)), " ")...)
		if strings.TrimSpace(r.Reason) != "" {
			lines = append(lines, prefixAll(wrap("why: "+r.Reason, emberDim), " ")...)
		}
	}
	if m.chat.questionError != "" {
		lines = append(lines, prefixAll(wrap(m.chat.questionError, emberCtxHot), " ")...)
	}
	// the options, the free-text row and the hint always fit: the text above them is cut to what is left (a long
	// hand-off body pushed the options off the screen, so you picked blind)
	if budget := m.promptBudget(*r); len(lines)-2 > budget {
		cut := len(lines) - 2 - (budget - 1)
		lines = append(lines[:2+budget-1], " "+emberDim.Render(fmt.Sprintf("… %d more lines", cut)))
	}
	lines = append(lines, "")
	opts := promptOptions(*r)
	accent := lipgloss.NewStyle().Foreground(cAccent).Bold(true)
	for i, o := range opts {
		mark, st := "   ", lipgloss.NewStyle().Foreground(cText)
		if i == m.chat.questionChoice && !m.chat.questionCustom {
			mark, st = " › ", accent
		}
		row := mark + st.Render(fmt.Sprintf("%d. %s", i+1, residentText(o)))
		if r.Recommended != "" && o == r.Recommended {
			row += accent.Render(" (Recommended)")
		}
		lines = append(lines, emberClip(row, w))
	}
	lines = append(lines, " "+emberDim.Render(strings.Repeat("─", max(4, min(inner, 40)))))
	free := len(opts)
	if m.chat.questionCustom {
		input := m.chat.questionInput
		input.SetWidth(max(16, inner-2))
		lines = append(lines, " "+accent.Render(fmt.Sprintf("%d. %s", free+1, freeTextLabel(*r))), " "+input.View())
	} else {
		mark, st := "   ", lipgloss.NewStyle().Foreground(cText)
		if m.chat.questionChoice == free {
			mark, st = " › ", accent
		}
		lines = append(lines, mark+st.Render(fmt.Sprintf("%d. %s…", free+1, freeTextLabel(*r))))
	}
	hint := fmt.Sprintf("Enter to select · ↑/↓ to navigate · 1-%d · Esc later", free+1)
	if m.chat.questionCustom {
		hint = "Enter sends it · Esc back to the options"
	}
	lines = append(lines, " "+emberDim.Render(emberClip(hint, inner)))
	return strings.Join(lines, "\n")
}

func prefixAll(lines []string, p string) []string {
	for i := range lines {
		lines[i] = p + lines[i]
	}
	return lines
}

// waitingLine is the line above the composer while requests wait that were put off with Esc.
func (m emberModel) waitingLine() string {
	if m.chat.question != nil {
		return ""
	}
	n := len(m.sessionRequests())
	if n == 0 {
		return ""
	}
	return " " + emberWarn.Render(fmt.Sprintf("%s %d waiting", g.root, n)) + emberDim.Render(" · Tab answers")
}

// reopenPrompt is Tab on the composer: the oldest waiting request comes back.
func (m emberModel) reopenPrompt() (emberModel, bool) {
	pending := m.sessionRequests()
	if m.chat.question != nil || len(pending) == 0 {
		return m, false
	}
	m.chat.later = nil
	m.openPrompt(pending[0])
	return m, true
}

func (m emberModel) questionKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	r := *m.chat.question
	k := msg.String()
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "pgup":
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		m.chat.transcript.PageDown()
		return m, nil
	}
	free := len(promptOptions(r))
	if m.chat.questionCustom {
		switch k {
		case "esc":
			m.chat.questionCustom = false
			m.chat.questionInput.Blur()
			return m, nil
		case "enter":
			return m.decide(r, free, strings.TrimSpace(m.chat.questionInput.Value()))
		}
		var cmd tea.Cmd
		m.chat.questionInput, cmd = m.chat.questionInput.Update(msg)
		return m, cmd
	}
	switch k {
	case "esc": // later: every waiting request steps back (one Esc gives the composer back); Tab brings them back
		if m.chat.later == nil {
			m.chat.later = map[string]bool{}
		}
		for _, p := range m.sessionRequests() {
			m.chat.later[p.ID] = true
		}
		m.chat.later[r.ID] = true
		m.closePrompt()
		return m, nil
	case "up", "k":
		m.chat.questionChoice = max(0, m.chat.questionChoice-1)
	case "down", "j":
		m.chat.questionChoice = min(free, m.chat.questionChoice+1)
	case "enter":
		if time.Since(m.chat.questionAt) < promptGrace {
			return m, nil
		}
		return m.decide(r, m.chat.questionChoice, "")
	default:
		if len(k) == 1 && k[0] >= '1' && k[0] <= '9' && time.Since(m.chat.questionAt) >= promptGrace {
			if i := int(k[0] - '1'); i <= free {
				m.chat.questionChoice = i
				return m.decide(r, i, "")
			}
		}
	}
	return m, nil
}

// decide acts on option i: a choice answers the question; Yes approves (runs the command as you, or the hand-off),
// No denies; the free-text option first opens its field, then answers with the text (for a command: a denial that
// tells the agent what to do instead).
func (m emberModel) decide(r emberReq, i int, text string) (tea.Model, tea.Cmd) {
	opts := promptOptions(r)
	if i == len(opts) && text == "" {
		m.chat.questionCustom = true
		return m, m.chat.questionInput.Focus()
	}
	idx := -1
	for j, x := range m.reqs {
		if x.ID == r.ID {
			idx = j
		}
	}
	if idx < 0 {
		m.closePrompt()
		return m, nil
	}
	m.sel = idx
	m.closePrompt()
	if r.kind() == "question" {
		answer := text
		if i < len(opts) {
			answer = opts[i]
		}
		return m.answerQuestion(r, answer)
	}
	if i == len(opts) { // "tell it something else": the denial carries the instruction, filed as a steer
		m.mode, m.edit, m.editRequest = "denywhy", text, &r
		return m.submitInput()
	}
	if opts[i] == "Yes" {
		return m.reqKeys("a")
	}
	return m.reqKeys("d")
}

func (m emberModel) answerQuestion(r emberReq, answer string) (tea.Model, tea.Cmd) {
	if strings.TrimSpace(answer) == "" {
		return m, nil
	}
	if err := writeAnswer(m.bridge, currentUser(), r, "answered", "", nil, "", "", answer); err != nil {
		m.msg = "the question changed or was answered elsewhere: " + err.Error()
		m.refreshLocal()
		return m, nil
	}
	_ = appendAudit(m.auditPath, map[string]any{"event": "decision", "id": r.ID, "decision": "answered", "by": currentUser(), "answer": answer})
	m.seen[r.ID] = true
	m.msg = "answered: " + oneLine(answer)
	m.refreshLocal()
	m.syncChat()
	return m, residentRefresh(m)
}

// askedRecommended is the recommended choice of the ask_user call that asked this question (by its text).
func (m *emberModel) askedRecommended(question string) string {
	for i := len(m.residentEvents) - 1; i >= 0; i-- {
		e := m.residentEvents[i]
		if e.Kind != "assistant" {
			continue
		}
		calls, _ := e.Data["tool_calls"].([]any)
		for _, raw := range calls {
			c, _ := raw.(map[string]any)
			f, _ := c["function"].(map[string]any)
			if n, _ := f["name"].(string); n != "ask_user" {
				continue
			}
			var args struct {
				Question    string `json:"question"`
				Recommended string `json:"recommended"`
			}
			if a, _ := f["arguments"].(string); json.Unmarshal([]byte(a), &args) == nil && args.Question == question {
				return args.Recommended
			}
		}
	}
	return ""
}

// outcomeText is a request's state for its record from a decision (the res file's or the tool result's fields).
func outcomeText(question bool, decision, answer string, rc any, by string) string {
	answer = oneLine(residentText(answer))
	switch {
	case by == "timeout" && question:
		return "recommended (timed out)"
	case by == "timeout":
		return "timed out (no)"
	case question && strings.TrimSpace(answer) != "":
		return answer + " (you)"
	case question:
		return "answered"
	case decision == "approved":
		if code, ok := rc.(float64); ok {
			return fmt.Sprintf("yes · exit %d", int(code))
		}
		return "yes"
	case decision == "denied" && strings.TrimSpace(answer) != "":
		return "no: " + answer
	case decision == "denied":
		return "no"
	}
	return ""
}

// requestOutcomes: each request's state for its one-line record, by request id. The bridge is the truth: a request on
// disk without an answer waits; an answer in res/ (the owner's, read once and kept) is what came of it; then the
// resident's own events (request_answered, a tool result, a timeout) for one whose files are gone.
func (m *emberModel) requestOutcomes() map[string]string {
	out := map[string]string{}
	question := map[string]bool{}
	timedOut := map[string]bool{}
	rec := map[string]string{} // a question's recommended choice: a timeout names it
	var ids []string
	for _, e := range m.residentEvents {
		str := func(k string) string { v, _ := e.Data[k].(string); return v }
		switch e.Kind {
		case "approval_requested", "question":
			if id := str("id"); id != "" {
				ids = append(ids, id)
				question[id] = e.Kind == "question"
				if r := str("recommended"); r != "" {
					rec[id] = r
				}
			}
		case "request_answered":
			// a timeout already recorded stays (an approval question's question_unanswered says "no" and comes first)
			if id := str("request"); id != "" && !timedOut[id] && (out[id] == "" || str("by") == "timeout") {
				out[id] = outcomeText(question[id], str("decision"), "", nil, str("by"))
				if str("by") == "timeout" {
					timedOut[id] = true
				}
			}
		case "question_unanswered":
			if id := str("id"); id != "" {
				if r := str("recommended"); r != "" {
					rec[id] = r
				}
				out[id] = outcomeText(question[id] && e.Data["approval"] != true, "", "", nil, "timeout")
				timedOut[id] = true
			}
		case "tool_result":
			name := str("name")
			if name != "ask_user" && name != "ask_root" && name != "github_handoff" {
				continue
			}
			res, _ := e.Data["result"].(map[string]any)
			id, _ := res["id"].(string)
			if id == "" {
				id, _ = res["request"].(string)
			}
			if id == "" || res["pending"] == true {
				continue
			}
			by, _ := res["by"].(string)
			answer, _ := res["answer"].(string)
			decision, _ := res["decision"].(string)
			if t := outcomeText(name == "ask_user", decision, answer, res["rc"], by); t != "" {
				out[id] = t
				if by == "timeout" {
					timedOut[id] = true
				}
			}
		}
	}
	if m.chat.resSeen == nil {
		m.chat.resSeen = map[string]string{}
	}
	for _, id := range ids {
		if timedOut[id] {
			continue // nobody answered: the timeout is the outcome, whatever a late file says
		}
		if t, ok := m.chat.resSeen[id]; ok {
			out[id] = t // the owner's answer is the most exact (it has the words)
			continue
		}
		if m.bridge == "" || !validBridgeID(id) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(m.bridge, "res", id+".json"))
		if err != nil {
			continue
		}
		var res map[string]any
		if json.Unmarshal(data, &res) != nil {
			continue
		}
		decision, _ := res["decision"].(string)
		answer, _ := res["answer"].(string)
		if t := outcomeText(question[id], decision, answer, res["rc"], ""); t != "" {
			m.chat.resSeen[id] = t
			out[id] = t
		}
	}
	for id, r := range rec {
		if out[id] == "recommended (timed out)" { // which option it went with, not only that it timed out
			out[id] = oneLine(residentText(r)) + " (recommended, timed out)"
		}
	}
	for _, r := range m.sessionRequests() {
		out[r.ID] = "waiting"
	}
	return out
}

// promptBudget is how many rows the prompt's text (the command, the question, a hand-off's body and reason) may take:
// the options, the free-text row, the hint and the chat's own header, footer and a few transcript rows always fit.
func (m emberModel) promptBudget(r emberReq) int {
	fixed := 2 + 1 + len(promptOptions(r)) + 2 + 1 + 8 // rule, header · blank · options · rule, free text · hint · the chat around it
	if m.chat.questionCustom {
		fixed++
	}
	return max(2, m.height-fixed)
}
