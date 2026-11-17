// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

// Keep the exact reviewed question snapshot until the human answers or defers
// it. A queue refresh must never redirect Enter to a different request.
func (m *emberModel) syncQuestion() {
	if !m.chat.ready || m.chat.sending {
		return
	}
	if q := m.chat.question; q != nil {
		for _, r := range m.reqs {
			if r.ID == q.ID && r.Digest == q.Digest {
				return
			}
		}
		m.chat.questionError = "This question changed or was answered elsewhere. Esc returns to the conversation."
		return
	}
	g := m.selectedResidentGoal()
	if g == nil || g.State != "waiting_user" {
		return
	}
	for _, r := range m.reqs {
		if r.kind() != "question" || r.Agent != m.agentName || r.Goal != g.ID || r.ID == m.chat.questionDeferred {
			continue
		}
		copy := r
		m.chat.question = &copy
		m.chat.questionChoice = 0
		m.chat.questionError = ""
		m.chat.questionInput = textarea.New()
		m.chat.questionInput.Prompt = "› "
		m.chat.questionInput.ShowLineNumbers = false
		m.chat.questionInput.CharLimit = 0
		m.chat.questionInput.MaxHeight = 0
		m.chat.questionInput.Placeholder = "Type your answer"
		m.chat.questionInput.SetWidth(max(16, m.width-4))
		m.chat.questionInput.SetHeight(3)
		m.chat.questionCustom = len(r.Choices) == 0
		// Arming is for a human who is looking at the conversation with an empty composer. A blurred
		// composer means they are browsing: Enter there returns to typing, so it must not answer.
		m.chat.questionArmed = m.chat.compose.Value() == "" && m.chat.compose.Focused()
		if m.chat.questionArmed {
			m.chat.compose.Blur()
		}
		if m.chat.questionCustom && m.chat.questionArmed {
			m.chat.questionInput.Focus()
		}
		return
	}
}

func (m emberModel) questionBox() string {
	r := m.chat.question
	if r == nil {
		return ""
	}
	w := max(16, m.width-4)
	wrap := func(s string) string { return lipgloss.NewStyle().Width(w).Render(residentText(s)) }
	header := emberWarn.Render("YOUR ANSWER") + "\n" + wrap(r.Text)
	if m.chat.questionError != "" {
		header += "\n" + wrap(m.chat.questionError)
	}
	var body, hint string
	if m.chat.questionCustom {
		input := m.chat.questionInput
		input.SetWidth(w)
		input.SetHeight(min(3, max(1, m.height-12)))
		view := viewport.New(viewport.WithWidth(w), viewport.WithHeight(min(lipgloss.Height(header), max(1, m.height-16))))
		view.SetContent(header)
		body = view.View() + "\n" + input.View()
		hint = "Enter answer · Alt+Enter newline · Esc back"
	} else {
		parts := []string{header, ""}
		selectedEnd := 0
		for i := 0; i <= len(r.Choices); i++ {
			text := "Write your own answer…"
			if i < len(r.Choices) {
				text = r.Choices[i]
			}
			mark := "  "
			if i == m.chat.questionChoice {
				mark = "› "
			}
			row := wrap(fmt.Sprintf("%s%d. %s", mark, i+1, text))
			if i == m.chat.questionChoice {
				row = bold(cAccent).Render(row)
			}
			parts = append(parts, row)
			if i == m.chat.questionChoice {
				selectedEnd = lipgloss.Height(strings.Join(parts, "\n"))
			}
		}
		text := strings.Join(parts, "\n")
		h := min(lipgloss.Height(text), max(3, m.height-12))
		view := viewport.New(viewport.WithWidth(w), viewport.WithHeight(h))
		view.SetContent(text)
		view.SetYOffset(max(0, selectedEnd-h))
		body = view.View()
		hint = "↑/↓ select · Enter answer · e custom · Esc defer"
	}
	return emberBorder.BorderForeground(cAccent).Width(max(18, m.width-2)).Render(body + "\n" + emberDim.Render(emberClip(hint, w)))
}

func (m emberModel) questionKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	if !m.chat.questionArmed {
		switch k {
		case "f2":
			m.chat.questionArmed = true
			m.chat.compose.Blur()
			if m.chat.questionCustom {
				return m, m.chat.questionInput.Focus()
			}
			return m, nil
		case "enter":
			m.msg = "Question waiting. F2 answers it; your steering draft is preserved."
			if !m.chat.compose.Focused() {
				return m, m.chat.compose.Focus() // Enter returns to typing, as the footer says
			}
			return m, nil
		case "esc":
			return m.deferQuestion()
		default:
			return m.chatDraftKey(msg)
		}
	}
	switch k {
	case "ctrl+c":
		return m, tea.Quit
	case "tab", "shift+tab":
		return m.onKey(k)
	case "pgup":
		m.chat.transcript.PageUp()
		return m, nil
	case "pgdown":
		m.chat.transcript.PageDown()
		return m, nil
	case "ctrl+l":
		m.chat.transcript.GotoBottom()
		return m, nil
	case "esc":
		if m.chat.questionCustom && len(m.chat.question.Choices) > 0 {
			m.chat.questionCustom = false
			return m, nil
		}
		return m.deferQuestion()
	}
	if m.chat.questionCustom {
		switch k {
		case "enter":
			return m.answerQuestion(m.chat.questionInput.Value())
		case "alt+enter", "shift+enter":
			m.chat.questionInput.InsertString("\n")
			return m, nil
		}
		var cmd tea.Cmd
		m.chat.questionInput, cmd = m.chat.questionInput.Update(msg)
		return m, cmd
	}
	switch k {
	case "up", "k":
		m.chat.questionChoice = max(0, m.chat.questionChoice-1)
	case "down", "j":
		m.chat.questionChoice = min(len(m.chat.question.Choices), m.chat.questionChoice+1)
	case "home":
		m.chat.questionChoice = 0
	case "end":
		m.chat.questionChoice = len(m.chat.question.Choices)
	case "e":
		m.chat.questionCustom = true
		return m, m.chat.questionInput.Focus()
	case "enter":
		if m.chat.questionChoice == len(m.chat.question.Choices) {
			m.chat.questionCustom = true
			return m, m.chat.questionInput.Focus()
		}
		return m.answerQuestion(m.chat.question.Choices[m.chat.questionChoice])
	default:
		if len(k) == 1 && k[0] >= '1' && k[0] <= '9' {
			i := int(k[0] - '1')
			if i <= len(m.chat.question.Choices) {
				m.chat.questionChoice = i
			}
		}
	}
	return m, nil
}

func (m emberModel) deferQuestion() (tea.Model, tea.Cmd) {
	m.chat.questionDeferred = m.chat.question.ID
	m.chat.question = nil
	m.chat.questionError = ""
	m.msg = "Question deferred; /approvals answers it. Steering does not answer a pending question."
	return m, m.chat.compose.Focus()
}

func (m emberModel) answerQuestion(answer string) (tea.Model, tea.Cmd) {
	r := m.chat.question
	if r == nil || strings.TrimSpace(answer) == "" {
		return m, nil
	}
	if r.kind() != "question" {
		m.chat.questionError = "Root requests require explicit review in /approvals (Ctrl+A)"
		return m, nil
	}
	if err := writeAnswer(m.bridge, currentUser(), *r, "answered", "", nil, "", "", answer); err != nil {
		m.chat.questionError = err.Error()
		m.msg = "Question changed or was answered elsewhere; review /approvals."
		return m, nil
	}
	_ = appendAudit(m.auditPath, map[string]any{"event": "decision", "id": r.ID, "decision": "answered", "by": currentUser(), "answer": answer})
	m.chat.questionDeferred = r.ID
	m.chat.question = nil
	m.chat.questionError = ""
	m.msg = "Answered question " + r.ID
	m.chat.compose.Focus()
	m.refreshLocal()
	m.syncChat()
	return m, residentRefresh(m)
}
