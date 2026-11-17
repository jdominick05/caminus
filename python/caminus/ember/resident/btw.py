"""/btw: a side question about a goal, answered by one model call that stores nothing in the goal.

The brief is the goal's prompt, its done check and its todo, plus the newest messages as context and the
question itself. No tools are offered, so the answer is talk, not work: the goal's transcript, steps and
state are untouched (an event btw keeps the question and the answer's first 300 characters for the
journal). The runtime's own model answers inside its own window: older context is trimmed until the
request and a full answer fit.
"""
KEEP = 20          # newest goal messages that ride along as context
HEADROOM = 512     # tokens kept for the answer and the request frame
PROMPT_LIMIT = 8000   # the goal's own words, bounded: a huge goal must not make the brief itself overflow


def brief(rt, gid):
    """The system part of a side question: what the goal is and where it stands."""
    g = rt.store.goal(gid)
    system = ['GOAL: ' + (g['prompt'] or '')[:PROMPT_LIMIT]]
    # what the human said since the goal began: a goal re-aimed by a message (a game, then "review the design") had its
    # old prompt and new done check side by side, and the side answer called the goal contradictory
    said = [r['data'] for r in rt.store.db.execute(
        "SELECT data FROM events WHERE goal=? AND kind='user_message' ORDER BY seq DESC LIMIT 5", (gid,))]
    told = []
    for raw in reversed(said):
        try:
            t = str(__import__('json').loads(raw).get('text') or '').strip()
        except (ValueError, AttributeError):
            t = ''
        if t:
            told.append('- ' + t[:600])
    if told:
        system.append('LATER WORDS FROM YOUR HUMAN (newest last; they redirect the goal\'s first words):\n' + '\n'.join(told))
    if g.get('done_check'):
        system.append('DONE CHECK: ' + g['done_check'])
    todo = rt.store.todo(gid)
    if todo['items']:
        system.append('TODO: ' + ' | '.join(f"[{it['state']}] {it['text']}" for it in todo['items']))
    if todo['blockers']:
        system.append('BLOCKED BY: ' + ' | '.join(todo['blockers']))
    return dict(role='system', content='\n'.join(system))


def ask(rt, gid, question):
    """One model call about the goal; returns the answer as text."""
    question_msg = dict(role='user', content='BTW, a side question about this goal (your answer is shown to '
                                             'your human and kept nowhere else; you have no tools): '
                                             + question)
    max_tokens = min(rt._reply_limit(), rt.model.context_size() // 4)
    # The recent conversation goes as text, not as chat turns: a slice of the history can start with a tool result
    # whose call was cut off, or end with a call still running, and the APIs refuse either.
    lines = [_line(m) for m in rt.store.goal(gid)['messages'][-KEEP:]]
    lines = [l for l in lines if l]
    while True:
        context = dict(role='user', content='RECENT CONVERSATION (oldest first):\n' + '\n'.join(lines))
        request = [brief(rt, gid), context, question_msg] if lines else [brief(rt, gid), question_msg]
        if rt.model.tokens(request, []) + max_tokens + HEADROOM <= rt.model.context_size():
            break
        if not lines:
            raise ValueError('the goal does not fit the model window for a side question: compact it first')
        lines = lines[1:]                 # trim older context until the request and its answer fit
    answer = rt.model.complete(request, None, max_tokens=max_tokens).get('content') or ''
    try:
        rt._meter(gid)                    # a paid brain's side question shows in the spend like any turn
    except Exception:
        pass
    rt.store.event('btw', dict(question=question, answer=answer[:300]), gid)
    return answer


def _line(m):
    """One message as a line of text for the side question's context."""
    role, content = m.get('role'), m.get('content') or ''
    if isinstance(content, list):
        content = ' '.join(p.get('text', '') for p in content if isinstance(p, dict))
    content = ' '.join(str(content).split())[:1200]
    if role == 'assistant':
        calls = ', '.join((c.get('function') or {}).get('name', '?') for c in m.get('tool_calls') or [])
        return 'agent: ' + content + ((' [called ' + calls + ']') if calls else '')
    if role == 'tool':
        return 'tool result: ' + content[:600]
    if role == 'user':
        return 'human: ' + content
    return ''
