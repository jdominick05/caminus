"""Plan-first goals: a goal submitted with --plan starts in plan mode.

While the plan is unapproved (goals.plan == 1) the goal only explores: the changing tools are refused
and `terminal` stays allowed on the prompt's promise of read-only use — the runtime cannot see what a
shell command actually touched, so that limit is advisory there. `sandbox` stays allowed: its box changes
nothing of the agent's outside its own folder (with hosts it can still send data to them, logged for the human).
`browser` may look (open, snapshot, scroll, back, screenshot, console) but not act (click, type, select, press,
eval: a form sent is a change outside). `propose_plan` files the plan and
parks the goal (state waiting_plan, which never times out) until the human answers: approve sets
goals.plan = 2 and lifts plan mode, reject keeps plan mode with the note in the inbox.
"""
# (everything that acts outside reading: a message gives a peer work, a hand-off posts upstream, remember writes the
# notes, schedule and lease commit future time and cards; MCP tools are refused by their mcp__ prefix in the runtime)
PLAN_REFUSED = ('write', 'edit', 'checkpoint', 'rollback', 'ask_root', 'send_message', 'github_handoff', 'remember',
                'schedule', 'lease')


def propose(gid, plan):
    """The filed plan's tool result; the caller stores the text, the event and the parking state."""
    return dict(plan_filed=True,
                note="your human now reviews the plan; you continue when it is approved, or after their revision note")


def refused_text(name):
    if name == 'terminal':
        return 'plan mode: use the terminal read-only (observing, no side effects) and call propose_plan; ' \
               'changes wait for the plan to be approved'
    if name == 'browser':
        return 'plan mode: the browser may open, snapshot, scroll, go back, screenshot and read the console; ' \
               'click, type, select, press and eval wait for the plan to be approved'
    return 'plan mode: the goal is exploring for a plan, so ' + name + ' is refused. ' \
           'Call propose_plan with what you would do; your human approves it before anything changes'


def loop_line(g):
    """The plan-mode note in the goal's turn context (empty once plan mode is off)."""
    if g.get('plan') != 1:
        return ''
    if g.get('plan_text'):
        return ('PLAN MODE: your plan was rejected and your human sent a revision note in the inbox; '
                'propose a revised plan with propose_plan. Until it is approved the changing tools are refused.')
    return ('PLAN MODE: this goal is plan-first. Explore read-only (read, web, history_search, terminal for '
            'observing only, sandbox, browser without acting) and call propose_plan with the plan; write, edit, checkpoint, rollback and ask_root '
            'are refused until your human approves it.')


def approval_text(note):
    return 'Plan approved' + (': ' + note.strip() if (note or '').strip() else '')


def rejection_text(note):
    return 'Plan rejected, revise it' + (': ' + note.strip() if (note or '').strip() else '')
