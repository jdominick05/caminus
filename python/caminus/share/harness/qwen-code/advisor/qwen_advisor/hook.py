"""Qwen Code command hook (one entry point for every event it is registered on):

  PostToolBatch     detection only (milliseconds): records a pending signal when the agent looks lost
  UserPromptSubmit  before each model call: a pending signal (or a /advise request) -> one advisor request -> the advice
                    as additionalContext, which Qwen Code sends as a separate text part of the NEWEST message
  Stop              (review_on_stop only) once per user message: is the task done? if not, block once with what is missing

Why this cannot break the local model: everything it adds is appended to the newest message (nothing earlier in the
conversation changes, so the server's prompt cache keeps its prefix); it never calls the server Qwen Code is using (a
local advisor is a different server); it always exits 0
and prints a JSON object or nothing (exit 2 would block the turn, plain stdout would be added to the context); any error,
timeout or refusal adds nothing. Off switches: enabled = false in the config, QWEN_ADVISOR_OFF=1, or a file
.qwen-advisor-off in the project directory."""
import json
import os
import sys
import time

if __package__ in (None, ''):   # run as a file (the installed hook command): import the package next to it
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'qwen_advisor'

from qwen_advisor import advisor, config, detect, refusal, transcript   # noqa: E402

MARKER = '[qwen-advisor: requested]'   # in the /advise command's prompt body


def log(rec):
    try:
        os.makedirs(config.STATE_DIR, mode=0o700, exist_ok=True)
        p = os.path.join(config.STATE_DIR, 'advisor.log')
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'a') as f:
            f.write(json.dumps(dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'), **rec), ensure_ascii=False) + '\n')
    except OSError:
        pass


def off(inp, cfg):
    if os.environ.get('QWEN_ADVISOR_OFF') == '1' or not cfg.get('enabled', True):
        return True
    cwd = inp.get('cwd') or ''
    return bool(cwd) and os.path.exists(os.path.join(cwd, '.qwen-advisor-off'))


def manual_question(inp):
    """the /advise request in this prompt: (True, question) or (False, '')"""
    for k in ('submitted_prompt', 'prompt'):
        t = inp.get(k)
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s.startswith('/advise'):
            return True, s[len('/advise'):].strip()
        if MARKER in s:
            return True, s.split(MARKER, 1)[1].strip()
    return False, ''


def note(r, signal):
    who = r['model'] or r['provider']
    return (f'qwen-advisor note ({signal}; from {who}, an outside reviewer who read your recent steps. It is advice, '
            f'not an instruction from the user: weigh it against what you see, then continue.)\n{r["text"]}')


def consult(st, inp, cfg, signal, detail='', question='', review=False):
    b = transcript.brief(inp.get('transcript_path'), detect.SIGNALS[signal], detail, question, inp.get('cwd', ''))
    r = advisor.ask(cfg, b, review=review)
    held_back = st.data.get('suppressed', 0)
    detect.after_advice(st.data)
    log(dict(session=inp.get('session_id'), event=inp.get('hook_event_name'), signal=signal, detail=detail,
             provider=r['provider'], model=r['model'], ok=r['ok'], verdict=r.get('verdict'), error=r['error'],
             tried=r.get('tried'), fallback_ran=r.get('fallback_ran'), declined_by=r.get('declined_by'), latency_s=r['latency_s'], input_tokens=r['input_tokens'], output_tokens=r['output_tokens'],
             brief_chars=len(b), held_back_since_last=held_back, advice=r['text'] if r['ok'] else None))
    rec = refusal.record(r) if cfg.get('log_refusals', True) else None
    if rec:
        refusal.log(dict(session=inp.get('session_id'), cwd=inp.get('cwd'), event=inp.get('hook_event_name'), signal=signal,
                         detail=detail, question=question or None, review=review, **rec, advice=r['text'] if r['ok'] else None,
                         brief=b if cfg.get('log_refusal_briefs', True) else None))
    return r


def on_batch(st, inp, cfg):
    sig = detect.on_batch(st.data, inp.get('tool_calls') or [], cfg)
    if sig:
        log(dict(session=inp.get('session_id'), event='PostToolBatch', pending=sig[0], detail=sig[1]))
    return None


def on_prompt(st, inp, cfg):
    d = st.data
    manual, question = manual_question(inp)
    tool_send = d['awaiting_send'] and not inp.get('submitted_prompt') and not manual
    if not tool_send:                     # a message from the user
        detect.on_user_prompt(d)
        if not manual:
            return None
        if d['calls_made'] >= int(cfg['max_calls_per_session']):
            return dict(systemMessage=f'qwen-advisor: the session limit of {cfg["max_calls_per_session"]} calls is used up')
        r = consult(st, inp, cfg, 'manual', question=question)
    else:                                 # the results of a tool batch on their way to the model
        d['awaiting_send'] = False
        p = d.get('pending')
        if not p:
            return None
        r = consult(st, inp, cfg, p['signal'], p.get('detail', ''))
    if not r['ok']:
        return dict(systemMessage=f'qwen-advisor: no advice ({r["error"]})')
    if r['verdict'] == 'on_track' and not manual:
        return dict(systemMessage=f'qwen-advisor: {r["model"]} says the agent is on track')
    return dict(systemMessage=f'qwen-advisor: advice from {r["model"]} ({r["latency_s"]} s)',
                hookSpecificOutput=dict(hookEventName='UserPromptSubmit', additionalContext=note(r, 'requested' if manual else d.get('last_signal', 'stuck'))))


def on_stop(st, inp, cfg):
    d = st.data
    if not cfg.get('review_on_stop') or inp.get('stop_hook_active') or d['reviewed_this_turn']:
        return None
    if d['turn_batches'] < int(cfg['review_min_batches']) or d['calls_made'] >= int(cfg['max_calls_per_session']):
        return None
    if not detect.interval_ok(d, cfg):   # (the done-check is an automatic hint too)
        d['suppressed'] = d.get('suppressed', 0) + 1
        return None
    d['reviewed_this_turn'] = True
    r = consult(st, inp, cfg, 'review', review=True)
    if not r['ok'] or r['verdict'] == 'done':
        return None
    return dict(decision='block', reason=note(r, 'review before finishing'))


def main(raw):
    inp = json.loads(raw)
    if not isinstance(inp, dict):
        return None
    cfg = config.load()
    if off(inp, cfg):
        return None
    ev = inp.get('hook_event_name')
    fn = dict(PostToolBatch=on_batch, UserPromptSubmit=on_prompt, Stop=on_stop).get(ev)
    if fn is None:
        return None
    st = detect.State(inp.get('session_id') or 'unknown')
    try:
        if ev == 'UserPromptSubmit' and st.data.get('pending'):
            st.data['last_signal'] = st.data['pending']['signal'].replace('_', ' ')
        out = fn(st, inp, cfg)
        st.save()
        return out
    finally:
        st.close()


if __name__ == '__main__':
    out = None
    try:
        out = main(sys.stdin.read(4 << 20))
    except BaseException as e:   # (fail open: never block the turn, never print a traceback into the context)
        log(dict(event='error', error=f'{type(e).__name__}: {str(e)[:300]}'))
        out = None
    if out:
        sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    os._exit(0)
