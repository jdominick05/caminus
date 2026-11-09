"""Refusals: detecting a soft one in advice text, and the refusal log (refusals.jsonl in the state directory, mode 600):
one record per advisor call that met a refusal anywhere: the whole chain refused, a model refused and Anthropic's
server-side fallback or the fallback provider answered, or the delivered advice itself declines. Each record keeps the
brief that was sent (already redacted), so it can be rerun against other models and settings."""
import json
import os
import re
import time

from . import config

# anchored on "I" so advice like "you can't push until you pull" or "the test can't find the fixture" never counts,
# and "I can't see the file" is not a refusal either (no help verb)
REFUSAL_RES = [re.compile(p, re.I) for p in (
    r"\bI(?:'m| am)\s+(?:sorry|afraid)\b[^.\n]{0,80}\b(?:can(?:no|')t|unable|won't|not able)\b",
    r"\bI\s+(?:can(?:'|’)t|cannot|won(?:'|’)t|will not|must decline to|am not able to|am unable to|'m not able to|'m unable to)\s+"
    r"(?:help|assist|provide|support|continue|comply|do that|engage|advise|give (?:advice|guidance|instructions))\b",
    r"\bI\s+(?:must|have to|need to)\s+decline\b",
    r"\b(?:not (?:able|comfortable|willing) to|unable to)\s+(?:help|assist|provide guidance)\s+(?:with|on)\s+(?:this|that|cracking|exploit|hacking|attack)",
)]


# a hedge that still helps ("I can't provide the exact line without the file, but the error points at ...") is advice
HEDGE_RE = re.compile(r"\b(?:but|without|unless|until|exact|from (?:here|this|the brief|what)|more context|seeing|the full)\b", re.I)


def _rest_of_sentence(text, i):
    ends = [k for k in (text.find('.', i), text.find('\n', i), text.find('!', i)) if k != -1]
    return text[i:min(ends) if ends else len(text)]


DECLINE_RE = re.compile(r'^[\s*_`>#\-]*DECLINE\b', re.I)   # the prompts' explicit way to decline
# policy steering: the advice judges the task (circumvention, not authorised, ...) and orders the agent off it in the same
# sentence ("this is circumvention; stop and do not weaponize it"). That is a refusal dressed as advice.
POLICY_RE = re.compile(r"\b(?:weaponi[sz]\w*|circumvent\w*|illegal(?:ly)?|unethical|unlawful|not (?:legally )?(?:authori[sz]ed|permitted|allowed)|"
                       r"(?:doesn't|does not|don't|do not|wouldn't|won't) cover (?:that|this|it)|proceed only (?:on|if)|"
                       r"only (?:on|for) (?:a |your )?(?:own )?(?:devices?|systems?|hardware) you (?:own|are authori[sz]ed))\b", re.I)
DIRECTIVE_RE = re.compile(r"\b(?:stop|do not|don't|never|refuse|abandon|drop|cease|avoid|must not|should not|shouldn't)\b", re.I)


def _sentences(text):
    return [x for x in re.split(r'(?<=[.!?])\s+|\n+', text or '') if x.strip()]


def refusal_kind(text):
    """'decline' (the explicit DECLINE reply), 'refusal' (declines in words), 'policy' (judges the task and orders the
    agent off it), or None"""
    t = text or ''
    if DECLINE_RE.match(t):
        return 'decline'
    for r in REFUSAL_RES:
        for m in r.finditer(t):
            if not HEDGE_RE.search(_rest_of_sentence(t, m.end())):   # (only what follows: "I'm sorry, but I can't" is a refusal)
                return 'refusal'
    for sent in _sentences(t):
        if POLICY_RE.search(sent) and DIRECTIVE_RE.search(sent):
            return 'policy'
    return None


def soft_refusal(text):
    """the reply declines to help, in any of the refusal_kind() forms; a hedge followed by help does not count"""
    return refusal_kind(text) is not None


def is_hard(err):
    return (err or '').startswith('refusal')


def record(r):
    """the refusal summary of one advisor result (advisor.ask's dict), or None when nothing refused"""
    refused = [t for t in (r.get('tried') or []) if is_hard(t.get('error')) or t.get('error') == 'soft refusal']
    if is_hard(r.get('error')) or r.get('error') == 'soft refusal':
        refused.append(dict(provider=r.get('provider'), model=r.get('model'), error=r.get('error'), category=r.get('stop_category'),
                            soft_text=r.get('soft_text')))
    declined = [m for m in (r.get('declined_by') or []) if m]
    soft = bool(r.get('ok')) and soft_refusal(r.get('text'))
    if not (refused or declined or soft):
        return None
    n_try = len(r.get('tried') or []) + 1
    if not r.get('ok'):
        outcome = (f'refused by every advisor tried ({len(refused)} of {n_try}): no advice delivered' if len(refused) == n_try else
                   f'{len(refused)} refusal(s), then {r.get("error")}: no advice delivered' if refused else f'not delivered ({r.get("error")})')
    elif declined:
        outcome = f'declined by {", ".join(declined)}; answered by {r.get("model")} (server-side fallback)'
    elif refused:
        outcome = (f'refused by ' + ', '.join(f'{t.get("provider")}:{t.get("model")} ({"soft" if t.get("error") == "soft refusal" else "hard"})' for t in refused)
                   + f'; answered by {r.get("provider")}:{r.get("model")} (advisor chain)')
    else:
        outcome = f'soft refusal in the advice from {r.get("model")}'
    if soft and r.get('ok') and not outcome.startswith('soft'):
        outcome += '; the delivered advice also declines'
    return dict(outcome=outcome, refused=[dict(provider=t.get('provider'), model=t.get('model'), category=t.get('category'),
                                              kind='soft' if t.get('error') == 'soft refusal' else 'hard', error=t.get('error'),
                                              soft_text=t.get('soft_text')) for t in refused],
                server_declined=declined, served_by=f'{r.get("provider")}:{r.get("model")}' if r.get('ok') else None,
                delivered=bool(r.get('ok')), soft=soft)


def log(entry, path=None):
    p = path or os.path.join(config.STATE_DIR, 'refusals.jsonl')
    try:
        os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'a') as f:
            f.write(json.dumps(dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'), **entry), ensure_ascii=False) + '\n')
    except OSError:
        pass
