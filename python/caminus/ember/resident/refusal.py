"""Soft refusals: advice text that declines to help. A copy of the Qwen Code advisor's detector
(share/harness/qwen-code/advisor/qwen_advisor/refusal.py) for the resident's stdlib zipapp, which cannot import the
advisor package; keep the two in step."""
import re

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
# "illegal instruction" (SIGILL), "illegal option" (getopt) and the like name an error, not a judgment of the task.
POLICY_RE = re.compile(r"\b(?:weaponi[sz]\w*|circumvent\w*|"
                       r"illegal(?:ly)?\b(?!\s+(?:hardware\s+)?(?:instruction|opcode|operand|option|argument|seek|char(?:acter)?|byte|escape))|"
                       r"unethical|unlawful|not (?:legally )?(?:authori[sz]ed|permitted|allowed)|"
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
