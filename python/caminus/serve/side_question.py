"""Side questions: a harness's question about the conversation that must leave it as it was (Qwen Code's /btw). The
harness sends the conversation's opening and recent history plus the question, so the prompt cache's routing would take
it for the conversation's next turn (it matches the main line) and run it in the main slot: on a long conversation it
reads the history tail again there, and the conversation's real next turn is restored from a checkpoint afterwards. A
request recognized here goes to the side slot instead whenever it fits (cache/prompt.py route(): reason 'side_question'):
the side call starts from the deepest main checkpoint inside the shared prefix, and the main slot comes back untouched.

Recognized by the harness's own fixed prompt text at the start of the last user message, never by guessing at content:
  Qwen Code 0.26.x  packages/core/src/utils/btwUtils.ts buildBtwPrompt: '<system-reminder>' then this sentence.
A side question's reply is capped at SIDE_QUESTION_MAX_NEW tokens (the harness asks for its model's output limit, up to
64K), so that a longer prompt still fits the side slot with its reply."""

# the harnesses' side-question prompts: the sentence that opens the question's system reminder
MARKERS = ('This is a side question from the user. Answer directly in a single response.',)
HEAD = 512                       # characters of the last user message searched (the marker opens it)
SIDE_QUESTION_MAX_NEW = 16384    # a side question's reply: the harness asks for one answer, no tools


def is_side_question(messages):
    """messages: the normalized chat (serve/normalize.py: [{'role', 'content': str, ...}]). True when the last message is
    the user's and its text opens with a harness's side-question prompt"""
    if not messages: return False
    m = messages[-1]
    if not isinstance(m, dict) or m.get('role') != 'user' or not isinstance(m.get('content'), str): return False
    head = m['content'].lstrip()[:HEAD]
    return any(k in head for k in MARKERS)
