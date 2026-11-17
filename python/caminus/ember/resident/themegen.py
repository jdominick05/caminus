"""A colour theme for the owner's panel, written by the resident's brain from a description (the theme screen's
"describe a theme"). One call, no tools, no goal context. The answer is untrusted text: the panel keeps only lines it
can validate (known roles, #rrggbb colours, a known code style) and refuses an answer missing roles."""

ROLES = [
    ('text', 'what you read: replies, prompts, values (high contrast on the terminal background)'),
    ('muted', 'resting text: hints, echoes, tool output (clearly dimmer than text, still readable)'),
    ('think', "the model's thinking: between muted and text"),
    ('dim', 'rules, borders, empty slots: recedes'),
    ('accent', 'titles and the focused element: the theme\'s signature colour'),
    ('ok', 'healthy, passing, the agent speaking: a green-ish role'),
    ('warn', 'needs you: questions, pauses: a yellow-ish role'),
    ('action', 'tools running: an orange-ish role'),
    ('hot', 'failure, errors, a hot context: a red-ish role'),
    ('purple', 'a distinct second stream of data'),
    ('user', "the human's words: goals, steering"),
    ('teal', 'inline code and quiet positive chrome'),
    ('chrome', 'selection background and box borders: dark, low contrast'),
    ('code_bg', 'the plate under inline code: a little lighter than a dark background'),
    ('codeblock_bg', 'the plate under fenced code blocks: near the background'),
]
CODE_THEMES = ['dracula', 'monokai', 'nord', 'onedark', 'github-dark', 'gruvbox', 'catppuccin-mocha', 'rose-pine',
               'solarized-dark', 'tokyonight-night', 'doom-one', 'native', 'vulcan', 'paraiso-dark', 'hrdark',
               # for light backgrounds
               'github', 'solarized-light', 'catppuccin-latte', 'rose-pine-dawn', 'tokyonight-day']
BACKGROUND = [
    ('bg', 'the background: ANY colour that fits the description (dark, light, pink, cream, deep green...)'),
    ('bg2', 'optional: a second background colour, for bg_pattern = gradient (top bg, bottom bg2)'),
    ('pattern', 'optional: the ink of a background pattern, a faint shade near bg'),
]
PATTERNS = ['none', 'dots', 'grid', 'diagonal', 'waves', 'stars', 'gradient']

PROMPT = """You design colour themes for a terminal app: a chat with an AI agent, with a side panel and a status line.
The theme sets the whole terminal's background too, so be creative: any background colour (light, dark, pink, cream,
deep green, a gradient...), and optionally a faint pattern behind the text.
Write one theme matching the owner's description. Reply with ONLY these lines, nothing else:

name = a-short-kebab-case-name
{background}
bg_pattern = one of: {patterns}
{roles}
code_theme = one of: {themes} (a light one on a light background)

Every colour is #rrggbb. Read every text role against YOUR bg: contrast at least 7:1 for text, 4.5:1 for muted and
the semantic roles (dark text on a light background, light text on a dark one). Keep the semantic roles recognisable
(ok green-ish, warn yellow-ish, hot red-ish) unless the description asks otherwise, make accent the signature colour,
and keep a pattern subtle (its ink close to bg) so it never competes with the text."""


def write(rt, description):
    """the brain's theme text for a description (validated by the panel, not here)"""
    roles = '\n'.join(f'{k} = #rrggbb   ({why})' for k, why in ROLES)
    background = '\n'.join(f'{k} = #rrggbb   ({why})' for k, why in BACKGROUND)
    request = [dict(role='system', content=PROMPT.format(roles=roles, background=background,
                                                        patterns=', '.join(PATTERNS), themes=', '.join(CODE_THEMES))),
               dict(role='user', content='The theme: ' + description)]
    answer = rt.model.complete(request, None, max_tokens=min(2048, rt._reply_limit())).get('content') or ''
    try:
        rt._meter('theme')                # a paid brain's theme shows in the spend like any turn
    except Exception:
        pass
    return answer
