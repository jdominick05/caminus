"""The agent's GitHub account: the link, the environment its terminal gets, and hand-offs to its human.

- link: the owner points an agent at a token file in the agent's own account (`caminus ember agent github NAME link
  --token-file PATH`). The token is checked against the GitHub API (who it is, what it may do on the named repos) and
  config.json keeps the file's path and the account (login, id, the noreply email), never the token itself.
- env: every terminal and job of a linked agent gets GH_TOKEN / GITHUB_TOKEN (read from the file when the worker
  starts), git's author and committer as that account, and gh as git's credential helper for github.com. A real agent
  opened ~20 commands with `export GH_TOKEN=$(cat ...)`, and committed as user@localhost, which a CLA check could
  not link to the account that signed.
- redact: the token's text never comes back in a tool result.
- handoff: what the agent's token may not do upstream (open a pull request from its fork, comment, approve a CI run
  from its fork) goes to its human as a structured request. The owner's side builds the gh command from these fields
  only (no shell, no free-form command), shows it, and runs it as the owner on approval.
"""
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

API = 'https://api.github.com'
REPO = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}')
BRANCH = re.compile(r'(?!-)[A-Za-z0-9._/-]{1,200}')
HEAD = re.compile(r'([A-Za-z0-9][A-Za-z0-9-]{0,38}:)?(?!-)[A-Za-z0-9._/-]{1,200}')
ACTIONS = ('open_pr', 'comment', 'approve_ci')
TITLE_MAX, BODY_MAX = 256, 60000


def _token(path):
    p = Path(path).expanduser()
    text = p.read_text().strip()
    if not text or '\n' in text or len(text) > 400:
        raise ValueError(f'{p} does not hold one token')
    return text


def _api(token, path, opener=urllib.request.urlopen):
    req = urllib.request.Request(API + path, headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'caminus-ember',
                                                      'Accept': 'application/vnd.github+json'})
    try:
        with opener(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ValueError(f'GitHub said {e.code} for {path}') from e
    except (OSError, ValueError) as e:
        raise ValueError(f'GitHub did not answer for {path}: {e}') from e


def link(cfg, token_file, repos=(), opener=urllib.request.urlopen):
    """Check the token and return the github section for config.json (the account, the file's path, per-repo
    permissions). Raises ValueError with the reason when the token is unusable."""
    token = _token(token_file)
    user = _api(token, '/user', opener)
    login, uid = user.get('login'), user.get('id')
    if not login or not isinstance(uid, int):
        raise ValueError('the token does not name an account')
    perms = {}
    for repo in repos:
        if not REPO.fullmatch(repo):
            raise ValueError(f'{repo!r} is not OWNER/NAME')
        info = _api(token, f'/repos/{repo}', opener)
        p = info.get('permissions') or {}
        perms[repo] = 'write' if p.get('push') else ('read' if p.get('pull') or not info.get('private') else 'none')
    return dict(token_file=str(token_file), login=login, id=uid, name=user.get('name') or login,
                email=f'{uid}+{login}@users.noreply.github.com', repos=perms)


def env(cfg):
    """The environment a linked agent's commands get; {} when it is not linked or the file is gone."""
    gh = (cfg or {}).get('github') or {}
    if not gh.get('token_file'):
        return {}
    try:
        token = _token(gh['token_file'])
    except (OSError, ValueError):
        return {}
    who, email = gh.get('login') or 'agent', gh.get('email') or ''
    out = dict(GH_TOKEN=token, GITHUB_TOKEN=token, GIT_AUTHOR_NAME=who, GIT_COMMITTER_NAME=who,
               GH_PROMPT_DISABLED='1', GIT_TERMINAL_PROMPT='0',
               GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='credential.https://github.com.helper',
               GIT_CONFIG_VALUE_0='!gh auth git-credential')
    if email:
        out.update(GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_EMAIL=email)
    return out


def redact(text, cfg):
    """The token's text replaced in anything a tool returns."""
    gh = (cfg or {}).get('github') or {}
    if not gh.get('token_file') or not isinstance(text, str):
        return text
    try:
        token = _token(gh['token_file'])
    except (OSError, ValueError):
        return text
    return text.replace(token, '[github token]') if token in text else text


def handoff(args):
    """The checked, canonical form of a hand-off request. Every field is validated; nothing here is ever a command."""
    action = args.get('action')
    if action not in ACTIONS:
        raise ValueError(f'action is one of {", ".join(ACTIONS)}')
    repo = str(args.get('repo') or '')
    if not REPO.fullmatch(repo):
        raise ValueError('repo is OWNER/NAME, e.g. jdominick05/caminus')
    out = dict(action=action, repo=repo)
    if action == 'open_pr':
        head, base = str(args.get('head') or ''), str(args.get('base') or '')
        if not HEAD.fullmatch(head):
            raise ValueError('head is your branch as FORK_OWNER:BRANCH (or BRANCH in the same repo)')
        if not BRANCH.fullmatch(base):
            raise ValueError('base is the branch to merge into, e.g. testing')
        title, body = str(args.get('title') or '').strip(), str(args.get('body') or '')
        if not title or len(title) > TITLE_MAX or '\n' in title:
            raise ValueError(f'title is one line of at most {TITLE_MAX} characters')
        if len(body) > BODY_MAX:
            raise ValueError(f'body is at most {BODY_MAX} characters')
        out.update(head=head, base=base, title=title, body=body)
    elif action == 'comment':
        number, body = args.get('number'), str(args.get('body') or '')
        if not isinstance(number, int) or number <= 0:
            raise ValueError('number is the issue or pull request number')
        if not body.strip() or len(body) > BODY_MAX:
            raise ValueError(f'body is the comment, at most {BODY_MAX} characters')
        out.update(number=number, body=body)
    else:
        run = args.get('run_id')
        if not isinstance(run, int) or run <= 0:
            raise ValueError('run_id is the workflow run waiting for approval (gh run list shows it)')
        out.update(run_id=run)
    return out


# what must never go public from a hand-off: a machine path, a private decision id, a session link, a token
PRIVATE = (('a home-directory path', re.compile(r'(/home/|/Users/|/root/)[\w.-]')),
           ('a decision id', re.compile(r'\bD-\d{2,}\b')),
           ('a session link', re.compile(r'claude\.ai/(code|chat|share)/|chatgpt\.com/(c|share)/')),
           ('a token', re.compile(r'\b(ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|\bsk-[A-Za-z0-9-]{20,}|\bAKIA[0-9A-Z]{16}\b')))


def private_text(req):
    """What in a hand-off's title or body looks private, as reasons; [] when it reads clean. The owner's side refuses
    a hand-off with any: approving posts the agent's text under the owner's name, and it cannot be unposted."""
    h = handoff(req)
    text = (h.get('title') or '') + '\n' + (h.get('body') or '')
    return [why for why, rx in PRIVATE if rx.search(text)]


def command(req):
    """(argv, stdin) for the owner to run with their own gh, from a checked hand-off only."""
    h = handoff(req)
    repo = h['repo']
    if h['action'] == 'open_pr':
        # --no-maintainer-edit: the owner's account cannot grant edits on the agent's fork (GitHub refused it)
        return (['gh', 'pr', 'create', '-R', repo, '--base', h['base'], '--head', h['head'], '--title', h['title'],
                 '--body-file', '-', '--no-maintainer-edit'], h['body'])
    if h['action'] == 'comment':
        return (['gh', 'api', '-X', 'POST', f'repos/{repo}/issues/{h["number"]}/comments', '-F', 'body=@-'], h['body'])
    return (['gh', 'api', '-X', 'POST', f'repos/{repo}/actions/runs/{h["run_id"]}/approve'], '')


def describe(req):
    """One line for the approval card."""
    h = handoff(req)
    if h['action'] == 'open_pr':
        return f'open a pull request on {h["repo"]}: {h["head"]} -> {h["base"]}: {h["title"]}'
    if h['action'] == 'comment':
        return f'comment on {h["repo"]}#{h["number"]} ({len(h["body"])} characters)'
    return f'approve CI run {h["run_id"]} on {h["repo"]}'
