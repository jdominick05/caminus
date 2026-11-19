"""Is a pull request into `testing` (or `ember-next`) ready to merge? (Reads GitHub through the `gh` CLI; changes nothing, posts nothing.)

  python -I tools/pr_ready.py N [-R OWNER/REPO] [--wait SECS] [--reviewer LOGIN] [--allow-base-merges]

Checks, each printed PASS / FAIL, then RESULT PASS / RESULT FAIL (exit 0 / 1):
- base: the pull request targets one of `BASES` (testing, or ember-next for the Ember harness), is open, is not a draft and has no merge conflict;
- ci: every check on the head commit has finished and succeeded;
- review: the review bot has looked at the HEAD commit and left nothing open there. It does not look again after a push:
  a head it has not seen needs a comment asking for a review (the script names the commit it last saw). A merge of
  `testing` into the branch is a push like any other; --allow-base-merges accepts commits whose headline says they
  merge `testing`, made after its look, as they are (their content is not inspected);
- threads: no review thread is unresolved (the branch rules refuse the merge otherwise);
- gates: a change under kernels/ or to code that builds a tape names its GPU gates and the cards in the body
  (AGENTS.md, Code Review Rules; CONTRIBUTING.md);
- size: the lines and files the pull request adds and removes, and what it adds to AGENTS.md (GitHub's file list). One
  that adds more lines than it removes says in its body what it builds on and what it took out (AGENTS.md, Invariants,
  Growth): a `Reuses:` line (what it extends, or `nothing existing` and what was searched for) and a `Removes:` line
  (files it deleted or made shorter, or `nothing:` and why). One that adds lines to tools/entropy.json, the list of
  what the tree checks accept as already wrong, says why in a `Baseline:` line;
- title: the pull request title and every authored commit headline read `<area> : <what>` (a merge of `testing` is not authored);
- trailers: each authored commit carries exactly one `Assisted-by:` trailer and no `Co-Authored-By:`, or, when the body
  says `AI assistance: none`, no `Assisted-by:` at all;
- exempt: a Dependabot pull request has no title, trailer or `Reuses:` / `Removes:` rule (its bumps are made by the bot);
- text: the title, body and commit messages carry no decision id, machine path or session link.

--metadata checks only base, title, trailers, gates, size and text (no CI, review or thread wait): what CI runs on each pull
request, since the review bot's verdict is not known there.

--wait SECS polls every 30 s (--poll), for at most SECS, while only `ci` or `review` are still pending on the head
commit: a bounded wait, to be run in the background. It never waits on a failure, and never past SECS."""
import argparse
import json
import re
import subprocess
import sys
import time

TAPE = ('kernels/', 'python/caminus/engine/', 'python/caminus/decode/', 'python/caminus/prefill/')
GATE = re.compile(r'bitwise|\bKLD\b|G-acc|G-model|G-self|G-speed', re.I)
CARD = re.compile(r'\b(XTX|XT|RX ?\d{4}|both cards)\b')
PRIVATE = (('decision id', re.compile(r'\bD-\d{2,}\b')), ('machine path', re.compile(r'(/home/|/Users/|/run/medi[a]/|/mnt/)\w')),
           ('session link', re.compile(r'claude\.ai/(code|chat|share)/|chatgpt\.com/(c|share)/')))
TITLE = re.compile(r'[a-z][a-z0-9-]* : \S')                 # <area> : <what>, as CONTRIBUTING.md and AGENTS.md write it
MERGE = re.compile(r'Merge ')                               # merges are not authored: no title or trailer rules
TRAILER = re.compile(r'^Assisted-by:\s*\S', re.M)
COAUTHOR = re.compile(r'^Co-Authored-By:', re.I | re.M)
NO_AI = re.compile(r'^AI assistance: none\s*$', re.M)      # the body's declaration for a pull request with no AI-assisted commit
NAMED = re.compile(r'\w[\w.-]*(?:/[\w.-]+)+|\w[\w-]+\.[A-Za-z]\w*')   # a file's name in a Removes: line: a path, or a name with its extension
BASELINE = 'tools/entropy.json'                             # what tools/entropy.py accepts as already wrong: adding to it needs a reason
BOTS = ('app/dependabot', 'dependabot[bot]', 'dependabot')  # gh reports the bot as app/dependabot, the REST API as dependabot[bot]
METADATA = ('base', 'title', 'trailers', 'gates', 'size', 'text')   # what --metadata checks: the pull request's text, not CI or review
THREADS = '''query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){pullRequest(number:$n){
  reviewThreads(first:100){nodes{isResolved isOutdated path comments(first:1){nodes{author{login} body}}}}}}}'''


def gh(*args):
    try: r = subprocess.run(['gh', *args], capture_output=True, text=True)
    except FileNotFoundError: raise SystemExit('pr_ready: the gh CLI is not on PATH (https://cli.github.com)')
    if r.returncode: raise SystemExit(f'pr_ready: gh {" ".join(args[:3])}... failed: {r.stderr.strip()[:300]}')
    return json.loads(r.stdout)


def bot_reviews(pr, login):
    """[(commit prefix, clean?)] in time order: what the review bot said, and about which commit"""
    out = []
    notes = [(c['createdAt'], c['body']) for c in pr['comments'] if c['author']['login'].startswith(login)]
    notes += [(r['submittedAt'], r['body']) for r in pr['reviews'] if r['author']['login'].startswith(login)]
    for _, body in sorted(notes):
        m = re.search(r'codex-review-state:v1 (\{.*?\}) -->', body, re.S)
        if m:
            st = json.loads(m.group(1)).get('code_review') or {}
            if st.get('commit_sha') and st.get('status') == 'completed':
                out.append((st['commit_sha'], not st.get('has_unresolved_findings')))
            continue
        m = re.search(r'Reviewed commit:\*\*\s*`([0-9a-f]{7,40})`', body)
        if m: out.append((m.group(1), "Didn't find any major issues" in body))
    return out


# The branches a pull request may target: testing (everything that ships), and ember-next, the Ember harness's
# integration branch, which lands in testing as one reviewed pull request after the release it waits for.
BASES = ('testing', 'ember-next')


def look(a):
    """({check: (ok, text)}, the checks that are only pending)"""
    pr = gh('pr', 'view', str(a.n), '-R', a.repo, '--json',
            'baseRefName,headRefOid,state,isDraft,mergeable,title,body,author,files,commits,comments,reviews,statusCheckRollup')
    head, res, pending = pr['headRefOid'], {}, set()
    bad = [w for w, ok in (('base is ' + pr['baseRefName'], pr['baseRefName'] in BASES), ('state ' + pr['state'], pr['state'] in ('OPEN', 'MERGED')),
                           ('a draft', not pr['isDraft']), ('merge conflict', pr['mergeable'] != 'CONFLICTING')) if not ok]
    res['base'] = (not bad, f'into {pr["baseRefName"]}, head {head[:10]}' + (': ' + ', '.join(bad) if bad else ''))
    if a.metadata:   # the text checks only: no CI, review or thread wait, so CI can run it without the review bot
        content(pr, res)
        return res, set()

    runs, by = pr['statusCheckRollup'] or [], {}
    for c in runs: by.setdefault(c.get('name') or c.get('context'), set()).add((c.get('conclusion') or c.get('state') or c.get('status') or '').upper())
    good, run = {'SUCCESS', 'NEUTRAL', 'SKIPPED'}, {'', 'PENDING', 'QUEUED', 'IN_PROGRESS', 'EXPECTED'}
    wait = sorted(n for n, st in by.items() if st & run)                       # (a cancelled run beside a success of the same check: superseded)
    fail = sorted(n for n, st in by.items() if n not in wait and (st - good - {'CANCELLED'} or not st & good))
    if wait and not fail: pending.add('ci')
    res['ci'] = (bool(runs) and not wait and not fail, f'{len(runs)} checks' + (f'; failed: {fail}' if fail else '') + (f'; running: {wait}' if wait else '') + ('' if runs else ': none reported'))

    seen = bot_reviews(pr, a.reviewer)
    oids = [c['oid'] for c in pr['commits']]
    def covers(sha):   # the head itself; with --allow-base-merges also a commit followed only by merges of the base branch
        i = next((k for k, o in enumerate(oids) if o.startswith(sha) or sha.startswith(o)), None)
        later = pr['commits'][i + 1:] if i is not None else None
        return later is not None and (not later or (a.allow_base_merges and all(re.match(r'Merge .*\btesting\b', c['messageHeadline']) for c in later)))
    at_head = [clean for sha, clean in seen if covers(sha)]
    if not seen: res['review'] = (False, f'{a.reviewer} has not reviewed this pull request'); pending.add('review')
    elif not at_head:
        res['review'] = (False, f'last reviewed commit {seen[-1][0][:10]}, head is {head[:10]}: it does not look again after a push (ask with a comment)')
        if pr['comments'] and '@codex review' in pr['comments'][-1]['body']: pending.add('review')
    else: res['review'] = (any(at_head), 'reviewed at head' + (' (merges of testing after it accepted)' if a.allow_base_merges else '') + ('' if any(at_head) else ': findings open (fix them, or answer each in its thread, then ask again)'))

    owner, name = a.repo.split('/')
    th = gh('api', 'graphql', '-f', 'query=' + THREADS, '-F', f'o={owner}', '-F', f'r={name}', '-F', f'n={a.n}')['data']['repository']['pullRequest']['reviewThreads']['nodes']
    open_ = [t for t in th if not t['isResolved']]
    res['threads'] = (not open_, f'{len(th)} review threads, {len(open_)} unresolved' + ''.join(
        f'\n       - {t["path"]}: {t["comments"]["nodes"][0]["author"]["login"]}: {" ".join(re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", "", t["comments"]["nodes"][0]["body"]).split())[:110]}' for t in open_))

    content(pr, res)
    hard = [k for k, (ok, _) in res.items() if not ok and k not in pending]
    return res, (pending if not hard else set())


def said(body, key):
    """the text after `Key:` at the start of a line of the body, '' without one"""
    m = re.search(rf'^{key}:[ \t]*(\S.*)$', body, re.M)
    return m.group(1).strip() if m else ''


def size(pr, body, bot):
    """(ok, text): what the pull request adds and removes; one that grows the tree names what it reused and removed"""
    fs = pr['files']
    add, rem = sum(f.get('additions', 0) for f in fs), sum(f.get('deletions', 0) for f in fs)
    new, gone = (sum(f.get('changeType') == t for f in fs) for t in ('ADDED', 'DELETED'))
    shorter = [f['path'] for f in fs if f.get('changeType') == 'DELETED' or f.get('deletions', 0) > f.get('additions', 0)]
    miss = []
    if add > rem and not bot:
        if not said(body, 'Reuses'): miss.append('no `Reuses:` line (what existing code or doc this extends, or `nothing existing` and what was searched for)')
        removes = said(body, 'Removes')
        named = [n.rstrip('.') for n in NAMED.findall(removes)]
        if not removes: miss.append('no `Removes:` line (the files this deleted or made shorter, or `nothing:` and why)')
        elif re.match(r'nothing\b', removes, re.I):
            if not re.search(r'\w', removes[7:]): miss.append('`Removes: nothing` without its reason')
        elif not named: miss.append('`Removes:` names no file')
        else:
            bad = [n for n in named if not any(p == n or p.endswith('/' + n) for p in shorter)]
            if bad: miss.append(f'`Removes:` names {bad}: not a file this pull request deleted or made shorter')
    if any(f['path'] == BASELINE and f.get('additions', 0) for f in fs) and not said(body, 'Baseline'):
        miss.append(f'adds to {BASELINE} with no `Baseline:` line (why each new entry is not fixed here instead)')
    ag = next((f for f in fs if f['path'] == 'AGENTS.md'), None)
    return (not miss, f'+{add} -{rem} lines, +{new} -{gone} files' + (f', AGENTS.md +{ag.get("additions", 0)} -{ag.get("deletions", 0)}' if ag else '')
            + (': ' + '; '.join(miss) if miss else ''))


def content(pr, res):
    """the pull request's own text: the gate block, what it adds and removes, the title format, the trailers, private text"""
    files, body = [f['path'] for f in pr['files']], pr['body'] or ''
    bot = ((pr.get('author') or {}).get('login') or '') in BOTS   # Dependabot's bumps carry neither a title nor a trailer
    tape = [f for f in files if f.startswith(TAPE)]
    miss = []
    if tape and not GATE.search(body): miss.append('no GPU gate named')
    if tape and not CARD.search(body): miss.append('no card named')
    res['gates'] = (not miss, (f'{len(tape)} kernel / tape files' if tape else 'no kernel or tape file changed') + (': ' + ', '.join(miss) if miss else ''))
    res['size'] = size(pr, body, bot)

    authored = [c for c in pr['commits'] if not MERGE.match(c['messageHeadline'])]   # (merges of testing are not authored)
    if bot:
        res['title'] = (True, f'exempt: a Dependabot pull request ({len(authored)} commits)')
        res['trailers'] = (True, 'exempt: a Dependabot pull request')
    else:
        bad = [] if TITLE.match(pr['title']) else ['pull request title']
        bad += [f'commit {c["oid"][:10]}' for c in authored if not TITLE.match(c['messageHeadline'])]
        res['title'] = (not bad, f'pull request title and {len(authored)} commit headlines are <area> : <what>' + (': ' + ', '.join(bad) if bad else ''))

        none = NO_AI.search(body) is not None   # a commit written without AI help has no trailer, and the body says so
        bad = []
        for c in authored:
            n = len(TRAILER.findall(c['messageBody']))
            if n > 1: bad.append(f'commit {c["oid"][:10]} has {n} Assisted-by trailers')
            elif n == 0 and not none: bad.append(f'commit {c["oid"][:10]} has no Assisted-by trailer')
            elif n == 1 and none: bad.append(f'commit {c["oid"][:10]} has an Assisted-by trailer, but the body says AI assistance: none')
            if COAUTHOR.search(c['messageBody']): bad.append(f'commit {c["oid"][:10]} has Co-Authored-By')
        res['trailers'] = (not bad, f'one Assisted-by per commit (none when the body says AI assistance: none), no Co-Authored-By, {len(authored)} commits' + (': ' + '; '.join(bad) if bad else ''))

    texts = [('title', pr['title']), ('body', body)] + [(c['oid'][:10], c['messageHeadline'] + '\n' + c['messageBody']) for c in pr['commits']]
    hits = sorted({f'{what} in {where}' for where, t in texts for what, rx in PRIVATE if rx.search(t)})
    res['text'] = (not hits, f'title, body and {len(pr["commits"])} commit messages' + (': ' + '; '.join(hits) if hits else ''))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('n', type=int); ap.add_argument('-R', '--repo'); ap.add_argument('--wait', type=int, default=0, metavar='SECS')
    ap.add_argument('--reviewer', default='chatgpt-codex-connector'); ap.add_argument('--poll', type=int, default=30, metavar='SECS')
    ap.add_argument('--allow-base-merges', action='store_true', help='accept merges of testing made after the review without a new review')
    ap.add_argument('--metadata', action='store_true', help='check only the text: base, title, trailers, gates, size, private text (no CI, review or threads; for CI)')
    a = ap.parse_args(argv)
    a.repo = a.repo or gh('repo', 'view', '--json', 'nameWithOwner')['nameWithOwner']
    end = time.monotonic() + a.wait
    while True:
        res, pending = look(a)
        if not pending or time.monotonic() + a.poll > end: break
        print(f'waiting on {sorted(pending)} ({int(end - time.monotonic())} s left)', flush=True)
        time.sleep(a.poll)
    for k, (ok, text) in res.items(): print(('PASS ' if ok else 'FAIL ') + f'{k}: {text}')
    ok = all(v[0] for v in res.values())
    print(f'RESULT {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
