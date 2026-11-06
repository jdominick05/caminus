"""Long-context probes and llama.cpp references (QSA work, checked at <= 32K): CPython's standard library source (PSF licence,
from the env this runs in) with synthetic needles and rare strings. Not all of it is public text: the haystack takes the
modules in sorted order, and past the default size (a large --min-chars, e.g. the 262K sweep's 1,300,000) it reaches
_sysconfigdata__linux_x86_64-linux-gnu.py and _sysconfigdata_x86_64_conda_linux_gnu.py, which the local conda env generates
with its own build and install paths (absolute paths of that machine). Contexts, token ids and logits built with such a haystack stay
private (never published or committed; docs/agentic/sweep-262k.md, Text). The default 420,000 chars (5 files, qsa/lc) does not
reach them.
  python -I tools/lc_probes.py build OUT [--depths 1792,8192,32768]
      the haystack, the probes per depth, their token ids (gg_golden --tokenize under guard.sh: one GPU-free vocab load) and
      OUT/jobs.txt for gg_golden --lc. Per depth D, one context of D tokens (so one fill serves every probe): the haystack
      with one block of two lines at 10 / 50 / 90 % of D, a needle (VAULT_NAME_CODE = 'dddd-dddd') and a rare string
      (TOKEN_TAG_KEY = 'RARE', 14 random letters and digits); then
        tf      the next 256 haystack tokens, teacher forced (logits at positions D .. D + 255)
        q_*     after the context, "assert VAULT_NAME_CODE == '" (or TOKEN_...): code completion, so the greedy answer is the
                copy itself (a question in prose sends the model into <think>), then 24 greedy tokens
      --min-chars N: the haystack's size (default 420,000 chars: enough for 32K; the 262K sweep uses 1,300,000);
      --seq: also OUT/seq.ids, the needle-free haystack (every chunk's ids, no probe block) of at least 262,144 + 257 tokens:
      one sequence whose prefixes give the sweep's G4 windows at every depth (sweep-262k)
  python -I tools/lc_probes.py jobs OUT        (re)write OUT/jobs.txt
  python -I tools/lc_probes.py decode OUT IDS...   the text of token ids (byte-level BPE from the GGUF vocab, host only)
Files: OUT/probes.json (meta: depths, items with token positions, questions, answers), OUT/d<D>/ctx.ids (context +
continuation), OUT/d<D>/<probe>.ids (context + question)."""
import glob, json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'python'))
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
MODEL = devpaths.coder()
GUARD = os.path.join(HERE, 'guard.sh')
GG = os.path.join(REPO, 'build', 'gg_golden')
STDLIB = os.environ.get('LC_STDLIB') or __import__('sysconfig').get_paths()['stdlib']   # the haystack: this python's stdlib
NTF, NGEN, FRACS = 256, 24, (0.10, 0.50, 0.90)
NAMES = ['Kestrel', 'Marlin', 'Osprey', 'Heron', 'Falcon', 'Condor', 'Puffin', 'Plover', 'Avocet', 'Curlew', 'Gannet', 'Merlin']
TAGS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'theta', 'iota', 'kappa', 'lambda', 'sigma', 'omega']


def haystack(min_chars=420_000):
    """CPython stdlib modules (sorted, each >= 20 KB), each under a file header line, until min_chars"""
    out, files, n = [], [], 0
    for p in sorted(glob.glob(os.path.join(STDLIB, '*.py'))):
        t = open(p, encoding='utf-8', errors='strict').read()
        if len(t) < 20_000: continue
        t = f'# ---- file: {os.path.basename(p)} ----\n' + t.replace('\x1e', '') + ('' if t.endswith('\n') else '\n')
        out.append(t); files.append(os.path.basename(p)); n += len(t)
        if n >= min_chars: break
    return ''.join(out), files


def chunks(text, size=400):
    """text cut at newlines into pieces of about `size` chars (each ends with a newline)"""
    out, a = [], 0
    while a < len(text):
        b = text.find('\n', min(a + size, len(text) - 1))
        b = len(text) if b < 0 else b + 1
        out.append(text[a:b]); a = b
    return out


def tokenize(pieces, log):
    """token ids of each piece (each alone), one guarded gg_golden --tokenize call"""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    src, dst = log + '.in', log + '.out'
    with open(src, 'w', encoding='utf-8') as f: f.write('\x1e'.join(pieces))
    r = subprocess.run([GUARD, log, '300', GG, MODEL, '--tokenize', src, dst], capture_output=True, text=True)
    print(r.stdout.strip(), open(log).read().strip().splitlines()[-1:] if os.path.exists(log) else '', file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f'tokenize: guard rc {r.returncode} (98 = refused, 99 = GPU fault: stop)')
    rows = [[int(x) for x in ln.split()] for ln in open(dst).read().split('\n')[:len(pieces)]]
    assert len(rows) == len(pieces), (len(rows), len(pieces))
    return rows


def build(out, depths, min_chars=420_000, want_seq=False):
    rng = random.Random(20261102)
    text, files = haystack(min_chars)
    ch = chunks(text)
    alnum = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
    plan = []
    for di, D in enumerate(depths):
        items = []
        for fi, f in enumerate(FRACS):
            j = di * 3 + fi
            name, tag = NAMES[j], f'{TAGS[j]}_key'
            code = f'{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}'
            rare = ''.join(rng.choice(alnum) for _ in range(14))
            assert rare not in text and code not in text and name not in text.split() and tag not in text
            items.append(dict(frac=f, name=name, code=code, tag=tag, rare=rare,
                              block=f'VAULT_{name.upper()}_CODE = \'{code}\'  # the access code of vault {name}\nTOKEN_{tag.upper()} = \'{rare}\'\n',
                              q_needle=f'\n\nassert VAULT_{name.upper()}_CODE == \'', q_rare=f'\n\nassert TOKEN_{tag.upper()} == \''))
        plan.append(dict(D=D, items=items))
    pieces = list(ch)
    for p in plan:
        for it in p['items']: pieces += [it['block'], it['q_needle'], it['q_rare']]
    ids = tokenize(pieces, os.path.join(out, 'logs', 'tokenize.log'))
    cid, rest = ids[:len(ch)], ids[len(ch):]
    cum = [0]
    for r in cid: cum.append(cum[-1] + len(r))
    meta = dict(model=MODEL, haystack_files=files, stdlib=STDLIB, ntf=NTF, ngen=NGEN, depths=[], haystack_chars=len(text), haystack_tokens=cum[-1])
    if want_seq:
        allids = [t for r in cid for t in r]
        assert len(allids) >= 262144 + NTF + 1, f'haystack {len(allids)} tokens: too short for the 262K sequence (raise --min-chars)'
        open(os.path.join(out, 'seq.ids'), 'w').write(' '.join(map(str, allids)) + '\n')
        meta['seq'] = dict(path=os.path.join(out, 'seq.ids'), tokens=len(allids), note='needle-free: every chunk of the haystack, no probe block')
    k = 0
    for p in plan:
        D = p['D']; blocks = []
        for it in p['items']:
            it['block_ids'], it['qn_ids'], it['qr_ids'] = rest[k], rest[k + 1], rest[k + 2]; k += 3
            ci = min(range(len(cum)), key=lambda i: abs(cum[i] - it['frac'] * D))   # the chunk boundary nearest frac * D
            blocks.append((ci, it))
        seq, pos_of = [], {}
        bi = 0
        for ci in range(len(cid)):
            while bi < len(blocks) and blocks[bi][0] == ci:
                pos_of[blocks[bi][1]['name']] = len(seq); seq += blocks[bi][1]['block_ids']; bi += 1
            seq += cid[ci]
            if len(seq) >= D + NTF + 1: break
        assert len(seq) >= D + NTF + 1, f'haystack too short for depth {D}'
        ctx = seq[:D + NTF + 1]
        dd = os.path.join(out, f'd{D}'); os.makedirs(dd, exist_ok=True)
        open(os.path.join(dd, 'ctx.ids'), 'w').write(' '.join(map(str, ctx)) + '\n')
        probes = []
        for it in p['items']:
            for kind, q, ans in (('needle', it['qn_ids'], it['code']), ('rare', it['qr_ids'], it['rare'])):
                name = f'q_{kind}_{int(it["frac"] * 100)}'
                toks = ctx[:D] + q
                open(os.path.join(dd, f'{name}.ids'), 'w').write(' '.join(map(str, toks)) + '\n')
                probes.append(dict(name=name, kind=kind, frac=it['frac'], item_pos=pos_of[it['name']], answer=ans, n_q=len(q),
                                   P=len(toks), key=it['name'] if kind == 'needle' else it['tag']))
        meta['depths'].append(dict(D=D, ctx_tokens=len(ctx), items=[{k2: v for k2, v in it.items() if not k2.endswith('_ids')}
                                                                    | dict(pos=pos_of[it['name']], block_ntok=len(it['block_ids']))
                                                                    for it in p['items']], probes=probes))
        print(f'depth {D}: items at ' + ', '.join(f'{it["name"]}@{pos_of[it["name"]]} ({pos_of[it["name"]] / D:.0%})' for it in p['items']),
              file=sys.stderr)
    json.dump(meta, open(os.path.join(out, 'probes.json'), 'w'), indent=1)
    jobs(out)


def jobs(out):
    meta = json.load(open(os.path.join(out, 'probes.json')))
    D_max = max(d['D'] for d in meta['depths'])
    need = max(max(d['D'] + meta['ntf'], max(p['P'] + meta['ngen'] for p in d['probes'])) for d in meta['depths'])
    lines = [f'nctx {(need + 255) // 256 * 256}']
    ref = os.path.join(out, 'ref'); os.makedirs(ref, exist_ok=True)
    for d in meta['depths']:
        D = d['D']; dd = os.path.join(out, f'd{D}')
        for r in (1, 2):   # twice: llama.cpp's own run-to-run drift (radix top-k ties above ~4K)
            lines.append(f'tf tf{D}_r{r} {dd}/ctx.ids {D} {meta["ntf"]} {ref}/tf{D}_r{r}')
        for p in d['probes']:
            lines.append(f'greedy {p["name"]}_{D} {dd}/{p["name"]}.ids {D} {p["P"]} {meta["ngen"]} {ref}/{p["name"]}_{D}')
    open(os.path.join(out, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f'{out}/jobs.txt: {len(lines) - 1} jobs, {lines[0]} (deepest {D_max})', file=sys.stderr)


_DEC = None


def decode(ids):
    """text of token ids: the GGUF vocab's byte-level BPE strings mapped back to bytes (GPT-2 bytes_to_unicode)"""
    global _DEC
    if _DEC is None:
        sys.path.insert(0, os.path.join(REPO, 'python'))
        from caminus import flashnext  # noqa: F401  (gguf-py on the path)
        from gguf import GGUFReader
        f = {x.name: x for x in GGUFReader(MODEL, 'r').fields.values()}
        toks = f['tokenizer.ggml.tokens'].contents()
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]; n = 0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256 + n); n += 1
        u2b = {chr(c): b for b, c in zip(bs, cs)}
        _DEC = (toks, u2b)
    toks, u2b = _DEC
    out = bytearray()
    for i in ids:
        s = toks[int(i)]
        if all(ch in u2b for ch in s): out += bytes(u2b[ch] for ch in s)
        else: out += s.encode('utf-8')   # special / added tokens
    return out.decode('utf-8', errors='replace')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    if a[0] == 'build':
        dp = [1792, 8192, 32768]; mc = 420_000; sq = '--seq' in a
        if sq: a.remove('--seq')
        if '--depths' in a: i = a.index('--depths'); dp = [int(x) for x in a[i + 1].split(',')]; del a[i:i + 2]
        if '--min-chars' in a: i = a.index('--min-chars'); mc = int(a[i + 1]); del a[i:i + 2]
        build(a[1], dp, mc, sq)
    elif a[0] == 'jobs': jobs(a[1])
    elif a[0] == 'decode': print(repr(decode(a[2:])))
    else: print(__doc__); sys.exit(2)
