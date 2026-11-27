"""Per-tensor quantization planning for a dense qwen35 GGUF (the Qwen3.8-27B line) on one card, CPU only: given the GGUF
header, a cost table and a sensitivity table, choose a format per tensor that minimizes the PREDICTED decode time per
token (a cost model, labelled as one everywhere it prints — never a measurement) under a VRAM cap and a quality-loss cap.

The inventory comes from the header through engine/q35_file.py (module_of, block kinds, projection terms) and the tape's
structure from engine/q35_dense.py _build_tape: one matvec segment per matrix, the feed-forward's gate | up fused into one
segment when q35_file.gu_fused says so, the embedding's one row and the head once per token. The menu is exactly the
(op, format) pairs of the module's k3.MODULE_CAPS; anything else is refused. Tables: load_costs / load_sensitivity below;
the default cost table is tools/q35_format_plan.costs.json (every number carries its published source, unmeasured entries
say so), the sensitivity schema is documented in load_sensitivity and would be filled by a per-tensor arm of G-model
(docs/specs/spec-own-quant.md; tools/gates/q35_gmodel.py) — the shipped file is a synthetic example.

Used by tools/q35_format_plan.py; the CPU tests are tests/tools/test_q35_format_plan.py."""
import json
import sys

from .. import ggufio
from . import k3
from . import q35_file

SCHEMA_COSTS = 'q35-format-plan.costs/1'
SCHEMA_SENS = 'q35-format-plan.sensitivity/1'


class PlanError(ValueError):
    """a refused plan: an unsupported (op, format) pair, a split fusion group, an infeasible budget, a bad table"""


QUANT_FMTS = tuple(sorted({v[0] for v in ggufio.TYPES.values() if v[1] > 1}))   # the block formats (bs > 1)
CARD_FORMATS = ('F32',) + QUANT_FMTS          # a tensor's format on the card: F32 (F32 / BF16 widened) or a block format


def one_rt_of(module):
    """the formats whose matvecs Dense35Own runs on the one-round-trip tiles (its self.one_rt)"""
    return (k3.OWN_Q35R_1RT if module in k3.Q35R_MODULES else k3.OWN_Q35T_1RT) if module in k3.Q35T_MODULES else k3.OWN_1RT_FMT


def card_fmt(t):
    """the format a tensor has on the card in the file as it is: F32 for F32 / BF16 (widened on load), else its GGUF type"""
    return 'F32' if t.type in ('F32', 'BF16') else t.type


def tape_pair(module, t, fmt, fused=False):
    """the (op, format) pair Dense35Own's tape runs tensor t with, in card format fmt (k3.MODULE_CAPS' keys): the
    embedding on OP_OWN_EMB (a quantized type only — the tape always reads it through own_emb), F32 matrices on
    OP_F32_MV, a one-round-trip format on OP_OWN_MV ('..._gu' when the gate | up tile is fused), any other block format
    on OP_OWN_MV16; None where the tape has no op for the tensor (a 1-D vector)"""
    if t.name == 'token_embd.weight':
        return ('OWN_EMB', fmt) if fmt != 'F32' else None
    if fmt == 'F32':
        return ('F32_MV', 'F32') if len(t.shape) == 2 else None
    if len(t.shape) != 2:
        return None
    if fmt in one_rt_of(module):
        return ('OWN_MV', f'{fmt}/a16_1rt_gu') if fused else ('OWN_MV', f'{fmt}/a16_1rt')
    return ('OWN_MV16', fmt)


def tensor_menu(module, t, fused=False):
    """the card formats the module's k3.MODULE_CAPS pairs allow for tensor t (sorted); the planner's format menu"""
    caps = k3.MODULE_CAPS[module]
    out = []
    for fmt in CARD_FORMATS:
        p = tape_pair(module, t, fmt, fused=fused)
        if p is not None and p in caps:
            out.append(fmt)
    return out


def inventory(F, module=None):
    """the decode inventory of a dense qwen35 GGUF's header: module (module_of unless given), shapes, block kinds, and
    per tensor a record {name, t, role, fmt (card format now), rows, K, stage, group}; groups are the fusion groups
    (the feed-forward's gate | up of a block, where the module's tape fuses them). Raises PlanError when the file's own
    formats are outside the module's pairs (a file this module cannot serve)."""
    module = module or q35_file.module_of(F)
    if module not in k3.MODULE_CAPS:
        raise PlanError(f'module {module}: no k3.MODULE_CAPS entry (the qwen35 decode modules: {k3.Q35_MODULES})')
    sh = q35_file._shapes(F)
    kinds = q35_file.block_kinds(F)
    fuses = module not in k3.Q35T_MODULES        # (Dense35Own.__init__'s self.gu = gu and not tcap, gu default True)
    recs = {}

    def add(name, role, stage, group=None):
        t = F.T[name]
        recs[name] = dict(name=name, t=t, role=role, fmt=card_fmt(t), rows=t.rows()[0], K=int(t.shape[0]),
                          stage=stage, group=group)

    for il, gdn in enumerate(kinds):
        p = f'blk.{il}.'
        for role in (q35_file.GDN_MAT_ROLES if gdn else q35_file.ATTN_MAT_ROLES):
            add(p + role + '.weight', role, 'mv')
        for role in q35_file.FFN_MAT_ROLES:
            add(p + role + '.weight', role, 'mv', group=(f'ffn|{il}' if fuses and role != 'ffn_down' else None))
        add(p + 'attn_norm.weight', 'norm', 'glue')
        add(p + 'post_attention_norm.weight', 'norm', 'glue')
        if gdn:
            add(p + 'ssm_alpha.weight', 'ssm_alpha', 'mv')
            add(p + 'ssm_beta.weight', 'ssm_beta', 'mv')
            for nm in ('ssm_norm.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_conv1d.weight'):
                add(p + nm, 'gdn_small', 'glue')
        else:
            add(p + 'attn_q_norm.weight', 'norm', 'glue')
            add(p + 'attn_k_norm.weight', 'norm', 'glue')
    add('token_embd.weight', 'emb', 'emb')
    add('output.weight', 'head', 'mv')
    add('output_norm.weight', 'norm', 'glue')
    groups = {}
    for r in recs.values():
        if r['group'] is not None:
            groups.setdefault(r['group'], []).append(r['name'])
    for name, r in recs.items():
        if r['stage'] == 'glue':      # (a 1-D vector read inside a glue / norm op: no op of its own, no format choice)
            continue
        fused = bool(r['group']) and r['fmt'] in k3.OWN_1RT_FMT
        if r['fmt'] not in tensor_menu(module, r['t'], fused=fused):
            p = tape_pair(module, r['t'], r['fmt'], fused=fused)
            raise PlanError(f'{name}: its format {r["fmt"]} is pair {p}, not in k3.MODULE_CAPS[{module}] '
                            f'(the module cannot run this file; its menu: {tensor_menu(module, r["t"], fused=fused)})')
    return dict(module=module, shapes=sh, kinds=kinds, recs=recs, groups=groups,
                kv_row=k3.kv_pos_bytes(sh['nkv'] * sh['hd'], False), n_attn=sum(1 for g in kinds if not g))


def shape_keys(rows, K):
    """the shape-class keys of a stage, most specific first: the exact (rows x K), then the rows bucket"""
    exact = f'{rows}x{K}'
    bucket = 'wide' if rows >= 16384 else 'mid' if rows >= 4096 else 'narrow'
    return (exact, bucket)


def load_costs(path):
    """a cost table file (schema q35-format-plan.costs/1): {cards: {name: {read_Bps, source, measured, ...}}, entries:
    [{stage, card, format, shape, fixed_us | us+bytes, per_byte_us, measured, source, note}]}. A stage's time is a fixed
    part plus a per-byte part; entries with null / absent numbers say so (measured: false or "total-only") and every
    prediction that uses them is flagged. Returns the parsed table with per-byte values resolved."""
    with open(path) as f:
        tab = json.load(f)
    if tab.get('schema') != SCHEMA_COSTS:
        raise PlanError(f'{path}: cost table schema {tab.get("schema")!r}, expected {SCHEMA_COSTS}')
    for card, c in tab['cards'].items():
        if not c.get('read_Bps') or not c.get('source'):
            raise PlanError(f'{path}: card {card} needs read_Bps and source')
    entries = []
    for e in tab['entries']:
        e = dict(e)
        for k in ('stage', 'card', 'format', 'shape'):
            if k not in e:
                raise PlanError(f'{path}: entry {e} lacks {k}')
        if 'per_byte_us' not in e:
            if 'us' in e and 'bytes' in e:
                fixed = e.get('fixed_us') or 0.0
                e['per_byte_us'] = (e['us'] - fixed) * 1e-6 / e['bytes']
            else:
                e['per_byte_us'] = None
        if 'fixed_us' not in e:
            e['fixed_us'] = None
        if not e.get('source'):
            raise PlanError(f'{path}: entry {e} has no source (every number must cite its published figure)')
        entries.append(e)
    tab['entries'] = entries
    return tab


def _entry_for(costs, stage, card, fmt, rows, K):
    """the most specific entry for a stage: (card, format, exact shape), then bucket, then wildcards, then the stage
    wildcard; None when the table has none (the caller then uses the card's read rate / 0 and flags it)"""
    keys = shape_keys(rows, K)
    for c in (card, '*'):
        for f in (fmt, '*'):
            for s in keys + ('*',):
                for e in costs['entries']:
                    if e['stage'] == stage and e['card'] == c and e['format'] == f and e['shape'] == s:
                        return e
    return None


def stage_cost(costs, card, stage, fmt, rows, K, bytes_read):
    """(us, used_entries) the cost model gives one stage: fixed + bytes_read * per-byte; per-byte defaults to the card's
    published read rate. Entries that are not plain measurements (measured != true, an assumption note, or a missing
    field) are returned so the caller can flag every result that depends on them"""
    e = _entry_for(costs, stage, card, fmt, rows, K)
    fixed = e['fixed_us'] if e is not None and e['fixed_us'] is not None else 0.0
    per_byte = e['per_byte_us'] if e is not None and e['per_byte_us'] is not None else 1.0 / costs['cards'][card]['read_Bps']
    used, shape = [], shape_keys(rows, K)[0]
    if e is None or e['fixed_us'] is None:
        used.append(dict(stage=stage, card=card, format=fmt, shape=shape, field='fixed_us', value=None,
                         source=e['source'] if e else None, note='unmeasured: no entry gives a fixed part; counted as 0'))
    if e is None or e['per_byte_us'] is None:
        used.append(dict(stage=stage, card=card, format=fmt, shape=shape, field='per_byte_us', value=per_byte,
                         source=costs['cards'][card]['source'],
                         note='the card\'s published read rate; per-format deviations are only measured where an entry says so'))
    if e is not None and (e.get('measured') is not True or e.get('assumption')):
        used.append(dict(stage=stage, card=card, format=fmt, shape=shape, field='entry', value=e.get('measured'),
                         source=e['source'], note=e.get('note', '') or e.get('assumption', '')))
    return fixed + bytes_read * per_byte * 1e6, used


def load_sensitivity(path):
    """a sensitivity table file (schema q35-format-plan.sensitivity/1):
    {"schema": ..., "unit": ..., "filled_by": ..., "synthetic": bool, "default": loss or null,
     "tensors": {name: {fmt: loss}}} — per tensor and format, the quality loss of that tensor in that format (the file's
    own format is the zero baseline). The real table needs the model and a card: a per-tensor arm of G-model
    (docs/specs/spec-own-quant.md, tools/gates/q35_gmodel.py) measuring one decode's mean KLD against the reference mode
    with one tensor alone requantized. The shipped file is a synthetic example and says so."""
    with open(path) as f:
        tab = json.load(f)
    if tab.get('schema') != SCHEMA_SENS:
        raise PlanError(f'{path}: sensitivity schema {tab.get("schema")!r}, expected {SCHEMA_SENS}')
    if not isinstance(tab.get('tensors'), dict):
        raise PlanError(f'{path}: sensitivity table needs a "tensors" object')
    return tab


def loss_of(sens, name, fmt):
    """the quality loss of tensor name in format fmt: the table's per-tensor entry, else its '*' entry, else default;
    refused when the table says nothing (an unknown loss is never guessed)"""
    for src in (sens['tensors'].get(name), sens['tensors'].get('*')):
        if src and fmt in src:
            return float(src[fmt])
    if sens.get('default') is not None:
        return float(sens['default'])
    raise PlanError(f'sensitivity table has no entry for {name} in {fmt} (and no default): the loss is unknown')


def read_bytes(rec, fmt):
    """bytes one token reads of a tensor at card format fmt: the whole matrix (upload_bytes: what the card holds),
    or one embedding row"""
    if rec['stage'] == 'emb':
        if fmt == 'F32':
            return 4 * rec['K']
        bs, ts = next((v[1], v[2]) for v in ggufio.TYPES.values() if v[0] == fmt)
        return rec['K'] // bs * ts
    return q35_file.upload_bytes(rec['t'], fmt)


def stages_of(inv, choice):
    """the decode stages of one token at assignment choice {name: card format}: [(label, kind, fmt, rows, K, names)].
    The stage skeleton is q35_dense.py _build_tape's: per block a norm, the block's matvecs, its glue, the attention
    (kind 'attn') in an attention block, the feed-forward (one fused 'mv' stage for gate | up when q35_file.gu_fused,
    else two matvecs plus the SwiGLU glue), the down matvec, the post norm; then the head's norm and matvec; the
    embedding's row first."""
    module, kinds, recs = inv['module'], inv['kinds'], inv['recs']
    fmt = lambda n: choice.get(n, recs[n]['fmt'])
    out = [('emb', 'emb', fmt('token_embd.weight'), recs['token_embd.weight']['rows'], recs['token_embd.weight']['K'],
            ['token_embd.weight'])]
    for il, gdn in enumerate(kinds):
        p = f'blk.{il}.'
        out.append((f'blk.{il}.add_rms', 'norm', 'F32', 0, 0, []))
        if gdn:
            for role in q35_file.GDN_MAT_ROLES + ('ssm_alpha', 'ssm_beta'):
                r = recs[p + role + '.weight']
                out.append((f'blk.{il}.{role}', 'mv', fmt(r['name']), r['rows'], r['K'], [r['name']]))
            out += [(f'blk.{il}.gdn_gates', 'glue', 'F32', 0, 0, []), (f'blk.{il}.gdn_conv', 'glue', 'F32', 0, 0, []),
                    (f'blk.{il}.gdn_l2', 'glue', 'F32', 0, 0, []), (f'blk.{il}.gdn_step_save_norm', 'glue', 'F32', 0, 0, [])]
        else:
            for role in ('attn_k', 'attn_v', 'attn_q', 'attn_output'):
                r = recs[p + role + '.weight']
                out.append((f'blk.{il}.{role}', 'mv', fmt(r['name']), r['rows'], r['K'], [r['name']]))
            out += [(f'blk.{il}.qk_nr_k', 'glue', 'F32', 0, 0, []), (f'blk.{il}.kv_append', 'glue', 'F32', 0, 0, []),
                    (f'blk.{il}.qk_nr_q', 'glue', 'F32', 0, 0, []), (f'blk.{il}.attn', 'attn', 'F32', 0, 0, []),
                    (f'blk.{il}.gate_mul', 'glue', 'F32', 0, 0, [])]
        out.append((f'blk.{il}.post_add_rms', 'norm', 'F32', 0, 0, []))
        gate, up = recs[p + 'ffn_gate.weight'], recs[p + 'ffn_up.weight']
        if q35_file.gu_fused(module, fmt(gate['name']), fmt(up['name'])):
            out.append((f'blk.{il}.ffn_gu', 'mv', fmt(gate['name']), gate['rows'], gate['K'], [gate['name'], up['name']]))
        else:
            out.append((f'blk.{il}.ffn_gate', 'mv', fmt(gate['name']), gate['rows'], gate['K'], [gate['name']]))
            out.append((f'blk.{il}.ffn_up', 'mv', fmt(up['name']), up['rows'], up['K'], [up['name']]))
            out.append((f'blk.{il}.swiglu', 'glue', 'F32', 0, 0, []))
        r = recs[p + 'ffn_down.weight']
        out.append((f'blk.{il}.ffn_down', 'mv', fmt(r['name']), r['rows'], r['K'], [r['name']]))
    out.append(('head.add_rms', 'norm', 'F32', 0, 0, []))
    r = recs['output.weight']
    out.append(('head', 'mv', fmt(r['name']), r['rows'], r['K'], [r['name']]))
    return out


def predict(inv, choice, costs, card, depth):
    """the predicted decode time per token at assignment choice, with the per-stage breakdown — A COST-MODEL PREDICTION,
    not a measurement. Returns dict(total_us, stages [(label, kind, fmt, us, bytes, names)], by_kind, weights_bytes,
    unmeasured [the entries used that are not plain measurements])"""
    total, stages, by_kind, unmeasured = 0.0, [], {}, []
    for label, kind, fmt, rows, K, names in stages_of(inv, choice):
        if kind == 'attn':
            us, used = stage_cost(costs, card, 'attn', 'F32', 0, 0, (depth + 1) * inv['kv_row'])
            b = (depth + 1) * inv['kv_row']
        elif kind in ('mv', 'emb'):
            b = sum(read_bytes(inv['recs'][n], choice.get(n, inv['recs'][n]['fmt'])) for n in names)
            us, used = stage_cost(costs, card, kind, fmt, rows, K, b)
        else:
            b, us, used = 0, *stage_cost(costs, card, kind, 'F32', 0, 0, 0)
        total += us
        by_kind[kind] = by_kind.get(kind, 0.0) + us
        stages.append((label, kind, fmt, us, b, names))
        for u in used:
            if not any(x['stage'] == u['stage'] and x['card'] == u['card'] and x['format'] == u['format'] and
                       x['shape'] == u['shape'] and x['field'] == u['field'] for x in unmeasured):
                unmeasured.append(u)
    weights = sum(q35_file.upload_bytes(r['t'], choice.get(n, r['fmt'])) for n, r in inv['recs'].items())
    return dict(total_us=total, stages=stages, by_kind=by_kind, weights_bytes=weights, unmeasured=unmeasured)


def vram_bytes(inv, F, choice, max_ctx):
    """the VRAM the assignment needs on the card at max_ctx: the weights at the chosen formats plus the projection's
    KV / GDN-state / attention-record terms (q35_file._projection_terms — those do not depend on the choice)"""
    terms = q35_file._projection_terms(F, max_ctx)
    weights = sum(q35_file.upload_bytes(r['t'], choice.get(n, r['fmt'])) for n, r in inv['recs'].items())
    return weights + terms['kv'] + terms['gdn'] + terms['part']


def _mv_us(costs, card, kind, rec, fmt):
    """the predicted us of one matvec / embedding stage of tensor rec at card format fmt"""
    return stage_cost(costs, card, kind, fmt, rec['rows'], rec['K'], read_bytes(rec, fmt))[0]


def group_options(inv, costs, sens, card, depth, var):
    """the options of one variable: a fusion group name ('ffn|<il>', one shared format for all its members) or a
    singleton tensor name. Each option: dict(fmts {name: fmt}, cost_us, bytes, quality). Refuses (PlanError) when a
    member's format has no pair in the module's MODULE_CAPS, or a group has no shared option."""
    module, recs = inv['module'], inv['recs']
    members = inv['groups'][var] if var in inv['groups'] else [var]
    is_group = var in inv['groups']
    menus = {}
    for n in members:
        r = recs[n]
        m = tensor_menu(module, r['t'], fused=False)   # (the plain matrix menu; whether the chosen shared format
        if not m:                                       #  runs fused is decided per option below)
            p = tape_pair(module, r['t'], r['fmt'], fused=is_group)
            raise PlanError(f'{n}: no (op, format) pair in k3.MODULE_CAPS[{module}] (tape pair {p} for {r["fmt"]})')
        menus[n] = m
    if is_group:
        shared = [f for f in menus[members[0]]
                  if all(f in menus[n] for n in members)
                  and (f not in k3.OWN_1RT_FMT or ('OWN_MV', f'{f}/a16_1rt_gu') in k3.MODULE_CAPS[module])]
        if not shared:
            raise PlanError(f'fusion group {var} {members}: no format all members share with a pair in '
                            f'k3.MODULE_CAPS[{module}] (menus: {menus})')
        combos = [{n: f for n in members} for f in shared]
    else:
        combos = [{members[0]: f} for f in menus[members[0]]]
    opts = []
    for c in combos:
        ok = True
        for n, f in c.items():
            fused = is_group and f in k3.OWN_1RT_FMT and q35_file.gu_fused(module, f, f)
            p = tape_pair(module, recs[n]['t'], f, fused=fused)
            if p is None or p not in k3.MODULE_CAPS[module]:
                ok = False
                break
        if not ok:
            continue
        if is_group:   # (the fused gate | up tile is one stage reading both matrices; unfused: two matvecs + SwiGLU)
            g, u = recs[members[0]], recs[members[1]]
            f = c[members[0]]
            if q35_file.gu_fused(module, f, f):
                cost = stage_cost(costs, card, 'mv', f, g['rows'], g['K'],
                                  read_bytes(g, f) + read_bytes(u, f))[0]
            else:
                cost = (_mv_us(costs, card, 'mv', g, f) + _mv_us(costs, card, 'mv', u, f)
                        + stage_cost(costs, card, 'glue', 'F32', 0, 0, 0)[0])
        else:
            r = recs[members[0]]
            cost = _mv_us(costs, card, r['stage'], r, c[members[0]])
        opts.append(dict(fmts=c, cost_us=cost,
                         bytes=sum(q35_file.upload_bytes(recs[n]['t'], f) for n, f in c.items()),
                         quality=sum(loss_of(sens, n, f) for n, f in c.items())))
    if not opts:
        raise PlanError(f'{var}: no option with a pair in k3.MODULE_CAPS[{module}]')
    return opts


def _variables(inv, costs, sens, card, depth):
    """every decision variable: the fusion groups, then the singleton matvec / embedding tensors"""
    out = [(gid, group_options(inv, costs, sens, card, depth, gid)) for gid in sorted(inv['groups'])]
    out += [(n, group_options(inv, costs, sens, card, depth, n)) for n in sorted(inv['recs'])
            if inv['recs'][n]['group'] is None and inv['recs'][n]['stage'] in ('mv', 'emb')]
    return out


def _drop_dominated(opts):
    """remove options dominated by another (cost, bytes, quality all no worse, one strictly): the optimum is unchanged"""
    out = []
    for a in opts:
        if any(b is not a and b['cost_us'] <= a['cost_us'] and b['bytes'] <= a['bytes'] and b['quality'] <= a['quality']
               and (b['cost_us'] < a['cost_us'] or b['bytes'] < a['bytes'] or b['quality'] < a['quality']) for b in opts):
            continue
        out.append(a)
    return out


def solve(inv, F, costs, sens, card, depth, max_ctx, vram_cap, quality_cap, grid=1024):
    """the exact minimum of the predicted decode time per token over the menu, under the VRAM and quality caps: a
    branch-and-bound over the variables, each option's cost / bytes / quality precomputed (group_options), dominated
    options dropped, bounded by the suffix minima and a suffix dynamic program over the quality cap (a valid lower
    bound: option losses are floored to a grid of `grid` cells, which can only relax the cap). Visited states are
    memoized only when neither cap can bind below them, so equal keys have identical feasible completion sets. Returns
    dict(choice, total_us, by_kind, stages, weights_bytes, vram, quality, binding, unconstrained_us, unmeasured).
    Raises PlanError when no assignment satisfies the caps (which cap, and its minimum possible value)."""
    variables = [(name, _drop_dominated(opts)) for name, opts in _variables(inv, costs, sens, card, depth)]
    n = len(variables)
    # the tensors that are not variables (the 1-D glue / norm vectors) still count towards the caps, at their own
    # format, and so do the projection's KV / GDN-state / attention-record terms at max_ctx: carry all of that as
    # constants of every assignment, so the VRAM constraint is the full projection (vram_bytes) against the cap
    var_names = {k for _, opts in variables for o in opts for k in o['fmts']}
    terms = q35_file._projection_terms(F, max_ctx)
    const_bytes = sum(q35_file.upload_bytes(r['t'], r['fmt']) for n_, r in inv['recs'].items() if n_ not in var_names)
    const_bytes += terms['kv'] + terms['gdn'] + terms['part'] + terms['extra']
    const_qual = sum(loss_of(sens, n_, r['fmt']) for n_, r in inv['recs'].items() if n_ not in var_names)
    min_cost = [min(o['cost_us'] for o in opts) for _, opts in variables]
    min_bytes = [min(o['bytes'] for o in opts) for _, opts in variables]
    min_qual = [min(o['quality'] for o in opts) for _, opts in variables]
    suf_cost = [0.0] * (n + 1); suf_bytes = [0.0] * (n + 1); suf_qual = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf_cost[i] = suf_cost[i + 1] + min_cost[i]
        suf_bytes[i] = suf_bytes[i + 1] + min_bytes[i]
        suf_qual[i] = suf_qual[i + 1] + min_qual[i]
    if const_bytes + suf_bytes[0] > vram_cap:
        raise PlanError(f'infeasible VRAM cap {vram_cap} B: the minimum is {const_bytes + suf_bytes[0]:.0f} B '
                        f'(the projection at max_ctx plus the smallest format of every matrix)')
    if const_qual + suf_qual[0] > quality_cap:
        raise PlanError(f'infeasible quality-loss cap {quality_cap}: the minimum total loss is {const_qual + suf_qual[0]} '
                        f'({const_qual} of fixed tensors plus the smallest loss the sensitivity table gives)')
    order = sorted(range(n), key=lambda i: -(max(o['cost_us'] for o in variables[i][1]) - min_cost[i]))
    variables = [variables[i] for i in order]
    min_cost = [min_cost[i] for i in order]; min_bytes = [min_bytes[i] for i in order]; min_qual = [min_qual[i] for i in order]
    suf_cost = [0.0] * (n + 1); suf_bytes = [0.0] * (n + 1); suf_qual = [0.0] * (n + 1)
    suf_bmax = [0.0] * (n + 1); suf_qmax = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf_cost[i] = suf_cost[i + 1] + min_cost[i]
        suf_bytes[i] = suf_bytes[i + 1] + min_bytes[i]
        suf_qual[i] = suf_qual[i + 1] + min_qual[i]
        suf_bmax[i] = suf_bmax[i + 1] + max(o['bytes'] for o in variables[i][1])
        suf_qmax[i] = suf_qmax[i + 1] + max(o['quality'] for o in variables[i][1])
    # the suffix DP over the quality cap: LB[i][q] = the least cost of variables i.. with floored losses summing to <= q
    delta = quality_cap / grid if quality_cap > 0 else 1.0
    Q = int(quality_cap / delta) if quality_cap > 0 else 0
    qf = [[int(o['quality'] // delta) for o in opts] for _, opts in variables]
    INF = float('inf')
    LB = [[INF] * (Q + 1) for _ in range(n + 1)]
    LB[n] = [0.0] * (Q + 1)
    for i in range(n - 1, -1, -1):
        row, nxt, opts = LB[i], LB[i + 1], variables[i][1]
        for q in range(Q + 1):
            b = INF
            for oi in range(len(opts)):
                qq = qf[i][oi]
                if qq <= q:
                    v = opts[oi]['cost_us'] + nxt[q - qq]
                    if v < b: b = v
            row[q] = b
    # a greedy feasible assignment as the first upper bound (starting from the constants)
    best = [None]
    c = 0.0; b = const_bytes; q = const_qual; qi = int(const_qual // delta) if quality_cap > 0 else 0; assign = {}
    for i, (name, opts) in enumerate(variables):
        cand = [o for o in opts if b + o['bytes'] <= vram_cap and q + o['quality'] <= quality_cap
                and qi + min(qf[i]) <= Q]
        if not cand:
            break
        o = min(cand, key=lambda o: o['cost_us'])
        c += o['cost_us']; b += o['bytes']; q += o['quality']
        qi += int(o['quality'] // delta)
        assign.update(o['fmts'])
    else:
        best[0] = (c, dict(assign))
    seen = {}   # a search state -> the least cost it was reached with (a costlier visit adds nothing)

    def rec(i, cost, byts, qual, qacc, assign):
        if byts > vram_cap or qual > quality_cap or qacc > Q:
            return
        dp = LB[i][Q - qacc]
        if dp == INF:
            return                                   # the remaining variables cannot fit the quality budget
        if best[0] is not None and max(cost + suf_cost[i], cost + dp) >= best[0][0] - 1e-9:
            return
        if byts + suf_bytes[i] > vram_cap or qual + suf_qual[i] > quality_cap:
            return
        # a cap that cannot bind below this node (even the largest remaining options fit) drops out of the memo
        # key; otherwise the exact running bytes / loss stay in it, so equal keys have identical completion sets
        if byts + suf_bmax[i] <= vram_cap and qual + suf_qmax[i] <= quality_cap:
            key = (i,)
        elif byts + suf_bmax[i] <= vram_cap:
            key = (i, qual)
        elif qual + suf_qmax[i] <= quality_cap:
            key = (i, byts)
        else:
            key = (i, byts, qual)
        if seen.get(key, INF) <= cost + 1e-9:
            return
        seen[key] = cost
        if i == n:
            if best[0] is None or cost < best[0][0] - 1e-9:
                best[0] = (cost, dict(assign))
            return
        for o, qq in sorted(zip(variables[i][1], qf[i]), key=lambda t: (t[0]['cost_us'], sorted(t[0]['fmts'].items()))):
            rec(i + 1, cost + o['cost_us'], byts + o['bytes'], qual + o['quality'], qacc + qq, {**assign, **o['fmts']})

    sys.setrecursionlimit(max(10000, 4 * n + 1000))
    rec(0, 0.0, const_bytes, const_qual, int(const_qual // delta) if quality_cap > 0 else 0, {})
    if best[0] is None:
        raise PlanError(f'infeasible: no assignment in the menu satisfies the VRAM cap {vram_cap} B and the '
                        f'quality-loss cap {quality_cap}')
    choice = {n_: r['fmt'] for n_, r in inv['recs'].items()}
    choice.update(best[0][1])
    pred = predict(inv, choice, costs, card, depth)
    vram = vram_bytes(inv, F, choice, max_ctx)
    qual = sum(loss_of(sens, n_, choice[n_]) for n_ in inv['recs'])
    binding = []
    if vram >= vram_cap - 1e-6:
        binding.append(f'VRAM cap ({vram:.0f} of {vram_cap} B)')
    if qual >= quality_cap - 1e-15:
        binding.append(f'quality-loss cap ({qual} of {quality_cap})')
    if not binding:
        binding = ['none: the menu alone decides (the unconstrained optimum is feasible)']
    return dict(choice=choice, total_us=pred['total_us'], by_kind=pred['by_kind'], stages=pred['stages'],
                weights_bytes=pred['weights_bytes'], vram=vram, quality=qual, binding=binding,
                unconstrained_us=suf_cost[0], unmeasured=pred['unmeasured'])


def check_request(inv, request):
    """validate a requested assignment {tensor: card format}: every name exists, every (op, format) pair is in the
    module's k3.MODULE_CAPS, and no fusion group is split (one launch reads its tensors together, so they share one
    format). Raises PlanError with the reason; returns True."""
    recs, module = inv['recs'], inv['module']
    for name, fmt in request.items():
        if name not in recs:
            raise PlanError(f'{name}: no such tensor in the file\'s decode inventory')
        r = recs[name]
        fused = bool(r['group']) and fmt in k3.OWN_1RT_FMT
        menu = [r['fmt']] if r['stage'] == 'glue' else tensor_menu(module, r['t'], fused=fused)
        if fmt not in menu:
            p = tape_pair(module, r['t'], fmt, fused=fused)
            raise PlanError(f'unsupported pair for {name}: ({p[0]}, {p[1]}) is not in k3.MODULE_CAPS[{module}] '
                            f'(the menu for it: {menu})' if p else
                            f'unsupported format {fmt} for {name}: the tape has no op for it on {module} (menu: {menu})')
    for gid, members in inv['groups'].items():
        fmts = {request.get(n, recs[n]['fmt']) for n in members}
        if len(fmts) > 1:
            raise PlanError(f'fusion group {gid} {members} is read by one launch and shares one format '
                            f'(q35_file.gu_fused, engine/q35_dense.py _build_tape); the request splits it: '
                            f'{ {n: request.get(n, recs[n]["fmt"]) for n in members} }')
    return True
