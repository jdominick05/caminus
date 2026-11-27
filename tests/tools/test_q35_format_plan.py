"""CPU tests of the per-tensor quantization planner (python/caminus/engine/q35_plan.py, used by
tools/q35_format_plan.py; docs/dev/q35-format-plan.md): synthetic qwen35 GGUF headers built with tests/tiny_gguf.py (no
model, no card), the solver checked against brute force on cases small enough to enumerate (320 random
instances, binding and non-binding caps, with ties), the
refusals (an unsupported (op, format) pair, a split fusion group, an infeasible budget), the cost table's sources and
unmeasured flags, and the tool's CLI end to end.

  python -I tests/tools/test_q35_format_plan.py --scratch DIR      RESULT PASS
"""
import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import tiny_gguf                                    # noqa: E402
from caminus import ggufio                           # noqa: E402
from caminus.engine import k3, q35_file, q35_plan    # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


# a tiny dense qwen35 header: one Gated DeltaNet block (block_count 1, full_attention_interval 4 -> block 0 is GDN)
D, FFN, S, NK, NV, HD, NH, NKV, VOCAB = 256, 512, 16, 2, 4, 64, 4, 2, 1024
CH = 2 * S * NK + S * NV


def meta(blocks=1, nextn=0):
    return {'general.architecture': 'qwen35', 'qwen35.block_count': blocks, 'qwen35.nextn_predict_layers': nextn,
            'qwen35.embedding_length': D, 'qwen35.feed_forward_length': FFN,
            'qwen35.attention.head_count': NH, 'qwen35.attention.head_count_kv': NKV,
            'qwen35.attention.key_length': HD, 'qwen35.rope.dimension_count': 32,
            'qwen35.ssm.state_size': S, 'qwen35.ssm.group_count': NK, 'qwen35.ssm.time_step_rank': NV,
            'qwen35.ssm.conv_kernel': 4, 'qwen35.ssm.inner_size': NV * S,
            'qwen35.full_attention_interval': 4, 'qwen35.context_length': 4096,
            'qwen35.attention.layer_norm_rms_epsilon': 1e-6, 'qwen35.rope.freq_base': 10000.0}


def mat(t, name, K, R, fmt):
    tid = [k for k, v in ggufio.TYPES.items() if v[0] == fmt][0]
    bs, ts = ggufio.TYPES[tid][1:]
    assert K * R % bs == 0
    t[name] = ((K, R), b'\x01' * (K * R // bs * ts), tid)


def vec(t, name, n):
    t[name] = ((n,), b'\x00' * (4 * n), 0)


def write_file(path, fmt_mat, blocks=1, nextn=0, fmt_down=None, fmt_emb='Q4_K'):
    """a tiny qwen35 GGUF whose trunk matrices are fmt_mat (all GDN blocks when blocks small)"""
    tensors = {}
    for il in range(blocks + nextn):
        p = f'blk.{il}.'
        gdn = (il % 4) != 3
        if gdn:
            mat(tensors, p + 'attn_qkv.weight', D, CH, fmt_mat)
            mat(tensors, p + 'attn_gate.weight', D, NV * S, fmt_mat)
            mat(tensors, p + 'ssm_out.weight', NV * S, D, fmt_mat)
            mat(tensors, p + 'ssm_alpha.weight', D, NV, 'F32')
            mat(tensors, p + 'ssm_beta.weight', D, NV, 'F32')
            vec(tensors, p + 'ssm_norm.weight', S); vec(tensors, p + 'ssm_dt.bias', NV)
            vec(tensors, p + 'ssm_a', NV); vec(tensors, p + 'ssm_conv1d.weight', CH * 4)
        else:
            mat(tensors, p + 'attn_q.weight', D, NH * 2 * HD, fmt_mat)
            mat(tensors, p + 'attn_k.weight', D, NKV * HD, fmt_mat)
            mat(tensors, p + 'attn_v.weight', D, NKV * HD, fmt_mat)
            mat(tensors, p + 'attn_output.weight', NH * HD, D, fmt_mat)
            vec(tensors, p + 'attn_q_norm.weight', HD); vec(tensors, p + 'attn_k_norm.weight', HD)
        mat(tensors, p + 'ffn_gate.weight', D, FFN, fmt_mat)
        mat(tensors, p + 'ffn_up.weight', D, FFN, fmt_mat)
        mat(tensors, p + 'ffn_down.weight', FFN, D, fmt_down or fmt_mat)
        vec(tensors, p + 'attn_norm.weight', D); vec(tensors, p + 'post_attention_norm.weight', D)
    mat(tensors, 'output.weight', D, VOCAB, 'Q4_K')
    mat(tensors, 'token_embd.weight', D, VOCAB, fmt_emb)
    vec(tensors, 'output_norm.weight', D)
    tiny_gguf.write_gguf(path, meta(blocks, nextn), tensors)


def test_costs(path):
    """a small synthetic cost table: per-byte differs per format, fixed per stage kind"""
    tab = {"schema": q35_plan.SCHEMA_COSTS,
           "cards": {"XTX": {"read_Bps": 858e9, "source": "docs/dev/qwen38-27b.md:32", "measured": True}},
           "entries": [
               {"stage": "mv", "card": "*", "format": "*", "shape": "*", "fixed_us": 5.0, "measured": True,
                "source": "docs/dev/qwen3-30b-a3b.md#where-the-remaining-time-is"},
               {"stage": "mv", "card": "XTX", "format": "Q8_0", "shape": "*", "per_byte_us": 1.0 / 700e9,
                "measured": "total-only", "source": "synthetic test table"},
               {"stage": "mv", "card": "XTX", "format": "Q4_K", "shape": "*", "per_byte_us": 1.0 / 800e9,
                "measured": "total-only", "source": "synthetic test table"},
               {"stage": "mv", "card": "XTX", "format": "Q6_K", "shape": "*", "per_byte_us": 1.0 / 850e9,
                "measured": "total-only", "source": "synthetic test table"},
               {"stage": "norm", "card": "*", "format": "*", "shape": "*", "fixed_us": 3.6, "measured": "other-model",
                "source": "docs/dev/qwen3-30b-a3b.md#where-the-remaining-time-is"},
               {"stage": "attn", "card": "*", "format": "*", "shape": "*", "fixed_us": 11.9, "measured": "other-model",
                "source": "docs/dev/qwen3-30b-a3b.md#where-the-remaining-time-is"},
               {"stage": "glue", "card": "*", "format": "*", "shape": "*", "fixed_us": 1.0, "measured": False,
                "source": "synthetic test table"},
               {"stage": "emb", "card": "*", "format": "*", "shape": "*", "fixed_us": 2.0, "measured": False,
                "source": "synthetic test table"}]}
    with open(path, 'w') as f:
        json.dump(tab, f)
    return path


def test_sens(path, per_tensor=None):
    """a small synthetic sensitivity table: loss per format (F32 zero), optionally per tensor"""
    losses = {'F32': 0.0, 'Q4_K': 4.3e-4, 'Q5_K': 3.6e-4, 'Q6_K': 3.5e-4, 'Q8_0': 3.7e-4}
    t = {"schema": q35_plan.SCHEMA_SENS, "unit": "synthetic KLD", "filled_by": "test", "synthetic": True,
         "default": None, "tensors": {"*": losses}}
    if per_tensor:
        t["tensors"].update(per_tensor)
    with open(path, 'w') as f:
        json.dump(t, f)
    return path


def brute_force(inv, costs, sens, card, depth, vram_cap, quality_cap):
    """enumerate every assignment of every variable (small cases only): the min option-cost sum over the feasible
    ones, then the predicted total_us of that assignment; None when none is feasible"""
    variables = q35_plan._variables(inv, costs, sens, card, depth)
    best = None
    for combo in itertools.product(*[opts for _, opts in variables]):
        byts = sum(o['bytes'] for o in combo)
        qual = sum(o['quality'] for o in combo)
        if byts > vram_cap or qual > quality_cap:
            continue
        c = sum(o['cost_us'] for o in combo)
        if best is None or c < best[0] - 1e-12:
            best = (c, {k: v for o in combo for k, v in o['fmts'].items()})
    if best is None:
        return None
    choice = {n: r['fmt'] for n, r in inv['recs'].items()}
    choice.update(best[1])
    return (q35_plan.predict(inv, choice, costs, card, depth)['total_us'], choice)


def brute_force_pruned(variables, vram_cap, quality_cap):
    """exhaustive enumeration of every feasible assignment (small instances), with only cap-feasibility pruning:
    returns (min option-cost sum, how many assignments reach it within 1e-9, one such assignment) or None"""
    n = len(variables)
    suf_b = [0.0] * (n + 1); suf_q = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf_b[i] = suf_b[i + 1] + min(o['bytes'] for o in variables[i][1])
        suf_q[i] = suf_q[i + 1] + min(o['quality'] for o in variables[i][1])
    best = [None, 0, None]   # cost, count of optima, assignment
    assign = {}

    def rec(i, cost, b, q):
        if b > vram_cap or q > quality_cap or b + suf_b[i] > vram_cap or q + suf_q[i] > quality_cap:
            return
        if i == n:
            if best[0] is None or cost < best[0] - 1e-9:
                best[0], best[1], best[2] = cost, 1, dict(assign)
            elif cost <= best[0] + 1e-9:
                best[1] += 1
            return
        for o in variables[i][1]:
            assign.update(o['fmts'])
            rec(i + 1, cost + o['cost_us'], b + o['bytes'], q + o['quality'])
            for k in o['fmts']:
                del assign[k]

    rec(0, 0.0, 0.0, 0.0)
    return (best[0], best[1], best[2]) if best[0] is not None else None


def randomized_solver_checks(inv, F, scratch, n_instances=320, seed=0x51A7):
    """solver against brute force on n_instances random small instances of one fixture: a random cost table and a
    random sensitivity table per instance (values drawn from small sets, so ties are common), and random VRAM and
    quality caps — non-binding, binding, and infeasible. Returns the stats dict; the caller checks it."""
    import random
    rng = random.Random(seed)
    menu_fmts = sorted({f for name, r in inv['recs'].items() if r['stage'] in ('mv', 'emb')
                        for f in q35_plan.tensor_menu(inv['module'], r['t'], fused=bool(r['group']))})
    fixed_vals = [2.0, 5.0, 5.0, 9.0]                       # ties
    pb_vals = [1 / 900e9, 1 / 858e9, 1 / 858e9, 1 / 700e9]  # ties
    loss_vals = [0.0, 1e-4, 2e-4, 2e-4, 5e-4]              # ties
    modes_seen = {}
    stats = dict(instances=0, ties=0, infeasible=0, feasible=0, mismatches=[])
    for it in range(n_instances):
        entries = [{"stage": "mv", "card": "*", "format": "*", "shape": "*",
                    "fixed_us": rng.choice(fixed_vals), "measured": True, "source": "randomized test table"}]
        for f in menu_fmts:      # a per-format per-byte override on a random half of the menu
            if rng.random() < 0.5:
                entries.append({"stage": "mv", "card": "XTX", "format": f, "shape": "*",
                                "per_byte_us": rng.choice(pb_vals), "measured": "total-only",
                                "source": "randomized test table"})
        for kind in ('norm', 'attn', 'glue', 'emb'):
            entries.append({"stage": kind, "card": "*", "format": "*", "shape": "*",
                            "fixed_us": rng.choice(fixed_vals), "measured": False, "source": "randomized test table"})
        tab = {"schema": q35_plan.SCHEMA_COSTS,
               "cards": {"XTX": {"read_Bps": 858e9, "source": "docs/dev/qwen38-27b.md:32", "measured": True}},
               "entries": entries}
        cp = os.path.join(scratch, f'rcosts{it}.json')
        with open(cp, 'w') as f:
            json.dump(tab, f)
        costs = q35_plan.load_costs(cp)
        losses = {f: rng.choice(loss_vals) for f in menu_fmts}
        tensors = {"*": losses}
        if rng.random() < 0.3:   # a per-tensor override with its own (possibly tied) values
            name = rng.choice(sorted(inv['recs']))
            tensors[name] = {f: rng.choice(loss_vals) for f in menu_fmts}
        sp = os.path.join(scratch, f'rsens{it}.json')
        with open(sp, 'w') as f:
            json.dump({"schema": q35_plan.SCHEMA_SENS, "unit": "synthetic", "synthetic": True,
                       "default": None, "tensors": tensors}, f)
        sens = q35_plan.load_sensitivity(sp)
        variables = q35_plan._variables(inv, costs, sens, 'XTX', 128)
        min_b = sum(min(o['bytes'] for o in opts) for _, opts in variables)
        max_b = sum(max(o['bytes'] for o in opts) for _, opts in variables)
        min_q = sum(min(o['quality'] for o in opts) for _, opts in variables)
        max_q = sum(max(o['quality'] for o in opts) for _, opts in variables)
        vmode = rng.choice(['free', 'bind', 'inf'])
        qmode = rng.choice(['free', 'bind', 'inf'])
        vram_cap = {'free': 1e18, 'bind': min_b + (max_b - min_b) * rng.random(), 'inf': min_b - 1.0}[vmode]
        qcap = {'free': 1e18, 'bind': min_q + (max_q - min_q) * rng.random(), 'inf': min_q - 1e-12}[qmode]
        modes_seen[(vmode, qmode)] = modes_seen.get((vmode, qmode), 0) + 1
        bf = brute_force_pruned(variables, vram_cap, qcap)
        try:
            res = q35_plan.solve(inv, F, costs, sens, 'XTX', 128, 512, vram_cap, qcap)
            got = ('ok', res)
        except q35_plan.PlanError:
            got = ('refused', None)
        stats['instances'] += 1
        if bf is None:
            stats['infeasible'] += 1
            if got[0] != 'refused':
                stats['mismatches'].append((it, vmode, qmode, 'brute force infeasible, solver returned a plan'))
            continue
        stats['feasible'] += 1
        if bf[1] > 1:
            stats['ties'] += 1
        if got[0] != 'ok':
            stats['mismatches'].append((it, vmode, qmode, 'brute force feasible, solver refused'))
            continue
        bf_choice = {n_: r['fmt'] for n_, r in inv['recs'].items()}
        bf_choice.update(bf[2])
        bf_total = q35_plan.predict(inv, bf_choice, costs, 'XTX', 128)['total_us']
        if abs(got[1]['total_us'] - bf_total) > 1e-9 or got[1]['vram'] > vram_cap + 1e-6 or got[1]['quality'] > qcap + 1e-15:
            stats['mismatches'].append((it, vmode, qmode, f'solver {got[1]["total_us"]:.9f} vs brute {bf_total:.9f}, '
                                        f'vram {got[1]["vram"]:.0f}/{vram_cap:.0f}, q {got[1]["quality"]:.6g}/{qcap:.6g}'))
    stats['modes_seen'] = modes_seen
    return stats



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    S = a.scratch
    os.makedirs(S, exist_ok=True)
    gsq = os.path.join(S, 'gsq.gguf')       # a trunk matrix in Q2_K: the GSQ module
    swift = os.path.join(S, 'swift.gguf')   # all Q4_K: the Swift module
    write_file(gsq, 'IQ3_S', fmt_down='Q2_K')
    write_file(swift, 'Q4_K')
    costs_p = test_costs(os.path.join(S, 'costs.json'))
    sens_p = test_sens(os.path.join(S, 'sens.json'))
    costs = q35_plan.load_costs(costs_p)
    sens = q35_plan.load_sensitivity(sens_p)

    # -- the inventory: module, roles, fusion groups, menu from MODULE_CAPS -------------------------------
    F = ggufio.GGUFFile(gsq)
    inv = q35_plan.inventory(F)
    check(inv['module'] == 'dec_q35_gsq' and q35_file.module_of(F) == 'dec_q35_gsq',
          'inventory: a Q2_K trunk matrix picks the GSQ module (module_of)')
    check(inv['groups'] == {} and 'blk.0.ffn_gate.weight' in inv['recs'],
          'inventory: no fusion groups on the GSQ module (gu off), ffn tensors present')
    check(all(inv['recs'][n]['fmt'] in q35_plan.tensor_menu(inv['module'], inv['recs'][n]['t'])
              for n in inv['recs'] if inv['recs'][n]['stage'] != 'glue'),
          'inventory: every current format is inside the module\'s MODULE_CAPS menu')
    F.close()
    F = ggufio.GGUFFile(swift)
    invs = q35_plan.inventory(F)
    check(invs['module'] == 'dec_q35_swift', 'inventory: an all-Q4_K file picks the Swift module')
    check(invs['groups'] == {'ffn|0': ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight']},
          'inventory: the gate | up fusion group on the Swift module (q35_dense.py _build_tape)')
    check(q35_plan.tensor_menu('dec_q35_swift', invs['recs']['blk.0.ffn_gate.weight']['t']) ==
          ['F32', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0'], 'menu: a matrix on dec_q35_swift, from k3.MODULE_CAPS')
    check(q35_plan.tensor_menu('dec_q35_swift', invs['recs']['blk.0.attn_norm.weight']['t']) == [],
          'menu: a 1-D norm tensor has no op of its own (its menu is its own format only)')
    check(len(q35_plan.stages_of(invs, {})) > 0 and
          any(s[0] == 'blk.0.ffn_gu' for s in q35_plan.stages_of(invs, {})),
          'stages: equal Q4_K gate | up run as one fused ffn_gu stage')
    F.close()

    # -- gu_fused matches the engine's fusion test (q35_dense.py _build_tape) ------------------------------
    check(q35_file.gu_fused('dec_q35_swift', 'Q4_K', 'Q4_K') and
          q35_file.gu_fused('dec_q35_swift', 'Q6_K', 'Q6_K') and
          not q35_file.gu_fused('dec_q35_swift', 'Q4_K', 'Q6_K') and
          not q35_file.gu_fused('dec_q35_swift', 'Q8_0', 'Q8_0') and
          not q35_file.gu_fused('dec_q35_gsq', 'Q4_K', 'Q4_K') and
          not q35_file.gu_fused('dec_q35_swift', 'Q4_K', 'Q4_K', rpt=8),
          'gu_fused: the tape\'s fusion predicate (Swift + equal one-round-trip format + rpt 16 only)')

    # -- upload_bytes / the projection terms ---------------------------------------------------------------
    F = ggufio.GGUFFile(gsq)
    terms = q35_file._projection_terms(F, 512)
    check(q35_file._projection_bytes(F, 512) ==
          terms['weights'] + terms['kv'] + terms['gdn'] + terms['part'] + terms['extra'],
          'projection: _projection_bytes equals its terms (q35_file, moved from q35_dense)')
    check(terms['weights'] == sum(q35_file.upload_bytes(t) for t in F.tensors
                                  if not t.name.startswith('blk.1.')),  # blocks 1.. are the nextn block here? no: 1 block
          'projection: the weights term equals the sum of upload_bytes per tensor')
    F.close()

    # -- prediction is the stated model: fixed + per-byte --------------------------------------------------
    F = ggufio.GGUFFile(swift)
    invs = q35_plan.inventory(F)
    cur = {n: r['fmt'] for n, r in invs['recs'].items()}
    pred = q35_plan.predict(invs, cur, costs, 'XTX', 128)
    mv = [s for s in pred['stages'] if s[1] == 'mv']
    ok = all(abs(s[3] - (5.0 + s[4] / 858e9 * 1e6)) < 1e-9 for s in mv if s[2] not in ('Q8_0', 'Q4_K', 'Q6_K'))
    check(ok, 'predict: a wildcard mv stage costs fixed 5 us + bytes / the card read rate')
    q8 = [s for s in mv if s[2] == 'Q8_0']
    check(all(abs(s[3] - (5.0 + s[4] / 700e9 * 1e6)) < 1e-9 for s in q8),
          'predict: a format-specific entry overrides the per-byte part')
    F.close()

    # -- the solver against brute force --------------------------------------------------------------------
    F = ggufio.GGUFFile(swift)
    invs = q35_plan.inventory(F)
    vars_ = q35_plan._variables(invs, costs, sens, 'XTX', 128)
    n_opts = 1
    for _, opts in vars_:
        n_opts *= len(opts)
    check(len(vars_) <= 12 and n_opts <= 1000000, f'solver test setup: small enough to enumerate ({len(vars_)} variables, {n_opts} assignments)')
    for vram_cap, qcap in [(10 ** 12, 1.0), (10 ** 12, 1.2e-3), (30 * 2 ** 20, 1.0), (24 * 2 ** 20, 1.5e-3)]:
        res = q35_plan.solve(invs, F, costs, sens, 'XTX', 128, 512, vram_cap, qcap)
        bf = brute_force(invs, costs, sens, 'XTX', 128, vram_cap, qcap)
        check(bf is not None and abs(res['total_us'] - bf[0]) < 1e-9,
              f'solver == brute force (vram cap {vram_cap}, quality cap {qcap}): {res["total_us"]:.6f} vs {bf[0] if bf else None:.6f}')
        check(res['vram'] <= vram_cap and res['quality'] <= qcap + 1e-12,
              f'solver: the solution satisfies both caps (vram {res["vram"]:.0f} <= {vram_cap}, quality {res["quality"]:.6g} <= {qcap})')
    st = randomized_solver_checks(invs, F, S, n_instances=320)
    check(st['instances'] == 320 and not st['mismatches'],
          f'randomized: solver == brute force on {st["instances"]} random instances ({st["feasible"]} feasible, '
          f'{st["infeasible"]} infeasible, {st["ties"]} with tied optima; mismatches: {st["mismatches"][:3]})')
    check(len(st['modes_seen']) == 9 and min(st['modes_seen'].values()) >= 5,
          f'randomized: all 9 cap modes (vram x quality, free / binding / infeasible) covered, >= 5 instances each: '
          f'{sorted(st["modes_seen"].items())}')
    check(st['ties'] > 0, 'randomized: instances with tied optima are present')
    F.close()

    # -- refusals ------------------------------------------------------------------------------------------
    F = ggufio.GGUFFile(swift)
    invs = q35_plan.inventory(F)
    try:
        q35_plan.check_request(invs, {'blk.0.attn_norm.weight': 'Q8_0'})
        check(False, 'refusal: an unsupported pair is refused')
    except q35_plan.PlanError as e:
        check('not in k3.MODULE_CAPS' in str(e) or 'no op for it' in str(e),
              f'refusal: an unsupported pair names the module caps ({e})')
    try:
        q35_plan.check_request(invs, {'blk.0.ffn_gate.weight': 'Q4_K', 'blk.0.ffn_up.weight': 'Q6_K'})
        check(False, 'refusal: a split fusion group is refused')
    except q35_plan.PlanError as e:
        check('shares one format' in str(e) and 'ffn|0' in str(e), f'refusal: a split fusion group ({e})')
    try:
        q35_plan.check_request(invs, {'blk.0.ffn_gate.weight': 'Q3_K'})
        check(False, 'refusal: a pair outside the module is refused')
    except q35_plan.PlanError as e:
        check('Q3_K' in str(e), 'refusal: Q3_K is not in dec_q35_swift\'s MODULE_CAPS')
    # an infeasible quality budget: every format in the menu loses more than the cap
    sens_bad = test_sens(os.path.join(S, 'sens_bad.json'))
    tab = json.load(open(sens_bad))
    tab['tensors']['*'] = {k: 1.0 for k in tab['tensors']['*']}
    json.dump(tab, open(sens_bad, 'w'))
    try:
        q35_plan.solve(invs, F, costs, q35_plan.load_sensitivity(sens_bad), 'XTX', 128, 512, 10 ** 12, 0.5)
        check(False, 'refusal: an infeasible quality budget is refused')
    except q35_plan.PlanError as e:
        check('quality-loss cap' in str(e), f'refusal: infeasible quality cap names the cap and the minimum ({e})')
    try:
        q35_plan.solve(invs, F, costs, sens, 'XTX', 128, 512, 1.0, 1.0)
        check(False, 'refusal: an infeasible VRAM budget is refused')
    except q35_plan.PlanError as e:
        check('VRAM cap' in str(e), f'refusal: infeasible VRAM cap names the cap and the minimum ({e})')
    F.close()

    # -- the default cost table: sources everywhere, unmeasured entries flagged ----------------------------
    dtab = q35_plan.load_costs(os.path.join(ROOT, 'tools', 'q35_format_plan.costs.json'))
    check(all(e.get('source') for e in dtab['entries']), 'default cost table: every entry cites its source')
    check(all(c.get('source') for c in dtab['cards'].values()), 'default cost table: every card cites its read rate source')
    check(any(e['fixed_us'] is None and e.get('measured') is False for e in dtab['entries']),
          'default cost table: unmeasured entries say so explicitly (fixed_us null, measured false)')
    F = ggufio.GGUFFile(gsq)
    inv = q35_plan.inventory(F)
    dsens = q35_plan.load_sensitivity(os.path.join(ROOT, 'tools', 'q35_format_plan.sensitivity.example.json'))
    check(dsens.get('synthetic') is True, 'default sensitivity table: marked synthetic')
    pred = q35_plan.predict(inv, {n: r['fmt'] for n, r in inv['recs'].items()}, dtab, 'XTX', 128)
    check(len(pred['unmeasured']) > 0, 'predict: results that depend on unmeasured entries are flagged')
    try:
        q35_plan.load_costs(os.path.join(S, 'sens.json'))
        check(False, 'cost table: a wrong schema is refused')
    except q35_plan.PlanError:
        check(True, 'cost table: a wrong schema is refused')
    try:
        bad = {'schema': q35_plan.SCHEMA_SENS, 'tensors': {}}
        p = os.path.join(S, 'sens_nodefault.json'); json.dump(bad, open(p, 'w'))
        q35_plan.loss_of(q35_plan.load_sensitivity(p), 'blk.0.ffn_gate.weight', 'Q8_0')
        check(False, 'sensitivity: an unknown loss is refused, never guessed')
    except q35_plan.PlanError:
        check(True, 'sensitivity: an unknown loss is refused, never guessed')
    F.close()

    # -- the tool end to end --------------------------------------------------------------------------------
    import importlib.util
    spec = importlib.util.spec_from_file_location('q35_format_plan_tool', os.path.join(ROOT, 'tools', 'q35_format_plan.py'))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = os.path.join(S, 'plan.json')
    rc = tool.main([gsq, '--card', 'XTX', '--depth', '128', '--max-ctx', '512', '--vram-cap', '100',
                    '--quality-cap', '1.0', '--json', out])
    check(rc == 0, 'tool: planning the GSQ fixture exits 0')
    plan = json.load(open(out))
    check('planned_ms_per_token' in plan and 'as_is_ms_per_token' in plan and plan['choice'] and
          isinstance(plan['stages'], list) and plan['binding'],
          'tool: the json has the planned and as-is predicted ms/token, the choice, the stages and the binding constraint')
    check(plan['synthetic_sensitivity'] is True, 'tool: a synthetic sensitivity table is flagged in the json')
    rc = tool.main([swift, '--set', 'blk.0.attn_norm.weight=Q8_0'])
    check(rc == 1, 'tool: --set with an unsupported pair exits 1 (REFUSED)')
    rc = tool.main([swift, '--set', 'blk.0.ffn_gate.weight=Q4_K', '--set', 'blk.0.ffn_up.weight=Q6_K'])
    check(rc == 1, 'tool: --set splitting a fusion group exits 1 (REFUSED)')
    rc = tool.main([gsq, '--vram-cap', '0.0000001'])
    check(rc == 1, 'tool: an infeasible VRAM cap exits 1 (REFUSED)')
    rc = tool.main([swift, '--set', 'blk.0.ffn_gate.weight=Q8_0', '--set', 'blk.0.ffn_up.weight=Q8_0'])
    check(rc == 0, 'tool: --set with a supported shared format on the fusion group exits 0')

    print(f'RESULT {"PASS" if not FAILS else "FAIL"}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
