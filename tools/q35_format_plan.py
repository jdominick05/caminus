"""Choose a quantization format per tensor of a dense qwen35 GGUF (the Qwen3.8-27B line) to minimize the PREDICTED decode
time per token on one card, under a VRAM cap and a quality-loss cap — instead of minimizing file size. CPU only: it reads
the GGUF header (caminus.ggufio), the menu from the decode module's k3.MODULE_CAPS pairs, the cost table and the
sensitivity table, and solves exactly (branch and bound; tests/tools/test_q35_format_plan.py checks it against brute
force). Every time it prints is a cost-model prediction, labelled as one; the measured yardsticks are printed separately.
Detail: docs/dev/q35-format-plan.md.

  python -I tools/q35_format_plan.py GGUF [--card XTX] [--depth 128] [--max-ctx 32768]
      [--vram-cap GiB] [--quality-cap 1e-2] [--costs FILE] [--sensitivity FILE]
      [--set TENSOR=FORMAT ...] [--module dec_q35_gsq] [--json OUT.json]

Defaults: the card's VRAM cap is its guard line less the display reserve (the cost table's cards), the cost table is
tools/q35_format_plan.costs.json, the sensitivity table is the SYNTHETIC example
tools/q35_format_plan.sensitivity.example.json (the real one needs the model and a GPU: a per-tensor arm of G-model,
docs/specs/spec-own-quant.md). --set checks a requested assignment instead of solving: an unsupported (op, format) pair,
a split fusion group or an infeasible budget is refused with the reason (exit 1). Exit 0 with the plan printed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))

from caminus import ggufio                                   # noqa: E402
from caminus.engine import q35_plan, q35_file, k3            # noqa: E402

DEFAULT_COSTS = os.path.join(HERE, 'q35_format_plan.costs.json')
DEFAULT_SENS = os.path.join(HERE, 'q35_format_plan.sensitivity.example.json')


def gib(n):
    return n / 2 ** 30


def breakdown_lines(pred, indent='  '):
    """the per-stage-kind breakdown of a prediction, most expensive first"""
    return [f'{indent}{kind:5s} {us / 1e3:9.3f} ms' for kind, us in sorted(pred['by_kind'].items(), key=lambda kv: -kv[1])]


def report(path, inv, res, cur_pred, args, costs, sens, card):
    """the printed plan: chosen formats, totals, the predicted ms/token with per-stage breakdown for the plan and for
    the file's current formats, the binding constraint, the unmeasured entries every number depends on"""
    L = []
    L.append(f'q35 format plan: {os.path.basename(path)}')
    L.append(f'  module {inv["module"]} (menu from k3.MODULE_CAPS[{inv["module"]}], {len(k3.MODULE_CAPS[inv["module"]])} pairs), '
             f'card {card}, depth {args.depth}, max_ctx {args.max_ctx}')
    L.append(f'  cost table {os.path.basename(args.costs)}, sensitivity {os.path.basename(args.sensitivity)}'
             + (' (SYNTHETIC — not measured)' if sens.get('synthetic') else ''))
    L.append('predicted decode time per token (cost model — a model, not a measurement):')
    L.append(f'  planned   {res["total_us"] / 1e3:9.3f} ms')
    L += breakdown_lines(res, '          ')
    L.append(f'  as-is     {cur_pred["total_us"] / 1e3:9.3f} ms   (the file\'s current formats, same model)')
    L += breakdown_lines(cur_pred, '          ')
    stages = sorted(res['stages'], key=lambda s: -s[3])
    L.append('  per-stage breakdown, planned, the 12 most expensive stages (us, predicted):')
    for label, kind, fmt, us, b, names in stages[:12]:
        L.append(f'    {label:24s} {kind:4s} {fmt:7s} {us:9.2f} us  {b / 2**20:8.3f} MiB read')
    L.append(f'    ... {len(stages) - 12} more stages, {sum(s[3] for s in stages[12:]):.2f} us in all')
    changed = {n: (inv['recs'][n]['fmt'], res['choice'][n]) for n in inv['recs'] if res['choice'][n] != inv['recs'][n]['fmt']}
    L.append(f'chosen formats: {len(changed)} of {len(inv["recs"])} tensors change')
    for n, (a, b) in sorted(changed.items())[:20]:
        L.append(f'  {n:34s} {a:7s} -> {b}')
    if len(changed) > 20:
        L.append(f'  ... {len(changed) - 20} more')
    L.append(f'total bytes on card (weights at the chosen formats): {gib(res["weights_bytes"]):.3f} GiB; '
             f'VRAM at max_ctx incl. KV / GDN states / attention records: {gib(res["vram"]):.3f} GiB of the cap {gib(args.vram_cap):.3f} GiB')
    L.append(f'quality loss (sensitivity table): {res["quality"]:.6g} of the cap {args.quality_cap:g}')
    L.append(f'binding constraint: {"; ".join(res["binding"])}')
    if res['unmeasured']:
        L.append('NOT plain measurements — every predicted number above depends on these cost entries '
                 f'({len(res["unmeasured"])} stage lookups, aggregated):')
        agg = {}
        for u in res['unmeasured']:
            k = (u['stage'], u['card'], u['format'], u['field'], u['note'], u['source'])
            agg[k] = agg.get(k, 0) + 1
        for (stage, card, fmt, field, note, src), cnt in sorted(agg.items()):
            L.append(f'  {stage}/{card}/{fmt} {field} x{cnt}: {(note or "")[:100]} [source: {src}]')
    else:
        L.append('every cost entry used is a plain published measurement')
    ref = costs.get('reference_tokens', {})
    for name, per_card in ref.items():
        if card in per_card:
            L.append(f'published whole-token yardstick (measured, not from this model): {name}, {card}: '
                     f'{per_card[card]["ms_per_token"]} ms/token [{per_card[card]["source"]}]')
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gguf')
    ap.add_argument('--card', default='XTX', help='card label in the cost table (XTX / XT)')
    ap.add_argument('--depth', type=int, default=128, help='context depth the predicted token runs at (attention KV reads)')
    ap.add_argument('--max-ctx', type=int, default=32768, help='context the VRAM cap is checked at')
    ap.add_argument('--vram-cap', type=float, default=None, help='GiB the card may hold (default: guard line - reserve)')
    ap.add_argument('--quality-cap', type=float, default=1e-2, help='total quality-loss cap')
    ap.add_argument('--costs', default=DEFAULT_COSTS)
    ap.add_argument('--sensitivity', default=DEFAULT_SENS)
    ap.add_argument('--set', action='append', default=[], metavar='TENSOR=FORMAT', help='check this assignment instead of solving')
    ap.add_argument('--module', default=None, help='decode module (default: module_of(F))')
    ap.add_argument('--json', default=None, help='write the machine-readable plan here')
    args = ap.parse_args(argv)
    try:
        costs = q35_plan.load_costs(args.costs)
        if args.card not in costs['cards']:
            raise q35_plan.PlanError(f'card {args.card}: not in the cost table (cards: {sorted(costs["cards"])})')
        sens = q35_plan.load_sensitivity(args.sensitivity)
        if args.vram_cap is None:
            c = costs['cards'][args.card]
            args.vram_cap = (c['guard_line_mib'] - c['reserve_mib']) * 2 ** 20
        else:
            args.vram_cap = args.vram_cap * 2 ** 30
        F = ggufio.GGUFFile(args.gguf)
        try:
            if F.arch != 'qwen35':
                raise q35_plan.PlanError(f'{args.gguf}: architecture {F.arch!r}, not the dense qwen35 line')
            inv = q35_plan.inventory(F, module=args.module)
            cur = {n: r['fmt'] for n, r in inv['recs'].items()}
            cur_pred = q35_plan.predict(inv, cur, costs, args.card, args.depth)
            if args.set:
                request = {}
                for s in args.set:
                    if '=' not in s:
                        raise q35_plan.PlanError(f'--set {s!r}: expected TENSOR=FORMAT')
                    n, f = s.split('=', 1)
                    request[n] = f
                q35_plan.check_request(inv, request)      # (refuses: unsupported pair, split fusion group)
                choice = {**cur, **request}
                pred = q35_plan.predict(inv, choice, costs, args.card, args.depth)
                res = dict(choice=choice, total_us=pred['total_us'], by_kind=pred['by_kind'], stages=pred['stages'],
                           weights_bytes=pred['weights_bytes'], vram=q35_plan.vram_bytes(inv, F, choice, args.max_ctx),
                           quality=sum(q35_plan.loss_of(sens, n, choice[n]) for n in inv['recs']),
                           binding=['the requested assignment (checked, not solved)'], unmeasured=pred['unmeasured'])
                if res['vram'] > args.vram_cap:
                    raise q35_plan.PlanError(f'infeasible VRAM cap {args.vram_cap} B: the requested assignment needs '
                                            f'{res["vram"]:.0f} B at max_ctx {args.max_ctx}')
                if res['quality'] > args.quality_cap:
                    raise q35_plan.PlanError(f'infeasible quality-loss cap {args.quality_cap}: the requested assignment '
                                            f'loses {res["quality"]:.6g}')
            else:
                res = q35_plan.solve(inv, F, costs, sens, args.card, args.depth, args.max_ctx,
                                     args.vram_cap, args.quality_cap)
        finally:
            F.close()
    except q35_plan.PlanError as e:
        print(f'REFUSED: {e}')
        print('RESULT FAIL')
        return 1
    for line in report(args.gguf, inv, res, cur_pred, args, costs, sens, args.card):
        print(line)
    if args.json:
        out = dict(gguf=args.gguf, module=inv['module'], card=args.card, depth=args.depth, max_ctx=args.max_ctx,
                   planned_ms_per_token=res['total_us'] / 1e3, as_is_ms_per_token=cur_pred['total_us'] / 1e3,
                   choice=res['choice'], current={n: inv['recs'][n]['fmt'] for n in inv['recs']},
                   weights_bytes=res['weights_bytes'], vram_bytes=res['vram'], vram_cap=args.vram_cap,
                   quality=res['quality'], quality_cap=args.quality_cap, binding=res['binding'],
                   synthetic_sensitivity=bool(sens.get('synthetic')),
                   stages=[dict(label=s[0], kind=s[1], fmt=s[2], us=s[3], bytes=s[4], tensors=s[5]) for s in res['stages']],
                   unmeasured=res['unmeasured'])
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
            f.write('\n')
        print(f'json written to {args.json}')
    print('RESULT PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
