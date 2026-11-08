#!/usr/bin/env bash
# tools/gates/gates.sh OUT STAGE...: the bitwise and D-281 gates against the stored current references
# ($ARTIFACTS/refs/MANIFEST.md: what each reference is, where it came from, which tree produced it).
# OUT: a fresh directory for this run's logs and dumps (never a reference directory; refdump.py refuses to overwrite).
# Every GPU command runs under tools/guard.sh; a guard exit 98 / 99 (GPU fault) stops everything at once with that code.
# Before a stage compares, the references it reads are checked against $ARTIFACTS/refs/MANIFEST.sha256.
# Stages, run in the order given (wall times measured on the two cards, 11-03; `all` took ~14 min):
#   trunk           Coder refdump, FlashNextTP defaults (max_ctx 2048: the dense trunk) == sp_g                       (~20 s)
#   grids           the race check: K3_PER_WGP 3 / 6 / 7 == sp_g_w3 / _w6 / _w7                                       (~1 min)
#   qsa             the QSA engine at max_ctx 4096, tapes dual and always, grids default / 3 / 6 / 7 == sp_dual* /
#                   sp_always* (all equal sp_g: at short positions QSA is the trunk)                                   (~2.5 min)
#   reap            REAP-308 refdump (the regression reference) == sp_reap_g                                           (~20 s)
#   multi           T=4 == one-token: trunk 24 and 200 positions, QSA dual and always 200 positions                    (~2.5 min)
#   forcebail       Coder forced bails, T=1 and T=4, 40 passes each: every pass bitwise the uninterrupted run          (~45 s)
#   forcebail-reap  the same on REAP-308                                                                                (~45 s)
#   short           D-281 at short context: tests/engine/test_flashnext_e2e.py on refs/d281-short (768 positions vs llama.cpp
#                   one-token decode, floor 0.0076, top-1 >= 95.9 %), and every result line == the stored log           (~45 s)
#   lc              long-context D-281 at 1792 + 8192 (tools/eval/lc_baseline.py --qsa) == the stored k3_qsa.json            (~1.5 min)
#   lc32            the same at 32768                                                                                    (~3.5 min)
#   all             trunk grids qsa reap multi forcebail short lc lc32
# Env: ARTIFACTS, CODER (both required), REAP (the reap stages), PY (a python with ROCm PyTorch; CLAUDE.md, Models and data).
# Ends with a SUMMARY (one line per check) and exits 0 only when every check passed.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ARTIFACTS=${ARTIFACTS:?set ARTIFACTS (CLAUDE.md, Models and data)}
CODER=${CODER:?set CODER (CLAUDE.md, Models and data)}
REAP=${REAP:-'$REAP'}
PY=${PY:-$HOME/miniforge3/envs/drafter/bin/python}
export ARTIFACTS CODER REAP
G=$REPO/tools/guard.sh; T=$REPO/tools/gates
REFS=$ARTIFACTS/refs; SHA=$REFS/MANIFEST.sha256
RC=$ARTIFACTS/m1/s3/ref; RR=$ARTIFACTS/m1/s6/verify/ref; LC=$ARTIFACTS/qsa/lc; CUR=$REFS/current
if [ $# -lt 2 ]; then sed -n '2,22p' "$0"; exit 2; fi
OUT=$1; shift
mkdir -p "$OUT/dumps" || exit 2
SUM=(); NFAIL=0
note() { SUM+=("$1 $2"); [ "$1" = PASS ] || NFAIL=$((NFAIL + 1)); echo "== $1 $2"; }
stop() { echo "GUARD EXIT $1: GPU fault recorded or refused -- STOP ALL GPU WORK, re-run nothing, report to the lead"
         printf '%s\n' "${SUM[@]}"; exit "$1"; }
g() {   # g LOG SECS CMD...: one guarded GPU run; stops on 98 / 99; returns the run's exit code
  local log=$1 secs=$2; shift 2
  "$G" "$log" "$secs" "$@"; local rc=$?
  echo "   $(basename "$log") rc=$rc: $(grep -E 'RESULT|PASS|FAIL|saved |positions bitwise|SUMMARY|^D=|Traceback|Error|GUARD|GPU FAULT' "$log" | tail -3 | tr '\n' ' ')"
  { [ $rc -eq 98 ] || [ $rc -eq 99 ]; } && stop $rc
  return $rc; }
refs_ok() {   # refs_ok PATTERN WHAT: the MANIFEST.sha256 lines whose path matches PATTERN all verify
  local n; n=$(grep -cE "$1" "$SHA")
  if [ "$n" -gt 0 ] && grep -E "$1" "$SHA" | sha256sum -c --quiet - > "$OUT/sha_$2.txt" 2>&1; then note PASS "refs $2: $n files match MANIFEST.sha256"
  else note FAIL "refs $2: MANIFEST.sha256 check ($n lines; $OUT/sha_$2.txt)"; return 1; fi; }
rd() {   # rd NAME REF LOG_SECS SAVE_ARGS...: save OUT/dumps/NAME.pt under the guard, then cmp against the stored REF (CPU)
  local name=$1 ref=$2 secs=$3; shift 3
  if g "$OUT/rd_$name.log" "$secs" "$PY" -I -B "$T/refdump.py" save "$OUT/dumps/$name.pt" "$@"; then
    if HIP_VISIBLE_DEVICES= "$PY" -I -B "$T/refdump.py" cmp "$OUT/dumps/$name.pt" "$ref" > "$OUT/cmp_$name.txt" 2>&1
    then note PASS "refdump $name == $(basename "$ref")"; else note FAIL "refdump $name vs $(basename "$ref"): $(tail -1 "$OUT/cmp_$name.txt")"; fi
  else note FAIL "refdump $name: save failed ($OUT/rd_$name.log)"; fi; }
chk() {   # chk NAME LOG SECS CMD...: a guarded run whose exit code is the verdict
  local name=$1 log=$2 secs=$3; shift 3
  if g "$log" "$secs" "$@"; then note PASS "$name"; else note FAIL "$name ($log)"; fi; }
lcdir() {   # a fresh lc_baseline OUT dir: the probe files and llama.cpp references by symlink (results land here, not in $LC)
  mkdir -p "$1"; for f in probes.json d1792 d8192 d32768 ref; do [ -e "$1/$f" ] || ln -s "$LC/$f" "$1/$f"; done; }
echo "gates.sh: repo $REPO at $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)$([ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || echo ' + uncommitted changes'); OUT $OUT"
echo "compaction_proactiveness $(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null); stages: $*"
ST=("$@"); [ "${ST[*]}" = all ] && ST=(trunk grids qsa reap multi forcebail short lc lc32)
cd "$REPO" || exit 2
for s in "${ST[@]}"; do echo "== stage $s ($(date +%T))"; case $s in
trunk) refs_ok '/m1/s3/ref/sp_g\.pt$' trunk && rd g "$RC/sp_g.pt" 900 --model "$CODER" ;;
grids) refs_ok '/m1/s3/ref/sp_g_w[367]\.pt$' grids && for w in 3 6 7; do rd g_w$w "$RC/sp_g_w$w.pt" 900 --model "$CODER" --per-wgp $w; done ;;
qsa)   refs_ok '/m1/s3/ref/sp_(dual|always)(_w[367])?\.pt$' qsa && for tape in dual always; do for w in "" 3 6 7; do
         sfx=${w:+_w$w}; rd $tape$sfx "$RC/sp_$tape$sfx.pt" 900 --model "$CODER" --max-ctx 4096 --tape $tape ${w:+--per-wgp $w}; done; done ;;
reap)  refs_ok '/m1/s6/verify/ref/sp_reap_g\.pt$' reap && rd reap_g "$RR/sp_reap_g.pt" 900 --model "$REAP" ;;
multi) chk "T=4 == one-token, trunk, 24 positions" "$OUT/multi4.log" 900 "$PY" -I -B "$T/tp_multi.py" 4 24 --model "$CODER"
       chk "T=4 == one-token, trunk, 200 positions" "$OUT/multi4_200.log" 900 "$PY" -I -B "$T/tp_multi.py" 4 200 --model "$CODER"
       for tape in dual always; do
         chk "T=4 == one-token, QSA $tape, 200 positions" "$OUT/qmulti4_$tape.log" 900 "$PY" -I -B "$T/tp_multi.py" 4 200 --model "$CODER" --max-ctx 4096 --tape $tape
       done ;;
forcebail) for t in 1 4; do chk "forced bails, Coder, T=$t, 40 passes" "$OUT/fb_t$t.log" 600 "$PY" -I -B "$T/forcebail.py" $t 40 --model "$CODER"; done ;;
forcebail-reap) for t in 1 4; do chk "forced bails, REAP-308, T=$t, 40 passes" "$OUT/fb_reap_t$t.log" 600 "$PY" -I -B "$T/forcebail.py" $t 40 --model "$REAP"; done ;;
short) refs_ok '/refs/d281-short/' d281-short || continue
       chk "D-281 short (768 positions vs llama.cpp)" "$OUT/e2e_tp.log" 1800 "$PY" -I -B tests/engine/test_flashnext_e2e.py \
           "$REFS/d281-short/e2e_prose_decode" "$REFS/d281-short/e2e_code_decode" "$REFS/d281-short/e2e_shell_decode" --model "$CODER" --tp
       E2E='^(e2e_|   engine top-1 flips|replica mismatches|ALL:|PASS|FAIL)'   # the result lines (not load times or stderr)
       if diff <(grep -E "$E2E" "$OUT/e2e_tp.log") <(grep -E "$E2E" "$CUR/e2e_tp.log") > "$OUT/e2e_tp.diff" 2>&1
       then note PASS "D-281 short: every result line == the stored log (refs/current/e2e_tp.log)"
       else note FAIL "D-281 short: lines differ from refs/current/e2e_tp.log ($OUT/e2e_tp.diff)"; fi ;;
lc|lc32) refs_ok '/qsa/lc/' lc-probes || continue
       if [ $s = lc ]; then dep=1792,8192; secs=2400; else dep=32768; secs=3000; fi
       lcdir "$OUT/$s"
       chk "long-context D-281 $dep (tools/eval/lc_baseline.py --qsa)" "$OUT/lc_run_$s.log" $secs "$PY" -I -B tools/eval/lc_baseline.py run "$OUT/$s" --qsa --depths $dep
       if HIP_VISIBLE_DEVICES= "$PY" -I -B "$T/lc_cmp.py" "$OUT/$s/k3_qsa.json" "$CUR/${s}_k3_qsa.json" > "$OUT/lc_cmp_$s.txt" 2>&1
       then note PASS "long-context $dep: every non-timing number == refs/current/${s}_k3_qsa.json"
       else note FAIL "long-context $dep vs refs/current/${s}_k3_qsa.json: $(tail -1 "$OUT/lc_cmp_$s.txt")"; fi ;;
*) note FAIL "unknown stage $s" ;;
esac; done
echo "SUMMARY ($(date +%T)): $(( ${#SUM[@]} - NFAIL )) passed, $NFAIL failed"
printf '%s\n' "${SUM[@]}"
[ $NFAIL -eq 0 ]
