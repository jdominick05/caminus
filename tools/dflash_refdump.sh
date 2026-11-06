#!/usr/bin/env bash
# tools/dflash_refdump.sh TEXT N DUMP(0|1) (SUFFIX=-N128: output dir suffix; DFLASH_DUMP: the binary, default build/dflash_dump): dflash_dump (speculative-simple + per-pass log + cb_eval dumps of the drafter's inject / draft
# graphs) on the Coder with the DFlash2 Q8_0 drafter, 3 drafts, greedy, the e2e prompt (200 ids). Run under guard.sh.
set -u
T=$1; N=$2; DUMP=$3
A=${ARTIFACTS:?set ARTIFACTS (CLAUDE.md, Models and data)}; M=${CODER:?set CODER}; DF=${DFLASH:?set DFLASH}
O=$A/drafters/df-refdump/$T-n3${SUFFIX:-}; mkdir -p $O
export GGML_CUDA_MMVQ_MAX_BATCH=8 SPEC_TOKFILE=$A/refs/d281-short/e2e_${T}_decode/tokens.txt SPEC_NTOK=200 SPEC_LOG=$O/passes.jsonl
[ $DUMP = 1 ] && export DF_DUMP_DIR=$O DF_DUMP_DFT=3
head -c 100000 $SPEC_TOKFILE | tr ' ' '\n' | head -200 | tr '\n' ' ' > $O/prompt_ids.txt
${DFLASH_DUMP:-$(dirname "$0")/../build/dflash_dump} -m $M -lm dio -dev ROCm0,ROCm1 -sm layer -ts 550,450 -ngl 99 --fit off -fa on -ctk q8_0 -ctv q8_0 -c 4096 -b 2048 -ub 256 -t 8 \
  --spec-type draft-dflash -md $DF -devd ROCm1 -ngld 99 --spec-draft-n-max 3 \
  -n $N --ignore-eos --temp 0 --top-k 1 --seed 1 -p x
