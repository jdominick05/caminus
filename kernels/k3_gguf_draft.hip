// the DFlash 2 drafter's engine (python/caminus/dflash_k3.py; docs/specs/spec-drafters.md 1): the GSQ-RCO Coder's
// multi-token module with only the formats the drafter's tapes use (its Q8_0 layers / selector, the target's Q6_K head
// shards, the Q8_0 gate/up pair as OP_GG_GU_DENSE) and the drafter ops (kernels/k3_draft_ops.h: conv, top-k, selector walk,
// embed, the ring KV append, windowed / non-causal OP_ATTN_DEC, the permuted q8_1 of the inject's features). A separate code
// object: the trunk's k3_gguf* modules stay byte for byte the same
#define K3_DRAFT
#define K3_FMTS_MV GG_Q6_K, GG_Q8_0
#define K3_FMTS_MOE_GU
#define K3_FMTS_MOE_DOWN
#define K3_FMTS_GU_DENSE GG_Q8_0
#define K3_MOE_N 0
#define K3_MOE_P 0
#include "k3_gguf_gsq_mt.hip"
