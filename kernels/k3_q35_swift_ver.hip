// kernels/k3_q35_swift.hip as the verify module (speculative decoding: engine/q35_dense_verify.py): the same ops with their
// T-token forms (K3_Q3M_VER: kernels/own_quant_mt.h for the one-round-trip matvecs, the rows tile per token in
// OP_OWN_MV16, one task per token in the embedding and the add + norm; the Gated DeltaNet, RoPE, KV and attention ops read
// the pass's tokens already). A module of its own: k3_q35_swift's code object is unchanged. Each token's sums are the
// decode's statements, so a pass of T tokens gives T decode steps' bits.
#define K3_Q3M_VER
#define K3_Q35_VER   // (its own T-token tiles: kernels/own_quant_mt.h's mt2 forms and the rows tiles' T forms)
#define K3_Q35_ATTN_PACK   // (the split pair's PART takes every pass row of a split in one task: kernels/k3_attn_split.h)
#include "k3_q35_swift.hip"
