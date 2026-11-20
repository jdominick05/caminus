// The qwen35moe decode module on Caminus's own code (Qwen3.6-35B-A3B), built without llama.cpp's headers: the K3 core
// with only this line's decode tape's ops (kernels/k3_q35_ops.h), no ggml include path. It is the dense qwen35 line's
// trunk with the MoE feed-forward:
//   the trunk       GDN (conv / l2 / gates / step / save / norm) and gated attention, the same ops and statements as
//                   kernels/k3_q35_swift.hip, with this model's geometry: 8 query heads per kv head (ATTN_G 8, one V
//                   pass of 8; the dense line is 6 and 6) and head dim 256
//   the MoE FFN     the 30B's OP_ROUTER (softmax over the logits, top-8, renormalised: kernels/k3_q3m_ops.h), the
//                   experts' gate / up and down through OP_OWN_MV's grouped one-round-trip form (OWN_GROUPED,
//                   OWN_X_PER_SLOT, OWN_GU) on the own Q2_0 tile (kernels/own_quant_q2_0.h, K3_OWN_Q2_0), and the
//                   k3_own_ops.h OP_OWN_ADD_RMS16, whose weighted slot sum takes the eight routed experts and the gated
//                   shared expert as a 9th (OWN_ADD_MAX_S: the dense line's any-length form has no slots)
// What it does NOT have, deliberately: no ggml format tiles, no Q8 KV (that is #81's object, as for the dense line), no
// verify (T-token) forms and no drafter ops. The host refuses a tape outside its set (caminus.k3.Q35MOE_OPS and
// MODULE_CAPS).
//
// Compile only for now: this object exists to answer the register question before the engine is written. One k3_run body
// holds the trunk's ~18 ops plus OP_ROUTER and the 9-slot add, against the 30B lean module's 7; an op inlined into
// k3_run adds live scalars, and past the SGPRs every segment hop of every wave pays the reloads (qwen3moe: 190 spilled
// scalars -> 79, XT 5.33 -> 4.78 ms per token at the same grid). Budget: 96 VGPRs, 0 spills (tools/gates/objcheck.py).
#define K3_OWN
#define K3_OWN_Q3M
#define K3_OWN_Q2_0
#define K3_Q3M_LEAN
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#define K3_MAX_VGPR 96
#define ATTN_G 8
#define ATTN_VH 8
#define OWN_ADD_MAX_S 9
#define K3_Q35_MOE
#define K3_Q35_ATTN_SPLIT   // decode attention as the split pair (kernels/k3_attn_split.h), as the dense line's modules
// The two pins the other modules found for this budget (measured on this object, compile only):
//   K3_Q3M_ADD_RMS_PIN    OP_OWN_ADD_RMS16 derives its row length and slot count itself instead of leaving values made
//                         from them live across the segment loop (k3_q3m_lean_kvq8: 54 -> 31 spilled scalars)
//   K3_Q35_GDN_LOCAL_BASE keeps the GDN replay-state base local to GDN instead of live through the dispatch
// SGPR spills: 117 with neither, 94 / 103 with one, 80 with both (k3_q35_swift 62, k3_q3m_lean 53). VGPRs 94 -> 88, no
// VGPR spills either way.
#define K3_Q3M_ADD_RMS_PIN
#define K3_Q35_GDN_LOCAL_BASE
#define K3_EXTRA_OPS_HEADER "k3_q35_ops.h"
#include "k3.hip"
