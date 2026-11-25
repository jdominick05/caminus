// The qwen35moe decode module on Caminus's own code (Qwen3.6-35B-A3B), built without llama.cpp's headers: the K3 core
// with only this line's decode tape's ops (kernels/ops_q35.h), no ggml include path. It is the dense qwen35 line's
// trunk with the MoE feed-forward:
//   the trunk       GDN (conv / l2 / gates / step / save / norm) and gated attention, the same ops and statements as
//                   kernels/dec_q35_swift.hip, with this model's geometry: 8 query heads per kv head (ATTN_G 8, one V
//                   pass of 8; the dense line is 6 and 6) and head dim 256
//   the MoE FFN     the 30B's OP_ROUTER (softmax over the logits, top-8, renormalised: kernels/ops_q3m.h), the
//                   experts' gate / up and down through OP_OWN_MV's grouped one-round-trip form (OWN_GROUPED,
//                   OWN_X_PER_SLOT, OWN_GU) on the own Q2_0 tile (kernels/quant_q2_0.h, K3_OWN_Q2_0), and the
//                   ops_own.h OP_OWN_ADD_RMS16, whose weighted slot sum takes the eight routed experts and the gated
//                   shared expert as a 9th (OWN_ADD_MAX_S: the dense line's any-length form has no slots)
// What it does NOT have, deliberately: no ggml format tiles, no verify (T-token) forms and no drafter ops. The Q8 KV
// cache is its own object (kernels/objects.tsv dec_q35m_kvq8, #define K3_Q35_KV_Q8), as on the dense line. The host refuses a
// tape outside its set (caminus.k3.Q35MOE_OPS and MODULE_CAPS).
//
// One k3_run body holds the trunk's ~18 ops plus OP_ROUTER and the 9-slot add, against the 30B lean module's 7; an op
// inlined into k3_run adds live scalars, and past the SGPRs every segment hop of every wave pays the reloads (qwen3moe:
// 190 spilled scalars -> 79, XT 5.33 -> 4.78 ms per token at the same grid). Budget: 96 VGPRs, 0 spills
// (tools/gates/objcheck.py).
#define K3_OWN
#define K3_OWN_Q3M
#define K3_OWN_Q2_0
#define K3_Q3M_LEAN
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#ifndef K3_PER_WGP           // the XT's grid, 5 x 42 WGPs: VGPR cap 144 (kernels/dec_core.hip); build.sh adds a _w4 object
#define K3_PER_WGP 5        // (-DK3_PER_WGP=4, cap 192) for the XTX's 4 x 48
#endif
#define ATTN_G 8
#define ATTN_VH 8
#define ATTN_SS 128          // list positions per split: 32 -> 64 -> 128 measured at 49,901 depth, 9.252 -> 8.730 ->
                             // 8.362 ms (-10.6 %), and 5.644 -> 5.561 at short context; 128 is the split header's max
#define ATTN_QL              // OP_ATTN_PART (kernels/ops_attn_split.h): the query heads in LDS, 4 V values per load round:
#ifndef ATTN_COMB_U
#define ATTN_COMB_U 12       // OP_ATTN_COMB: split records loaded ahead per round (the same additions in the same order);
#endif                       // the Q8 build 16 (dec_q35m_kvq8). F16: 16 spills 3 more SGPRs (short +0.7 %)
#define ATTN_VU 4            // the same bits; per 128-cell task 86 -> 61 (VU) -> 50 us (QL) at 32K (XT, Q8); VU 8 spills
#define OWN_ADD_MAX_S 9
#define OWN_ADD_S 3           // the 9-slot add's slots per load round trip (kernels/ops_own.h): 1 -> 3 measured (XT,
                             // q35_gmodel --speed) 6.27 -> 6.20 ms per token, the same bits, +1 spilled SGPR; 5: 6.25, +6 VGPRs
#define K3_Q35_MOE
#define K3_Q35_YARN          // (OP_QK_NR's YaRN form: Seg.aux = the per-dim factor table, Seg.work = mscale's bits)
#define K3_Q35_ATTN_SPLIT   // decode attention as the split pair (kernels/ops_attn_split.h), as the dense line's modules
#define K3_Q35_CONV_L2       // (ops_q35.h: the GDN conv with the q | k heads' L2 norm in one segment)
#define K3_Q35_MV16_ROW     // OP_OWN_MV16's Q8_0 at K = 1024 n as a wave per row (kernels/quant.h own_row_q8_0_w)
#define OWN_ROW_W_NT        // the Q8_0 row tile's weight loads nontemporal (quant.h): hit the prefetched lines, allocate none
// The two pins the other modules found for this budget (measured on this object, compile only):
//   K3_Q3M_ADD_RMS_PIN    OP_OWN_ADD_RMS16 derives its row length and slot count itself instead of leaving values made
//                         from them live across the segment loop (dec_q3m_kvq8: 54 -> 31 spilled scalars)
//   K3_Q35_GDN_LOCAL_BASE keeps the GDN replay-state base local to GDN instead of live through the dispatch
// SGPR spills: 117 with neither, 94 / 103 with one, 80 with both (dec_q35_swift 62, dec_q3m 53). VGPRs 94 -> 88, no
// VGPR spills either way.
#define K3_Q3M_ADD_RMS_PIN
#define K3_Q35_GDN_LOCAL_BASE
// The routed experts' tile (OP_OWN_MV, Q2_0) at 4 rows per wave: twice the bytes in flight per pass (the experts' down,
// K 512, one pass per row: latency-bound at 2 rows); a row's sum does not depend on RW (the logits bitwise at 2 and 4).
// XT: down 17.9 -> 12.3 us, gate | up 17.0 -> 13.8; 8 rows spill 55 VGPRs (8.0 ms per token). No Q4_K / Q6_K tiles:
// the line's files hold none (k3.OWN_Q35MOE_1RT); without them k3_run 88 -> 78 VGPRs, 78 -> 66 spilled SGPRs.
#define OWN_K3_RW 4
#define OWN_K3_NO_Q4
#define OWN_K3_NO_Q6
#define K3_EXTRA_OPS_HEADER "ops_q35.h"
#include "dec_core.hip"
