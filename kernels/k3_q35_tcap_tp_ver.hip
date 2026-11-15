// kernels/k3_q35_tcap_tp.hip as the two-card verify module (ThinkingCap's line; engine/q35_dense_verify_tp.py): K3_TP's
// OP_XCHG (kernels/k3_xchg.h: Seg.N is the payload's floats, so T * D exchanges as-is) with K3_Q3M_VER's T-token tiles
// (kernels/own_quant_tcap_mt.h, kernels/k3_q35t_ops.h, K3_Q35_VER) and K3_Q35_ATTN_PACK. The one-card verify module
// k3_q35_tcap_ver and the two-card decode module k3_q35_tcap_tp are unchanged.
#define K3_TP
#define K3_Q3M_VER
#define K3_Q35_VER
#define K3_Q35_ATTN_PACK
#define K3_Q35_GDN_LOCAL_BASE   // keep the replay-state base local to GDN instead of live through the dispatch
#include "k3_q35_tcap.hip"
