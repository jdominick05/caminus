// kernels/dec_q35_swift_tp.hip as the two-card verify module (two-card speculative decoding: engine/q35_dense_verify_tp.py):
// K3_TP's OP_XCHG (kernels/ops_xchg.h: its Seg.N is the payload's floats, so a T-token partial of T * D exchanges as-is)
// with the verify T-token forms (K3_Q3M_VER: kernels/quant_ver.h, K3_Q35_VER's tiles, K3_Q35_ATTN_PACK). The one-card
// verify module dec_q35_swift_ver and the two-card decode module dec_q35_swift_tp are unchanged: their code objects are
// built without the other's defines.
#define K3_TP
#define K3_Q3M_VER
#define K3_Q35_VER
#define K3_Q35_ATTN_PACK
#include "dec_q35_swift.hip"
