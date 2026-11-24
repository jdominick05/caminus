// A qwen3moe decode module of the row-scale study (docs/dev/row-scale-study.md): kernels/k3_q3m_lean.hip's defines plus
// K3_Q3M_RS = 4, which adds ONE tile to the own matvec op: row-scaled 4-bit codes (kernels/own_quant_rs.h). The bit
// width is fixed per code object (the IQ study measured what a runtime choice of format costs inside a tile). A
// measurement module: the official file runs on it too (the same Q4_K / Q6_K tiles), and the shipped module stays as it is.
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_Q3M_RS 4
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#ifndef K3_PER_WGP           // the XT's grid, 5 x 42 WGPs: VGPR cap 144 (kernels/k3.hip); build.sh adds a _w4 object
#define K3_PER_WGP 5        // (-DK3_PER_WGP=4, cap 192) for the XTX's 4 x 48
#endif
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
