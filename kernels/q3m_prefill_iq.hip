// The IQ study's batched prefill (docs/dev/iq-quant-study.md): kernels/q3m_prefill.hip's kernels as they are, plus the
// quantized GEMM for GGUF's codebook formats on the experts (IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS). A study object: the
// shipped prefill object is built from q3m_prefill.hip alone and does not change. No ggml-cuda code; the grids are format
// constants (kernels/iq_grids.h, ggml's data, MIT notice there).
//
// ---- q3p_mm_iq4xs / q3p_mm_iq3xxs / q3p_mm_iq3s / q3p_mm_iq2xxs: q3p_mm_q4k's contract (Y[ydst[i]][r] = X[xsrc[i]] . W_e[r]
// in fp32, fp16 activations through WMMA, the same work items and tiles) -------------------------------------------------
// The decode engine's own weight copies are read in place: IQ4_XS raw GGUF blocks (136 B per 256 values), the grid
// formats in the record layout (caminus.engine.own_layout.iq_records: plane A a record per unit of 32 values, plane B a
// header per block, 16 bytes of padding; Rtot = the rows of the whole buffer). The weights are EXACT in the B operand:
//   IQ4_XS   w = d (ls - 32) kv: B = kv (|kv| <= 127), one product P per unit of 32 values, acc += d (ls - 32) P
//            (kv (ls - 32) reaches 4,064: not every such integer is an fp16, so the unit's scale stays outside B)
//   IQ3_XXS  w = d 0.25 (1 + 2 s) (+-grid): B = (1 + 2 s) (+-grid), an integer of at most 31 * 62 = 1,922: exact in fp16
//   IQ3_S    w = d (1 + 2 s) (+-grid): at most 31 * 15 = 465
//   IQ2_XXS  w = d 0.125 (1 + 2 s) (+-grid): at most 31 * 43 = 1,333
//            the grid formats: one product P over the whole 256-value block, acc += (d c) P
// so a product's only rounding is the fp32 accumulation, as in q3p_mm_q4k. The grids sit in the workgroup's local memory
// (2 KB), staged by the 256 threads before the first block's barrier: IQ3 one word per entry (IQ3_S 512 entries),
// IQ2_XXS entry e as words e (values 0 .. 3) and 256 + e (4 .. 7), the decode tile's form.
#include "q3m_prefill.hip"
#include "iq_grids.h"
#include "q3m_prefill_iq.h"
