// The row-scale study's standalone matvec kernels (docs/dev/row-scale-study.md): out[r] = W[r] . x for row-scaled
// 4-bit and 2-bit codes (kernels/own_quant_rs.h: the codes and, in a buffer of their own, the row scales), fp16 activations, at the experts' K values
// (2048: gate / up, 768: down), 2 rows per wave. For the op test against float64 and a first run on a card; every
// speed in the study is measured in the decode module.
#include <hip/hip_runtime.h>
#include <type_traits>

#include "own_quant_rs.h"

#define OWN_RS_1RT(BITS, KK)                                                                                               \
extern "C" __global__ void __launch_bounds__(256) own_mv_rs##BITS##_1rt_k##KK(const uint8_t * W, const float * sc,     \
        int nrows, const _Float16 * x16, float * out) {                                                                 \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * 2;                                                  \
    if (row0 < nrows) own_rows_rs_1rt<BITS, KK, 2>(W, sc, row0, nrows, x16, out, threadIdx.x & 31);                      \
}
OWN_RS_1RT(4, 2048)
OWN_RS_1RT(4, 768)
OWN_RS_1RT(2, 2048)
OWN_RS_1RT(2, 768)
