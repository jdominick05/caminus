// The wide prefill GEMM tiles' skeleton, one source for the MoE models' wide objects (Qwen3-30B-A3B's
// kernels/pf_q3m_mm.hip: Q4_K / Q6_K; Qwen3.6-35B-A3B's kernels/pf_q35m_mm.hip: Q2_0): a workgroup's work item,
// its rows and its stores. The tiles: 8 waves of two 16-row tiles each, so each A operand (a token tile's 16 values, from
// LDS) feeds the WMMAs of both row tiles and each token tile's X is staged once per 256 rows instead of per 128. GU (the
// experts' gate and up in one launch): row tile 0 is rows 16 w of the gate tensor, row tile 1 the same rows of the up
// tensor (128 rows of each per workgroup), and the output is fp16(silu(g) u), q3p_swiglu's expression on the same fp32 g
// and u. The format's tile (its weight loads, B operands and sums, in the narrow tile's order per row) sits between these.
// Needs pf_q3m.hip's types (MMItem, f8_t, MM_M, MM_N).
#ifndef MMW_N
#define MMW_N 256   // (rows per workgroup of the wide tiles: 8 waves x 32 rows)
#endif

// the workgroup's work item: the grouped experts' (items: q3p_group's), or a dense matrix's 64 tokens (items null: expert 0,
// tokens 64 x .. of ntok); a workgroup past the device's item count returns (the grid is the host's bound). A statement
// macro: as a function returning a flag it changed the 30B tile's code (1,171 lines of its disassembly)
#define MMW_ITEM(it, items, nitems, ntok)                                                                             \
    if (items) { if ((int)blockIdx.x >= *(nitems)) return; (it) = (items)[blockIdx.x]; }                              \
    else { (it).e = 0; (it).off = blockIdx.x * MM_M; (it).cnt = min(MM_M, (ntok) - (it).off); }

// lane's row of row tile rt (GU: of the gate or the up tensor)
#define MMW_ROW(GU, rt, wave, lane) \
    ((GU) ? blockIdx.y * MM_N + (wave) * 16 + ((lane) & 15) : blockIdx.y * MMW_N + (wave) * 32 + (rt) * 16 + ((lane) & 15))

// the row's index in the tensor's rows (expert e's row r at e R + r; dense: r); rows past R read row 0 and are never
// written. (The dense form written as the dense line's tile had it, so the 30B's object kept its code.)
__device__ __forceinline__ size_t mmw_grow(const MMItem & it, bool grouped, int R, int r, bool rok) {
    return grouped ? (size_t)it.e * R + (rok ? r : 0) : (size_t)(rok ? r : 0);
}

// the stores, the tile's last statements: token tile mt's value v of a lane is token mt * 16 + 2 v + lane / 16 of the
// item (only its it.cnt tokens are written), at ydst's row (null: the item's own rows), column r[rt]. GU: Y16 =
// fp16(silu(g) u) of row tile 0 (g) and 1 (u). acc [2][4] f8_t, r [2] int, rok [2] bool
#define MMW_STORE(GU, acc, r, rok, it, Y, Y16, ldy, ydst, lane)                                                      \
    if (GU) {                                                                                                         \
        if (!(rok)[0]) return;                                                                                        \
        _Pragma("unroll")                                                                                             \
        for (int mt = 0; mt < 4; mt++)                                                                                \
            _Pragma("unroll")                                                                                         \
            for (int v = 0; v < 8; v++) {                                                                             \
                const int t = mt * 16 + 2 * v + ((lane) >> 4);                                                        \
                if (t < (it).cnt) {                                                                                   \
                    const float gv = (acc)[0][mt][v];                                                                 \
                    (Y16)[(size_t)((ydst) ? (ydst)[(it).off + t] : (it).off + t) * (ldy) + (r)[0]] =                  \
                        (_Float16)(gv / (1.0f + expf(-gv)) * (acc)[1][mt][v]);                                        \
                }                                                                                                     \
            }                                                                                                         \
        return;                                                                                                       \
    }                                                                                                                 \
    _Pragma("unroll")                                                                                                 \
    for (int rt = 0; rt < 2; rt++) {                                                                                  \
        if (!(rok)[rt]) continue;                                                                                     \
        _Pragma("unroll")                                                                                             \
        for (int mt = 0; mt < 4; mt++)                                                                                \
            _Pragma("unroll")                                                                                         \
            for (int v = 0; v < 8; v++) {                                                                             \
                const int t = mt * 16 + 2 * v + ((lane) >> 4);                                                        \
                if (t < (it).cnt) (Y)[(size_t)((ydst) ? (ydst)[(it).off + t] : (it).off + t) * (ldy) + (r)[rt]] = (acc)[rt][mt][v]; \
            }                                                                                                         \
    }
