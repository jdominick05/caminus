// K3 separate bound guard: the device-side defense behind the host checks (k3.check_span, FlashNextTP._check) for the
// launches k3_run's own guard does not cover (k3.Engine.run: k3_run2 tapes, and tapes with more than one position tensor;
// k3_run itself checks one position word and the commit in its prologue, kernels/k3.hip, and skips its whole tape).
// One single-thread launch on the stream right before the tape's launch: it reads the pass's position and lazy-commit
// words as the tape will read them (after the host's copy), and if the T positions pos .. pos + T - 1 leave the KV
// cache's rows [0, cap) or the commit leaves [0, T], it clamps them in place (pos = 0, commit = 0: every KV write / read
// and every replay of the previous pass's inputs / records stays inside its buffer), records the refusal in the sync
// block's bounds words (bnd = 1) and sets its err word. The launch that follows still runs every task of its tape once,
// with the clamped values (its waits give up at once, so it ends within its tasks: measured 8.8 - 9.0 ms at T = 1 and
// ~15 ms at T = 4 on the Coder TP path, far under the 200 ms bound); the KV rows 0 .. T - 1 and the recurrent states are
// garbage afterwards. The host raises k3.ContextBoundsError, never retries, and the engines refuse passes until reset().
// A separate dispatch costs ~0.05 - 0.10 ms per token on the Coder TP path, hence k3_run's prologue check there.
// Sync block (kernels/k3.hip struct Sync): claim [K3_MAXSEG], done [K3_MAXSEG], then the status words exit, err, err_seg,
// err_seen, bnd, bnd_what, bnd_val, bnd_cap. bnd_what: -1 = the position (bnd_val = pos, bnd_cap = the cache's rows),
// -2 = the lazy commit (bnd_cap = T).
#include <hip/hip_runtime.h>
#include <stdint.h>

// pos: int32 [1] (nullptr: no position check); cap: the KV cache's rows; T: tokens per pass; params: int32 [16], params[1]
// = the lazy commit (nullptr: no commit check); status: the sync block's status words (int32 [8] at 2 * K3_MAXSEG)
extern "C" __global__ void k3_guard(int * pos, int cap, int T, int * params, int * status) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int what = 0, val = 0, bound = 0;
    if (pos) {
        const int p = *pos;
        if (p < 0 || (long long)p + T > (long long)cap) { what = -1; val = p; bound = cap; *pos = 0; }
    }
    if (params) {
        const int m = params[1];
        if (m < 0 || m > T) { if (!what) { what = -2; val = m; bound = T; } params[1] = 0; }
    }
    if (what) {
        if (status[4] == 0) { status[5] = what; status[6] = val; status[7] = bound; status[4] = 1; }
        if (status[1] == 0) { status[2] = 0; status[3] = -1; status[1] = 1; }   // err: the launch's waits give up at once (its tasks still run)
        __threadfence_system();
    }
}
