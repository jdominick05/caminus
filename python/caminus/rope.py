"""YaRN RoPE scaling (the long-context mode of Qwen3-30B-A3B, caminus serve --long-context): the per-pair frequency table and
the attention factor the YaRN decode modules (kernels/dec_q3m*_yarn.hip: OP_OWN_ATTS under K3_Q3M_YARN) and the
prefill's q3p_qk_rope (pf_q3m*_yarn: Q3P_YARN) read from the end of their par block. Both kernels compute
theta = (float)pos * freq[j] and cos(theta) * mscale, sin(theta) * mscale with the same statements on the same table, so
decode and prefill agree bit for bit; the table is made once on the host, in float64, rounded to fp32.

The recipe is llama.cpp's (ggml rope_yarn, ggml_rope_yarn_corr_dims; transformers' _compute_yarn_parameters computes the
same frequencies and puts the factor on cos and sin too: checked against its 5.20 source), the yardstick's convention:
  pair j (dims j and j + 64 for NEOX), inv_j = base^(-2 j / n_rot)
  the correction range [lo, hi] = [floor(dim(beta_fast)), ceil(dim(beta_slow))] clamped to [0, n_rot - 1], with
    dim(beta) = n_rot ln(n_ctx_orig / (2 pi beta)) / (2 ln base)          (n_rot 128, base 1e6, 32,768: [23, 40])
  ramp_j = 1 - clamp((j - lo) / max(0.001, hi - lo), 0, 1)               (1: the pair keeps its trained frequency)
  freq_j = inv_j (ramp_j + (1 - ramp_j) / factor)                       (llama.cpp: theta_interp (1 - ramp) + theta_extrap ramp)
  mscale = 1 + 0.1 ln(factor) on cos and sin, so on q and on the stored k (the scores carry mscale^2). llama.cpp sets
    yarn_attn_factor = get_mscale(factor) / (1 + 0.1 ln factor) = 1 and rope_yarn multiplies it by 1 + 0.1 ln(1 / freq_scale)
    again: the same fp32 value, 1.1386294 at factor 4 (transformers: cos and sin times attention_factor, the same place).
The original context is the model card's original_max_position_embeddings (32,768), never the GGUF's context_length
(40,960): llama.cpp falls back to the latter when --yarn-orig-ctx is not given, so its yardstick runs pass it."""
import math

import numpy as np

BETA_FAST, BETA_SLOW = 32.0, 1.0
TABLE_FLOATS = 65          # freq [n_rot / 2] | mscale, appended to the decode block q_norm | k_norm | theta_scale | eps


def corr_dims(n_rot, n_ctx_orig, base, beta_fast=BETA_FAST, beta_slow=BETA_SLOW):
    """llama.cpp's ggml_rope_yarn_corr_dims: (lo, hi) as floats"""
    dim = lambda beta: n_rot * math.log(n_ctx_orig / (beta * 2 * math.pi)) / (2 * math.log(base))
    return max(0.0, float(math.floor(dim(beta_fast)))), min(float(n_rot - 1), float(math.ceil(dim(beta_slow))))


def yarn_freqs(n_rot, base, factor, n_ctx_orig, beta_fast=BETA_FAST, beta_slow=BETA_SLOW):
    """float64 [n_rot / 2]: each pair's YaRN frequency (the angle at position p is p * freq_j)"""
    lo, hi = corr_dims(n_rot, n_ctx_orig, base, beta_fast, beta_slow)
    j = np.arange(n_rot // 2, dtype=np.float64)
    inv = base ** (-2.0 * j / n_rot)
    ramp = 1.0 - np.clip((j - lo) / max(0.001, hi - lo), 0.0, 1.0)
    return inv * (ramp + (1.0 - ramp) / factor)


def mscale(factor):
    """the attention factor on cos and sin (1 at factor <= 1)"""
    return 1.0 if factor <= 1.0 else 1.0 + 0.1 * math.log(factor)


def yarn_table(n_rot, base, factor, n_ctx_orig):
    """fp32 [TABLE_FLOATS]: freq [n_rot / 2] then mscale, the block the YaRN kernels read after eps"""
    if n_rot != 128: raise ValueError(f'yarn_table: n_rot {n_rot} (the YaRN kernels are built for 128)')
    return np.concatenate([yarn_freqs(n_rot, base, factor, n_ctx_orig), [mscale(factor)]]).astype(np.float32)


class YarnMode:
    """the long-context mode's RoPE: factor and original context (from caminus.arch.RopeFacts), the table it makes"""
    kind = 'yarn'

    def __init__(self, factor, n_ctx_orig):
        self.factor, self.n_ctx_orig = float(factor), int(n_ctx_orig)
        if not self.factor > 1.0 or self.n_ctx_orig < 1: raise ValueError(f'YaRN: factor {factor} (> 1), original context {n_ctx_orig}')

    @property
    def max_ctx(self):
        """the most positions the mode serves: factor x the original context (131,072 at x4 from 32,768)"""
        return int(self.factor * self.n_ctx_orig)

    def key(self):
        """the engine key's rope entry (caminus.cache.prompt.engine_key): a YaRN park never restores into another RoPE"""
        return f'yarn:{self.factor:g}:{self.n_ctx_orig}:{BETA_FAST:g}:{BETA_SLOW:g}'

    def table(self, n_rot, base):
        return yarn_table(n_rot, base, self.factor, self.n_ctx_orig)

    def __repr__(self):
        return f'YarnMode(x{self.factor:g} from {self.n_ctx_orig})'
