"""The DFlash drafter of Qwen3-30B-A3B (RedHatAI/Qwen3-30B-A3B-speculator.dflash: speculators' DFlashDraftModel) in plain torch,
fp32: the reference the device drafter is checked against, and the offline acceptance measurement. Written from the
architecture (speculators' dflash core / model_definitions, read as a reference; nothing of it is imported or run):

  context   features f[p] = the target's residual stream at five layers, concatenated (5 x 2048) for each position p
            before the anchor; c[p] = hidden_norm(fc f[p])
  block     the anchor token (known: the last accepted one) at position a, then mask_token_id x (block_size - 1) at a + 1 ..;
            h = embed(block tokens) (the target's own token embedding: speculators reloads it from the verifier)
  layer x5  x = rms(h, input_layernorm); q = q_norm(q_proj x) (per head, 128), RoPE (theta 1e6, rotate_half) at the block
            positions; K / V over [context c[p] for p in (q_pos - window, a) | block x]: k = k_norm(k_proj .), RoPE at
            their positions, v = v_proj . (no norm); attention: every context row inside the query's window, block rows
            causal (j <= i); GQA 8 query heads per kv head; h += o_proj(attn); h += down(silu(gate x2) * up x2) with
            x2 = rms(h, post_attention_layernorm)
  head      logits = lm_head(rms(h, norm)) over the draft vocabulary (32,000 rows); a draft id d is target id d + d2t[d]
  Row i of the block (i >= 1) proposes the token at position a + i.
Weights read from the checkpoint's safetensors file by its header (bf16 widened to fp32); the config's numbers are passed in.
  DFlashQ3MRef(path, device).draft(feats [C, 5 D] (positions 0 .. C - 1 = every position before the anchor), anchor_token,
      a = C, n=7) -> the n proposed target ids"""
import json
import os
import struct

import numpy as np
import torch

from ..device import capacity

CFG = dict(layers=5, D=2048, heads=32, kv_heads=4, head_dim=128, ffn=6144, eps=1e-6, theta=1e6, window=2048, block=8,
           mask_token=151669, draft_vocab=32000, aux_layers=(1, 12, 23, 34, 45))


def safetensors_header(path):
    """(the header's {name: entry}, the data's file offset)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def loaded_bytes(head):
    """bytes read_safetensors's tensors take once loaded (BF16 widened to fp32: twice the file's)"""
    return sum((v['data_offsets'][1] - v['data_offsets'][0]) * (2 if v['dtype'] == 'BF16' else 1)
               for k, v in head.items() if k != '__metadata__')


def read_safetensors(path, device='cpu'):
    """{name: tensor} from a safetensors file (BF16 -> float32 on the host, I64, BOOL), parsed from its header"""
    out = {}
    head, base = safetensors_header(path)
    with open(path, 'rb') as f:
        for k, v in head.items():
            if k == '__metadata__': continue
            a, b = v['data_offsets']; f.seek(base + a); raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
            if v['dtype'] == 'BF16': t = torch.from_numpy((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy())
            elif v['dtype'] == 'I64': t = torch.from_numpy(raw.view(np.int64).copy())
            elif v['dtype'] == 'BOOL': t = torch.from_numpy(raw.view(np.bool_).copy())
            else: raise ValueError(f'{path}: {k} is {v["dtype"]}')
            out[k] = t.view(v['shape']).to(device)
    return out


class DFlashQ3MRef:
    def __init__(self, path, device='cuda', cfg=CFG, vram_check=True):
        self.c = c = dict(cfg)
        f = os.path.join(path, 'model.safetensors') if os.path.isdir(path) else path
        # VRAM, projected from the file's header before anything is on the card: the widened weights + one pass's
        # workspace over a full window (per layer: the sources, k and v and their per-query-head copies, the scores and
        # their softmax; the block's logits) + 256 MiB (the runtime's own memory, allocator rounding)
        rows = c['window'] + c['block']
        work = rows * (2 * c['D'] + (2 * c['kv_heads'] + 2 * c['heads']) * c['head_dim']) * 4 \
            + 3 * c['heads'] * c['block'] * rows * 4 + 2 * c['block'] * c['draft_vocab'] * 4
        self.vram_mib = (loaded_bytes(safetensors_header(f)[0]) + work) / 2 ** 20 + 256
        dv = torch.device(device)
        if vram_check and dv.type == 'cuda':
            capacity.check([torch.cuda.current_device() if dv.index is None else dv.index], [self.vram_mib],
                           f'DFlashQ3MRef {os.path.basename(os.path.dirname(f) or f)} ({self.vram_mib:.0f} MiB projected)')
        self.w = read_safetensors(f, device)
        self.dev = device
        d2t = self.w['d2t']
        self.d2t_ids = torch.arange(len(d2t), device=device) + d2t       # draft id -> target id (offsets)

    def rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.c['eps']) * w

    def rope(self, x, pos):
        """x [n, h, 128], pos [n]: rotate_half RoPE"""
        hd = x.shape[-1]
        inv = self.c['theta'] ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float64) / hd)
        fr = pos.to(torch.float64)[:, None] * inv[None]
        cos = torch.cat([fr.cos(), fr.cos()], -1).float()[:, None]; sin = torch.cat([fr.sin(), fr.sin()], -1).float()[:, None]
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    @torch.no_grad()
    def context(self, feats):
        """c [C, D] = hidden_norm(fc feats)"""
        return self.rms(feats.float() @ self.w['fc.weight'].T, self.w['hidden_norm.weight'])

    @torch.no_grad()
    def logits(self, ctx, ctx_pos, block_tokens, a):
        """draft logits [B, draft_vocab] for the block [anchor, masks...] at positions a .. a + B - 1 over context rows ctx
        [C, D] at positions ctx_pos [C] (all < a)"""
        c, w = self.c, self.w
        H, G, hd = c['heads'], c['kv_heads'], c['head_dim']
        Bn = len(block_tokens)
        bpos = torch.arange(a, a + Bn, device=self.dev)
        h = w['embed_tokens.weight'][torch.as_tensor(block_tokens, device=self.dev)]
        kpos = torch.cat([ctx_pos.to(self.dev), bpos])
        # mask [B, C + B]: context inside the window (kv > q - window, kv < a), block causal
        qp = bpos[:, None]
        m_ctx = (ctx_pos.to(self.dev)[None, :] > qp - c['window']) & (ctx_pos.to(self.dev)[None, :] < a)
        m_blk = torch.arange(Bn, device=self.dev)[None, :] <= torch.arange(Bn, device=self.dev)[:, None]
        mask = torch.cat([m_ctx, m_blk], 1)
        for il in range(c['layers']):
            p = f'layers.{il}.'
            x = self.rms(h, w[p + 'input_layernorm.weight'])
            q = self.rms((x @ w[p + 'self_attn.q_proj.weight'].T).view(Bn, H, hd), w[p + 'self_attn.q_norm.weight'])
            src = torch.cat([ctx, x], 0)
            k = self.rms((src @ w[p + 'self_attn.k_proj.weight'].T).view(-1, G, hd), w[p + 'self_attn.k_norm.weight'])
            v = (src @ w[p + 'self_attn.v_proj.weight'].T).view(-1, G, hd)
            q, k = self.rope(q, bpos), self.rope(k, kpos)
            kq = k.repeat_interleave(H // G, 1); vq = v.repeat_interleave(H // G, 1)     # [C + B, H, hd]
            s = torch.einsum('bhd,nhd->hbn', q, kq) / hd ** 0.5
            s = s.masked_fill(~mask[None], float('-inf'))
            o = torch.einsum('hbn,nhd->bhd', torch.softmax(s, -1), vq).reshape(Bn, H * hd)
            h = h + o @ w[p + 'self_attn.o_proj.weight'].T
            x2 = self.rms(h, w[p + 'post_attention_layernorm.weight'])
            g_ = x2 @ w[p + 'mlp.gate_proj.weight'].T; u_ = x2 @ w[p + 'mlp.up_proj.weight'].T
            h = h + (g_ * torch.sigmoid(g_) * u_) @ w[p + 'mlp.down_proj.weight'].T
        return self.rms(h, w['norm.weight']) @ w['lm_head.weight'].T

    def draft(self, feats, anchor_token, n=7, ctx=None):
        """the n proposed target ids after an anchor at position a = len(feats) (feats: every position before it)"""
        C = len(feats) if ctx is None else len(ctx)
        lo = max(0, C - self.c['window'])
        ctx = self.context(torch.as_tensor(feats[lo:], device=self.dev)) if ctx is None else ctx[lo:]
        lg = self.logits(ctx, torch.arange(lo, C, device=self.dev), [anchor_token] + [self.c['mask_token']] * n, C)
        return self.d2t_ids[lg[1:].argmax(-1)].tolist()
