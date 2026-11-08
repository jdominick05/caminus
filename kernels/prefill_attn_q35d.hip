// The batched prefill's attention core for qwen35 dense (AREX-2: 24 q / 4 kv heads, head dim 256; python/caminus/prefill35.py):
// prefill_attn.hip with 6 query heads per kv head, ATTN_VH 6: the split pair of k3_gguf_q35d (ATTN_G 6, ATTN_VH 6, 32-cell splits)
#define ATTN_G 6
#define ATTN_VH 6
#include "prefill_attn.hip"
