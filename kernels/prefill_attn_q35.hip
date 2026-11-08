// The batched prefill's attention core for qwen35moe (python/caminus/prefill35.py): prefill_attn.hip with 8 query heads per
// kv head (16 q / 2 kv heads, head dim 256), ATTN_VH 8: the split pair of k3_gguf_q35h / q35g (ATTN_G 8, ATTN_VH 8, 32-cell
// splits). Their merge (K3_ATTN_COMB_FAST) is bitwise the serial walk this file's pf_attn_comb runs.
#define ATTN_G 8
#define ATTN_VH 8
#include "prefill_attn.hip"
