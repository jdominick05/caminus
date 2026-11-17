// The qwen3moe decode module for the official GPTQ-Int4 checkpoint converted to Q4_0 (tools/eval/gptq_to_gguf.py):
// kernels/k3_q3m_lean.hip with K3_Q3M_Q40, which puts the Q4_0 tile (kernels/own_quant_q4_0.h) in the own matvec op
// in place of the Q4_K tile. A separate object family (k3.Q3M_Q40_OF: one twin per lean, YaRN, Q8 KV, two-card and
// verify build, each this define over its base's file), so the shipped modules keep their code.
#define K3_Q3M_Q40
#include "k3_q3m_lean.hip"
