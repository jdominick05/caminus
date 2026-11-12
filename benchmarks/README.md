# benchmarks/

Speed measurements, not gates: decode (`fn_bench.py`, `q3n_bench.py`, `fn_soak.py`), the prefill feed and the TP link
(`prefill_*_bench.py`), the split attention pair (`attn_bench.py`), the dense qwen35 line's attention at depth and prefill GEMMs /
attention next to outside yardsticks (`attn_depth.py`, `q35_prefill_gemm.py`, `q35_prefill_attn.py`, `rocke_attn.py`; results and the
yardsticks' versions in docs/dev/yardsticks.md), where a token's time goes (`fn_profile.py`, `k3_trace.py`),
and the engine cost model's microbenchmarks (`bench_*.py`, `bench_link.hip`). Every one runs on the GPU under
`tools/guard.sh`. A speed claim needs an interleaved A/B, the median of at least 3 runs and a difference above 0.05 ms, and
says whether it is decode or prefill.
