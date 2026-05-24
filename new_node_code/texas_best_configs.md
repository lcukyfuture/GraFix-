# Texas Best Configuration Summary

This summary records the best-validation configurations for Texas on split 0 with seed 42.

| Mode | Best params | hidden | WL | layers | hop | C | dropout | lr | best val loss | seed42 test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asymmetric | `True_1_False_WL_128_5_5_1l_3h_clustered_C16_0.0_0.01_32` | 128 | 5 | 1 | 3 | 16 | 0.0 | 0.01 | 0.6170 | 0.8649 |
| symmetric | `True_1_False_WL_32_3_5_3l_3h_clustered_C64_0.2_0.01_32` | 32 | 3 | 3 | 3 | 64 | 0.2 | 0.01 | 0.6090 | 0.8108 |
| asymmetric_mul | `True_1_False_WL_128_3_5_1l_1h_clustered_C16_0.0_0.01_32_gate-asymmetric_mul` | 128 | 3 | 1 | 1 | 16 | 0.0 | 0.01 | 0.6300 | 0.8649 |
| symmetric_mul | `True_1_False_WL_64_5_5_3l_2h_clustered_C16_0.2_0.001_32_gate-symmetric_mul` | 64 | 5 | 3 | 2 | 16 | 0.2 | 0.001 | 0.6060 | 0.8649 |

Common settings:

- Dataset: `Texas`
- Split: `0`
- Seed: `42`
- Kernel: `WL`
- Num heads: `1`
- `GL_k`: `5`
- Batch size: `32`
- Clustered WL features: enabled

Local source result files:

- `node_grid_search_results/texas_asym_no_gate_full_grid/grid_best.md`
- `node_grid_search_results/wl_multi_hop_symmetric_clustered/Texas_texas_symmetric_split0_seed42_20260522_154831/best_results_summary.txt`
- `node_ablation_grid_search_results/wl_multi_hop_gating_variants/asymmetric_mul/Texas_texas_grid_seed42_20260522_183809/best_results_summary.txt`
- `node_ablation_grid_search_results/wl_multi_hop_gating_variants/symmetric_mul/Texas_texas_grid_seed42_20260522_155009/best_results_summary.txt`
