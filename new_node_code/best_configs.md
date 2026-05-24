# Best Configuration Summary

This summary records the best-validation configurations on split 0 with seed 42.

| Dataset | Mode | Best params | hidden | WL | layers | hop | C | dropout | lr | best val loss | seed42 test acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | asymmetric | `True_1_False_WL_128_5_5_1l_3h_clustered_C32_0.0_0.001_32` | 128 | 5 | 1 | 3 | 32 | 0.0 | 0.001 | 0.7380 | 0.7800 |
| Cora | symmetric | `True_1_False_WL_64_5_5_2l_3h_clustered_C32_0.2_0.001_32` | 64 | 5 | 2 | 3 | 32 | 0.2 | 0.001 | 0.7200 | 0.7620 |
| Cora | asymmetric_mul | `True_1_False_WL_128_4_5_1l_2h_clustered_C32_0.2_0.001_32_gate-asymmetric_mul` | 128 | 4 | 1 | 2 | 32 | 0.2 | 0.001 | 0.7460 | 0.7760 |
| Cora | symmetric_mul | `True_1_False_WL_64_5_5_2l_3h_clustered_C32_0.2_0.001_32_gate-symmetric_mul` | 64 | 5 | 2 | 3 | 32 | 0.2 | 0.001 | 0.6920 | 0.7630 |
| CiteSeer | asymmetric | `True_1_False_WL_128_3_5_1l_2h_clustered_C32_0.2_0.001_32` | 128 | 3 | 1 | 2 | 32 | 0.2 | 0.001 | 1.0110 | 0.6880 |
| CiteSeer | symmetric | `True_1_False_WL_128_5_5_1l_2h_clustered_C32_0.0_0.001_32` | 128 | 5 | 1 | 2 | 32 | 0.0 | 0.001 | 1.0000 | 0.6570 |
| CiteSeer | asymmetric_mul | `True_1_False_WL_128_4_5_1l_2h_clustered_C32_0.2_0.001_32_gate-asymmetric_mul` | 128 | 4 | 1 | 2 | 32 | 0.2 | 0.001 | 1.0110 | 0.6880 |
| CiteSeer | symmetric_mul | `True_1_False_WL_128_5_5_1l_2h_clustered_C32_0.0_0.001_32_gate-symmetric_mul` | 128 | 5 | 1 | 2 | 32 | 0.0 | 0.001 | 1.0010 | 0.6580 |
| PubMed | asymmetric | `True_1_False_WL_64_4_5_1l_3h_clustered_C64_0.2_0.001_32` | 64 | 4 | 1 | 3 | 64 | 0.2 | 0.001 | 0.5810 | 0.7710 |
| PubMed | symmetric | `True_1_False_WL_64_3_5_1l_3h_clustered_C64_0.2_0.001_32` | 64 | 3 | 1 | 3 | 64 | 0.2 | 0.001 | 0.5970 | 0.7510 |
| PubMed | asymmetric_mul | `True_1_False_WL_64_4_5_1l_3h_clustered_C64_0.2_0.001_32_gate-asymmetric_mul` | 64 | 4 | 1 | 3 | 64 | 0.2 | 0.001 | 0.5820 | 0.7700 |
| PubMed | symmetric_mul | `True_1_False_WL_64_5_5_1l_3h_clustered_C64_0.2_0.001_32_gate-symmetric_mul` | 64 | 5 | 1 | 3 | 64 | 0.2 | 0.001 | 0.5950 | 0.7540 |
| Texas | asymmetric | `True_1_False_WL_128_5_5_1l_3h_clustered_C16_0.0_0.01_32` | 128 | 5 | 1 | 3 | 16 | 0.0 | 0.01 | 0.6170 | 0.8649 |
| Texas | symmetric | `True_1_False_WL_32_3_5_3l_3h_clustered_C64_0.2_0.01_32` | 32 | 3 | 3 | 3 | 64 | 0.2 | 0.01 | 0.6090 | 0.8108 |
| Texas | asymmetric_mul | `True_1_False_WL_128_3_5_1l_1h_clustered_C16_0.0_0.01_32_gate-asymmetric_mul` | 128 | 3 | 1 | 1 | 16 | 0.0 | 0.01 | 0.6300 | 0.8649 |
| Texas | symmetric_mul | `True_1_False_WL_64_5_5_3l_2h_clustered_C16_0.2_0.001_32_gate-symmetric_mul` | 64 | 5 | 3 | 2 | 16 | 0.2 | 0.001 | 0.6060 | 0.8649 |

Common settings:

- Split: `0`
- Seed: `42`
- Kernel: `WL`
- Num heads: `1`
- `GL_k`: `5`
- Batch size: `32`
- Clustered WL features: enabled
