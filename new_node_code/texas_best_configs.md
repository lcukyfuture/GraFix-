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
