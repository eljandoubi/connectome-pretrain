### Cohort by site

| site | n | age mean (sd) | FIQ mean (sd) | frames in file | lag-1 autocorr | held out |
|---|---|---|---|---|---|---|
| NYU | 172 | 15.3 (6.6) | 110.8 (14.8) | 176 | 0.78 | yes |
| UM_1 | 82 | 13.6 (2.9) | 107.3 (13.1) | 296 | 0.79 | yes |
| USM | 67 | 22.6 (8.3) | 105.3 (17.9) | 236 | 0.78 | yes |
| UCLA_1 | 64 | 13.3 (2.4) | 104.1 (10.8) | 116 | 0.55 | yes |
| PITT | 50 | 18.5 (6.7) | 110.3 (11.9) | 196 | 0.88 | yes |
| MAX_MUN | 44 | 25.9 (10.3) | 110.5 (11.7) | 116 | 0.56 | yes |
| TRINITY | 44 | 17.0 (3.5) | 109.7 (13.9) | 146 | 0.78 | yes |
| YALE | 41 | 13.3 (2.6) | 98.3 (20.0) | 196 | 0.79 | yes |
| UM_2 | 34 | 16.0 (3.3) | 112.3 (10.7) | 296 | 0.79 |  |
| KKI | 33 | 10.3 (1.3) | 107.0 (15.8) | 152 | 0.67 |  |
| LEUVEN_1 | 28 | 22.4 (3.4) | 112.4 (13.0) | 246 | 0.85 |  |
| SDSU | 27 | 14.4 (1.9) | 112.0 (12.7) | 176 | 0.78 |  |
| OLIN | 26 | 16.8 (3.2) | 113.5 (16.7) | 206 | 0.88 |  |
| STANFORD | 25 | 10.0 (1.6) | 114.0 (14.8) | 236 | 0.78 |  |
| UCLA_2 | 21 | 12.5 (1.6) | 102.9 (14.5) | 116 | 0.55 |  |
| CALTECH | 14 | 27.2 (10.7) | 107.7 (9.1) | 146 | 0.78 |  |
| CMU | 11 | 26.8 (4.6) | 112.6 (10.8) | 236 | 0.83 |  |
| SBL | 4 | 34.0 (5.0) | 106.5 (12.2) | 196 | 0.76 |  |

### age: pooled R2 by label count (leave-one-site-out)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | -0.05 [-0.06, -0.04] | -0.06 [-0.08, -0.06] | -0.05 [-0.06, -0.05] | -0.05 [-0.06, -0.05] |
| correlation + ridge | -0.02 [-0.05, 0.00] | 0.04 [0.01, 0.08] | 0.25 [0.20, 0.29] | 0.34 [0.29, 0.39] |
| tangent + ridge | 0.01 [-0.01, 0.03] | 0.10 [0.07, 0.13] | 0.28 [0.24, 0.31] | 0.39 [0.35, 0.43] |
| tangent + RBF kernel ridge | -0.01 [-0.03, 0.01] | 0.07 [0.04, 0.09] | 0.24 [0.20, 0.27] | 0.35 [0.31, 0.39] |
| probe, mean token, random encoder | -0.11 [-0.14, -0.08] | -0.09 [-0.11, -0.08] | -0.06 [-0.07, -0.05] | - |
| probe, all tokens, random encoder | -0.07 [-0.09, -0.06] | -0.07 [-0.09, -0.06] | -0.06 [-0.07, -0.05] | - |
| probe, mean token, pretrained | -0.08 [-0.10, -0.07] | -0.09 [-0.11, -0.08] | -0.06 [-0.07, -0.05] | - |
| probe, all tokens, pretrained | -0.08 [-0.11, -0.07] | -0.09 [-0.10, -0.08] | -0.06 [-0.07, -0.05] | - |
| fine-tune from scratch | -0.11 [-0.14, -0.10] | -0.10 [-0.12, -0.09] | -0.08 [-0.10, -0.07] | - |
| fine-tune from pretrained | -0.09 [-0.11, -0.08] | -0.09 [-0.11, -0.08] | -0.10 [-0.12, -0.09] | - |

### age: macro R2 by label count

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | -0.94 [-1.24, -0.79] | -1.00 [-1.31, -0.84] | -0.90 [-1.20, -0.76] | -0.89 [-1.19, -0.75] |
| correlation + ridge | -1.06 [-1.43, -0.89] | -1.08 [-1.47, -0.90] | -0.58 [-0.98, -0.39] | -0.40 [-0.76, -0.21] |
| tangent + ridge | -0.79 [-1.08, -0.65] | -0.52 [-0.76, -0.41] | -0.11 [-0.28, -0.02] | 0.03 [-0.14, 0.13] |
| tangent + RBF kernel ridge | -0.81 [-1.10, -0.67] | -0.59 [-0.83, -0.47] | -0.17 [-0.35, -0.08] | 0.00 [-0.16, 0.10] |
| probe, mean token, random encoder | -1.12 [-1.46, -0.97] | -1.19 [-1.57, -1.00] | -0.91 [-1.21, -0.76] | - |
| probe, all tokens, random encoder | -1.07 [-1.39, -0.92] | -1.03 [-1.36, -0.87] | -0.91 [-1.21, -0.76] | - |
| probe, mean token, pretrained | -1.02 [-1.34, -0.88] | -1.11 [-1.45, -0.93] | -0.91 [-1.21, -0.76] | - |
| probe, all tokens, pretrained | -1.07 [-1.40, -0.91] | -1.07 [-1.40, -0.91] | -0.91 [-1.21, -0.76] | - |
| fine-tune from scratch | -1.16 [-1.53, -0.98] | -1.07 [-1.40, -0.90] | -0.95 [-1.26, -0.80] | - |
| fine-tune from pretrained | -1.07 [-1.39, -0.91] | -1.02 [-1.35, -0.86] | -1.02 [-1.34, -0.85] | - |

### age: all metrics at N=100

| arm | pooled R2 | macro R2 | pooled MAE | macro Spearman |
|---|---|---|---|---|
| site mean (constant) | -0.06 [-0.08, -0.06] | -1.00 [-1.31, -0.84] | 5.56 [5.21, 5.95] | 0.00 [0.00, 0.00] |
| correlation + ridge | 0.04 [0.01, 0.08] | -1.08 [-1.47, -0.90] | 5.21 [4.91, 5.56] | 0.28 [0.22, 0.34] |
| tangent + ridge | 0.10 [0.07, 0.13] | -0.52 [-0.76, -0.41] | 5.00 [4.67, 5.35] | 0.40 [0.33, 0.46] |
| tangent + RBF kernel ridge | 0.07 [0.04, 0.09] | -0.59 [-0.83, -0.47] | 5.11 [4.77, 5.47] | 0.35 [0.28, 0.42] |
| probe, mean token, random encoder | -0.09 [-0.11, -0.08] | -1.19 [-1.57, -1.00] | 5.64 [5.30, 6.03] | -0.02 [-0.06, 0.02] |
| probe, all tokens, random encoder | -0.07 [-0.09, -0.06] | -1.03 [-1.36, -0.87] | 5.57 [5.23, 5.96] | -0.00 [-0.04, 0.04] |
| probe, mean token, pretrained | -0.09 [-0.11, -0.08] | -1.11 [-1.45, -0.93] | 5.63 [5.28, 6.02] | -0.01 [-0.05, 0.03] |
| probe, all tokens, pretrained | -0.09 [-0.10, -0.08] | -1.07 [-1.40, -0.91] | 5.62 [5.27, 6.01] | 0.01 [-0.03, 0.06] |
| fine-tune from scratch | -0.10 [-0.12, -0.09] | -1.07 [-1.40, -0.90] | 5.62 [5.28, 6.01] | -0.04 [-0.08, 0.02] |
| fine-tune from pretrained | -0.09 [-0.11, -0.08] | -1.02 [-1.35, -0.86] | 5.60 [5.24, 5.98] | -0.02 [-0.07, 0.03] |
| site mean of the held-out site (oracle) | 0.29 [0.21, 0.35] | -0.00 [-0.04, -0.00] | 4.44 [4.14, 4.76] | 0.00 [0.00, 0.00] |

### age: transductive encoders, pooled R2

| arm | N=50 | N=100 | N=300 |
|---|---|---|---|
| probe, mean token, transductive | -0.11 [-0.14, -0.10] | -0.10 [-0.11, -0.09] | -0.09 [-0.10, -0.07] |
| probe, all tokens, transductive | -0.11 [-0.13, -0.09] | -0.09 [-0.11, -0.08] | -0.08 [-0.10, -0.07] |
| fine-tune, transductive (pretrained rate) | -0.19 [-0.22, -0.17] | -0.15 [-0.18, -0.14] | -0.14 [-0.16, -0.12] |
| fine-tune from pretrained | -0.09 [-0.11, -0.08] | -0.09 [-0.11, -0.08] | -0.10 [-0.12, -0.09] |

### age: 90 percent conformal intervals, pooled coverage (width)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | 0.89 (22.4) | 0.89 (22.3) | 0.89 (22.3) | 0.89 (22.2) |
| correlation + ridge | 0.88 (21.6) | 0.88 (19.7) | 0.88 (17.7) | 0.87 (16.2) |
| tangent + ridge | 0.89 (21.4) | 0.89 (19.6) | 0.87 (16.6) | 0.86 (14.7) |
| tangent + RBF kernel ridge | 0.89 (21.7) | 0.89 (20.1) | 0.88 (17.4) | 0.87 (15.2) |
| probe, mean token, random encoder | 0.89 (23.3) | 0.89 (22.6) | 0.89 (22.3) | - |
| probe, all tokens, random encoder | 0.89 (22.9) | 0.89 (22.5) | 0.89 (22.3) | - |
| probe, mean token, pretrained | 0.89 (22.6) | 0.90 (22.7) | 0.89 (22.3) | - |
| probe, all tokens, pretrained | 0.89 (22.6) | 0.90 (22.6) | 0.89 (22.3) | - |
| probe, mean token, transductive | 0.89 (23.0) | 0.89 (22.4) | 0.89 (22.3) | - |
| probe, all tokens, transductive | 0.89 (22.8) | 0.89 (22.4) | 0.89 (22.3) | - |

### age: 90 percent conformal intervals of the fine-tunes, pooled coverage (width)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| fine-tune from scratch | 0.89 (22.5) | 0.89 (22.4) | 0.89 (22.4) | - |
| fine-tune from pretrained | 0.89 (22.7) | 0.89 (22.4) | 0.89 (22.3) | - |
| fine-tune, transductive (pretrained rate) | 0.89 (22.8) | 0.89 (22.5) | 0.89 (22.3) | - |

### fiq: pooled R2 by label count (leave-one-site-out)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | -0.03 [-0.05, -0.01] | -0.03 [-0.04, -0.02] | -0.02 [-0.03, -0.01] | -0.02 [-0.03, -0.01] |
| correlation + ridge | -0.06 [-0.09, -0.04] | -0.04 [-0.05, -0.02] | -0.00 [-0.03, 0.02] | 0.01 [-0.02, 0.03] |
| tangent + ridge | -0.03 [-0.05, -0.01] | -0.02 [-0.04, -0.00] | 0.02 [0.00, 0.04] | 0.02 [-0.01, 0.05] |
| tangent + RBF kernel ridge | -0.03 [-0.05, -0.01] | -0.02 [-0.04, -0.01] | 0.02 [0.00, 0.04] | 0.02 [-0.01, 0.04] |
| probe, mean token, random encoder | -0.04 [-0.07, -0.03] | -0.03 [-0.05, -0.02] | -0.02 [-0.04, -0.01] | - |
| probe, all tokens, random encoder | -0.04 [-0.07, -0.03] | -0.04 [-0.06, -0.02] | -0.02 [-0.03, -0.01] | - |
| probe, mean token, pretrained | -0.04 [-0.07, -0.03] | -0.04 [-0.06, -0.02] | -0.03 [-0.04, -0.02] | - |
| probe, all tokens, pretrained | -0.06 [-0.08, -0.04] | -0.06 [-0.08, -0.04] | -0.03 [-0.04, -0.02] | - |
| fine-tune from scratch | -0.04 [-0.07, -0.03] | -0.04 [-0.06, -0.02] | -0.02 [-0.04, -0.01] | - |
| fine-tune from pretrained | -0.05 [-0.08, -0.04] | -0.04 [-0.06, -0.03] | -0.03 [-0.04, -0.02] | - |

### fiq: macro R2 by label count

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | -0.10 [-0.21, -0.07] | -0.10 [-0.21, -0.07] | -0.09 [-0.19, -0.06] | -0.08 [-0.19, -0.05] |
| correlation + ridge | -0.16 [-0.28, -0.11] | -0.12 [-0.24, -0.09] | -0.07 [-0.18, -0.03] | -0.06 [-0.17, -0.01] |
| tangent + ridge | -0.11 [-0.22, -0.07] | -0.10 [-0.22, -0.06] | -0.05 [-0.16, -0.02] | -0.05 [-0.17, -0.01] |
| tangent + RBF kernel ridge | -0.11 [-0.22, -0.07] | -0.10 [-0.21, -0.06] | -0.05 [-0.16, -0.01] | -0.05 [-0.17, -0.01] |
| probe, mean token, random encoder | -0.12 [-0.24, -0.09] | -0.10 [-0.22, -0.07] | -0.09 [-0.20, -0.06] | - |
| probe, all tokens, random encoder | -0.12 [-0.24, -0.09] | -0.11 [-0.22, -0.08] | -0.09 [-0.19, -0.05] | - |
| probe, mean token, pretrained | -0.12 [-0.24, -0.08] | -0.11 [-0.23, -0.08] | -0.10 [-0.21, -0.06] | - |
| probe, all tokens, pretrained | -0.15 [-0.28, -0.11] | -0.14 [-0.26, -0.10] | -0.10 [-0.21, -0.07] | - |
| fine-tune from scratch | -0.13 [-0.25, -0.09] | -0.12 [-0.24, -0.08] | -0.09 [-0.20, -0.06] | - |
| fine-tune from pretrained | -0.14 [-0.27, -0.10] | -0.12 [-0.24, -0.08] | -0.09 [-0.20, -0.06] | - |

### fiq: all metrics at N=100

| arm | pooled R2 | macro R2 | pooled MAE | macro Spearman |
|---|---|---|---|---|
| site mean (constant) | -0.03 [-0.04, -0.02] | -0.10 [-0.21, -0.07] | 11.97 [11.23, 12.69] | 0.00 [0.00, 0.00] |
| correlation + ridge | -0.04 [-0.05, -0.02] | -0.12 [-0.24, -0.09] | 11.96 [11.24, 12.70] | 0.05 [-0.01, 0.12] |
| tangent + ridge | -0.02 [-0.04, -0.00] | -0.10 [-0.22, -0.06] | 11.89 [11.15, 12.62] | 0.06 [0.01, 0.11] |
| tangent + RBF kernel ridge | -0.02 [-0.04, -0.01] | -0.10 [-0.21, -0.06] | 11.89 [11.16, 12.61] | 0.05 [-0.00, 0.10] |
| probe, mean token, random encoder | -0.03 [-0.05, -0.02] | -0.10 [-0.22, -0.07] | 12.00 [11.26, 12.70] | 0.06 [0.01, 0.11] |
| probe, all tokens, random encoder | -0.04 [-0.06, -0.02] | -0.11 [-0.22, -0.08] | 12.01 [11.28, 12.72] | 0.06 [-0.00, 0.11] |
| probe, mean token, pretrained | -0.04 [-0.06, -0.02] | -0.11 [-0.23, -0.08] | 12.02 [11.29, 12.71] | 0.03 [-0.01, 0.08] |
| probe, all tokens, pretrained | -0.06 [-0.08, -0.04] | -0.14 [-0.26, -0.10] | 12.13 [11.41, 12.82] | 0.03 [-0.01, 0.08] |
| fine-tune from scratch | -0.04 [-0.06, -0.02] | -0.12 [-0.24, -0.08] | 12.04 [11.33, 12.77] | -0.01 [-0.05, 0.03] |
| fine-tune from pretrained | -0.04 [-0.06, -0.03] | -0.12 [-0.24, -0.08] | 12.05 [11.34, 12.78] | 0.01 [-0.04, 0.06] |
| site mean of the held-out site (oracle) | 0.06 [0.01, 0.10] | -0.00 [-0.04, -0.00] | 11.45 [10.70, 12.17] | 0.00 [0.00, 0.00] |

### fiq: transductive encoders, pooled R2

| arm | N=50 | N=100 | N=300 |
|---|---|---|---|
| probe, mean token, transductive | -0.07 [-0.10, -0.05] | -0.05 [-0.07, -0.04] | -0.03 [-0.04, -0.02] |
| probe, all tokens, transductive | -0.08 [-0.11, -0.06] | -0.06 [-0.08, -0.04] | -0.02 [-0.04, -0.02] |
| fine-tune, transductive (pretrained rate) | -0.08 [-0.11, -0.06] | -0.05 [-0.08, -0.04] | -0.03 [-0.05, -0.02] |
| fine-tune from pretrained | -0.05 [-0.08, -0.04] | -0.04 [-0.06, -0.03] | -0.03 [-0.04, -0.02] |

### fiq: 90 percent conformal intervals, pooled coverage (width)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| site mean (constant) | 0.89 (48.4) | 0.89 (48.2) | 0.89 (48.0) | 0.89 (47.7) |
| correlation + ridge | 0.88 (49.1) | 0.89 (48.3) | 0.90 (48.1) | 0.89 (47.6) |
| tangent + ridge | 0.88 (48.1) | 0.88 (47.7) | 0.88 (46.9) | 0.88 (47.1) |
| tangent + RBF kernel ridge | 0.88 (48.2) | 0.88 (47.7) | 0.88 (47.1) | 0.88 (47.1) |
| probe, mean token, random encoder | 0.89 (48.7) | 0.89 (48.1) | 0.89 (48.1) | - |
| probe, all tokens, random encoder | 0.89 (48.6) | 0.89 (47.9) | 0.89 (48.1) | - |
| probe, mean token, pretrained | 0.89 (48.8) | 0.89 (47.9) | 0.89 (48.0) | - |
| probe, all tokens, pretrained | 0.88 (49.0) | 0.88 (47.7) | 0.88 (47.8) | - |
| probe, mean token, transductive | 0.88 (48.7) | 0.88 (47.8) | 0.89 (48.1) | - |
| probe, all tokens, transductive | 0.88 (49.0) | 0.88 (47.9) | 0.89 (47.9) | - |

### fiq: 90 percent conformal intervals of the fine-tunes, pooled coverage (width)

| arm | N=50 | N=100 | N=300 | N=all |
|---|---|---|---|---|
| fine-tune from scratch | 0.89 (48.2) | 0.89 (48.4) | 0.89 (47.9) | - |
| fine-tune from pretrained | 0.88 (48.1) | 0.89 (48.4) | 0.89 (47.9) | - |
| fine-tune, transductive (pretrained rate) | 0.88 (48.5) | 0.89 (48.6) | 0.89 (47.9) | - |

### Paired differences, pooled R2

| difference | age N=50 | age N=100 | age N=300 | fiq N=50 | fiq N=100 | fiq N=300 |
|---|---|---|---|---|---|---|
| fine-tune from pretrained minus fine-tune from scratch | 0.02 [0.01, 0.03] | 0.01 [-0.00, 0.02] | -0.02 [-0.02, -0.01] | -0.01 [-0.02, -0.00] | -0.00 [-0.01, 0.00] | -0.00 [-0.01, 0.00] |
| fine-tune from pretrained minus tangent + ridge | -0.10 [-0.13, -0.09] | -0.19 [-0.22, -0.16] | -0.38 [-0.41, -0.35] | -0.03 [-0.04, -0.01] | -0.02 [-0.03, -0.01] | -0.05 [-0.07, -0.03] |
| fine-tune, transductive (pretrained rate) minus fine-tune from pretrained | -0.10 [-0.11, -0.08] | -0.06 [-0.07, -0.05] | -0.04 [-0.04, -0.03] | -0.02 [-0.03, -0.01] | -0.01 [-0.02, -0.01] | -0.00 [-0.01, 0.00] |
| fine-tune from scratch minus tangent + ridge | -0.12 [-0.14, -0.11] | -0.20 [-0.22, -0.18] | -0.36 [-0.40, -0.33] | -0.02 [-0.03, -0.01] | -0.02 [-0.03, -0.01] | -0.04 [-0.06, -0.02] |
| probe, all tokens, pretrained minus probe, all tokens, random encoder | -0.01 [-0.02, 0.00] | -0.02 [-0.03, -0.01] | -0.00 [-0.01, 0.00] | -0.01 [-0.03, -0.00] | -0.02 [-0.03, -0.01] | -0.01 [-0.01, -0.00] |
| probe, all tokens, pretrained minus tangent + ridge | -0.09 [-0.12, -0.07] | -0.18 [-0.22, -0.15] | -0.34 [-0.37, -0.31] | -0.03 [-0.05, -0.01] | -0.04 [-0.06, -0.02] | -0.05 [-0.07, -0.03] |
| probe, all tokens, transductive minus probe, all tokens, pretrained | -0.02 [-0.03, -0.01] | -0.00 [-0.02, 0.01] | -0.03 [-0.03, -0.02] | -0.03 [-0.04, -0.02] | -0.00 [-0.01, 0.01] | 0.00 [-0.00, 0.01] |
| probe, mean token, pretrained minus probe, mean token, random encoder | 0.03 [0.01, 0.05] | 0.00 [-0.01, 0.01] | 0.00 [-0.00, 0.00] | -0.00 [-0.01, 0.01] | -0.00 [-0.01, 0.00] | -0.00 [-0.01, 0.00] |
| probe, mean token, pretrained minus tangent + ridge | -0.09 [-0.12, -0.06] | -0.19 [-0.22, -0.16] | -0.34 [-0.37, -0.30] | -0.02 [-0.03, -0.00] | -0.02 [-0.03, -0.00] | -0.05 [-0.07, -0.03] |
| tangent + RBF kernel ridge minus tangent + ridge | -0.02 [-0.02, -0.01] | -0.03 [-0.04, -0.02] | -0.04 [-0.05, -0.04] | -0.00 [-0.00, 0.00] | 0.00 [-0.00, 0.00] | -0.00 [-0.00, 0.00] |
| tangent + ridge minus correlation + ridge | 0.03 [0.01, 0.05] | 0.05 [0.03, 0.08] | 0.03 [-0.00, 0.07] | 0.04 [0.02, 0.05] | 0.02 [0.01, 0.03] | 0.03 [0.01, 0.05] |

### Paired differences, macro R2

| difference | age N=50 | age N=100 | age N=300 | fiq N=50 | fiq N=100 | fiq N=300 |
|---|---|---|---|---|---|---|
| fine-tune from pretrained minus fine-tune from scratch | 0.09 [0.06, 0.15] | 0.04 [0.02, 0.07] | -0.07 [-0.10, -0.05] | -0.01 [-0.02, 0.00] | -0.00 [-0.01, 0.01] | -0.00 [-0.01, 0.00] |
| fine-tune from pretrained minus tangent + ridge | -0.27 [-0.36, -0.21] | -0.50 [-0.63, -0.41] | -0.91 [-1.13, -0.78] | -0.03 [-0.05, -0.02] | -0.01 [-0.03, 0.01] | -0.04 [-0.07, -0.01] |
| fine-tune, transductive (pretrained rate) minus fine-tune from pretrained | -0.24 [-0.32, -0.19] | -0.15 [-0.21, -0.12] | -0.04 [-0.08, -0.01] | -0.03 [-0.05, -0.02] | -0.03 [-0.05, -0.01] | -0.01 [-0.02, 0.00] |
| fine-tune from scratch minus tangent + ridge | -0.37 [-0.47, -0.30] | -0.54 [-0.69, -0.45] | -0.84 [-1.05, -0.72] | -0.02 [-0.04, -0.01] | -0.01 [-0.03, 0.01] | -0.04 [-0.07, -0.01] |
| probe, all tokens, pretrained minus probe, all tokens, random encoder | 0.00 [-0.05, 0.06] | -0.04 [-0.08, 0.00] | -0.00 [-0.01, 0.01] | -0.03 [-0.06, -0.01] | -0.03 [-0.05, -0.02] | -0.01 [-0.02, -0.00] |
| probe, all tokens, pretrained minus tangent + ridge | -0.28 [-0.38, -0.20] | -0.55 [-0.70, -0.46] | -0.80 [-0.99, -0.69] | -0.05 [-0.08, -0.03] | -0.04 [-0.06, -0.01] | -0.04 [-0.08, -0.01] |
| probe, all tokens, transductive minus probe, all tokens, pretrained | -0.28 [-0.41, -0.20] | -0.03 [-0.08, 0.00] | -0.02 [-0.03, -0.01] | -0.05 [-0.08, -0.03] | -0.01 [-0.02, 0.00] | 0.00 [-0.01, 0.01] |
| probe, mean token, pretrained minus probe, mean token, random encoder | 0.10 [0.06, 0.17] | 0.08 [0.02, 0.16] | -0.00 [-0.01, 0.00] | -0.00 [-0.01, 0.01] | -0.01 [-0.02, 0.00] | -0.01 [-0.01, -0.00] |
| probe, mean token, pretrained minus tangent + ridge | -0.23 [-0.32, -0.16] | -0.58 [-0.73, -0.49] | -0.80 [-1.00, -0.69] | -0.01 [-0.04, 0.00] | -0.01 [-0.03, 0.02] | -0.04 [-0.08, -0.01] |
| tangent + RBF kernel ridge minus tangent + ridge | -0.02 [-0.03, -0.00] | -0.07 [-0.09, -0.05] | -0.07 [-0.09, -0.04] | -0.00 [-0.00, 0.00] | 0.00 [-0.00, 0.01] | 0.00 [-0.00, 0.01] |
| tangent + ridge minus correlation + ridge | 0.27 [0.20, 0.39] | 0.55 [0.42, 0.80] | 0.48 [0.31, 0.77] | 0.05 [0.03, 0.08] | 0.02 [0.00, 0.04] | 0.02 [-0.01, 0.05] |

### Sites won on MAE (seed mean per site)

| pair | age N=50 | age N=100 | age N=300 | fiq N=50 | fiq N=100 | fiq N=300 |
|---|---|---|---|---|---|---|
| fine-tune from pretrained beats fine-tune from scratch | 6/8 | 5/8 | 3/8 | 3/8 | 4/8 | 3/8 |
| fine-tune from pretrained beats tangent + ridge | 0/8 | 0/8 | 0/8 | 2/8 | 2/8 | 1/8 |
| fine-tune, transductive (pretrained rate) beats fine-tune from pretrained | 1/8 | 2/8 | 3/8 | 2/8 | 2/8 | 3/8 |
| fine-tune from scratch beats tangent + ridge | 0/8 | 0/8 | 0/8 | 2/8 | 1/8 | 1/8 |
| fine-tune from scratch, 1000 steps beats fine-tune from scratch | - | - | 2/8 | - | - | - |
| fine-tune from transductive encoder, 1000 steps beats fine-tune, transductive (pretrained rate) | - | - | 2/8 | - | - | - |
| fine-tune from transductive encoder, 1000 steps beats fine-tune from scratch, 1000 steps | - | - | 3/8 | - | - | - |
| fine-tune from transductive encoder, 1000 steps beats tangent + ridge | - | - | 0/8 | - | - | - |
| probe, all tokens, pretrained beats probe, all tokens, random encoder | 2/8 | 2/8 | 3/8 | 3/8 | 2/8 | 3/8 |
| probe, all tokens, pretrained beats tangent + ridge | 0/8 | 0/8 | 0/8 | 1/8 | 1/8 | 1/8 |
| probe, all tokens, transductive beats probe, all tokens, pretrained | 2/8 | 3/8 | 2/8 | 2/8 | 3/8 | 3/8 |
| probe, mean token, pretrained beats probe, mean token, random encoder | 5/8 | 3/8 | 2/8 | 3/8 | 3/8 | 3/8 |
| probe, mean token, pretrained beats tangent + ridge | 2/8 | 0/8 | 0/8 | 3/8 | 2/8 | 1/8 |
| tangent + RBF kernel ridge beats tangent + ridge | 1/8 | 0/8 | 1/8 | 4/8 | 2/8 | 3/8 |
| tangent + ridge beats correlation + ridge | 7/8 | 6/8 | 5/8 | 8/8 | 5/8 | 6/8 |

### Longer fine-tuning, age, N=300

| arm | pooled R2 | macro R2 | pooled MAE | macro Spearman |
|---|---|---|---|---|
| site mean (constant) | -0.05 [-0.06, -0.05] | -0.90 [-1.20, -0.76] | 5.51 [5.16, 5.90] | 0.00 [0.00, 0.00] |
| tangent + ridge | 0.28 [0.24, 0.31] | -0.11 [-0.28, -0.02] | 4.42 [4.13, 4.73] | 0.57 [0.51, 0.62] |
| fine-tune from scratch | -0.08 [-0.10, -0.07] | -0.95 [-1.26, -0.80] | 5.56 [5.21, 5.95] | -0.03 [-0.09, 0.02] |
| fine-tune, transductive (pretrained rate) | -0.14 [-0.16, -0.12] | -1.06 [-1.39, -0.90] | 5.67 [5.32, 6.08] | -0.05 [-0.10, 0.01] |
| fine-tune from scratch, 1000 steps | -0.26 [-0.31, -0.23] | -1.57 [-2.09, -1.33] | 5.97 [5.60, 6.38] | -0.07 [-0.14, -0.00] |
| fine-tune from transductive encoder, 1000 steps | -0.38 [-0.44, -0.33] | -2.54 [-3.49, -2.10] | 6.30 [5.93, 6.70] | -0.06 [-0.12, 0.01] |

### Longer fine-tuning, paired differences, age, N=300

| difference (age) | pooled R2 | macro R2 | pooled MAE | macro Spearman |
|---|---|---|---|---|
| fine-tune from scratch, 1000 steps minus fine-tune from scratch | -0.18 [-0.21, -0.16] | -0.62 [-0.85, -0.51] | 0.41 [0.35, 0.47] | -0.04 [-0.10, 0.03] |
| fine-tune from transductive encoder, 1000 steps minus fine-tune, transductive (pretrained rate) | -0.25 [-0.29, -0.21] | -1.48 [-2.17, -1.14] | 0.63 [0.54, 0.72] | -0.02 [-0.07, 0.05] |
| fine-tune from transductive encoder, 1000 steps minus fine-tune from scratch, 1000 steps | -0.12 [-0.15, -0.09] | -0.96 [-1.44, -0.71] | 0.33 [0.25, 0.41] | 0.01 [-0.05, 0.07] |
| fine-tune from transductive encoder, 1000 steps minus tangent + ridge | -0.66 [-0.73, -0.60] | -2.43 [-3.29, -2.04] | 1.88 [1.69, 2.06] | -0.63 [-0.70, -0.54] |

### In-site reference versus leave-one-site-out, N=100, pooled R2

| arm | age LOSO | age in-site | fiq LOSO | fiq in-site |
|---|---|---|---|---|
| site mean (constant) | -0.06 [-0.08, -0.06] | -0.01 [-0.03, -0.01] | -0.03 [-0.04, -0.02] | -0.01 [-0.03, -0.00] |
| correlation + ridge | 0.04 [0.01, 0.08] | 0.22 [0.18, 0.25] | -0.04 [-0.05, -0.02] | -0.04 [-0.06, -0.02] |
| tangent + ridge | 0.10 [0.07, 0.13] | 0.22 [0.19, 0.24] | -0.02 [-0.04, -0.00] | -0.00 [-0.02, 0.01] |
| tangent + RBF kernel ridge | 0.07 [0.04, 0.09] | 0.18 [0.16, 0.20] | -0.02 [-0.04, -0.01] | -0.00 [-0.02, 0.01] |
| site mean of the labelled subjects of the site | - | 0.20 [0.10, 0.29] | - | -0.06 [-0.12, -0.01] |
| probe, mean token, random encoder | -0.09 [-0.11, -0.08] | - | -0.03 [-0.05, -0.02] | - |
| probe, all tokens, random encoder | -0.07 [-0.09, -0.06] | - | -0.04 [-0.06, -0.02] | - |
| probe, mean token, pretrained | -0.09 [-0.11, -0.08] | - | -0.04 [-0.06, -0.02] | - |
| probe, all tokens, pretrained | -0.09 [-0.10, -0.08] | - | -0.06 [-0.08, -0.04] | - |
| fine-tune from scratch | -0.10 [-0.12, -0.09] | - | -0.04 [-0.06, -0.02] | - |
| fine-tune from pretrained | -0.09 [-0.11, -0.08] | - | -0.04 [-0.06, -0.03] | - |

### Fine-tuning length on NYU, N=300, age

| init | lr | steps | train R2 | train Spearman | held-out R2 | held-out Spearman |
|---|---|---|---|---|---|---|
| scratch | 0.0001 | 300 | 0.021 | 0.18 | -0.054 | 0.03 |
| scratch | 0.0001 | 1500 | 0.071 | 0.33 | -0.024 | 0.08 |
| scratch | 0.0003 | 300 | 0.007 | 0.13 | -0.052 | 0.07 |
| scratch | 0.0003 | 1500 | 0.104 | 0.35 | -0.013 | 0.04 |
| transductive pretrained | 0.0001 | 300 | 0.018 | 0.24 | -0.052 | 0.12 |
| transductive pretrained | 0.0001 | 1500 | 0.139 | 0.42 | 0.013 | 0.14 |
| transductive pretrained | 0.0003 | 300 | 0.012 | 0.24 | -0.050 | 0.10 |
| transductive pretrained | 0.0003 | 1500 | 0.307 | 0.54 | 0.018 | 0.14 |

### Masked pretraining, ratio to the zero predictor

| fold | train loss | val ratio | held-out ratio | seconds |
|---|---|---|---|---|
| loso-MAX_MUN | 0.504 | 0.504 | 0.525 | 234 |
| loso-NYU | 0.501 | 0.493 | 0.519 | 162 |
| loso-PITT | 0.508 | 0.480 | 0.447 | 233 |
| loso-TRINITY | 0.500 | 0.507 | 0.445 | 234 |
| loso-UCLA_1 | 0.496 | 0.492 | 0.505 | 161 |
| loso-UM_1 | 0.527 | 0.520 | 0.534 | 162 |
| loso-USM | 0.503 | 0.489 | 0.488 | 160 |
| loso-YALE | 0.503 | 0.472 | 0.468 | 235 |
