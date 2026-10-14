# Frozen Transformer: bias and neighbor-use diagnosis

Run `PYTHONPATH=src python scripts/diagnose_transformer.py` from the repository root
with the deep-learning dependencies installed and the six 2022–2024 source CSVs available.
No retraining, checkpoint changes, or 2025 test access. All errors are in degrees Celsius.

## Findings

1. **The illustrated outage has no neighbor observations at all.** In the entire
128-hour example window, all four channels at station 41010 are missing. Removing
or perturbing observed neighbor temperatures therefore has exactly zero effect.
The hidden target mean is 21.9354°C and the model mean is 22.4212°C (bias +0.4858°C).
This case cannot demonstrate benefit from real neighbor measurements. The network
still receives the neighbor's learned embedding and missing-value mask.
2. **Time ordering is weakly used in the present checkpoint.** Shuffling observed
local water temperatures among their observed timestamps (holding the gap and masks
fixed) changes 24-hour predictions by only 0.0419°C on average, with MAE 0.7095°C
versus 0.7209°C originally. This suggests reliance on broad window-level conditions
rather than precise local evolution. It is evidence from an intervention, not proof
of one unique training failure. In the example this shuffle reduces bias to +0.3633°C.
3. **Neighbor values influence the network, but a net accuracy benefit is not established.**
Raising observed neighbor water temperatures by 1°C raises 24-hour predictions by
0.0778°C on average; raising local observed water temperatures by 1°C raises them
by 0.6183°C. Swapping complete neighbor windows worsens errors, but removing the
neighbor improves errors on these validation windows. Using an input does not mean
using it optimally. These are frozen-input interventions; they do not establish how
well a separately trained single-station model would perform.
4. **Bias varies by season rather than being one constant offset.** Mean window
bias in February is +0.9479°C; May is -0.4954°C and December -0.5380°C. Training-window
MAE is lower than validation MAE, so generalization is also a concern. The training
subset is the first 256 originally sampled fitting windows, not an unseen split.
5. **Basic data-path checks passed.** Source hashes match the training manifest;
normalization is exactly reproduced from 2022–2023; masked target values and their
observation flags remain zero in every intervention. There is no evidence here of
a normalization mismatch. This is not an exhaustive software audit.

## All 256 validation windows: macro averages over the four gap lengths

| variant | mae | bias | absolute_prediction_change |
| --- | --- | --- | --- |
| neighbor_removed | 0.6747 | 0.2594 | 0.1328 |
| neighbor_time_shuffled | 0.7103 | 0.1562 | 0.0474 |
| neighbor_window_swapped | 0.7667 | 0.3151 | 0.3309 |
| neighbor_wtmp_minus1 | 0.7179 | 0.0652 | 0.0924 |
| neighbor_wtmp_plus1 | 0.7050 | 0.2505 | 0.0929 |
| neighbor_wtmp_removed | 0.6748 | 0.1376 | 0.1316 |
| original | 0.7068 | 0.1576 | 0.0000 |
| target_context_minus1 | 0.8281 | -0.4584 | 0.6160 |
| target_context_plus1 | 0.9745 | 0.8026 | 0.6451 |
| target_observed_time_shuffled | 0.7021 | 0.1596 | 0.0480 |
| target_wtmp_removed | 2.0837 | 1.4253 | 1.9278 |

## Windows with at least 90% observed neighbor water temperature

165 of 256 windows qualify. This table averages windows; the gap mix may differ.

| variant | mae | bias | absolute_prediction_change |
| --- | --- | --- | --- |
| neighbor_removed | 0.5639 | 0.1496 | 0.2046 |
| neighbor_time_shuffled | 0.6183 | -0.0082 | 0.0726 |
| neighbor_window_swapped | 0.6241 | 0.1480 | 0.3607 |
| neighbor_wtmp_minus1 | 0.6289 | -0.1488 | 0.1418 |
| neighbor_wtmp_plus1 | 0.6115 | 0.1355 | 0.1425 |
| neighbor_wtmp_removed | 0.5640 | -0.0391 | 0.2030 |
| original | 0.6128 | -0.0070 | 0.0000 |
| target_context_minus1 | 0.8444 | -0.6962 | 0.6892 |
| target_context_plus1 | 0.8225 | 0.6527 | 0.6596 |
| target_observed_time_shuffled | 0.6061 | -0.0014 | 0.0490 |
| target_wtmp_removed | 0.9899 | -0.0315 | 0.9995 |

## Training fit, evaluated with dropout disabled

| gap_hours | mae | bias |
| --- | --- | --- |
| 1 | 0.4340 | -0.1112 |
| 6 | 0.4887 | -0.1524 |
| 24 | 0.3933 | -0.0784 |
| 72 | 0.3865 | -0.0776 |

## Likely contributors and next controlled experiment

The current network predicts absolute standardized temperature without a local
endpoint anchor; its learned positional embedding and temporal attention have not
produced strong sensitivity to local temperature ordering. The loss averages over
hidden points, so a 72-hour gap contributes far more labels than a 1-hour gap.
Masks are fixed across epochs, and training lasts 15 epochs. These are design facts
and plausible contributors, not experimentally isolated causes of the warm bias.
A missing station still produces learned tokens and participates in spatial attention.

Next, compare models trained with and without neighbor inputs under the same seeds
and masking schedule; explicitly gate unavailable stations. Test window-relative
normalization and balanced per-window loss one at a time. Use validation for these
choices. Do not claim improvement or inspect the test year during diagnosis.

Windows overlap, so these are descriptive paired results without independence-based
significance claims. Perturbations can create inputs unlike training data. 2024 was
already used for checkpoint selection. Detailed per-gap, per-window, monthly,
availability and example outputs are included next to this report.
