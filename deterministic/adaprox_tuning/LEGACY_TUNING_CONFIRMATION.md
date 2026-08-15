# Legacy AdaProx Tuning Confirmation

The legacy searches were rerun with their original lower-objective definitions.
Existing result files were not overwritten; new files use `legacy_` prefixes.

## Definitions

```text
Stage 1 coarse search:
  LOWER_SCALE = 1
  lower_divide_by_n = False
  exact_inner = True

Stage 2 refined search:
  LOWER_SCALE = 1
  lower_divide_by_n = True
  exact_inner = False
  N_inner = 20

PD local and micro searches:
  LOWER_SCALE = 1
  lower_divide_by_n = True
  exact_inner = False
  N_inner = 20
```

## Exact comparison with the old CSV files

Timing columns were excluded because they depend on machine load. Configuration
sets and all other shared columns were compared exactly, without numerical
tolerance.

| Search | Configurations | Non-time differences |
| --- | ---: | ---: |
| Stage 1 SG | 144 | 0 |
| Stage 1 PD | 576 | 0 |
| Stage 2 SG | 48 | 0 |
| Stage 2 PD | 108 | 0 |
| PD local | 20 | 0 |
| PD micro | 12 | 0 |
| Total | 908 | 0 |

All new grid rows have `status=ok`.

## Final parameter check

The final AdaProx-PD setting is the first-ranked configuration in the legacy
micro search:

```text
primal_lr = 0.0004
dual_lr = 0.003
final_total_mean = 0.4087597727775574
rank = 1 / 12
```

The final AdaProx-SG setting was evaluated in the legacy Stage 2 refined grid:

```text
gamma0 = 300
xi = 0.0001
sigma = 0.01
beta = 0.01
constraint_step_interval = 10
final_total_mean = 0.6016872525215149
```

Six configurations have a strictly smaller total loss. Six configurations,
including the final SG setting, share exactly the same value, so the final SG
setting occupies positions 7--12 depending on tie order. The minimum total loss
is `0.6016844511032104`, only about `2.8e-6` below the final SG setting.

## New output prefixes

```text
results/legacy_stage1_nodivn_upper_*
results/legacy_scale1_stage2_total_iterinner_N20_*
results/legacy_scale1_pd_local_iterinner_N20_*
results/legacy_scale1_pd_micro_iterinner_N20_*
legacy_scale1_logs/
```
