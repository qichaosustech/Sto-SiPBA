# AdaProx Parameter Search Record

This file records the parameter grids searched for AdaProx-SG and AdaProx-PD.
The original `README.md` is kept unchanged.

## Common experiment settings

```text
n = 100
K = 20
T = 50
K * T = 1000 iterations
N_inner = 20
base_seed = 42
search_num_seeds = 1
```

The Stage 1 search used the exact inner solution. Stage 2 and the PD local and
micro searches used the iterative inner solution:

```text
exact_inner = False
N_inner = 20
```

The lower-objective definitions used by the recorded searches were:

```text
Stage 1: LOWER_SCALE = 1, lower objective not divided by n
Stage 2: LOWER_SCALE = 1, lower objective divided by n
PD local: LOWER_SCALE = 1, lower objective divided by n
PD micro: LOWER_SCALE = 1, lower objective divided by n
```

## Stage 1: coarse upper-loss search

Ranking metric:

```text
final_upper_mean
```

AdaProx-SG grid:

```text
alpha = 0.001
xi = 0.0001, 0.001, 0.01, 0.1
sigma = 0.001, 0.01, 0.1
beta = 0.001, 0.01, 0.1
gamma0 = 1, 10, 100, 1000
threshold_scale = 1
constraint_step_interval = 0
```

Number of SG configurations:

```text
1 * 4 * 3 * 3 * 4 * 1 * 1 = 144
```

AdaProx-PD grid:

```text
alpha = 0.001
xi = 0.0001, 0.001, 0.01, 0.1
sigma = 0.001, 0.01, 0.1
beta = 0.001, 0.01, 0.1
primal_lr = 0.0001, 0.001, 0.01, 0.1
dual_lr = 0.0001, 0.001, 0.01, 0.1
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

Number of PD configurations:

```text
1 * 4 * 3 * 3 * 4 * 4 * 1 * 1 * 1 = 576
```

Output files:

```text
results/adaprox_upper_loss_grid_6proc_sg_grid.csv
results/adaprox_upper_loss_grid_6proc_pd_grid.csv
```

## Stage 2: refined total-loss search

Ranking metric:

```text
final_total_mean
```

AdaProx-SG grid:

```text
alpha = 0.001
xi = 0.0001, 0.001
sigma = 0.001, 0.01
beta = 0.0001, 0.001, 0.01
gamma0 = 30, 100, 300, 1000
threshold_scale = 1
constraint_step_interval = 10
```

Number of SG configurations:

```text
1 * 2 * 2 * 3 * 4 * 1 * 1 = 48
```

AdaProx-PD grid:

```text
alpha = 0.001
xi = 0.0001, 0.001
sigma = 0.001, 0.01
beta = 0.0001, 0.001, 0.01
primal_lr = 0.001, 0.003, 0.01
dual_lr = 0.001, 0.003, 0.01
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

Number of PD configurations:

```text
1 * 2 * 2 * 3 * 3 * 3 * 1 * 1 * 1 = 108
```

Output files:

```text
results/adaprox_lowerdivn_total_grid_small_iterinner_N20_sg_grid.csv
results/adaprox_lowerdivn_total_grid_small_iterinner_N20_pd_grid.csv
```

## Stage 3: PD local search

Ranking metric:

```text
final_total_mean
```

Fixed parameters and grid:

```text
alpha = 0.001
xi = 0.0001
sigma = 0.01
beta = 0.01
primal_lr = 0.0001, 0.0003, 0.0006, 0.001, 0.0015
dual_lr = 0.0003, 0.0006, 0.001, 0.0015
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

Number of PD configurations:

```text
5 * 4 = 20
```

Output file:

```text
results/adaprox_lowerdivn_pd_local_iterinner_N20_pd_grid.csv
```

## Stage 4: PD micro search

Ranking metric:

```text
final_total_mean
```

Fixed parameters and grid:

```text
alpha = 0.001
xi = 0.0001
sigma = 0.01
beta = 0.01
primal_lr = 0.0002, 0.0003, 0.0004, 0.0005
dual_lr = 0.0015, 0.002, 0.003
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

Number of PD configurations:

```text
4 * 3 = 12
```

Output file:

```text
results/adaprox_lowerdivn_pd_micro_iterinner_N20_pd_grid.csv
```

## Search totals

```text
AdaProx-SG configurations: 144 + 48 = 192
AdaProx-PD configurations: 576 + 108 + 20 + 12 = 716
All searched configurations: 192 + 716 = 908
```
