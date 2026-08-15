# AdaProx Tuning Notes for the Final Report

This folder records the tuning process used to choose the AdaProx-PD and
AdaProx-SG parameters used by the deterministic final-report experiments.

The tuning code is:

```text
run_adaprox_grid_and_compare.py
```

It imports the final deterministic implementation modules from the parent
folder:

```text
../AdaProx_common.py
../AdaProx_PD.py
../AdaProx_SG.py
../SiPBA.py
```

The final fixed parameters selected from this tuning process are used in:

```text
../run_final_deterministic.py
```

## 1. Tuning Target

The tuning was done for the deterministic pessimistic bilevel toy example. The
losses reported in the final table are:

```math
\mathrm{UpperLoss}
=
\frac{1}{n}\|x-x^\star\|^2,
\qquad x^\star=0.5\mathbf{1},
```

```math
\mathrm{LowerLoss}
=
\frac{1}{n}
\left\|
y-\frac{\|x\|^2}{n}\mathbf{1}
\right\|^2.
```

The total loss used during several tuning runs is:

```math
\mathrm{TotalLoss}=\mathrm{UpperLoss}+\mathrm{LowerLoss}.
```

The final implementation uses the same lower-level scaling as the final
deterministic report scripts:

```text
LOWER_SCALE = 10
```

AdaProx was evaluated with a fixed computational budget:

```text
K = 20
T = 50
K * T = 1000 total AdaProx inner iterations
n = 100 during tuning
base_seed = 42 during grid search
```

After choosing parameters, the final report runner evaluates both `n=100` and
`n=1000` with 10 repeated seeds:

```text
seeds = 42, 43, ..., 51
steps = 1000
T = 50
K = steps / T = 20
```

## 2. Why `N_inner=20` Was Used

AdaProx needs an approximate lower-level regularized solution
`\hat y(x)`. The code can either use an exact closed form or perform projected
gradient steps. To keep the comparison closer to an iterative first-order
setting, the final report uses:

```text
exact_inner = False
N_inner = 20
```

This is the setting used by `../run_final_deterministic.py`.

## 3. Search Script

The main tuning script does three things:

```text
1. Build an AdaProx-SG parameter grid.
2. Build an AdaProx-PD parameter grid.
3. Run each grid in parallel, save CSV files, then compare SiPBA with the best
   SG and PD configurations on the same seed sequence.
```

The script uses:

```python
ProcessPoolExecutor
torch.set_num_threads(1)
```

Each completed parameter configuration is written immediately to CSV. This was
useful because the searches were long enough that intermediate results needed
to survive interruption.

## 4. Stage 1: Coarse Upper-Loss Search

The first pass was a coarse search mainly to locate reasonable AdaProx ranges.
It used:

```text
search_metric = final_upper_mean
n = 100
K = 20
T = 50
search_num_seeds = 1
base_seed = 42
```

AdaProx-SG grid:

```text
gamma0 = 1, 10, 100, 1000
threshold_scale = 1
beta = 0.001, 0.01, 0.1
sigma = 0.001, 0.01, 0.1
xi = 0.0001, 0.001, 0.01, 0.1
alpha = 0.001
```

AdaProx-PD grid:

```text
primal_lr = 0.0001, 0.001, 0.01, 0.1
dual_lr = 0.0001, 0.001, 0.01, 0.1
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
beta = 0.001, 0.01, 0.1
sigma = 0.001, 0.01, 0.1
xi = 0.0001, 0.001, 0.01, 0.1
alpha = 0.001
```

The saved CSV snapshots are:

```text
results/adaprox_upper_loss_grid_6proc_sg_grid.csv
results/adaprox_upper_loss_grid_6proc_pd_grid.csv
```

This stage showed an important issue: sorting only by final upper loss can pick
PD parameters that drive the upper loss down while leaving lower loss and
constraint violation large. Therefore the later tuning did not rely only on
upper loss.

Example command for this stage:

```bash
cd final_report_experiments/deterministic/adaprox_tuning
python run_adaprox_grid_and_compare.py \
  --workers 6 \
  --device cpu \
  --search-metric final_upper_mean \
  --run-name adaprox_upper_loss_grid_6proc \
  --n 100 --K 20 --T 50 --N-inner 20 \
  --search-num-seeds 1 \
  --compare-num-seeds 10 \
  --compare-iterations 1000 \
  --compare-log-every 50 \
  --sg-gamma0-grid 1,10,100,1000 \
  --sg-threshold-scale-grid 1 \
  --sg-constraint-step-interval-grid 0 \
  --pd-primal-lr-grid 0.0001,0.001,0.01,0.1 \
  --pd-dual-lr-grid 0.0001,0.001,0.01,0.1 \
  --pd-lambda-max-grid 100 \
  --pd-dual-momentum-grid 0 \
  --pd-reset-lambda-grid false \
  --beta-grid 0.001,0.01,0.1 \
  --sigma-grid 0.001,0.01,0.1 \
  --xi-grid 0.0001,0.001,0.01,0.1 \
  --alpha-grid 0.001
```

## 5. Stage 2: Current Lower-Scale and Iterative-Inner Search

After switching to the final `LOWER_SCALE=10` setting and using iterative
inner solves, the grid was narrowed. The main ranking criterion was:

```text
search_metric = final_total_mean
```

The following diagnostics were checked together:

```text
final_upper_mean
final_lower_mean
final_total_mean
final_max_constraint_mean
```

AdaProx-SG refined grid:

```text
gamma0 = 30, 100, 300, 1000
threshold_scale = 1
constraint_step_interval = 10
beta = 0.0001, 0.001, 0.01
sigma = 0.001, 0.01
xi = 0.0001, 0.001
alpha = 0.001
```

AdaProx-PD refined grid:

```text
primal_lr = 0.001, 0.003, 0.01
dual_lr = 0.001, 0.003, 0.01
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
beta = 0.0001, 0.001, 0.01
sigma = 0.001, 0.01
xi = 0.0001, 0.001
alpha = 0.001
```

The saved CSV snapshots are:

```text
results/adaprox_lowerdivn_total_grid_small_iterinner_N20_sg_grid.csv
results/adaprox_lowerdivn_total_grid_small_iterinner_N20_pd_grid.csv
```

Example command:

```bash
cd final_report_experiments/deterministic/adaprox_tuning
python run_adaprox_grid_and_compare.py \
  --workers 6 \
  --device cpu \
  --search-metric final_total_mean \
  --run-name adaprox_lowerdivn_total_grid_small_iterinner_N20 \
  --n 100 --K 20 --T 50 --N-inner 20 \
  --no-exact-inner \
  --search-num-seeds 1 \
  --compare-num-seeds 10 \
  --compare-iterations 1000 \
  --compare-log-every 50 \
  --sg-gamma0-grid 30,100,300,1000 \
  --sg-threshold-scale-grid 1 \
  --sg-constraint-step-interval-grid 10 \
  --pd-primal-lr-grid 0.001,0.003,0.01 \
  --pd-dual-lr-grid 0.001,0.003,0.01 \
  --pd-lambda-max-grid 100 \
  --pd-dual-momentum-grid 0 \
  --pd-reset-lambda-grid false \
  --beta-grid 0.0001,0.001,0.01 \
  --sigma-grid 0.001,0.01 \
  --xi-grid 0.0001,0.001 \
  --alpha-grid 0.001
```

## 6. Stage 3: PD Local and Micro Refinement

PD was more sensitive to `primal_lr` and `dual_lr`, so an additional local
refinement was run around the better Stage 2 region.

Local PD grid:

```text
primal_lr = 0.0001, 0.0003, 0.0006, 0.001, 0.0015
dual_lr = 0.0003, 0.0006, 0.001, 0.0015
alpha = 0.001
xi = 0.0001
sigma = 0.01
beta = 0.01
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

Micro PD grid:

```text
primal_lr = 0.0002, 0.0003, 0.0004, 0.0005
dual_lr = 0.0015, 0.002, 0.003
alpha = 0.001
xi = 0.0001
sigma = 0.01
beta = 0.01
lambda_max = 100
dual_momentum = 0
reset_lambda_per_outer = False
```

The saved CSV snapshots are:

```text
results/adaprox_lowerdivn_pd_local_iterinner_N20_pd_grid.csv
results/adaprox_lowerdivn_pd_micro_iterinner_N20_pd_grid.csv
```

The micro refinement selected:

```text
primal_lr = 0.0004
dual_lr = 0.003
```

with the shared PD settings listed above.

## 7. Final Parameters Used in the Report

The final AdaProx-SG parameters are:

```python
BEST_SG = {
    "alpha": 0.001,
    "xi": 0.0001,
    "sigma": 0.01,
    "beta": 0.01,
    "gamma0": 300.0,
    "threshold_scale": 1.0,
    "constraint_step_interval": 10,
}
```

The final AdaProx-PD parameters are:

```python
BEST_PD = {
    "alpha": 0.001,
    "xi": 0.0001,
    "sigma": 0.01,
    "beta": 0.01,
    "primal_lr": 0.0004,
    "dual_lr": 0.003,
    "lambda_max": 100.0,
    "dual_momentum": 0.0,
    "reset_lambda_per_outer": False,
}
```

These dictionaries are fixed in:

```text
../run_final_deterministic.py
```

The final deterministic runner evaluates the fixed AdaProx parameters together
with SiPBA:

```bash
cd final_report_experiments/deterministic
python run_final_deterministic.py --device cpu
```

## 8. Output Files Produced by Final Runner

The final report runner saves the gradient-method results as:

```text
../result/final_gradient_n100_it1000_comparison.pt
../result/final_gradient_n100_it1000_summary.csv
../result/final_gradient_n1000_it1000_comparison.pt
../result/final_gradient_n1000_it1000_summary.csv
```

If Scholtes CSV files are present, `../run_final_deterministic.py` also builds
the final five-method LaTeX tables.

## 9. Reading the CSV Snapshots

Each CSV row is one parameter configuration. Important columns:

```text
alpha, xi, sigma, beta
gamma0, threshold_scale, constraint_step_interval       # SG only
primal_lr, dual_lr, lambda_max, dual_momentum           # PD only
final_upper_mean
final_lower_mean
final_total_mean
final_max_constraint_mean
time_mean
```

For the final report, the CSVs are used as tuning evidence. The actual final
numbers in the paper/table should be read from the final runner outputs, not
from the single-seed search CSVs.
