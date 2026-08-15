# Synthetic-example result map

Run the commands below from `final_report_experiments/`. Outputs are written to
the `deterministic/result/` and `stochastic/result/` directories.

| Result | Run file(s), in order | Canonical output |
| --- | --- | --- |
| Deterministic table, `n=100` | `python deterministic/run_scholtes_same_init_compare.py --n 100 --device cpu`<br>`python deterministic/run_final_deterministic.py --device cpu --dims 100 --steps 1000` | `deterministic/result/final_report_table_n100.tex` |
| Deterministic table, `n=1000` | `python deterministic/run_scholtes_same_init_compare.py --n 1000 --device cpu`<br>`python deterministic/run_final_deterministic.py --device cpu --dims 1000 --steps 1000` | `deterministic/result/final_report_table_n1000.tex` |
| Deterministic convergence figure | `python deterministic/run_final_deterministic.py --device cpu --dims 100 --steps 1000`<br>`python deterministic/plot_n100_iteration_losses.py` | `deterministic/result/n100_upper_lower_loss_iteration.png` |
| Stochastic convergence figure | `python stochastic/compare_vr_noise_std_levels.py --device cpu` | `stochastic/result/variance_compare_alpha0p1_beta0p1_it1000_iteration_ticks_1e0_1e4_1e8_1e12.png` |
| Stochastic hyperparameter-ablation table | `python stochastic/ablation_variance_reduction.py --device cpu` | Timestamped `stochastic/result/vr_ablation_*.csv` |
| AdaProx parameter tuning | `python deterministic/adaprox_tuning/run_adaprox_grid_and_compare.py ...` | `deterministic/adaprox_tuning/results/*.csv` |

The staged AdaProx tuning commands are recorded in
`deterministic/adaprox_tuning/README.md`.
