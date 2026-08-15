from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

import AdaProx_common as adaprox_common_module
import SiPBA as sipba_module
from AdaProx_PD import run_one_experiment as run_pd
from AdaProx_SG import run_one_experiment as run_sg
from AdaProx_common import AdaProxParams, build_summary, stack_and_trim
from SiPBA import LOWER_DIVIDE_BY_N as DEFAULT_LOWER_DIVIDE_BY_N
from SiPBA import LOWER_SCALE as DEFAULT_LOWER_SCALE
from SiPBA import run_one_experiment as run_sipba


def set_lower_definition(value: float, divide_by_n: bool) -> None:
    if value <= 0.0:
        raise ValueError(f"lower_scale must be positive, got {value}")
    sipba_module.LOWER_SCALE = float(value)
    sipba_module.LOWER_DIVIDE_BY_N = bool(divide_by_n)
    adaprox_common_module.LOWER_SCALE = float(value)


def parse_float_list(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.replace("，", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def parse_bool_list(text: str) -> List[bool]:
    values: List[bool] = []
    for part in text.replace("，", ",").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "yes", "y"}:
            values.append(True)
        elif token in {"0", "false", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"invalid boolean token: {part}")
    if not values:
        raise ValueError("empty bool list")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.replace("，", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("empty int list")
    return values


def tail_mean(values: Sequence[float], tail_fraction: float) -> float:
    if not values:
        return float("nan")
    tail_len = max(1, int(math.ceil(len(values) * tail_fraction)))
    return float(np.mean(np.asarray(values[-tail_len:], dtype=float)))


def positive_part(value: float) -> float:
    if not np.isfinite(value):
        return float("inf")
    return max(float(value), 0.0)


def trace_metrics(trace: Dict[str, List[float]], tail_fraction: float, constraint_weight: float) -> Dict[str, float]:
    final_total = float(trace["point_error"][-1])
    final_upper = float(trace["loss_upper_trace"][-1])
    final_lower = float(trace["loss_lower_trace"][-1])
    final_max_constraint = float(trace.get("max_constraint_trace", [float("nan")])[-1])
    tail_total = tail_mean(trace["point_error"], tail_fraction)
    tail_upper = tail_mean(trace["loss_upper_trace"], tail_fraction)
    tail_lower = tail_mean(trace["loss_lower_trace"], tail_fraction)
    tail_max_constraint = tail_mean(trace.get("max_constraint_trace", [float("nan")]), tail_fraction)

    return {
        "final_total": final_total,
        "final_upper": final_upper,
        "final_lower": final_lower,
        "final_max_constraint": final_max_constraint,
        "final_score": final_total + constraint_weight * positive_part(final_max_constraint),
        "tail_mean_total": tail_total,
        "tail_mean_upper": tail_upper,
        "tail_mean_lower": tail_lower,
        "tail_mean_max_constraint": tail_max_constraint,
        "tail_score": tail_total + constraint_weight * positive_part(tail_max_constraint),
        "best_total": float(np.min(np.asarray(trace["point_error"], dtype=float))),
        "best_upper": float(np.min(np.asarray(trace["loss_upper_trace"], dtype=float))),
        "best_lower": float(np.min(np.asarray(trace["loss_lower_trace"], dtype=float))),
        "best_max_constraint": float(np.min(np.asarray(trace.get("max_constraint_trace", [float("nan")]), dtype=float))),
        "time": float(trace["time_trace"][-1]),
    }


def aggregate_records(config: Dict[str, object], records: Sequence[Dict[str, float]]) -> Dict[str, object]:
    row: Dict[str, object] = dict(config)
    row["num_seeds"] = len(records)
    metric_keys = [
        "final_total",
        "final_upper",
        "final_lower",
        "final_max_constraint",
        "final_score",
        "tail_mean_total",
        "tail_mean_upper",
        "tail_mean_lower",
        "tail_mean_max_constraint",
        "tail_score",
        "best_total",
        "best_upper",
        "best_lower",
        "best_max_constraint",
        "time",
    ]

    for key in metric_keys:
        values = np.asarray([record[key] for record in records], dtype=float)
        row[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        row[f"{key}_std"] = float(np.std(values)) if values.size else float("nan")

    return row


def metric_value(row: Dict[str, object], metric: str) -> float:
    value = row.get(metric, float("inf"))
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value_float):
        return float("inf")
    return value_float


def make_params(config: Dict[str, object], args: argparse.Namespace) -> AdaProxParams:
    return AdaProxParams(
        n=args.n,
        K=args.K,
        T=args.T,
        N_inner=args.N_inner,
        alpha=float(config["alpha"]),
        xi=float(config["xi"]),
        sigma=float(config["sigma"]),
        beta=float(config["beta"]),
        w_max=args.w_max,
        exact_inner=args.exact_inner,
        inner_lr=args.inner_lr,
        inner_lr_scale=args.inner_lr_scale,
    )


def run_grid_job(job: Dict[str, object]) -> Dict[str, object]:
    torch.set_num_threads(int(job["torch_threads"]))
    method = str(job["method"])
    args_dict = dict(job["args"])
    args = argparse.Namespace(**args_dict)
    set_lower_definition(float(args.lower_scale), bool(args.lower_divide_by_n))
    config = dict(job["config"])
    params = make_params(config, args)
    records: List[Dict[str, float]] = []

    try:
        for seed_offset in range(args.search_num_seeds):
            seed = args.base_seed + seed_offset
            if method == "sg":
                runner = lambda: run_sg(
                    params=params,
                    device=args.device,
                    seed=seed,
                    log_every=args.search_log_every,
                    gamma0=float(config["gamma0"]),
                    threshold_scale=float(config["threshold_scale"]),
                    constraint_step_interval=int(config.get("constraint_step_interval", 0)),
                )
            elif method == "pd":
                runner = lambda: run_pd(
                    params=params,
                    device=args.device,
                    seed=seed,
                    log_every=args.search_log_every,
                    primal_lr=float(config["primal_lr"]),
                    dual_lr=float(config["dual_lr"]),
                    dual_momentum=float(config["dual_momentum"]),
                    lambda_max=float(config["lambda_max"]),
                    reset_lambda_per_outer=bool(config["reset_lambda_per_outer"]),
                )
            else:
                raise ValueError(f"unknown method: {method}")

            with contextlib.redirect_stdout(io.StringIO()):
                trace = runner()
            records.append(trace_metrics(trace, args.tail_fraction, args.constraint_weight))

        row = aggregate_records(config, records)
        row["lower_scale"] = float(args.lower_scale)
        row["lower_divide_by_n"] = bool(args.lower_divide_by_n)
        row["method"] = method.upper()
        row["status"] = "ok"
        row["error"] = ""
        return row
    except Exception as exc:  # pragma: no cover - returned to parent for long-running grid robustness.
        row = dict(config)
        row["method"] = method.upper()
        row["status"] = "error"
        row["error"] = repr(exc)
        row["final_upper_mean"] = float("inf")
        row["tail_mean_upper_mean"] = float("inf")
        row["final_score_mean"] = float("inf")
        row["tail_score_mean"] = float("inf")
        return row


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_parallel_grid(
    method: str,
    configs: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    output_path: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    args_payload = vars(args).copy()
    jobs = [
        {
            "method": method,
            "config": config,
            "args": args_payload,
            "torch_threads": args.torch_threads,
        }
        for config in configs
    ]

    print(f"Starting {method.upper()} grid: {len(jobs)} configs, workers={args.workers}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_grid_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: metric_value(item, args.search_metric))
            write_rows(output_path, rows)
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (len(jobs) - done) / rate if rate > 0 else float("nan")
            best = rows[0]
            print(
                f"{method.upper()} [{done}/{len(jobs)}] elapsed={elapsed/60:.1f}m "
                f"eta={remaining/60:.1f}m best_{args.search_metric}={metric_value(best, args.search_metric):.6g} "
                f"status={row.get('status')}"
            )

    rows.sort(key=lambda item: metric_value(item, args.search_metric))
    write_rows(output_path, rows)
    print(f"Saved {method.upper()} grid table: {output_path}")
    print(f"Best {method.upper()} row: {rows[0]}")
    return rows


def build_sg_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    configs = []
    for alpha, xi, sigma, beta, gamma0, threshold_scale, constraint_step_interval in itertools.product(
        parse_float_list(args.alpha_grid),
        parse_float_list(args.xi_grid),
        parse_float_list(args.sigma_grid),
        parse_float_list(args.beta_grid),
        parse_float_list(args.sg_gamma0_grid),
        parse_float_list(args.sg_threshold_scale_grid),
        parse_int_list(args.sg_constraint_step_interval_grid),
    ):
        configs.append(
            {
                "alpha": alpha,
                "xi": xi,
                "sigma": sigma,
                "beta": beta,
                "gamma0": gamma0,
                "threshold_scale": threshold_scale,
                "constraint_step_interval": constraint_step_interval,
            }
        )
    return configs


def build_pd_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    configs = []
    for alpha, xi, sigma, beta, primal_lr, dual_lr, lambda_max, dual_momentum, reset_lambda in itertools.product(
        parse_float_list(args.alpha_grid),
        parse_float_list(args.xi_grid),
        parse_float_list(args.sigma_grid),
        parse_float_list(args.beta_grid),
        parse_float_list(args.pd_primal_lr_grid),
        parse_float_list(args.pd_dual_lr_grid),
        parse_float_list(args.pd_lambda_max_grid),
        parse_float_list(args.pd_dual_momentum_grid),
        parse_bool_list(args.pd_reset_lambda_grid),
    ):
        configs.append(
            {
                "alpha": alpha,
                "xi": xi,
                "sigma": sigma,
                "beta": beta,
                "primal_lr": primal_lr,
                "dual_lr": dual_lr,
                "lambda_max": lambda_max,
                "dual_momentum": dual_momentum,
                "reset_lambda_per_outer": reset_lambda,
            }
        )
    return configs


def run_compare(
    best_sg: Dict[str, object],
    best_pd: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    sg_params = make_params(best_sg, args)
    pd_params = make_params(best_pd, args)
    sipba_outputs = []
    sg_outputs = []
    pd_outputs = []
    schedule_params = {
        "t": args.sipba_t,
        "alpha0": args.sipba_alpha0,
        "beta0": args.sipba_beta0,
        "rho0": args.sipba_rho0,
        "sigma0": args.sipba_sigma0,
        "delta0": args.sipba_delta0,
    }

    print(f"Running comparison with {args.compare_num_seeds} repeated seeds")
    for rep in range(args.compare_num_seeds):
        seed = args.base_seed + rep
        print(f"comparison repeat {rep + 1}/{args.compare_num_seeds}, seed={seed}")
        with contextlib.redirect_stdout(io.StringIO()):
            sipba_outputs.append(
                run_sipba(
                    n=args.n,
                    iterations=args.compare_iterations,
                    device=args.device,
                    schedule_params=schedule_params,
                    seed=seed,
                    log_every=args.compare_log_every,
                )
            )
            sg_outputs.append(
                run_sg(
                    params=sg_params,
                    device=args.device,
                    seed=seed,
                    log_every=args.compare_log_every,
                    gamma0=float(best_sg["gamma0"]),
                    threshold_scale=float(best_sg["threshold_scale"]),
                    constraint_step_interval=int(best_sg.get("constraint_step_interval", 0)),
                )
            )
            pd_outputs.append(
                run_pd(
                    params=pd_params,
                    device=args.device,
                    seed=seed,
                    log_every=args.compare_log_every,
                    primal_lr=float(best_pd["primal_lr"]),
                    dual_lr=float(best_pd["dual_lr"]),
                    dual_momentum=float(best_pd["dual_momentum"]),
                    lambda_max=float(best_pd["lambda_max"]),
                    reset_lambda_per_outer=bool(best_pd["reset_lambda_per_outer"]),
                )
            )

    return {
        "lower_scale": float(args.lower_scale),
        "lower_divide_by_n": bool(args.lower_divide_by_n),
        "best_sg": dict(best_sg),
        "best_pd": dict(best_pd),
        "sipba": {
            "eval_steps": sipba_outputs[0]["eval_steps"],
            "loss_upper_list_all": [out["loss_upper_trace"] for out in sipba_outputs],
            "loss_lower_list_all": [out["loss_lower_trace"] for out in sipba_outputs],
            "point_error_list_all": [out["point_error"] for out in sipba_outputs],
            "time_list_all": [out["time_trace"] for out in sipba_outputs],
        },
        "adaprox_sg": build_summary(sg_outputs, sg_params, {"method": "AdaProx-SG"}),
        "adaprox_pd": build_summary(pd_outputs, pd_params, {"method": "AdaProx-PD"}),
    }


def metric_matrix(summary: Dict[str, object], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = stack_and_trim(summary[key])
    steps = np.asarray(summary["eval_steps"], dtype=float)
    common_len = min(values.shape[1], steps.shape[0])
    return steps[:common_len], values[:, :common_len]


def plot_comparison(payload: Dict[str, object], fig_path: Path) -> None:
    methods = [
        ("SiPBA", payload["sipba"], plt.cm.tab10(0)),
        ("AdaProx-SG", payload["adaprox_sg"], plt.cm.tab10(1)),
        ("AdaProx-PD", payload["adaprox_pd"], plt.cm.tab10(2)),
    ]
    panels = [
        ("Upper Loss", "loss_upper_list_all"),
        ("Lower Loss", "loss_lower_list_all"),
        ("Total Loss", "point_error_list_all"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (title, key) in zip(axes, panels):
        for label, summary, color in methods:
            steps, mat = metric_matrix(summary, key)
            mean = np.maximum(mat.mean(axis=0), 1e-14)
            std = mat.std(axis=0)
            low = np.maximum(mean - std, 1e-14)
            high = np.maximum(mean + std, 1e-14)
            ax.plot(steps, mean, label=label, linewidth=2.0, color=color)
            ax.fill_between(steps, low, high, color=color, alpha=0.14)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel grid search for AdaProx SG/PD and compare against SiPBA.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--N-inner", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--search-num-seeds", type=int, default=1)
    parser.add_argument("--search-log-every", type=int, default=10**9)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--search-metric", type=str, default="final_upper_mean")
    parser.add_argument("--compare-num-seeds", type=int, default=10)
    parser.add_argument("--compare-iterations", type=int, default=1000)
    parser.add_argument("--compare-log-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default="adaprox_grid_upper_loss")

    parser.add_argument("--sg-gamma0-grid", type=str, default="1,10,100,1000")
    parser.add_argument("--sg-threshold-scale-grid", type=str, default="1")
    parser.add_argument("--sg-constraint-step-interval-grid", type=str, default="0")
    parser.add_argument("--pd-primal-lr-grid", type=str, default="0.0001,0.001,0.01,0.1")
    parser.add_argument("--pd-dual-lr-grid", type=str, default="0.0001,0.001,0.01,0.1")
    parser.add_argument("--pd-lambda-max-grid", type=str, default="100")
    parser.add_argument("--pd-dual-momentum-grid", type=str, default="0")
    parser.add_argument("--pd-reset-lambda-grid", type=str, default="false")
    parser.add_argument("--beta-grid", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--sigma-grid", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--xi-grid", type=str, default="0.0001,0.001,0.01,0.1")
    parser.add_argument("--alpha-grid", type=str, default="0.001")

    parser.add_argument("--w-max", type=float, default=100.0)
    parser.add_argument("--inner-lr", type=float, default=0.0)
    parser.add_argument("--inner-lr-scale", type=float, default=1.0)
    parser.add_argument("--exact-inner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lower-scale",
        type=float,
        default=DEFAULT_LOWER_SCALE,
        help="Lower-level objective scale. Legacy searches used 1.",
    )
    parser.add_argument(
        "--lower-divide-by-n",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOWER_DIVIDE_BY_N,
        help="Whether the lower-level objective is divided by n.",
    )

    parser.add_argument("--sipba-t", type=float, default=0.01)
    parser.add_argument("--sipba-alpha0", type=float, default=0.1)
    parser.add_argument("--sipba-beta0", type=float, default=0.001)
    parser.add_argument("--sipba-rho0", type=float, default=10.0)
    parser.add_argument("--sipba-sigma0", type=float, default=1e-4)
    parser.add_argument("--sipba-delta0", type=float, default=1e-4)
    args = parser.parse_args()
    set_lower_definition(args.lower_scale, args.lower_divide_by_n)
    print(
        f"Using lower_scale={args.lower_scale:g}, "
        f"lower_divide_by_n={args.lower_divide_by_n}"
    )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but CUDA device was requested")
        device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    out_dir = Path(__file__).resolve().parent / "results"
    sg_path = out_dir / f"{args.run_name}_sg_grid.csv"
    pd_path = out_dir / f"{args.run_name}_pd_grid.csv"
    compare_path = out_dir / f"{args.run_name}_comparison.pt"
    fig_path = out_dir / f"{args.run_name}_comparison.png"

    sg_rows = run_parallel_grid("sg", build_sg_configs(args), args, sg_path)
    pd_rows = run_parallel_grid("pd", build_pd_configs(args), args, pd_path)

    best_sg = sg_rows[0]
    best_pd = pd_rows[0]
    payload = run_compare(best_sg, best_pd, args)
    torch.save(payload, compare_path)
    plot_comparison(payload, fig_path)
    print(f"Saved comparison data: {compare_path}")
    print(f"Saved comparison figure: {fig_path}")
    print(f"Best SG params: {best_sg}")
    print(f"Best PD params: {best_pd}")


if __name__ == "__main__":
    main()
