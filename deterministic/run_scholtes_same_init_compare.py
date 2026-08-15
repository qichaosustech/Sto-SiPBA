from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from scholtes_common import FORM_COMPACT, FORM_DETAILED, ScholtesConfig, result_dir, solve_one_experiment


def final_row(method: str, seed: int, trace: Dict[str, List[float]]) -> Dict[str, float | int | str]:
    return {
        "method": method,
        "seed": seed,
        "upper_loss": trace["loss_upper_trace"][-1],
        "lower_loss": trace["loss_lower_trace"][-1],
        "point_error": trace["point_error"][-1],
        "objective": trace["objective_trace"][-1],
        "lower_objective": trace["lower_objective_trace"][-1],
        "max_residual": trace["max_residual_trace"][-1],
        "residual_norm": trace["residual_norm_trace"][-1],
        "max_upper_constraint": trace["max_upper_constraint_trace"][-1],
        "max_lower_constraint": trace["max_lower_constraint_trace"][-1],
        "min_u": trace["min_u_trace"][-1],
        "max_scholtes_constraint": trace["max_scholtes_constraint_trace"][-1],
        "fsolve_ier": int(trace["fsolve_ier_trace"][-1]),
        "time": trace["time_trace"][-1],
        "final_t": trace["t_trace"][-1],
    }


def write_csv(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "seed", "upper_loss", "lower_loss", "max_residual", "fsolve_ier", "time"]
    lines = [
        "# Scholtes same-initial-point comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.12g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact and detailed Scholtes methods on the same 10 initial points as deterministic SiPBA."
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=10)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=1e-1)
    parser.add_argument("--t-decay", type=float, default=5e-2)
    parser.add_argument("--xtol", type=float, default=1e-6)
    parser.add_argument("--maxfev", type=int, default=10000)
    parser.add_argument(
        "--solve-timeout",
        type=float,
        default=200.0,
        help="Seconds allowed for each fsolve call. 0 means no timeout.",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=2,
        help="Stop the current seed after this many consecutive fsolve stagnation exits (ier 4/5) or timeouts. 0 disables.",
    )
    parser.add_argument("--fb-epsilon", type=float, default=1e-2)
    parser.add_argument("--x-low", type=float, default=-0.9)
    parser.add_argument("--x-high", type=float, default=0.9)
    parser.add_argument("--y-low", type=float, default=-0.9)
    parser.add_argument("--y-high", type=float, default=0.9)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--methods",
        type=str,
        default=f"{FORM_COMPACT},{FORM_DETAILED}",
        help="Comma-separated methods to run: compact,detailed",
    )
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--md-path", type=str, default="")
    parser.add_argument("--data-path", type=str, default="")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ScholtesConfig:
    return ScholtesConfig(
        n=args.n,
        outer_iterations=args.outer_iterations,
        num_experiments=args.num_experiments,
        base_seed=args.base_seed,
        t0=args.t0,
        t_decay=args.t_decay,
        xtol=args.xtol,
        maxfev=args.maxfev,
        solve_timeout=args.solve_timeout,
        stagnation_patience=args.stagnation_patience,
        fb_epsilon=args.fb_epsilon,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_name = args.run_name or (
        f"scholtes_cd_n{config.n}_outer{config.outer_iterations}_"
        f"tdecay005_eps01_stag2_timeout{int(config.solve_timeout)}_"
        f"seed{config.base_seed}_{config.num_experiments}"
    )
    rows: List[Dict[str, float | int | str]] = []
    traces: Dict[str, List[Dict[str, List[float]]]] = {FORM_COMPACT: [], FORM_DETAILED: []}
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid_methods = [method for method in methods if method not in {FORM_COMPACT, FORM_DETAILED}]
    if invalid_methods:
        raise ValueError(f"Invalid methods: {invalid_methods}")

    for method in methods:
        for rep in range(config.num_experiments):
            seed = config.base_seed + rep
            print(f"method={method}, repeat {rep + 1}/{config.num_experiments}, seed={seed}", flush=True)
            trace = solve_one_experiment(method, config, seed)
            traces[method].append(trace)
            rows.append(final_row(method, seed, trace))

    out_dir = result_dir()
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else out_dir / f"{run_name}.csv"
    md_path = Path(args.md_path).expanduser().resolve() if args.md_path else out_dir / f"{run_name}.md"
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else out_dir / f"{run_name}.pt"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    torch.save({"rows": rows, "traces": traces, "config": config.__dict__, "run_name": run_name}, data_path)

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved Markdown: {md_path}", flush=True)
    print(f"Saved data: {data_path}", flush=True)


if __name__ == "__main__":
    main()
