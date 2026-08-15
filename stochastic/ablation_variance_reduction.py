from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stochastic_pessimistic_bilevel import run_until_threshold


# =============================
# User editable experiment area
# =============================
BASE_SCHEDULE_PARAMS: Dict[str, float] = {
    "t": 0.01,
    "s": 0.5,
    "alpha0": 0.1,
    "beta0": 0.1,
    "rho0": 10.0,
    "sigma0": 1e-4,
    "delta0": 1e-4,
    "eta0": 1.0,
}

# Cases match the hyperparameter table order.
USER_ABLATION_CASES: List[Dict[str, object]] = [
    {"case_name": "baseline", "overrides": {}},
    {"case_name": "alpha0.05", "overrides": {"alpha0": 0.05}},
    {"case_name": "alpha0.2", "overrides": {"alpha0": 0.2}},
    {"case_name": "beta0.05", "overrides": {"beta0": 0.05}},
    {"case_name": "beta0.02", "overrides": {"beta0": 0.02}},
    {"case_name": "rho5", "overrides": {"rho0": 5.0}},
    {"case_name": "rho20", "overrides": {"rho0": 20.0}},
    {"case_name": "sigma1e-5", "overrides": {"sigma0": 1e-5}},
    {"case_name": "sigma1e-3", "overrides": {"sigma0": 1e-3}},
    {"case_name": "delta1e-5", "overrides": {"delta0": 1e-5}},
    {"case_name": "delta1e-3", "overrides": {"delta0": 1e-3}},
    {"case_name": "eta0.8", "overrides": {"eta0": 0.8}},
    {"case_name": "eta0.5", "overrides": {"eta0": 0.5}},
]


def make_custom_cases(base_params: Dict[str, float], user_cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not user_cases:
        raise ValueError("USER_ABLATION_CASES is empty. Please add at least one case.")

    valid_keys = set(base_params.keys())
    cases: List[Dict[str, object]] = []

    for idx, case_cfg in enumerate(user_cases):
        case_name = str(case_cfg.get("case_name", f"case_{idx:02d}"))
        overrides = dict(case_cfg.get("overrides", {}))

        unknown_keys = sorted(set(overrides.keys()) - valid_keys)
        if unknown_keys:
            raise KeyError(f"Unknown hyperparameters in case '{case_name}': {unknown_keys}")

        params = dict(base_params)
        params.update(overrides)

        changed_keys = [key for key in base_params.keys() if not np.isclose(params[key], base_params[key])]

        if not changed_keys:
            changed_param = "none"
            scale = 1.0
        elif len(changed_keys) == 1 and base_params[changed_keys[0]] != 0:
            changed_param = changed_keys[0]
            scale = float(params[changed_param] / base_params[changed_param])
        else:
            changed_param = ",".join(changed_keys)
            scale = float("nan")

        cases.append(
            {
                "case_name": case_name,
                "changed_param": changed_param,
                "scale": scale,
                "schedule_params": params,
            }
        )

    return cases


def summarize_runs(case: Dict[str, object], run_results: List[Dict[str, float]], repeats: int) -> Dict[str, object]:
    valid_times = np.asarray([item["hit_time"] for item in run_results if np.isfinite(item["hit_time"])], dtype=float)
    valid_iters = np.asarray([item["hit_iteration"] for item in run_results if np.isfinite(item["hit_iteration"])], dtype=float)
    final_total = np.asarray([item["final_total_loss"] for item in run_results], dtype=float)
    success_count = int(sum(int(item["reached"]) for item in run_results))

    schedule_params = case["schedule_params"]
    changed_param = str(case["changed_param"])

    if changed_param == "none":
        tested_value = "-"
    elif "," in changed_param:
        keys = changed_param.split(",")
        tested_value = "; ".join(f"{k}={schedule_params[k]:.6g}" for k in keys)
    else:
        tested_value = schedule_params[changed_param]

    mean_hit_time = float(valid_times.mean()) if valid_times.size else float("nan")
    std_hit_time = float(valid_times.std()) if valid_times.size else float("nan")
    mean_hit_iteration = float(valid_iters.mean()) if valid_iters.size else float("nan")
    std_hit_iteration = float(valid_iters.std()) if valid_iters.size else float("nan")

    return {
        "case_name": str(case["case_name"]),
        "changed_param": changed_param,
        "scale": float(case["scale"]),
        "tested_value": tested_value,
        "success_count": success_count,
        "repeats": repeats,
        "mean_hit_time": mean_hit_time,
        "std_hit_time": std_hit_time,
        "mean_hit_iteration": mean_hit_iteration,
        "std_hit_iteration": std_hit_iteration,
        "time": format_mean_pm_std(mean_hit_time, std_hit_time),
        "iter": format_mean_pm_std(mean_hit_iteration, std_hit_iteration),
        "alpha": float(schedule_params["alpha0"]),
        "beta": float(schedule_params["beta0"]),
        "rho": float(schedule_params["rho0"]),
        "sigma": float(schedule_params["sigma0"]),
        "delta": float(schedule_params["delta0"]),
        "eta": float(schedule_params["eta0"]),
        "mean_final_total_loss": float(final_total.mean()),
        "var_final_total_loss": float(final_total.var()),
    }


def format_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def format_mean_pm_std(mean_value: float, std_value: float) -> str:
    if np.isnan(mean_value) or np.isnan(std_value):
        return "nan(+-nan)"
    return f"{mean_value:.6g}(+-{std_value:.6g})"


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "alpha",
        "beta",
        "rho",
        "sigma",
        "delta",
        "eta",
        "time",
        "iter",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def save_markdown(rows: List[Dict[str, object]], path: Path, metadata: Dict[str, object]) -> None:
    headers = [
        "alpha",
        "beta",
        "rho",
        "sigma",
        "delta",
        "eta",
        "time",
        "iter",
    ]

    lines = [
        "# Variance Reduction Ablation",
        "",
        f"- convergence criterion: max(loss_upper, loss_lower) <= {metadata['threshold']}",
        f"- repeats per case: {metadata['repeats']}",
        f"- iterations: {metadata['iterations']}",
        f"- n: {metadata['n']}",
        f"- noise_std: {metadata['noise_std']}",
        f"- device: {metadata['device']}",
        "- time/iter are formatted as mean(+-std) over successful runs.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(format_value(row[h]) for h in headers) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study for the variance-reduction method by perturbing one schedule hyperparameter at a time."
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated runs per case")
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum iterations per run")
    parser.add_argument("--n", type=int, default=100, help="Problem dimension")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Standard deviation of Gaussian noise")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Target threshold for max(loss_upper, loss_lower)",
    )
    parser.add_argument("--log-every", type=int, default=0, help="Unused placeholder for compatibility")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed shared across cases")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but a CUDA device was requested")
        device_index = 0
        if ":" in args.device:
            device_index = int(args.device.split(":", maxsplit=1)[1])
        torch.cuda.set_device(device_index)

    base_schedule_params = dict(BASE_SCHEDULE_PARAMS)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    cases = make_custom_cases(base_schedule_params, USER_ABLATION_CASES)
    rows: List[Dict[str, object]] = []
    raw_results: Dict[str, List[Dict[str, float]]] = {}

    for case_idx, case in enumerate(cases, start=1):
        case_name = str(case["case_name"])
        schedule_params = dict(case["schedule_params"])
        print(f"[{case_idx}/{len(cases)}] running {case_name}")

        run_results: List[Dict[str, float]] = []
        for rep in range(args.repeats):
            seed = args.base_seed + rep
            noise_seed = seed
            print(f"  repeat {rep + 1}/{args.repeats}, seed={seed}, noise_seed={noise_seed}")
            result = run_until_threshold(
                n=args.n,
                iterations=args.iterations,
                noise_std=args.noise_std,
                device=args.device,
                schedule_params=schedule_params,
                threshold=args.threshold,
                seed=seed,
                noise_seed=noise_seed,
                use_variance_reduction=True,
            )
            run_results.append(result)

        rows.append(summarize_runs(case, run_results, args.repeats))
        raw_results[case_name] = run_results

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_prefix = (
        f"vr_ablation_n{args.n}_it{args.iterations}_std{args.noise_std}_thr{args.threshold}_{timestamp}"
    )
    csv_path = result_dir / f"{file_prefix}.csv"
    md_path = result_dir / f"{file_prefix}.md"
    pt_path = result_dir / f"{file_prefix}.pt"

    save_csv(rows, csv_path)
    save_markdown(
        rows,
        md_path,
        metadata={
            "threshold": args.threshold,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "n": args.n,
            "noise_std": args.noise_std,
            "device": args.device,
            "noise_seed_mode": "same_as_initial_seed",
        },
    )
    torch.save(
        {
            "base_schedule_params": base_schedule_params,
            "cases": cases,
            "rows": rows,
            "raw_results": raw_results,
            "threshold": args.threshold,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "n": args.n,
            "noise_std": args.noise_std,
            "noise_seed_mode": "same_as_initial_seed",
            "device": args.device,
        },
        pt_path,
    )

    print("\nAblation summary table:")
    for row in rows:
        print(
            f"alpha={format_value(row['alpha'])}, beta={format_value(row['beta'])}, "
            f"rho={format_value(row['rho'])}, sigma={format_value(row['sigma'])}, "
            f"delta={format_value(row['delta'])}, eta={format_value(row['eta'])}, "
            f"time={row['time']}, iter={row['iter']}"
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved Raw PT: {pt_path}")


if __name__ == "__main__":
    main()
