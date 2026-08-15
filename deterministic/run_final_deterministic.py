from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

from AdaProx_PD import run_one_experiment as run_pd
from AdaProx_SG import run_one_experiment as run_sg
from AdaProx_common import AdaProxParams, build_summary
from SiPBA import LOWER_SCALE, run_one_experiment as run_sipba


BEST_SG = {
    "alpha": 0.001,
    "xi": 0.0001,
    "sigma": 0.01,
    "beta": 0.01,
    "gamma0": 300.0,
    "threshold_scale": 1.0,
    "constraint_step_interval": 10,
}

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

SIPBA_CURRENT = {
    "t": 0.01,
    "alpha0": 0.1,
    "beta0": 0.01,
    "rho0": 10.0,
    "sigma0": 1e-4,
    "delta0": 1e-4,
}


def parse_dims(text: str) -> List[int]:
    dims = [int(part.strip()) for part in text.replace("，", ",").split(",") if part.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("empty dimension list")
    return dims


def stack_and_trim(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(seq, dtype=float).reshape(-1) for seq in series_list]
    min_len = min(arr.shape[0] for arr in arrays)
    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def final_values(summary: Dict[str, object], key: str) -> np.ndarray:
    return stack_and_trim(summary[key])[:, -1]  # type: ignore[arg-type]


def time_values(summary: Dict[str, object]) -> np.ndarray:
    return np.asarray([trace[-1] for trace in summary["time_list_all"]], dtype=float)  # type: ignore[index]


def method_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, key in [
        ("SiPBA", "sipba"),
        ("AdaProx-PD", "adaprox_pd"),
        ("AdaProx-SG", "adaprox_sg"),
    ]:
        summary = payload[key]
        upper = final_values(summary, "loss_upper_list_all")  # type: ignore[arg-type]
        lower = final_values(summary, "loss_lower_list_all")  # type: ignore[arg-type]
        total = final_values(summary, "point_error_list_all")  # type: ignore[arg-type]
        times = time_values(summary)  # type: ignore[arg-type]
        rows.append(
            {
                "method": method,
                "num_runs": int(upper.size),
                "upper_loss_min": float(np.min(upper)),
                "upper_loss_max": float(np.max(upper)),
                "upper_loss_mean": float(np.mean(upper)),
                "upper_loss_std": float(np.std(upper)),
                "lower_loss_min": float(np.min(lower)),
                "lower_loss_max": float(np.max(lower)),
                "lower_loss_mean": float(np.mean(lower)),
                "lower_loss_std": float(np.std(lower)),
                "total_loss_min": float(np.min(total)),
                "total_loss_max": float(np.max(total)),
                "total_loss_mean": float(np.mean(total)),
                "total_loss_std": float(np.std(total)),
                "avg_time_s": float(np.mean(times)),
                "std_time_s": float(np.std(times)),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_adaprox_params(n: int, steps: int, args: argparse.Namespace, config: Dict[str, object]) -> AdaProxParams:
    return AdaProxParams(
        n=n,
        K=steps // args.T,
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


def build_sipba_summary(outputs: Sequence[Dict[str, List[float]]], schedule_params: Dict[str, float]) -> Dict[str, object]:
    return {
        "eval_steps": outputs[0]["eval_steps"],
        "point_error_list_all": [out["point_error"] for out in outputs],
        "loss_upper_list_all": [out["loss_upper_trace"] for out in outputs],
        "loss_lower_list_all": [out["loss_lower_trace"] for out in outputs],
        "objective_list_all": [out["objective_trace"] for out in outputs],
        "lower_objective_list_all": [out["lower_objective_trace"] for out in outputs],
        "time_list_all": [out["time_trace"] for out in outputs],
        "schedule_params": dict(schedule_params),
    }


def run_gradient_methods(n: int, steps: int, args: argparse.Namespace) -> Dict[str, object]:
    if steps % args.T != 0:
        raise ValueError(f"steps={steps} must be divisible by T={args.T}")

    log_every = max(args.T, steps // args.log_points)
    sipba_outputs = []
    pd_outputs = []
    sg_outputs = []
    pd_params = build_adaprox_params(n, steps, args, BEST_PD)
    sg_params = build_adaprox_params(n, steps, args, BEST_SG)
    schedule_params = dict(SIPBA_CURRENT)

    for rep in range(args.num_experiments):
        seed = args.base_seed + rep
        print(f"[gradient n={n}] repeat {rep + 1}/{args.num_experiments}, seed={seed}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            sipba_outputs.append(
                run_sipba(
                    n=n,
                    iterations=steps,
                    device=args.device,
                    schedule_params=schedule_params,
                    seed=seed,
                    log_every=log_every,
                )
            )
            pd_outputs.append(
                run_pd(
                    params=pd_params,
                    device=args.device,
                    seed=seed,
                    log_every=log_every,
                    primal_lr=float(BEST_PD["primal_lr"]),
                    dual_lr=float(BEST_PD["dual_lr"]),
                    dual_momentum=float(BEST_PD["dual_momentum"]),
                    lambda_max=float(BEST_PD["lambda_max"]),
                    reset_lambda_per_outer=bool(BEST_PD["reset_lambda_per_outer"]),
                )
            )
            sg_outputs.append(
                run_sg(
                    params=sg_params,
                    device=args.device,
                    seed=seed,
                    log_every=log_every,
                    gamma0=float(BEST_SG["gamma0"]),
                    threshold_scale=float(BEST_SG["threshold_scale"]),
                    constraint_step_interval=int(BEST_SG["constraint_step_interval"]),
                )
            )

    return {
        "run_name": f"final_gradient_n{n}_it{steps}",
        "n": n,
        "steps": steps,
        "base_seed": args.base_seed,
        "num_experiments": args.num_experiments,
        "lower_scale": LOWER_SCALE,
        "sipba": build_sipba_summary(sipba_outputs, schedule_params),
        "adaprox_pd": build_summary(pd_outputs, pd_params, {"method": "AdaProx-PD"}),
        "adaprox_sg": build_summary(sg_outputs, sg_params, {"method": "AdaProx-SG"}),
        "best_pd": dict(BEST_PD),
        "best_sg": dict(BEST_SG),
        "exact_inner": args.exact_inner,
        "N_inner": args.N_inner,
    }


def rerun_sipba_summary(n: int, steps: int, args: argparse.Namespace) -> Dict[str, object]:
    log_every = max(args.T, steps // args.log_points)
    schedule_params = dict(SIPBA_CURRENT)
    sipba_outputs = []

    for rep in range(args.num_experiments):
        seed = args.base_seed + rep
        print(f"[SiPBA-only n={n}] repeat {rep + 1}/{args.num_experiments}, seed={seed}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            sipba_outputs.append(
                run_sipba(
                    n=n,
                    iterations=steps,
                    device=args.device,
                    schedule_params=schedule_params,
                    seed=seed,
                    log_every=log_every,
                )
            )

    return build_sipba_summary(sipba_outputs, schedule_params)


def stats_from_scholtes_csv(path: Path) -> Dict[str, Dict[str, object]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    stats: Dict[str, Dict[str, object]] = {}
    for raw_method, label in [("compact", "Scholtes-C"), ("detailed", "Scholtes-D")]:
        method_rows_raw = [row for row in rows if row["method"] == raw_method]
        if not method_rows_raw:
            raise ValueError(f"No Scholtes rows found for method={raw_method!r} in {path}")
        upper = np.asarray([float(row["upper_loss"]) for row in method_rows_raw], dtype=float)
        lower = np.asarray([float(row["lower_loss"]) for row in method_rows_raw], dtype=float)
        times = np.asarray([float(row["time"]) for row in method_rows_raw], dtype=float)
        stats[label] = {
            "method": label,
            "upper_loss_min": float(upper.min()),
            "upper_loss_max": float(upper.max()),
            "lower_loss_min": float(lower.min()),
            "lower_loss_max": float(lower.max()),
            "avg_time_s": float(times.mean()),
        }
    return stats


def format_loss(value: float) -> str:
    return f"{value:.3e}"


def format_time(value: float) -> str:
    return f"{value:.3f}"


def maybe_bold(value: float, best_value: float, formatted: str) -> str:
    if np.isclose(value, best_value, rtol=1e-12, atol=1e-15):
        return rf"\textbf{{{formatted}}}"
    return formatted


def render_report_table(gradient_rows: Sequence[Dict[str, object]], scholtes_rows: Dict[str, Dict[str, object]], n: int) -> str:
    method_order = ["SiPBA", "AdaProx-PD", "AdaProx-SG", "Scholtes-C", "Scholtes-D"]
    by_method = {str(row["method"]): row for row in gradient_rows}
    by_method.update(scholtes_rows)

    def row(metric: str, time_metric: bool = False) -> str:
        raw_values = [float(by_method[method][metric]) for method in method_order]
        best_value = min(raw_values)
        formatted = []
        for value in raw_values:
            text = format_time(value) if time_metric else format_loss(value)
            formatted.append(maybe_bold(value, best_value, text))
        return " & ".join(formatted)

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            rf"	\caption{{Performance comparison of the SiPBA, AdaProx-PD, AdaProx-SG, Scholtes-C, and Scholtes-D with $n={n}$.}}",
            rf"	\label{{tab:dimension{n}}}",
            r"	\scriptsize",
            r"	\setlength{\tabcolsep}{3pt}",
            r"	\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc}",
            r"		\hline",
            r"		& SiPBA & AdaProx-PD & AdaProx-SG & Scholtes-C & Scholtes-D \\",
            r"		\hline",
            f"		Upper loss (Min.) & {row('upper_loss_min')} \\\\",
            f"		Upper loss (Max.) & {row('upper_loss_max')} \\\\",
            f"		Lower loss (Min.) & {row('lower_loss_min')} \\\\",
            f"		Lower loss (Max.) & {row('lower_loss_max')} \\\\",
            f"		Ave Time (s) & {row('avg_time_s', time_metric=True)} \\\\",
            r"		\hline",
            r"	\end{tabular*}",
            r"\end{table}",
            "",
        ]
    )


def write_report_table_if_possible(result_dir: Path, gradient_rows: Sequence[Dict[str, object]], n: int) -> None:
    scholtes_csv = result_dir / f"scholtes_cd_n{n}_outer10_tdecay005_eps01_stag2_timeout200_seed42_10.csv"
    if not scholtes_csv.exists():
        print(f"Skipping five-method table for n={n}: missing {scholtes_csv}", flush=True)
        return
    table = render_report_table(gradient_rows, stats_from_scholtes_csv(scholtes_csv), n)
    out_path = result_dir / f"final_report_table_n{n}.tex"
    out_path.write_text(table, encoding="utf-8")
    print(f"Saved five-method report table: {out_path}", flush=True)


def run(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but CUDA device was requested")
        device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    for n in args.dims:
        steps = args.steps
        payload_path = result_dir / f"final_gradient_n{n}_it{steps}_comparison.pt"
        summary_path = result_dir / f"final_gradient_n{n}_it{steps}_summary.csv"
        if args.skip_gradient:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        elif args.only_sipba:
            if not payload_path.exists():
                raise FileNotFoundError(f"Missing payload for --only-sipba: {payload_path}")
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            payload["sipba"] = rerun_sipba_summary(n, steps, args)
            payload.update(
                {
                    "run_name": f"final_gradient_n{n}_it{steps}",
                    "n": n,
                    "steps": steps,
                    "base_seed": args.base_seed,
                    "num_experiments": args.num_experiments,
                    "lower_scale": LOWER_SCALE,
                    "sipba_rerun_only": True,
                    "sipba_current": dict(SIPBA_CURRENT),
                    "adaprox_preserved": True,
                }
            )
            torch.save(payload, payload_path)
            print(f"Saved SiPBA-refreshed comparison data: {payload_path}", flush=True)
        else:
            payload = run_gradient_methods(n, steps, args)
            torch.save(payload, payload_path)
            print(f"Saved gradient comparison data: {payload_path}", flush=True)

        rows = method_rows(payload)
        write_csv(summary_path, rows)
        print(f"Saved gradient summary CSV: {summary_path}", flush=True)
        write_report_table_if_possible(result_dir, rows, n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final deterministic toy-example experiments for the report. "
            "This script covers SiPBA, AdaProx-PD, and AdaProx-SG; run "
            "run_scholtes_same_init_compare.py separately for Scholtes CSVs."
        )
    )
    parser.add_argument("--dims", type=parse_dims, default=[100, 1000])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--N-inner", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--log-points", type=int, default=20)
    parser.add_argument("--w-max", type=float, default=100.0)
    parser.add_argument("--inner-lr", type=float, default=0.0)
    parser.add_argument("--inner-lr-scale", type=float, default=1.0)
    parser.add_argument("--exact-inner", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-gradient",
        action="store_true",
        help="Only rebuild CSV/table files from existing final_gradient_*.pt files.",
    )
    parser.add_argument(
        "--only-sipba",
        action="store_true",
        help="Rerun only SiPBA and preserve existing AdaProx summaries in final_gradient_*.pt files.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
