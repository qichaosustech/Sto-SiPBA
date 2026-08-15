from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedLocator, FormatStrFormatter, LogFormatterMathtext, NullLocator

from stochastic_pessimistic_bilevel import run_one_experiment, set_random_seed


def parse_noise_stds(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("noise std list is empty")
    return values


def stack_and_trim(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(seq, dtype=float).reshape(-1) for seq in series_list]
    if not arrays:
        raise ValueError("series list is empty")

    min_len = min(arr.shape[0] for arr in arrays)
    if min_len == 0:
        raise ValueError("at least one series is empty")

    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def build_metric_matrix(run_outputs: Sequence[Dict[str, List]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not run_outputs:
        raise ValueError("run_outputs is empty")

    metric_list_all = [out[key] for out in run_outputs]
    metric_mat = stack_and_trim(metric_list_all)

    eval_steps = np.asarray(run_outputs[0]["eval_steps"], dtype=float).reshape(-1)
    common_len = min(metric_mat.shape[1], eval_steps.shape[0])

    return eval_steps[:common_len], metric_mat[:, :common_len]


def build_summary(run_outputs: Sequence[Dict[str, List]]) -> Dict[str, object]:
    if not run_outputs:
        raise ValueError("run_outputs is empty")

    return {
        "eval_steps": run_outputs[0]["eval_steps"],
        "point_error_list_all": [out["point_error"] for out in run_outputs],
        "loss_upper_list_all": [out["loss_upper_trace"] for out in run_outputs],
        "loss_lower_list_all": [out["loss_lower_trace"] for out in run_outputs],
        "time_list_all": [out["time_trace"] for out in run_outputs],
    }


def load_pt(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_metric_matrix_from_summary(summary: Dict[str, object], key: str) -> Tuple[np.ndarray, np.ndarray]:
    if "eval_steps" not in summary:
        raise KeyError("summary is missing key: eval_steps")
    if key not in summary:
        raise KeyError(f"summary is missing key: {key}")

    metric_mat = stack_and_trim(summary[key])
    eval_steps = np.asarray(summary["eval_steps"], dtype=float).reshape(-1)
    common_len = min(metric_mat.shape[1], eval_steps.shape[0])

    return eval_steps[:common_len], metric_mat[:, :common_len]


def std_label(std: float) -> str:
    if np.isclose(std, round(std)):
        value = str(int(round(std)))
    else:
        value = f"{std:g}"
    return f"Std={value}"


def final_figure_name(iterations: int) -> str:
    return f"variance_compare_alpha0p1_beta0p1_it{iterations}_iteration_ticks_1e0_1e4_1e8_1e12.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VR-only experiments under multiple noise std settings and plot all results in one figure."
    )
    parser.add_argument("--num-experiments", type=int, default=10, help="Number of repeated experiments for each std")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations per experiment")
    parser.add_argument("--n", type=int, default=100, help="Problem dimension")
    parser.add_argument("--noise-stds", type=str, default="0.0,0.1,0.5,1.0", help="Comma-separated std list")
    parser.add_argument("--log-every", type=int, default=50, help="Log interval")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--load-data-path",
        type=str,
        default="",
        help="Path to existing .pt result; if set, skip experiments and only plot",
    )
    parser.add_argument("--fig-path", type=str, default="", help="Optional path to output figure (.png)")
    args = parser.parse_args()

    load_data_path = Path(args.load_data_path).expanduser().resolve() if args.load_data_path else None
    if load_data_path is not None and not load_data_path.exists():
        raise FileNotFoundError(f"Load file not found: {load_data_path}")

    noise_stds = parse_noise_stds(args.noise_stds)

    schedule_params = {
        "t": 0.01,
        "s": 0.5,
        "alpha0": 0.1,
        "beta0": 0.1,
        "rho0": 10.0,
        "sigma0": 1e-4,
        "delta0": 1e-4,
        "eta0": 1.0,
    }

    summary_by_std: Dict[str, Dict[str, object]] = {}

    if load_data_path is not None:
        payload = load_pt(load_data_path)
        summary_raw = payload.get("summary_by_std")
        if not isinstance(summary_raw, dict) or not summary_raw:
            raise ValueError("Loaded data has empty or invalid summary_by_std")

        summary_by_std = {str(k): v for k, v in summary_raw.items()}

        config = payload.get("config")
        config_noise = config.get("noise_stds") if isinstance(config, dict) else None
        if isinstance(config_noise, (list, tuple)) and config_noise:
            noise_stds = [float(std) for std in config_noise]
        else:
            noise_stds = sorted(float(k) for k in summary_by_std.keys())
        if isinstance(config, dict) and "iterations" in config:
            args.iterations = int(config["iterations"])
    else:
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but CUDA device was requested")
            device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
            torch.cuda.set_device(device_index)

        outputs_by_std: Dict[float, List[Dict[str, List]]] = {}
        for std in noise_stds:
            print(f"Running VR-only experiments for std={std}")
            run_outputs: List[Dict[str, List]] = []

            for rep in range(args.num_experiments):
                seed = args.base_seed + rep
                noise_seed = seed
                print(f"  repeat {rep + 1}/{args.num_experiments}, seed={seed}, noise_seed={noise_seed}")

                set_random_seed(seed, args.device)
                out = run_one_experiment(
                    n=args.n,
                    iterations=args.iterations,
                    noise_std=std,
                    device=args.device,
                    schedule_params=schedule_params,
                    use_variance_reduction=True,
                    log_every=args.log_every,
                    seed=seed,
                    noise_seed=noise_seed,
                )
                run_outputs.append(out)

            outputs_by_std[std] = run_outputs

        for std, outputs in outputs_by_std.items():
            summary_by_std[str(std)] = build_summary(outputs)

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 22,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax_upper, ax_lower = axes

    std_key_map: Dict[float, str] = {}
    for key in summary_by_std.keys():
        std_key_map[float(key)] = key

    colors = ["red", "blue", "green", "purple"]

    for idx, std in enumerate(noise_stds):
        if std not in std_key_map:
            raise KeyError(f"Missing std={std} in summary_by_std")

        std_key = std_key_map[std]
        summary = summary_by_std[std_key]
        eval_steps, upper_mat = build_metric_matrix_from_summary(summary, "loss_upper_list_all")
        _, lower_mat = build_metric_matrix_from_summary(summary, "loss_lower_list_all")

        mean_upper = np.maximum(upper_mat.mean(axis=0), 1e-12)
        std_upper = upper_mat.std(axis=0)
        upper_low = np.maximum(mean_upper - std_upper, 1e-12)
        upper_up = np.maximum(mean_upper + std_upper, 1e-12)

        mean_lower = np.maximum(lower_mat.mean(axis=0), 1e-12)
        std_lower = lower_mat.std(axis=0)
        lower_low = np.maximum(mean_lower - std_lower, 1e-12)
        lower_up = np.maximum(mean_lower + std_lower, 1e-12)

        x_axis = eval_steps
        color = colors[idx % len(colors)]
        label = std_label(std)

        ax_upper.plot(x_axis, mean_upper, linewidth=5.0, color=color, label=label)
        ax_upper.fill_between(x_axis, upper_low, upper_up, color=color, alpha=0.12)

        ax_lower.plot(x_axis, mean_lower, linewidth=5.0, color=color, label=label)
        ax_lower.fill_between(x_axis, lower_low, lower_up, color=color, alpha=0.12)

    ax_upper.set_yscale("log")
    ax_upper.set_xlabel("Iteration")
    ax_upper.set_ylabel("Upper Loss")
    ax_upper.grid(True, which="major", color="#b0b0b0", linewidth=0.8)

    ax_lower.set_yscale("log")
    ax_lower.set_xlabel("Iteration")
    ax_lower.set_ylabel("Lower Loss")
    ax_lower.grid(True, which="major", color="#b0b0b0", linewidth=0.8)

    x_tick_step = 200 if args.iterations == 1000 else max(1, args.iterations // 5)
    x_ticks = np.arange(0, args.iterations + 1, x_tick_step, dtype=float)
    if x_ticks[-1] != args.iterations:
        x_ticks = np.append(x_ticks, float(args.iterations))
    for ax in (ax_upper, ax_lower):
        ax.set_xlim(-20.0, float(args.iterations) + 20.0)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.tick_params(axis="x", labelsize=18)

    y_ticks = np.array([1, 1e-4, 1e-8, 1e-12], dtype=float)
    for ax in (ax_upper, ax_lower):
        ax.set_ylim(float(y_ticks[-1]), float(y_ticks[0]))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(NullLocator())

    ax_upper.tick_params(axis="y", labelleft=True)
    ax_lower.tick_params(axis="y", labelleft=True)

    handles, labels = ax_upper.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(noise_stds), 4),
        frameon=True,
        handlelength=1.5,
        columnspacing=1.0,
    )

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.fig_path:
        fig_path = Path(args.fig_path).expanduser().resolve()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
    elif load_data_path is not None:
        fig_path = result_dir / final_figure_name(args.iterations)
    else:
        fig_path = result_dir / final_figure_name(args.iterations)

    data_path = result_dir / f"variance_compare_alpha0p1_beta0p1_it{args.iterations}.pt"

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    if load_data_path is None:
        torch.save(
            {
                "config": {
                    "num_experiments": args.num_experiments,
                    "iterations": args.iterations,
                    "n": args.n,
                    "noise_stds": noise_stds,
                    "log_every": args.log_every,
                    "base_seed": args.base_seed,
                    "noise_seed_mode": "same_as_initial_seed",
                    "device": args.device,
                    "schedule_params": schedule_params,
                    "use_variance_reduction": True,
                },
                "summary_by_std": summary_by_std,
            },
            data_path,
        )

    print(f"Saved figure: {fig_path}")
    if load_data_path is None:
        print(f"Saved data: {data_path}")
    else:
        print(f"Loaded data: {load_data_path}")


if __name__ == "__main__":
    main()
