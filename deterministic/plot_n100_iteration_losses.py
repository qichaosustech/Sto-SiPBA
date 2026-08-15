from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


METHODS = [
    ("SiPBA", "sipba", "red"),
    ("AdaProx-PD", "adaprox_pd", "blue"),
    ("AdaProx-SG", "adaprox_sg", "green"),
]


def load_pt(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stack_and_trim(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(series, dtype=float).reshape(-1) for series in series_list]
    if not arrays:
        raise ValueError("series list is empty")

    min_len = min(array.size for array in arrays)
    if min_len == 0:
        raise ValueError("at least one series is empty")

    return np.stack([array[:min_len] for array in arrays], axis=0)


def iteration_and_metric(summary: Dict[str, object], metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if "eval_steps" not in summary:
        raise KeyError("summary is missing key: eval_steps")
    if metric_key not in summary:
        raise KeyError(f"summary is missing key: {metric_key}")

    iterations = np.asarray(summary["eval_steps"], dtype=float).reshape(-1)
    metric_mat = stack_and_trim(summary[metric_key])  # type: ignore[arg-type]
    common_len = min(iterations.size, metric_mat.shape[1])

    return iterations[:common_len], metric_mat[:, :common_len]


def mean_min_max(values: np.ndarray, floor: float = 1e-14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.maximum(values.mean(axis=0), floor)
    low = np.maximum(values.min(axis=0), floor)
    high = np.maximum(values.max(axis=0), floor)
    return mean, low, high


def plot_panel(
    ax: plt.Axes,
    payload: Dict[str, object],
    metric_key: str,
    ylabel: str,
    y_ticks: Sequence[float],
) -> list[plt.Line2D]:
    lines: list[plt.Line2D] = []
    max_iteration = 0.0
    for label, payload_key, color in METHODS:
        summary = payload[payload_key]
        if not isinstance(summary, dict):
            raise ValueError(f"payload[{payload_key!r}] is not a summary dict")

        iterations, metric_mat = iteration_and_metric(summary, metric_key)
        metric_mean, metric_low, metric_high = mean_min_max(metric_mat)
        if iterations.size:
            max_iteration = max(max_iteration, float(iterations[-1]))

        (line,) = ax.semilogy(iterations, metric_mean, label=label, color=color, linewidth=5)
        ax.fill_between(iterations, metric_low, metric_high, color=color, alpha=0.2)
        lines.append(line)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    if max_iteration > 0:
        ax.set_xlim(0.0, max_iteration)
    ax.set_xlabel("Iteration", fontsize=25)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_ylim(float(y_ticks[-1]), float(y_ticks[0]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(
        ticker.FixedFormatter([rf"$10^{{{int(np.log10(tick))}}}$" for tick in y_ticks])
    )
    ax.grid(True, which="major")
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    return lines


def plot_iteration_losses(payload: Dict[str, object], fig_path: Path) -> None:
    n_value = payload.get("n")
    if n_value is not None and int(n_value) != 100:
        raise ValueError(f"Expected n=100 payload, got n={n_value}")

    fig, (ax_upper, ax_lower) = plt.subplots(1, 2, figsize=(10, 5))
    loss_ticks = [1.0, 1e-4, 1e-8, 1e-12]

    lines = plot_panel(
        ax_upper,
        payload,
        "loss_upper_list_all",
        "Upper Loss",
        y_ticks=loss_ticks,
    )
    plot_panel(
        ax_lower,
        payload,
        "loss_lower_list_all",
        "Lower Loss",
        y_ticks=loss_ticks,
    )

    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=25)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def default_input_path() -> Path:
    root_dir = Path(__file__).resolve().parent
    return root_dir / "result" / "final_gradient_n100_it1000_comparison.pt"


def parse_args() -> argparse.Namespace:
    result_dir = Path(__file__).resolve().parent / "result"
    parser = argparse.ArgumentParser(
        description="Plot the final deterministic n=100 upper/lower loss curves against iteration."
    )
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Input comparison .pt file")
    parser.add_argument(
        "--output",
        type=Path,
        default=result_dir / "n100_upper_lower_loss_iteration.png",
        help="Output figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_pt(args.input)
    plot_iteration_losses(payload, args.output)
    print(f"Input data: {args.input}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
