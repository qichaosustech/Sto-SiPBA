from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

import SiPBA as sipba_module
from SiPBA import LOWER_SCALE, F, analytic_solution, compute_total_loss, f, initialize, proj_X, proj_Y, set_random_seed


@dataclass
class AdaProxParams:
    n: int = 100
    K: int = 100
    T: int = 200
    N_inner: int = 20
    alpha: float = 1e-3
    xi: float = 1e-2
    sigma: float = 1e-3
    beta: float = 1e-1
    x_low: float = -0.9
    x_high: float = 0.9
    y_low: float = -0.9
    y_high: float = 0.9
    w_max: float = 100.0
    exact_inner: bool = True
    inner_lr: float = 0.0
    inner_lr_scale: float = 1.0

    @property
    def epsilon_sub(self) -> float:
        return self.beta / (2.0 * self.K)


@dataclass
class AdaProxState:
    x: torch.Tensor
    y: torch.Tensor
    w: torch.Tensor


def make_state(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, requires_grad: bool) -> AdaProxState:
    return AdaProxState(
        x.detach().clone().requires_grad_(requires_grad),
        y.detach().clone().requires_grad_(requires_grad),
        w.detach().clone().requires_grad_(requires_grad),
    )


def clone_state(state: AdaProxState, requires_grad: bool) -> AdaProxState:
    return make_state(state.x, state.y, state.w, requires_grad=requires_grad)


def zeros_like_state(state: AdaProxState) -> AdaProxState:
    return AdaProxState(torch.zeros_like(state.x), torch.zeros_like(state.y), torch.zeros_like(state.w))


def add_state(left: AdaProxState, right: AdaProxState) -> AdaProxState:
    return AdaProxState(left.x + right.x, left.y + right.y, left.w + right.w)


def scale_state(state: AdaProxState, scale: float) -> AdaProxState:
    return AdaProxState(state.x * scale, state.y * scale, state.w * scale)


def project_new_state(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    params: AdaProxParams,
    requires_grad: bool = True,
) -> AdaProxState:
    with torch.no_grad():
        x_new = proj_X(x, low=params.x_low, high=params.x_high)
        y_new = proj_Y(y, low=params.y_low, high=params.y_high)
        w_new = torch.clamp(w, min=0.0, max=params.w_max)
    return make_state(x_new, y_new, w_new, requires_grad=requires_grad)


def gradient_state(loss: torch.Tensor, state: AdaProxState) -> AdaProxState:
    grads = torch.autograd.grad(loss, (state.x, state.y, state.w), retain_graph=False)
    return AdaProxState(grads[0], grads[1], grads[2])


def gradient_step(state: AdaProxState, grads: AdaProxState, step_size: float, params: AdaProxParams) -> AdaProxState:
    return project_new_state(
        state.x - step_size * grads.x,
        state.y - step_size * grads.y,
        state.w - step_size * grads.w,
        params,
        requires_grad=True,
    )


def prox_norm2(state: AdaProxState, center: AdaProxState) -> torch.Tensor:
    return (
        torch.sum((state.x - center.x) ** 2)
        + torch.sum((state.y - center.y) ** 2)
        + torch.sum((state.w - center.w) ** 2)
    )


def objective_prox(state: AdaProxState, center: AdaProxState, sigma: float) -> torch.Tensor:
    return F(state.x, state.y) + 0.5 * sigma * prox_norm2(state, center)


def exact_inner_solution(x: torch.Tensor, params: AdaProxParams) -> torch.Tensor:
    n = x.shape[0]
    squared_norm = torch.sum(x.detach() ** 2)
    if sipba_module.LOWER_DIVIDE_BY_N:
        # Keep the current/default operation order unchanged.
        value = 2.0 * LOWER_SCALE * squared_norm / (n * (2.0 * LOWER_SCALE + params.alpha))
    else:
        value = 2.0 * LOWER_SCALE * squared_norm / (2.0 * LOWER_SCALE * n + params.alpha)
    y_hat = value * torch.ones_like(x)
    return proj_Y(y_hat, low=params.y_low, high=params.y_high).detach()


def iterative_inner_solution(x: torch.Tensor, y_init: torch.Tensor, params: AdaProxParams) -> torch.Tensor:
    n = x.shape[0]
    lr = params.inner_lr
    if lr <= 0.0:
        lr = params.inner_lr_scale / (2.0 + params.alpha)

    x_fixed = x.detach()
    y_hat = y_init.detach().clone()
    for _ in range(params.N_inner):
        y_req = y_hat.detach().clone().requires_grad_(True)
        loss = f(x_fixed, y_req) + 0.5 * params.alpha * torch.sum(y_req ** 2)
        grad_y = torch.autograd.grad(loss, y_req)[0]
        with torch.no_grad():
            y_hat = proj_Y(y_req - lr * grad_y, low=params.y_low, high=params.y_high)

    return y_hat.detach()


def inner_solution(x: torch.Tensor, y_init: torch.Tensor, params: AdaProxParams) -> torch.Tensor:
    if params.exact_inner:
        return exact_inner_solution(x, params)
    return iterative_inner_solution(x, y_init, params)


def lower_star_hat(x: torch.Tensor, y_hat: torch.Tensor, params: AdaProxParams) -> torch.Tensor:
    return f(x, y_hat) + 0.5 * params.alpha * torch.sum(y_hat ** 2)


def base_constraint_hat(state: AdaProxState, y_hat: torch.Tensor, params: AdaProxParams) -> torch.Tensor:
    lower_value = f(state.x, state.y)
    lower_gap = lower_value - lower_star_hat(state.x, y_hat, params) - params.xi

    upper_value = F(state.x, state.y)
    grad_upper_y = torch.autograd.grad(upper_value, state.y, create_graph=True, retain_graph=True)[0]
    grad_lower_y = torch.autograd.grad(lower_value, state.y, create_graph=True, retain_graph=True)[0]

    stationarity = -grad_upper_y + state.w.reshape(()) * grad_lower_y
    complementarity = state.w.reshape(()) * lower_gap

    return torch.cat(
        [
            lower_gap.reshape(1),
            stationarity.reshape(-1),
            (-stationarity).reshape(-1),
            complementarity.reshape(1),
            (-complementarity).reshape(1),
        ],
        dim=0,
    )


def constraint_hat(
    state: AdaProxState,
    center: AdaProxState,
    y_hat: torch.Tensor,
    outer_iter: int,
    params: AdaProxParams,
) -> torch.Tensor:
    relax = (outer_iter + 1.0) * params.beta / params.K
    prox = 0.5 * params.sigma * prox_norm2(state, center)
    return base_constraint_hat(state, y_hat, params) + prox - relax


def initial_state(params: AdaProxParams, device: str, seed: int) -> AdaProxState:
    set_random_seed(seed, device)
    x, y, _ = initialize(params.n, device, params.x_low, params.x_high, params.y_low, params.y_high)
    w = torch.zeros(1, device=device)
    return project_new_state(x, y, w, params, requires_grad=True)


def inner_residual(x: torch.Tensor, y_hat: torch.Tensor, params: AdaProxParams) -> float:
    y_req = y_hat.detach().clone().requires_grad_(True)
    loss = f(x.detach(), y_req) + 0.5 * params.alpha * torch.sum(y_req ** 2)
    grad_y = torch.autograd.grad(loss, y_req)[0]
    return float(torch.linalg.vector_norm(grad_y).detach().cpu())


def evaluate_state(state: AdaProxState, params: AdaProxParams, device: str) -> Dict[str, float]:
    eval_state = clone_state(state, requires_grad=True)
    x_star, _ = analytic_solution(params.n, device)
    loss_upper, loss_lower, total_loss = compute_total_loss(eval_state.x.detach(), eval_state.y.detach(), x_star)
    objective = float(F(eval_state.x.detach(), eval_state.y.detach()).cpu())
    lower_objective = float(f(eval_state.x.detach(), eval_state.y.detach()).cpu())

    y_hat = inner_solution(eval_state.x, eval_state.y, params)
    base_h = base_constraint_hat(eval_state, y_hat, params)
    max_constraint = float(torch.max(base_h).detach().cpu())
    residual = inner_residual(eval_state.x, y_hat, params)

    return {
        "loss_upper": loss_upper,
        "loss_lower": loss_lower,
        "point_error": total_loss,
        "objective": objective,
        "lower_objective": lower_objective,
        "max_constraint": max_constraint,
        "inner_residual": residual,
        "w": float(eval_state.w.detach().cpu().reshape(())),
    }


def empty_trace() -> Dict[str, List[float]]:
    return {
        "point_error": [],
        "eval_steps": [],
        "loss_upper_trace": [],
        "loss_lower_trace": [],
        "objective_trace": [],
        "lower_objective_trace": [],
        "max_constraint_trace": [],
        "inner_residual_trace": [],
        "w_trace": [],
        "time_trace": [],
    }


def append_trace(trace: Dict[str, List[float]], step: int, total_time: float, metrics: Dict[str, float]) -> None:
    trace["eval_steps"].append(step)
    trace["time_trace"].append(total_time)
    trace["point_error"].append(metrics["point_error"])
    trace["loss_upper_trace"].append(metrics["loss_upper"])
    trace["loss_lower_trace"].append(metrics["loss_lower"])
    trace["objective_trace"].append(metrics["objective"])
    trace["lower_objective_trace"].append(metrics["lower_objective"])
    trace["max_constraint_trace"].append(metrics["max_constraint"])
    trace["inner_residual_trace"].append(metrics["inner_residual"])
    trace["w_trace"].append(metrics["w"])


def stack_and_trim(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(seq, dtype=float).reshape(-1) for seq in series_list]
    min_len = min(arr.shape[0] for arr in arrays)
    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def build_summary(run_outputs: Sequence[Dict[str, List[float]]], params: AdaProxParams, metadata: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "eval_steps": run_outputs[0]["eval_steps"],
        "point_error_list_all": [out["point_error"] for out in run_outputs],
        "loss_upper_list_all": [out["loss_upper_trace"] for out in run_outputs],
        "loss_lower_list_all": [out["loss_lower_trace"] for out in run_outputs],
        "objective_list_all": [out["objective_trace"] for out in run_outputs],
        "lower_objective_list_all": [out["lower_objective_trace"] for out in run_outputs],
        "max_constraint_list_all": [out["max_constraint_trace"] for out in run_outputs],
        "inner_residual_list_all": [out["inner_residual_trace"] for out in run_outputs],
        "w_list_all": [out["w_trace"] for out in run_outputs],
        "time_list_all": [out["time_trace"] for out in run_outputs],
        "params": params.__dict__,
    }
    summary.update(metadata)
    return summary


def plot_summary(summary: Dict[str, object], fig_path: Path, label: str) -> None:
    eval_steps = np.asarray(summary["eval_steps"], dtype=float)
    upper = stack_and_trim(summary["loss_upper_list_all"])
    lower = stack_and_trim(summary["loss_lower_list_all"])
    constraint = stack_and_trim(summary["max_constraint_list_all"])

    common_len = min(eval_steps.shape[0], upper.shape[1], lower.shape[1], constraint.shape[1])
    eval_steps = eval_steps[:common_len]
    upper = upper[:, :common_len]
    lower = lower[:, :common_len]
    constraint = constraint[:, :common_len]

    metrics = [
        ("Upper Loss", np.maximum(upper, 1e-14)),
        ("Lower Loss", np.maximum(lower, 1e-14)),
        ("Max Constraint", np.maximum(constraint, 1e-14)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    color = plt.cm.tab10(1)
    for ax, (title, values) in zip(axes, metrics):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        low = np.maximum(mean - std, 1e-14)
        high = np.maximum(mean + std, 1e-14)
        ax.plot(eval_steps, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(eval_steps, low, high, color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Value")
    axes[0].legend()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def result_dir() -> Path:
    path = Path(__file__).resolve().parent / "result"
    path.mkdir(parents=True, exist_ok=True)
    return path


def timed_step_start() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return time.perf_counter() - start
