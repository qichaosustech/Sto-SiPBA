import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


LOWER_SCALE = 10.0
LOWER_DIVIDE_BY_N = True


def F(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Deterministic upper-level objective, i.e. the stochastic toy objective with w=0."""
    n = x.shape[0]
    e = torch.ones_like(x)
    return torch.linalg.vector_norm(x - e) ** 2 - (n ** 0.5) * torch.linalg.vector_norm(y - e)


def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Deterministic lower-level objective with configurable dimension normalization."""
    n = x.shape[0]
    e = torch.ones_like(y)
    denominator = n if LOWER_DIVIDE_BY_N else 1.0
    return LOWER_SCALE * (torch.dot(y, e) - torch.linalg.vector_norm(x) ** 2) ** 2 / denominator


def psi(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rho: float,
    sigma: float,
    delta: float,
) -> torch.Tensor:
    """Smooth approximation psi_{rho,sigma,delta}(x,y,z)."""
    return (
        F(x, y)
        - rho * (f(x, y) - f(x, z))
        + 0.5 * sigma * torch.sum(z ** 2)
        - sigma * torch.sum(y * z)
        - 0.5 * delta * torch.sum(y ** 2)
    )


def proj_X(x: torch.Tensor, low: float = -0.9, high: float = 0.9) -> torch.Tensor:
    return torch.clamp(x, min=low, max=high)


def proj_Y(y: torch.Tensor, low: float = -0.9, high: float = 0.9) -> torch.Tensor:
    return torch.clamp(y, min=low, max=high)


def schedules(k: int, params: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    """Deterministic SiPBA schedules from the paper."""
    step = k + 1
    t = params["t"]
    alpha_k = params["alpha0"] * step ** (-8 * t)
    beta_k = params["beta0"] * step ** (-3 * t)
    rho_k = params["rho0"] * step ** t
    sigma_k = params["sigma0"] * step ** (-t)
    delta_k = params["delta0"] * step ** (-t)
    return alpha_k, beta_k, rho_k, sigma_k, delta_k


def set_random_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def initialize(
    n: int,
    device: str,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (x_high - x_low) * torch.rand(n, device=device) + x_low
    y = (y_high - y_low) * torch.rand(n, device=device) + y_low
    z = y.detach().clone()
    return x, y, z


def analytic_solution(n: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    e = torch.ones(n, device=device)
    return 0.5 * e, 0.25 * e


def compute_total_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    x_star: torch.Tensor,
) -> Tuple[float, float, float]:
    n = x.shape[0]
    e = torch.ones_like(x)
    y_target = torch.linalg.vector_norm(x) ** 2 * e / n
    loss_upper = torch.linalg.vector_norm(x - x_star) ** 2 / n
    loss_lower = torch.linalg.vector_norm(y - y_target) ** 2 / n
    total_loss = loss_upper + loss_lower
    return loss_upper.item(), loss_lower.item(), total_loss.item()


def run_until_threshold(
    n: int,
    iterations: int,
    device: str,
    schedule_params: Dict[str, float],
    threshold: float,
    seed: int,
    x_low: float = -0.9,
    x_high: float = 0.9,
    y_low: float = -0.9,
    y_high: float = 0.9,
) -> Dict[str, float]:
    set_random_seed(seed, device)
    x, y, z = initialize(n, device, x_low, x_high, y_low, y_high)
    x_star, _ = analytic_solution(n, device)
    total_time = 0.0

    for k in range(iterations):
        loss_upper, loss_lower, total_loss = compute_total_loss(x, y, x_star)
        if total_loss <= threshold:
            return {
                "reached": 1.0,
                "hit_time": total_time,
                "hit_iteration": float(k),
                "final_total_loss": total_loss,
                "final_loss_upper": loss_upper,
                "final_loss_lower": loss_lower,
            }

        alpha_k, beta_k, rho_k, sigma_k, delta_k = schedules(k, schedule_params)
        start = time.perf_counter()
        x, y, z = sipba_step(x, y, z, alpha_k, beta_k, rho_k, sigma_k, delta_k, x_low, x_high, y_low, y_high)
        total_time += time.perf_counter() - start

    loss_upper, loss_lower, total_loss = compute_total_loss(x, y, x_star)
    reached = 1.0 if total_loss <= threshold else 0.0
    return {
        "reached": reached,
        "hit_time": total_time if reached else float("nan"),
        "hit_iteration": float(iterations) if reached else float("nan"),
        "final_total_loss": total_loss,
        "final_loss_upper": loss_upper,
        "final_loss_lower": loss_lower,
    }


def sipba_step(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    alpha_k: float,
    beta_k: float,
    rho_k: float,
    sigma_k: float,
    delta_k: float,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)
    z_req = z.detach().clone().requires_grad_(True)

    psi_yz = psi(x_req, y_req, z_req, rho_k, sigma_k, delta_k)
    d_y = torch.autograd.grad(psi_yz, y_req, retain_graph=True)[0]
    d_z = torch.autograd.grad(psi_yz, z_req)[0]

    y_new = proj_Y(y + beta_k * d_y.detach(), low=y_low, high=y_high)
    z_new = proj_Y(z - beta_k * d_z.detach(), low=y_low, high=y_high)

    x_req = x.detach().clone().requires_grad_(True)
    psi_x = psi(x_req, y_new.detach(), z_new.detach(), rho_k, sigma_k, delta_k)
    d_x = torch.autograd.grad(psi_x, x_req)[0].detach()

    x_new = proj_X(x - alpha_k * d_x, low=x_low, high=x_high)
    return x_new.detach(), y_new.detach(), z_new.detach()


def run_one_experiment(
    n: int = 100,
    iterations: int = 5000,
    device: str = "cpu",
    schedule_params: Optional[Dict[str, float]] = None,
    seed: int = 123,
    log_every: int = 50,
    x_low: float = -0.9,
    x_high: float = 0.9,
    y_low: float = -0.9,
    y_high: float = 0.9,
) -> Dict[str, List[float]]:
    if schedule_params is None:
        schedule_params = {
            "t": 0.01,
            "alpha0": 0.1,
            "beta0": 0.01,
            "rho0": 10.0,
            "sigma0": 1e-4,
            "delta0": 1e-4,
        }

    set_random_seed(seed, device)
    x, y, z = initialize(n, device, x_low, x_high, y_low, y_high)
    x_star, _ = analytic_solution(n, device)

    point_error: List[float] = []
    loss_upper_trace: List[float] = []
    loss_lower_trace: List[float] = []
    objective_trace: List[float] = []
    lower_objective_trace: List[float] = []
    eval_steps: List[int] = []
    time_trace: List[float] = []

    total_time = 0.0

    for k in range(iterations):
        if k % log_every == 0 or k == iterations - 1:
            loss_upper, loss_lower, total_loss = compute_total_loss(x, y, x_star)
            objective = F(x, y).item()
            lower_objective = f(x, y).item()
            loss_upper_trace.append(loss_upper)
            loss_lower_trace.append(loss_lower)
            point_error.append(total_loss)
            objective_trace.append(objective)
            lower_objective_trace.append(lower_objective)
            eval_steps.append(k)
            time_trace.append(total_time)
            print(
                f"Iters:{k}, loss_upper:{loss_upper:.10f}, "
                f"loss_lower:{loss_lower:.6f}, loss:{total_loss:.6f}, "
            )

        alpha_k, beta_k, rho_k, sigma_k, delta_k = schedules(k, schedule_params)
        start = time.perf_counter()
        x, y, z = sipba_step(x, y, z, alpha_k, beta_k, rho_k, sigma_k, delta_k, x_low, x_high, y_low, y_high)
        total_time += time.perf_counter() - start

    return {
        "point_error": point_error,
        "eval_steps": eval_steps,
        "loss_upper_trace": loss_upper_trace,
        "loss_lower_trace": loss_lower_trace,
        "objective_trace": objective_trace,
        "lower_objective_trace": lower_objective_trace,
        "time_trace": time_trace,
    }


def plot_summary(summary: Dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eval_steps = np.array(summary["eval_steps"])
    point_error = np.array(summary["point_error_list_all"], dtype=float)
    time_trace = np.array(summary["time_list_all"], dtype=float)

    mean_error = point_error.mean(axis=0)
    min_error = point_error.min(axis=0)
    max_error = point_error.max(axis=0)
    mean_time = time_trace.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].semilogy(eval_steps, mean_error, color="red", linewidth=2, label="SiPBA")
    axes[0].fill_between(eval_steps, min_error, max_error, color="red", alpha=0.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total error")
    axes[0].grid(True, which="major")

    axes[1].semilogy(mean_time, mean_error, color="red", linewidth=2, label="SiPBA")
    axes[1].fill_between(mean_time, min_error, max_error, color="red", alpha=0.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, which="major")

    fig.legend(loc="upper center", ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic SiPBA toy example without gradient noise.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--alpha0", type=float, default=0.1)
    parser.add_argument("--beta0", type=float, default=0.01)
    parser.add_argument("--rho0", type=float, default=10.0)
    parser.add_argument("--sigma0", type=float, default=1e-4)
    parser.add_argument("--delta0", type=float, default=1e-4)
    parser.add_argument("--x-low", type=float, default=-0.9)
    parser.add_argument("--x-high", type=float, default=0.9)
    parser.add_argument("--y-low", type=float, default=-0.9)
    parser.add_argument("--y-high", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    schedule_params = {
        "t": args.t,
        "alpha0": args.alpha0,
        "beta0": args.beta0,
        "rho0": args.rho0,
        "sigma0": args.sigma0,
        "delta0": args.delta0,
    }

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    point_error_list_all = []
    loss_upper_list_all = []
    loss_lower_list_all = []
    objective_list_all = []
    lower_objective_list_all = []
    time_list_all = []
    eval_steps_ref = None

    for exp in range(args.num_experiments):
        print(f"experiment:{exp}/{args.num_experiments}")
        out = run_one_experiment(
            n=args.n,
            iterations=args.iterations,
            device=args.device,
            schedule_params=schedule_params,
            seed=args.seed + exp,
            log_every=args.log_every,
            x_low=args.x_low,
            x_high=args.x_high,
            y_low=args.y_low,
            y_high=args.y_high,
        )
        point_error_list_all.append(out["point_error"])
        loss_upper_list_all.append(out["loss_upper_trace"])
        loss_lower_list_all.append(out["loss_lower_trace"])
        objective_list_all.append(out["objective_trace"])
        lower_objective_list_all.append(out["lower_objective_trace"])
        time_list_all.append(out["time_trace"])
        if eval_steps_ref is None:
            eval_steps_ref = out["eval_steps"]

    run_name = f"deterministic_sipba_n{args.n}_it{args.iterations}_seed{args.seed}"
    summary = {
        "run_name": run_name,
        "eval_steps": eval_steps_ref,
        "point_error_list_all": point_error_list_all,
        "loss_upper_list_all": loss_upper_list_all,
        "loss_lower_list_all": loss_lower_list_all,
        "objective_list_all": objective_list_all,
        "lower_objective_list_all": lower_objective_list_all,
        "time_list_all": time_list_all,
        "schedule_params": schedule_params,
        "n": args.n,
        "iterations": args.iterations,
        "num_experiments": args.num_experiments,
        "device": args.device,
        "bounds": {
            "x_low": args.x_low,
            "x_high": args.x_high,
            "y_low": args.y_low,
            "y_high": args.y_high,
        },
    }

    summary_path = result_dir / f"{run_name}_summary.pt"
    torch.save(summary, summary_path)
    print(f"saved summary: {summary_path}")

    if not args.no_plot:
        plot_path = result_dir / f"{run_name}.png"
        plot_summary(summary, plot_path)
        print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
