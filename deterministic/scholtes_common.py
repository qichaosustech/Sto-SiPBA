from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue as queue_module
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from scipy.optimize import fsolve


FORM_COMPACT = "compact"
FORM_DETAILED = "detailed"


@dataclass
class ScholtesConfig:
    n: int = 100
    outer_iterations: int = 10
    num_experiments: int = 10
    base_seed: int = 42
    t0: float = 1e-1
    t_decay: float = 5e-2
    xtol: float = 1e-6
    maxfev: int = 10000
    fb_epsilon: float = 1e-2
    solve_timeout: float = 200.0
    stagnation_patience: int = 2
    x_low: float = -0.9
    x_high: float = 0.9
    y_low: float = -0.9
    y_high: float = 0.9

    @property
    def p(self) -> int:
        return 2 * self.n

    @property
    def q(self) -> int:
        return 2 * self.n


def upper_objective(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    e = np.ones(n)
    return float(np.linalg.norm(x - e) ** 2 - np.sqrt(n) * np.linalg.norm(y - e))


def lower_objective(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    return float((np.sum(y) - np.dot(x, x)) ** 2 / n)


def upper_constraints(x: np.ndarray, config: ScholtesConfig) -> np.ndarray:
    return np.concatenate([config.x_low - x, x - config.x_high])


def lower_constraints(x: np.ndarray, y: np.ndarray, config: ScholtesConfig) -> np.ndarray:
    return np.concatenate([config.y_low - y, y - config.y_high])


def grad_upper_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    del y
    return 2.0 * (x - 1.0)


def grad_upper_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    diff = y - 1.0
    norm = np.linalg.norm(diff)
    if norm <= 1e-12:
        return np.zeros_like(y)
    return -np.sqrt(n) * diff / norm


def jac_upper_constraints_x(config: ScholtesConfig) -> np.ndarray:
    n = config.n
    return np.vstack([-np.eye(n), np.eye(n)])


def jac_lower_constraints_x(config: ScholtesConfig) -> np.ndarray:
    return np.zeros((config.q, config.n))


def jac_lower_constraints_y(config: ScholtesConfig) -> np.ndarray:
    n = config.n
    return np.vstack([-np.eye(n), np.eye(n)])


def lower_stationarity(x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    u_low = u[:n]
    u_high = u[n:]
    gap = np.sum(y) - np.dot(x, x)
    return (2.0 / n) * gap * np.ones(n) - u_low + u_high


def jac_lower_stationarity_x(x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    del y, u
    n = x.shape[0]
    return (-4.0 / n) * np.outer(np.ones(n), x)


def jac_lower_stationarity_y(x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    n = len(u) // 2
    del x, y, u
    return (2.0 / n) * np.ones((n, n))


def analytic_x_star(n: int) -> np.ndarray:
    return 0.5 * np.ones(n)


def compute_losses(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    n = x.shape[0]
    x_star = analytic_x_star(n)
    y_target = np.dot(x, x) * np.ones(n) / n
    loss_upper = np.linalg.norm(x - x_star) ** 2 / n
    loss_lower = np.linalg.norm(y - y_target) ** 2 / n
    total_loss = loss_upper + loss_lower
    return float(loss_upper), float(loss_lower), float(total_loss)


def fb(a: np.ndarray, b: np.ndarray, epsilon: float) -> np.ndarray:
    return np.sqrt(a**2 + b**2 + 2.0 * epsilon) - a + b


def split_vars(values: np.ndarray, form: str, config: ScholtesConfig) -> Dict[str, np.ndarray]:
    n = config.n
    p = config.p
    q = config.q
    idx = 0
    x = values[idx : idx + n]
    idx += n
    y = values[idx : idx + n]
    idx += n
    u = values[idx : idx + q]
    idx += q
    alpha = values[idx : idx + p]
    idx += p
    beta = values[idx : idx + n]
    idx += n
    gamma = values[idx : idx + q]
    idx += q
    delta = values[idx : idx + q]
    idx += q

    parts = {
        "x": x,
        "y": y,
        "u": u,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }

    if form == FORM_DETAILED:
        mu = values[idx : idx + q]
        idx += q
        parts["mu"] = mu

    if idx != values.shape[0]:
        raise ValueError(f"Unexpected vector length for {form}: consumed {idx}, got {values.shape[0]}")

    return parts


def pack_initial_guess(form: str, config: ScholtesConfig, seed: int) -> np.ndarray:
    n = config.n
    p = config.p
    q = config.q
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tensor = (config.x_high - config.x_low) * torch.rand(n) + config.x_low
    y_tensor = (config.y_high - config.y_low) * torch.rand(n) + config.y_low
    x = x_tensor.numpy().astype(float)
    y = y_tensor.numpy().astype(float)
    u = np.zeros(q)
    alpha = np.zeros(p)
    beta = np.zeros(n)
    gamma = np.zeros(q)
    delta = np.zeros(q)
    values = [x, y, u, alpha, beta, gamma, delta]
    if form == FORM_DETAILED:
        values.append(np.zeros(q))
    return np.concatenate(values)


def residual(values: np.ndarray, t: float, epsilon: float, form: str, config: ScholtesConfig) -> np.ndarray:
    parts = split_vars(values, form, config)
    x = parts["x"]
    y = parts["y"]
    u = parts["u"]
    alpha = parts["alpha"]
    beta = parts["beta"]
    gamma = parts["gamma"]
    delta = parts["delta"]

    g_val = lower_constraints(x, y, config)
    G_val = upper_constraints(x, config)
    ell_val = lower_stationarity(x, y, u)

    n = config.n
    alpha_low = alpha[:n]
    alpha_high = alpha[n:]
    delta_u_minus_gamma = delta * u - gamma
    dug_low = delta_u_minus_gamma[:n]
    dug_high = delta_u_minus_gamma[n:]
    beta_sum = np.sum(beta)

    eq_x = grad_upper_x(x, y) - alpha_low + alpha_high - (4.0 / n) * beta_sum * x
    eq_y = grad_upper_y(x, y) + (2.0 / n) * beta_sum * np.ones(n) - dug_low + dug_high

    eq_alpha = fb(alpha, G_val, epsilon)
    eq_gamma = fb(gamma, g_val, epsilon)
    eq_delta = fb(delta, -u * g_val - t * np.ones(config.q), epsilon)

    if form == FORM_DETAILED:
        mu = parts["mu"]
        grad_g_y_beta = np.concatenate([-beta, beta])
        eq_mu_relation = mu - grad_g_y_beta + delta * g_val
        eq_mu_u = fb(mu, -u, epsilon)
        return np.concatenate([eq_x, eq_y, eq_mu_relation, eq_alpha, eq_gamma, eq_mu_u, eq_delta, ell_val])

    if form == FORM_COMPACT:
        grad_g_y_beta = np.concatenate([-beta, beta])
        mu_proxy = grad_g_y_beta - delta * g_val
        # Algebraic compact form: eliminate mu from Eq. (5.6) and keep FB(mu, -u).
        # The printed Eq. (5.7) uses FB(u, mu_proxy), but with phi(a,b)=sqrt(...)-a+b
        # that enforces mu_proxy <= 0 and is not equivalent to the detailed system.
        eq_u = fb(mu_proxy, -u, epsilon)
        return np.concatenate([eq_x, eq_y, eq_alpha, eq_gamma, eq_u, eq_delta, ell_val])

    raise ValueError(f"Unknown Scholtes form: {form}")


class TimedResidual:
    def __init__(
        self,
        t: float,
        epsilon: float,
        form: str,
        config: ScholtesConfig,
        shared_best: Any = None,
        shared_calls: Any = None,
        shared_best_max_residual: Any = None,
    ) -> None:
        self.t = t
        self.epsilon = epsilon
        self.form = form
        self.config = config
        self.shared_best = shared_best
        self.shared_calls = shared_calls
        self.shared_best_max_residual = shared_best_max_residual
        self.calls = 0
        self.best_values: np.ndarray | None = None
        self.best_max_residual = float("inf")
        self.latest_values: np.ndarray | None = None
        self.latest_max_residual = float("inf")

    def _publish_best(self, values: np.ndarray, max_residual: float) -> None:
        if self.shared_best is not None:
            with self.shared_best.get_lock():
                shared_array = np.frombuffer(self.shared_best.get_obj(), dtype=np.float64)
                shared_array[:] = values
        if self.shared_calls is not None:
            self.shared_calls.value = self.calls
        if self.shared_best_max_residual is not None:
            self.shared_best_max_residual.value = max_residual

    def __call__(self, values: np.ndarray) -> np.ndarray:
        res = residual(values, self.t, self.epsilon, self.form, self.config)
        max_residual = float(np.max(np.abs(res)))
        self.calls += 1
        self.latest_values = values.copy()
        self.latest_max_residual = max_residual
        if max_residual < self.best_max_residual:
            self.best_values = values.copy()
            self.best_max_residual = max_residual
            self._publish_best(self.best_values, self.best_max_residual)
        return res


def _fsolve_worker(
    output_queue: Any,
    shared_best: Any,
    shared_calls: Any,
    shared_best_max_residual: Any,
    t: float,
    epsilon: float,
    form: str,
    config: ScholtesConfig,
    solution: np.ndarray,
) -> None:
    fun = TimedResidual(
        t,
        epsilon,
        form,
        config,
        shared_best=shared_best,
        shared_calls=shared_calls,
        shared_best_max_residual=shared_best_max_residual,
    )
    try:
        solution, _info, ier, msg = fsolve(
            fun,
            solution,
            full_output=True,
            xtol=config.xtol,
            maxfev=config.maxfev,
        )
        output_queue.put(("ok", solution, int(ier), str(msg), fun.calls, float(fun.best_max_residual)))
    except BaseException as exc:
        fallback = fun.best_values if fun.best_values is not None else fun.latest_values
        if fallback is None:
            fallback = solution
        output_queue.put(("error", fallback, -8, repr(exc), fun.calls, float(fun.best_max_residual)))


def run_fsolve_with_optional_timeout(
    fun: TimedResidual,
    solution: np.ndarray,
    config: ScholtesConfig,
) -> Tuple[np.ndarray, int, str, bool]:
    if config.solve_timeout <= 0:
        solution, _info, ier, msg = fsolve(
            fun,
            solution,
            full_output=True,
            xtol=config.xtol,
            maxfev=config.maxfev,
        )
        return solution, int(ier), str(msg), False

    shared_best = mp.Array("d", int(solution.size), lock=True)
    with shared_best.get_lock():
        np.frombuffer(shared_best.get_obj(), dtype=np.float64)[:] = solution
    shared_calls = mp.Value("i", 0)
    shared_best_max_residual = mp.Value("d", float("inf"))
    output_queue: Any = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_fsolve_worker,
        args=(
            output_queue,
            shared_best,
            shared_calls,
            shared_best_max_residual,
            fun.t,
            fun.epsilon,
            fun.form,
            config,
            solution,
        ),
    )
    process.start()
    process.join(config.solve_timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        with shared_best.get_lock():
            fallback = np.frombuffer(shared_best.get_obj(), dtype=np.float64).copy()
        fun.calls = int(shared_calls.value)
        fun.best_values = fallback
        fun.best_max_residual = float(shared_best_max_residual.value)
        return fallback, -9, f"fsolve exceeded hard timeout of {config.solve_timeout:g}s", True

    try:
        status, values, ier, msg, calls, best_max_residual = output_queue.get_nowait()
    except queue_module.Empty:
        with shared_best.get_lock():
            fallback = np.frombuffer(shared_best.get_obj(), dtype=np.float64).copy()
        fun.calls = int(shared_calls.value)
        fun.best_values = fallback
        fun.best_max_residual = float(shared_best_max_residual.value)
        return fallback, -8, "fsolve subprocess exited without returning a result", True

    fun.calls = int(calls)
    fun.best_values = values.copy()
    fun.best_max_residual = float(best_max_residual)
    return values, int(ier), str(msg), status != "ok"


def evaluate(values: np.ndarray, t: float, epsilon: float, form: str, config: ScholtesConfig) -> Dict[str, float]:
    parts = split_vars(values, form, config)
    x = parts["x"]
    y = parts["y"]
    u = parts["u"]
    g_val = lower_constraints(x, y, config)
    G_val = upper_constraints(x, config)
    res = residual(values, t, epsilon, form, config)
    loss_upper, loss_lower, point_error = compute_losses(x, y)
    return {
        "loss_upper": loss_upper,
        "loss_lower": loss_lower,
        "point_error": point_error,
        "objective": upper_objective(x, y),
        "lower_objective": lower_objective(x, y),
        "max_residual": float(np.max(np.abs(res))),
        "residual_norm": float(np.linalg.norm(res)),
        "max_upper_constraint": float(np.max(G_val)),
        "max_lower_constraint": float(np.max(g_val)),
        "min_u": float(np.min(u)),
        "max_scholtes_constraint": float(np.max(-u * g_val - t)),
    }


def empty_trace() -> Dict[str, List[float]]:
    return {
        "point_error": [],
        "loss_upper_trace": [],
        "loss_lower_trace": [],
        "objective_trace": [],
        "lower_objective_trace": [],
        "max_residual_trace": [],
        "residual_norm_trace": [],
        "max_upper_constraint_trace": [],
        "max_lower_constraint_trace": [],
        "min_u_trace": [],
        "max_scholtes_constraint_trace": [],
        "time_trace": [],
        "eval_steps": [],
        "t_trace": [],
        "fsolve_ier_trace": [],
    }


def append_trace(
    trace: Dict[str, List[float]],
    step: int,
    t: float,
    total_time: float,
    metrics: Dict[str, float],
    ier: int,
) -> None:
    trace["eval_steps"].append(float(step))
    trace["t_trace"].append(float(t))
    trace["time_trace"].append(float(total_time))
    trace["point_error"].append(metrics["point_error"])
    trace["loss_upper_trace"].append(metrics["loss_upper"])
    trace["loss_lower_trace"].append(metrics["loss_lower"])
    trace["objective_trace"].append(metrics["objective"])
    trace["lower_objective_trace"].append(metrics["lower_objective"])
    trace["max_residual_trace"].append(metrics["max_residual"])
    trace["residual_norm_trace"].append(metrics["residual_norm"])
    trace["max_upper_constraint_trace"].append(metrics["max_upper_constraint"])
    trace["max_lower_constraint_trace"].append(metrics["max_lower_constraint"])
    trace["min_u_trace"].append(metrics["min_u"])
    trace["max_scholtes_constraint_trace"].append(metrics["max_scholtes_constraint"])
    trace["fsolve_ier_trace"].append(float(ier))


def fb_epsilon_value(config: ScholtesConfig) -> float:
    return config.fb_epsilon


def solve_one_experiment(form: str, config: ScholtesConfig, seed: int) -> Dict[str, List[float]]:
    solution = pack_initial_guess(form, config, seed)
    trace = empty_trace()
    total_time = 0.0
    t = config.t0
    epsilon = fb_epsilon_value(config)

    metrics = evaluate(solution, t, epsilon, form, config)
    append_trace(trace, 0, t, total_time, metrics, ier=0)
    print(
        f"outer:0, t:{t:.3g}, loss_upper:{metrics['loss_upper']:.6g}, "
        f"loss_lower:{metrics['loss_lower']:.6g}, max_res:{metrics['max_residual']:.3g}",
        flush=True,
    )

    consecutive_stagnation = 0
    for outer in range(config.outer_iterations):
        epsilon = fb_epsilon_value(config)
        fun = TimedResidual(t, epsilon, form, config)
        start = time.perf_counter()
        solution, ier, msg, timed_out = run_fsolve_with_optional_timeout(fun, solution, config)
        total_time += time.perf_counter() - start

        metrics = evaluate(solution, t, epsilon, form, config)
        append_trace(trace, outer + 1, t, total_time, metrics, ier=ier)
        if timed_out:
            print(
                f"warning: fsolve timed out at outer {outer + 1}; "
                f"calls={fun.calls}, best_max_res={fun.best_max_residual:.3g}, msg={msg}",
                flush=True,
            )
        elif ier != 1:
            print(f"warning: fsolve did not converge at outer {outer + 1}; ier={ier}, msg={msg}", flush=True)
        print(
            f"outer:{outer + 1}, t:{t:.3g}, loss_upper:{metrics['loss_upper']:.6g}, "
            f"loss_lower:{metrics['loss_lower']:.6g}, max_res:{metrics['max_residual']:.3g}, "
            f"time:{total_time:.3g}s",
            flush=True,
        )
        stagnant_exit = timed_out or ier in {4, 5}
        if stagnant_exit:
            consecutive_stagnation += 1
        else:
            consecutive_stagnation = 0
        if config.stagnation_patience > 0 and consecutive_stagnation >= config.stagnation_patience:
            print(
                f"early stop: {consecutive_stagnation} consecutive stagnant/timeout outer solves "
                f"(timeout or ier in {{4, 5}}) at outer {outer + 1}",
                flush=True,
            )
            break
        t *= config.t_decay

    return trace


def build_summary(form: str, traces: List[Dict[str, List[float]]], config: ScholtesConfig) -> Dict[str, object]:
    return {
        "method": f"Scholtes-{form}",
        "run_name": f"scholtes_{form}_n{config.n}_outer{config.outer_iterations}_seed{config.base_seed}",
        "eval_steps": traces[0]["eval_steps"],
        "t_trace": traces[0]["t_trace"],
        "point_error_list_all": [trace["point_error"] for trace in traces],
        "loss_upper_list_all": [trace["loss_upper_trace"] for trace in traces],
        "loss_lower_list_all": [trace["loss_lower_trace"] for trace in traces],
        "objective_list_all": [trace["objective_trace"] for trace in traces],
        "lower_objective_list_all": [trace["lower_objective_trace"] for trace in traces],
        "max_residual_list_all": [trace["max_residual_trace"] for trace in traces],
        "residual_norm_list_all": [trace["residual_norm_trace"] for trace in traces],
        "max_upper_constraint_list_all": [trace["max_upper_constraint_trace"] for trace in traces],
        "max_lower_constraint_list_all": [trace["max_lower_constraint_trace"] for trace in traces],
        "min_u_list_all": [trace["min_u_trace"] for trace in traces],
        "max_scholtes_constraint_list_all": [trace["max_scholtes_constraint_trace"] for trace in traces],
        "fsolve_ier_list_all": [trace["fsolve_ier_trace"] for trace in traces],
        "time_list_all": [trace["time_trace"] for trace in traces],
        "config": config.__dict__,
    }


def stack_and_trim(series_list: List[List[float]]) -> np.ndarray:
    arrays = [np.asarray(series, dtype=float).reshape(-1) for series in series_list]
    min_len = min(array.shape[0] for array in arrays)
    return np.stack([array[:min_len] for array in arrays], axis=0)


def plot_summary(summary: Dict[str, object], fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    eval_steps = np.asarray(summary["eval_steps"], dtype=float)
    point_error = stack_and_trim(summary["point_error_list_all"])
    max_residual = stack_and_trim(summary["max_residual_list_all"])
    time_trace = stack_and_trim(summary["time_list_all"])

    common_len = min(eval_steps.shape[0], point_error.shape[1], max_residual.shape[1], time_trace.shape[1])
    eval_steps = eval_steps[:common_len]
    point_error = point_error[:, :common_len]
    max_residual = max_residual[:, :common_len]
    time_trace = time_trace[:, :common_len]
    mean_time = time_trace.mean(axis=0)

    metrics = [
        ("Point Error", eval_steps, np.maximum(point_error, 1e-16), "Outer Iteration"),
        ("Max Residual", eval_steps, np.maximum(max_residual, 1e-16), "Outer Iteration"),
        ("Point Error vs Time", mean_time, np.maximum(point_error, 1e-16), "Time (s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    color = plt.cm.tab10(2 if summary["method"] == "Scholtes-compact" else 3)
    for ax, (title, x_axis, values, xlabel) in zip(axes, metrics):
        mean = values.mean(axis=0)
        low = np.maximum(values.min(axis=0), 1e-16)
        high = np.maximum(values.max(axis=0), 1e-16)
        ax.plot(x_axis, mean, color=color, linewidth=2.0, label=summary["method"])
        ax.fill_between(x_axis, low, high, color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
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


def parse_args(form: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {form} Scholtes relaxation on the deterministic PBO example.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=10)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=1e-1)
    parser.add_argument("--t-decay", type=float, default=5e-2)
    parser.add_argument("--xtol", type=float, default=1e-6)
    parser.add_argument("--maxfev", type=int, default=10000)
    parser.add_argument("--fb-epsilon", type=float, default=1e-2)
    parser.add_argument(
        "--solve-timeout",
        type=float,
        default=200.0,
        help="Seconds per fsolve call. If > 0, keep the best residual point seen before timeout.",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=2,
        help="Stop the current seed after this many consecutive fsolve stagnation exits (ier 4/5) or timeouts. 0 disables.",
    )
    parser.add_argument("--x-low", type=float, default=-0.9)
    parser.add_argument("--x-high", type=float, default=0.9)
    parser.add_argument("--y-low", type=float, default=-0.9)
    parser.add_argument("--y-high", type=float, default=0.9)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--fig-path", type=str, default="")
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
        fb_epsilon=args.fb_epsilon,
        solve_timeout=args.solve_timeout,
        stagnation_patience=args.stagnation_patience,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
    )


def main(form: str) -> None:
    if form not in {FORM_COMPACT, FORM_DETAILED}:
        raise ValueError(f"Unknown Scholtes form: {form}")

    args = parse_args(form)
    config = config_from_args(args)
    traces: List[Dict[str, List[float]]] = []

    for rep in range(config.num_experiments):
        seed = config.base_seed + rep
        print(f"repeat {rep + 1}/{config.num_experiments}, seed={seed}, method={form}")
        traces.append(solve_one_experiment(form, config, seed))

    summary = build_summary(form, traces, config)
    out_dir = result_dir()
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else out_dir / (
        f"scholtes_{form}_n{config.n}_outer{config.outer_iterations}.pt"
    )
    torch.save({"summary": summary}, data_path)
    print(f"Saved data: {data_path}")

    if not args.no_plot:
        fig_path = Path(args.fig_path).expanduser().resolve() if args.fig_path else out_dir / (
            f"scholtes_{form}_n{config.n}_outer{config.outer_iterations}.png"
        )
        plot_summary(summary, fig_path)
        print(f"Saved figure: {fig_path}")
