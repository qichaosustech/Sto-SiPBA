from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from AdaProx_common import (
    AdaProxParams,
    AdaProxState,
    add_state,
    append_trace,
    build_summary,
    clone_state,
    constraint_hat,
    elapsed_since,
    empty_trace,
    evaluate_state,
    gradient_state,
    gradient_step,
    initial_state,
    inner_solution,
    objective_prox,
    plot_summary,
    result_dir,
    scale_state,
    timed_step_start,
    zeros_like_state,
)


def solve_subproblem_pd(
    state: AdaProxState,
    center: AdaProxState,
    lambda_vec: torch.Tensor,
    outer_iter: int,
    params: AdaProxParams,
    primal_lr: float,
    dual_lr: float,
    dual_momentum: float,
    lambda_max: float,
) -> tuple[AdaProxState, torch.Tensor]:
    weighted_sum = zeros_like_state(state)
    sum_weight = 0.0
    h_prev = None

    for t in range(params.T):
        y_hat = inner_solution(state.x, state.y, params)
        h_hat = constraint_hat(state, center, y_hat, outer_iter, params)

        if h_prev is None or dual_momentum == 0.0:
            dual_direction = h_hat
        else:
            dual_direction = (1.0 + dual_momentum) * h_hat - dual_momentum * h_prev

        with torch.no_grad():
            lambda_vec += dual_lr * dual_direction.detach()
            lambda_vec.clamp_(min=0.0, max=lambda_max)

        lagrangian = objective_prox(state, center, params.sigma) + torch.dot(lambda_vec.detach(), h_hat)
        grads = gradient_state(lagrangian, state)
        state = gradient_step(state, grads, primal_lr, params)

        weight = t + 1.0
        weighted_sum = add_state(weighted_sum, scale_state(clone_state(state, requires_grad=False), weight))
        sum_weight += weight
        h_prev = h_hat.detach()

    averaged = scale_state(weighted_sum, 1.0 / sum_weight)
    return clone_state(averaged, requires_grad=True), lambda_vec


def run_one_experiment(
    params: AdaProxParams,
    device: str,
    seed: int,
    log_every: int,
    primal_lr: float,
    dual_lr: float,
    dual_momentum: float,
    lambda_max: float,
    reset_lambda_per_outer: bool,
) -> Dict[str, List[float]]:
    state = initial_state(params, device, seed)
    center = clone_state(state, requires_grad=False)
    q = 2 * params.n + 3
    lambda_vec = torch.zeros(q, device=device)
    trace = empty_trace()
    total_time = 0.0

    metrics = evaluate_state(state, params, device)
    append_trace(trace, 0, total_time, metrics)
    print(
        f"Iters:0, loss_upper:{metrics['loss_upper']:.6g}, "
        f"loss_lower:{metrics['loss_lower']:.6g}, max_h:{metrics['max_constraint']:.6g}"
    )

    for k in range(params.K):
        if reset_lambda_per_outer:
            lambda_vec.zero_()

        start = timed_step_start()
        state, lambda_vec = solve_subproblem_pd(
            state=state,
            center=center,
            lambda_vec=lambda_vec,
            outer_iter=k,
            params=params,
            primal_lr=primal_lr,
            dual_lr=dual_lr,
            dual_momentum=dual_momentum,
            lambda_max=lambda_max,
        )
        total_time += elapsed_since(start)
        center = clone_state(state, requires_grad=False)

        global_step = (k + 1) * params.T
        if global_step % log_every == 0 or k == params.K - 1:
            metrics = evaluate_state(state, params, device)
            append_trace(trace, global_step, total_time, metrics)
            lambda_norm = float(torch.linalg.vector_norm(lambda_vec.detach(), ord=1).cpu())
            print(
                f"Iters:{global_step}, loss_upper:{metrics['loss_upper']:.6g}, "
                f"loss_lower:{metrics['loss_lower']:.6g}, max_h:{metrics['max_constraint']:.6g}, "
                f"lambda_l1:{lambda_norm:.6g}"
            )

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaProx-PD on the deterministic PBO example.")
    parser.add_argument("--num-experiments", type=int, default=1)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--K", type=int, default=100, help="AdaProx outer iterations")
    parser.add_argument("--T", type=int, default=200, help="PD iterations per AdaProx subproblem")
    parser.add_argument("--N-inner", type=int, default=20, help="Inner GD steps if --no-exact-inner is used")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--xi", type=float, default=1e-2)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--primal-lr", type=float, default=1e-3)
    parser.add_argument("--dual-lr", type=float, default=1e-3)
    parser.add_argument("--dual-momentum", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=100.0)
    parser.add_argument("--reset-lambda-per-outer", action="store_true")
    parser.add_argument("--w-max", type=float, default=100.0)
    parser.add_argument("--inner-lr", type=float, default=0.0)
    parser.add_argument("--inner-lr-scale", type=float, default=1.0)
    parser.add_argument("--exact-inner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fig-path", type=str, default="")
    parser.add_argument("--data-path", type=str, default="")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but CUDA device was requested")
        device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    params = AdaProxParams(
        n=args.n,
        K=args.K,
        T=args.T,
        N_inner=args.N_inner,
        alpha=args.alpha,
        xi=args.xi,
        sigma=args.sigma,
        beta=args.beta,
        w_max=args.w_max,
        exact_inner=args.exact_inner,
        inner_lr=args.inner_lr,
        inner_lr_scale=args.inner_lr_scale,
    )

    run_outputs: List[Dict[str, List[float]]] = []
    for rep in range(args.num_experiments):
        seed = args.base_seed + rep
        print(f"repeat {rep + 1}/{args.num_experiments}, seed={seed}")
        run_outputs.append(
            run_one_experiment(
                params=params,
                device=args.device,
                seed=seed,
                log_every=args.log_every,
                primal_lr=args.primal_lr,
                dual_lr=args.dual_lr,
                dual_momentum=args.dual_momentum,
                lambda_max=args.lambda_max,
                reset_lambda_per_outer=args.reset_lambda_per_outer,
            )
        )

    summary = build_summary(
        run_outputs,
        params,
        {
            "method": "AdaProx-PD",
            "num_experiments": args.num_experiments,
            "base_seed": args.base_seed,
            "device": args.device,
            "primal_lr": args.primal_lr,
            "dual_lr": args.dual_lr,
            "dual_momentum": args.dual_momentum,
            "lambda_max": args.lambda_max,
            "reset_lambda_per_outer": args.reset_lambda_per_outer,
        },
    )

    out_dir = result_dir()
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else out_dir / (
        f"adaprox_pd_n{args.n}_K{args.K}_T{args.T}.pt"
    )
    torch.save({"summary": summary}, data_path)
    print(f"Saved data: {data_path}")

    fig_path = Path(args.fig_path).expanduser().resolve() if args.fig_path else out_dir / (
        f"adaprox_pd_n{args.n}_K{args.K}_T{args.T}.png"
    )
    plot_summary(summary, fig_path, label="AdaProx-PD")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
