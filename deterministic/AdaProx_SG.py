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


def solve_subproblem_sg(
    state: AdaProxState,
    center: AdaProxState,
    outer_iter: int,
    params: AdaProxParams,
    gamma0: float,
    threshold: float,
    constraint_step_interval: int = 0,
) -> tuple[AdaProxState, float]:
    weighted_sum = zeros_like_state(state)
    sum_weight = 0.0

    for t in range(params.T):
        gamma = gamma0 * (t + 1.0)
        y_hat = inner_solution(state.x, state.y, params)
        h_hat = constraint_hat(state, center, y_hat, outer_iter, params)
        max_value, max_index = torch.max(h_hat, dim=0)
        violates_constraint = float(max_value.detach().cpu()) > threshold
        if constraint_step_interval > 0 and violates_constraint:
            violates_constraint = t % constraint_step_interval == 0

        if not violates_constraint:
            if float(max_value.detach().cpu()) <= threshold:
                weighted_sum = add_state(weighted_sum, scale_state(clone_state(state, requires_grad=False), gamma))
                sum_weight += gamma
            loss = objective_prox(state, center, params.sigma)
        else:
            loss = h_hat[max_index]

        grads = gradient_state(loss, state)
        state = gradient_step(state, grads, 1.0 / gamma, params)

    if sum_weight > 0.0:
        averaged = scale_state(weighted_sum, 1.0 / sum_weight)
        return clone_state(averaged, requires_grad=True), sum_weight

    return state, sum_weight

def run_one_experiment(
    params: AdaProxParams,
    device: str,
    seed: int,
    log_every: int,
    gamma0: float,
    threshold_scale: float,
    constraint_step_interval: int = 0,
) -> Dict[str, List[float]]:
    state = initial_state(params, device, seed)
    center = clone_state(state, requires_grad=False)
    trace = empty_trace()
    total_time = 0.0
    threshold = threshold_scale * params.epsilon_sub

    metrics = evaluate_state(state, params, device)
    append_trace(trace, 0, total_time, metrics)
    print(
        f"Iters:0, loss_upper:{metrics['loss_upper']:.6g}, "
        f"loss_lower:{metrics['loss_lower']:.6g}, max_h:{metrics['max_constraint']:.6g}"
    )

    for k in range(params.K):
        start = timed_step_start()
        state, sum_weight = solve_subproblem_sg(
            state=state,
            center=center,
            outer_iter=k,
            params=params,
            gamma0=gamma0,
            threshold=threshold,
            constraint_step_interval=constraint_step_interval,
        )
        total_time += elapsed_since(start)
        center = clone_state(state, requires_grad=False)

        global_step = (k + 1) * params.T
        if global_step % log_every == 0 or k == params.K - 1:
            metrics = evaluate_state(state, params, device)
            append_trace(trace, global_step, total_time, metrics)
            print(
                f"Iters:{global_step}, loss_upper:{metrics['loss_upper']:.6g}, "
                f"loss_lower:{metrics['loss_lower']:.6g}, max_h:{metrics['max_constraint']:.6g}, "
                f"feasible_weight:{sum_weight:.3g}"
            )

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaProx-SG on the deterministic PBO example.")
    parser.add_argument("--num-experiments", type=int, default=1)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--K", type=int, default=100, help="AdaProx outer iterations")
    parser.add_argument("--T", type=int, default=200, help="SG iterations per AdaProx subproblem")
    parser.add_argument("--N-inner", type=int, default=20, help="Inner GD steps if --no-exact-inner is used")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--xi", type=float, default=1e-2)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--gamma0", type=float, default=10.0)
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    parser.add_argument(
        "--constraint-step-interval",
        type=int,
        default=0,
        help=(
            "If > 0, when constraints are violated, take a max-constraint step only every N "
            "SG iterations and use objective descent on the other violated iterations."
        ),
    )
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
                gamma0=args.gamma0,
                threshold_scale=args.threshold_scale,
                constraint_step_interval=args.constraint_step_interval,
            )
        )

    summary = build_summary(
        run_outputs,
        params,
        {
            "method": "AdaProx-SG",
            "num_experiments": args.num_experiments,
            "base_seed": args.base_seed,
            "device": args.device,
            "gamma0": args.gamma0,
            "threshold_scale": args.threshold_scale,
            "constraint_step_interval": args.constraint_step_interval,
        },
    )

    out_dir = result_dir()
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else out_dir / (
        f"adaprox_sg_n{args.n}_K{args.K}_T{args.T}.pt"
    )
    torch.save({"summary": summary}, data_path)
    print(f"Saved data: {data_path}")

    fig_path = Path(args.fig_path).expanduser().resolve() if args.fig_path else out_dir / (
        f"adaprox_sg_n{args.n}_K{args.K}_T{args.T}.png"
    )
    plot_summary(summary, fig_path, label="AdaProx-SG")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
