import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


# Upper-level objective sample: F(x, y; w)
def F(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    e = torch.ones_like(x)
    return  torch.norm(x + w - e, p=2) ** 2 -  (n ** 0.5)*torch.norm(y - e, p=2)


# Lower-level objective sample: f(x, y; v)
def f(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    e= torch.ones_like(x)
    LOWER_SCALE = 10.0
    return LOWER_SCALE * torch.norm(torch.dot(y + v, e) - torch.norm(x, p=2) ** 2)**2 / n

# Aggregated objective psi(x,y,z; w,v,rho,sigma,delta)
def aggregation(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    rho: float,
    sigma: float,
    delta: float,
) -> torch.Tensor:
    return (
        F(x, y, w)
        - rho * (f(x, y, v) - f(x, z, v))
        - sigma * torch.sum(y * z)
        + 0.5 * sigma * torch.sum(z ** 2)
        - 0.5 * delta * torch.sum(y ** 2)
    )


def sample_single_noise(
    dim: int,
    std: float = 0.1,
    device: str = "cuda:0",
    noise_type: str = "gaussian",
    noise_df: float = 5.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if noise_type == "gaussian":
        return torch.randn(dim, device=device, generator=generator) * std
    if noise_type == "uniform":
        bound = np.sqrt(3.0) * std
        return (2.0 * torch.rand(dim, device=device, generator=generator) - 1.0) * bound
    if noise_type == "laplace":
        scale = std / np.sqrt(2.0)
        uniform = torch.rand(dim, device=device, generator=generator) - 0.5
        return -scale * torch.sign(uniform) * torch.log1p(-2.0 * torch.abs(uniform))
    if noise_type == "rademacher":
        return (
            2.0
            * torch.randint(0, 2, (dim,), device=device, dtype=torch.int64, generator=generator).float()
            - 1.0
        ) * std
    if noise_type == "student_t":
        if noise_df <= 2.0:
            raise ValueError("noise_df must be > 2 for student_t noise to have finite variance")
        scale = std * np.sqrt((noise_df - 2.0) / noise_df)
        dist = torch.distributions.StudentT(
            df=torch.tensor(noise_df, device=device),
            loc=torch.tensor(0.0, device=device),
            scale=torch.tensor(scale, device=device),
        )
        return dist.sample((dim,))
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def sample_noise(
    dim: int,
    std: float = 0.1,
    device: str = "cuda:0",
    noise_type: str = "gaussian",
    noise_df: float = 5.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = sample_single_noise(
        dim,
        std=std,
        device=device,
        noise_type=noise_type,
        noise_df=noise_df,
        generator=generator,
    )
    v = sample_single_noise(
        dim,
        std=std,
        device=device,
        noise_type=noise_type,
        noise_df=noise_df,
        generator=generator,
    )
    return w, v


def proj_X(x: torch.Tensor, low: float = -10, high: float = 10.0) -> torch.Tensor:
    return torch.clamp(x, min=low, max=high)


def proj_Y(y: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.clamp(y, min=low, max=high)


def schedules(k: int, params: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    t = params["t"]
    s = params["s"]
    alpha0 = params["alpha0"]
    beta0 = params["beta0"]
    rho0 = params["rho0"]
    sigma0 = params["sigma0"]
    delta0 = params["delta0"]
    eta0 = params["eta0"]

    alpha_k = alpha0 * (k + 1) ** (-9 * t - s)
    beta_k = beta0 * (k + 1) ** (-4 * t - s)
    rho_k = rho0 * (k + 1) ** t
    sigma_k = sigma0 * (k + 1) ** (-t)
    delta_k = delta0 * (k + 1) ** (-t)
    eta_k = eta0 * (k + 1) ** (-5*t-s)
    return alpha_k, beta_k, rho_k, sigma_k, delta_k, eta_k


def set_random_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed: Optional[int], device: str) -> Optional[torch.Generator]:
    if seed is None:
        return None

    torch_device = torch.device(device)
    generator_device = "cuda" if torch_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def compute_total_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    x_star: torch.Tensor,
    y_star: torch.Tensor,
) -> Tuple[float, float, float]:
    n = x.shape[0]
    loss_upper = (torch.norm(x - x_star, p=2) ** 2) / n
    e = torch.ones_like(x)
    y_star=torch.norm(x, p=2)**2*e/n
    loss_lower = (torch.norm(y - y_star, p=2) ** 2) / n
    total_loss = loss_upper + loss_lower
    return loss_upper.item(), loss_lower.item(), total_loss.item()


def run_until_threshold(
    n: int,
    iterations: int,
    noise_std: float,
    device: str,
    schedule_params: Dict[str, float],
    threshold: float,
    seed: int,
    noise_seed: Optional[int] = None,
    use_variance_reduction: bool = True,
    noise_type: str = "gaussian",
    noise_df: float = 5.0,
    x_low: float = -0.9,
    x_high: float = 0.9,
    y_low: float = -0.9,
    y_high: float = 0.9,
) -> Dict[str, float]:
    set_random_seed(seed, device)
    init_generator = make_torch_generator(seed, device)
    noise_generator = make_torch_generator(noise_seed, device)

    x = 1.8 * torch.rand(n, device=device, generator=init_generator) -0.9* torch.ones(n, device=device)
    y = 1.8 * torch.rand(n, device=device, generator=init_generator) - 0.9*torch.ones(n, device=device)
    z = y.detach().clone()

    e = torch.ones(n, device=device)
    x_star = 0.5 * e
    y_star = 0.25 * e

    total_time = 0.0
    d_x_prev = None
    x_prev = None
    y_prev = None
    z_prev = None

    for k in range(iterations):
        loss_upper, loss_lower, total_loss = compute_total_loss(x, y, x_star, y_star)
        if max(loss_upper, loss_lower) <= threshold:
            return {
                "reached": 1.0,
                "hit_time": total_time,
                "hit_iteration": float(k),
                "final_total_loss": total_loss,
                "final_loss_upper": loss_upper,
                "final_loss_lower": loss_lower,
            }

        alpha_k, beta_k, rho_k, sigma_k, delta_k, eta_k = schedules(k, schedule_params)
        start = time.perf_counter()

        w_u, v_u = sample_noise(
            n,
            std=noise_std,
            device=device,
            noise_type=noise_type,
            noise_df=noise_df,
            generator=noise_generator,
        )

        x_req = x.detach().clone().requires_grad_(True)
        y_req = y.detach().clone().requires_grad_(True)
        z_req = z.detach().clone().requires_grad_(True)

        psi_u = aggregation(x_req, y_req, z_req, w_u, v_u, rho_k, sigma_k, delta_k)
        d_y = torch.autograd.grad(psi_u, y_req, retain_graph=True)[0]
        d_z = torch.autograd.grad(psi_u, z_req)[0]

        y_new = proj_Y(y + beta_k * d_y.detach(), low=y_low, high=y_high)
        z_new = proj_Y(z - beta_k * d_z.detach(), low=y_low, high=y_high)

        w_x, v_x = sample_noise(
            n,
            std=noise_std,
            device=device,
            noise_type=noise_type,
            noise_df=noise_df,
            generator=noise_generator,
        )

        x_cur = x.detach().clone().requires_grad_(True)
        y_cur = y_new.detach().clone()
        z_cur = z_new.detach().clone()
        psi_x_cur = aggregation(x_cur, y_cur, z_cur, w_x, v_x, rho_k, sigma_k, delta_k)
        g_cur = torch.autograd.grad(psi_x_cur, x_cur)[0].detach()

        if use_variance_reduction:
            if k == 0:
                d_x = g_cur
            else:
                x_old = x_prev.detach().clone().requires_grad_(True)
                _, _, rho_pre, sigma_pre, delta_pre, _ = schedules(k - 1, schedule_params)
                psi_x_old = aggregation(x_old, y_prev, z_prev, w_x, v_x, rho_pre, sigma_pre, delta_pre)
                g_old = torch.autograd.grad(psi_x_old, x_old)[0].detach()
                d_x = g_cur + (1.0 - eta_k) * (d_x_prev - g_old)
        else:
            d_x = g_cur

        x_new = proj_X(x - alpha_k * d_x, low=x_low, high=x_high)

        x_prev = x.detach().clone()
        y_prev = y_new.detach().clone()
        z_prev = z_new.detach().clone()
        d_x_prev = d_x.detach().clone()
        x, y, z = x_new.detach(), y_new.detach(), z_new.detach()

        total_time += time.perf_counter() - start

    loss_upper, loss_lower, total_loss = compute_total_loss(x, y, x_star, y_star)
    reached = 1.0 if max(loss_upper, loss_lower) <= threshold else 0.0
    hit_time = total_time if reached else float("nan")
    hit_iteration = float(iterations) if reached else float("nan")
    return {
        "reached": reached,
        "hit_time": hit_time,
        "hit_iteration": hit_iteration,
        "final_total_loss": total_loss,
        "final_loss_upper": loss_upper,
        "final_loss_lower": loss_lower,
    }


def run_one_experiment(
    n: int = 100,
    iterations: int = 20000,
    noise_std: float = 0.1,
    device: str = "cuda:0",
    schedule_params: Optional[Dict[str, float]] = None,
    use_variance_reduction: bool = True,
    noise_type: str = "gaussian",
    noise_df: float = 5.0,
    log_every: int = 100,
    seed: Optional[int] = None,
    noise_seed: Optional[int] = None,
) -> Dict[str, List]:
    if schedule_params is None:
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

    init_generator = make_torch_generator(seed, device)
    noise_generator = make_torch_generator(noise_seed, device)

    x = 1.8 * torch.rand(n, device=device, generator=init_generator) - 0.9 * torch.ones(n, device=device)
    y = 1.8*torch.rand(n, device=device, generator=init_generator)-0.9*torch.ones(n, device=device)
    z = y.detach().clone()

    e = torch.ones(n, device=device)
    x_star = 1/2*e 
    y_star = 1/4*e 
    x_low = -0.9
    x_high = 0.9
    y_low = -0.9
    y_high = 0.9

    point_error: List[float] = []
    eval_steps: List[int] = []
    loss_upper_trace: List[float] = []
    loss_lower_trace: List[float] = []
    time_trace: List[float] = []

    total_time = 0.0
    loss_upper = (torch.norm(x - x_star, p=2) ** 2) / n
    loss_lower = (torch.norm(y - y_star, p=2) ** 2) / n
    d_x_prev = None
    x_prev = None
    y_prev = None
    z_prev = None

    for k in range(iterations):
        if k % log_every == 0 or k == iterations - 1:
            loss_upper = (torch.norm(x - x_star, p=2) ** 2 ) / n
            y_star=torch.norm(x, p=2)**2*e/n
            loss_lower = (torch.norm(y - y_star, p=2) ** 2) / n
            loss_upper_trace.append(loss_upper.item())
            loss_lower_trace.append(loss_lower.item())
            point_error.append(loss_upper_trace[-1] + loss_lower_trace[-1])
            eval_steps.append(k)
            time_trace.append(total_time)
            print(f"Iters:{k},loss_upper:{loss_upper.item():.6f}, loss_lower:{loss_lower.item():.6f}, loss:{point_error[-1]:.6f}")

        alpha_k, beta_k, rho_k, sigma_k, delta_k, eta_k = schedules(k, schedule_params)
        start = time.time()

        # Sample for (y,z)-update direction d_y^k, d_z^k.
        w_u, v_u = sample_noise(
            n,
            std=noise_std,
            device=device,
            noise_type=noise_type,
            noise_df=noise_df,
            generator=noise_generator,
        )

        x_req = x.detach().clone().requires_grad_(True)
        y_req = y.detach().clone().requires_grad_(True)
        z_req = z.detach().clone().requires_grad_(True)

        psi_u = aggregation(x_req, y_req, z_req, w_u, v_u, rho_k, sigma_k, delta_k)
        d_y = torch.autograd.grad(psi_u, y_req, retain_graph=True)[0]
        d_z = torch.autograd.grad(psi_u, z_req)[0]

        # y^{k+1} = Proj_Y(y^k + beta_k d_y^k), z^{k+1} = Proj_Y(z^k - beta_k d_z^k)
        y_new = proj_Y(y + beta_k * d_y.detach(), low=y_low, high=y_high)
        z_new = proj_Y(z - beta_k * d_z.detach(), low=y_low, high=y_high)

        # Sample for x-gradient estimator.
        w_x, v_x = sample_noise(
            n,
            std=noise_std,
            device=device,
            noise_type=noise_type,
            noise_df=noise_df,
            generator=noise_generator,
        )

        x_cur = x.detach().clone().requires_grad_(True)
        y_cur = y_new.detach().clone()
        z_cur = z_new.detach().clone()
        psi_x_cur = aggregation(x_cur, y_cur, z_cur, w_x, v_x, rho_k, sigma_k, delta_k)
        g_cur = torch.autograd.grad(psi_x_cur, x_cur)[0].detach()

        if use_variance_reduction:
            if k == 0:
                d_x = g_cur
            else:
                # Variance-reduced estimator for x-update.
                x_old = x_prev.detach().clone().requires_grad_(True)
                alpha_pre, beta_pre, rho_pre, sigma_pre, delta_pre, eta_pre = schedules(k - 1, schedule_params)
                psi_x_old = aggregation(x_old, y_prev, z_prev, w_x, v_x, rho_pre, sigma_pre, delta_pre)
                g_old = torch.autograd.grad(psi_x_old, x_old)[0].detach()
                d_x = g_cur + (1.0 - eta_k) * (d_x_prev - g_old)
        else:
            # Plain gradient descent on x without variance reduction.
            d_x = g_cur

        # x^{k+1} = Proj_X(x^k - alpha_k d_x^k)
        x_new = proj_X(x - alpha_k * d_x, low=x_low, high=x_high)

        x_prev = x.detach().clone()
        y_prev = y_new.detach().clone()
        z_prev = z_new.detach().clone()
        d_x_prev = d_x.detach().clone()

        x, y, z = x_new.detach(), y_new.detach(), z_new.detach()

        total_time += time.time() - start

    return {
        "point_error": point_error,
        "eval_steps": eval_steps,
        "loss_upper_trace": loss_upper_trace,
        "loss_lower_trace": loss_lower_trace,
        "time_trace": time_trace
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot use GPU 0.")
    torch.cuda.set_device(0)

    num_experiments = 10
    iterations = 5000
    n = 100
    std = 1
    log_every = 50
    base_seed = 42
    device = "cuda:0"
    use_variance_reduction = True
    vr_tag = "vr_on" if use_variance_reduction else "vr_off"

    schedule_params = {
        "t": 0.01,
        "s": 0.5,
        "alpha0": 0.1,
        "beta0": 0.1,
        "rho0": 10,
        "sigma0": 1e-4,
        "delta0": 1e-4,
        "eta0": 1.0,
    }

    point_error_list_all = []
    loss_upper_list_all = []
    loss_lower_list_all = []
    time_list_all = []
    x_list_all = []
    y_list_all = []

    run_name = f"n{n}_it{iterations}_std{std}_{vr_tag}"

    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    eval_steps_ref = None
    for exp in range(num_experiments):
        print(f"experiment:{exp}/{num_experiments}")
        seed = base_seed + exp
        noise_seed = seed
        out = run_one_experiment(
            n=n,
            iterations=iterations,
            noise_std=std,
            device=device,
            schedule_params=schedule_params,
            use_variance_reduction=use_variance_reduction,
            log_every=log_every,
            seed=seed,
            noise_seed=noise_seed,
        )

        exp_save_data = {
            "experiment_index": exp,
            "seed": seed,
            "noise_seed": noise_seed,
            "run_name": run_name,
            "use_variance_reduction": use_variance_reduction,
            "device": device,
            "n": n,
            "iterations": iterations,
            "noise_std": std,
            "log_every": log_every,
            "schedule_params": schedule_params,
            **out,
        }

        point_error_list_all.append(out["point_error"])
        loss_upper_list_all.append(out["loss_upper_trace"])
        loss_lower_list_all.append(out["loss_lower_trace"])
        time_list_all.append(out["time_trace"])
        if eval_steps_ref is None:
            eval_steps_ref = out["eval_steps"]

    summary_save_data = {
        "run_name": run_name,
        "use_variance_reduction": use_variance_reduction,
        "eval_steps": eval_steps_ref,
        "point_error_list_all": point_error_list_all,
        "loss_upper_list_all": loss_upper_list_all,
        "loss_lower_list_all": loss_lower_list_all,
        "time_list_all": time_list_all,
        "schedule_params": schedule_params,
        "base_seed": base_seed,
        "noise_seed_mode": "same_as_initial_seed",
    }
    summary_path = result_dir / f"{run_name}_summary.pt"
    torch.save(summary_save_data, summary_path)
    print(f"saved summary: {summary_path}")
