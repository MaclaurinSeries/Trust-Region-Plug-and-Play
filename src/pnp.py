from typing import Callable, Dict, List, Optional, Tuple, Union
import time

import string
import pandas as pd
from pathlib import Path
import numpy as np

import torch
import torchvision

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.physics import Physics

from .denoiser import Denoiser, RelaxedDenoiser, GradientStepDenoiser
from .utils import DEVICE, to_image
from .preconditioner import get_blur_preconditioner, _M_norm_sq, _M_inner
from .second_order import SecondOrderFBE, SecondOrderDRE

from .steihaug import Steihaug
from .LBFGS import LBFGSSolver

psnr_real = dinv.metric.PSNR(max_pixel=1, min_pixel=0)
ssim_real = dinv.metric.SSIM(max_pixel=1, min_pixel=0)
psnr_cplx = dinv.metric.PSNR(max_pixel=1, min_pixel=0, complex_abs=True)
ssim_cplx = dinv.metric.PSNR(max_pixel=1, min_pixel=0, complex_abs=True)


def _random_name(length: int = 10) -> str:
    import numpy as np

    return "".join(
        np.random.choice(list(string.ascii_letters + string.digits), size=length)
    )


def _init_run(
    x_target: torch.Tensor,
    physics: Physics,
    denoiser: Denoiser,
    device: torch.device,
    save_dir: Optional[Union[str, Path]],
    name: Optional[str],
    y_observed: Optional[torch.Tensor] = None,
    v_noise: float = 0.01,
    weighted: bool = False,
):
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser.nfe = 0
    if name is None:
        name = _random_name()

    if save_dir is not None and not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    x_true = x_target.clone().to(device)
    if y_observed is None:
        y = physics(x_true)
    else:
        y = y_observed.to(device)
    x_init = physics.A_adjoint(y).clone().to(device)

    denoiser = denoiser.to(device)

    if save_dir is not None:
        torchvision.utils.save_image(y, save_dir / f"measured_{name}.png")
        torchvision.utils.save_image(x_true, save_dir / f"target_{name}.png")

    if weighted:
        F = L2(sigma=v_noise)
    else:
        F = L2()
    metrics = MetricsTracker(name)

    return name, x_true, y, x_init, denoiser, save_dir, F, metrics


def _save_iterate(
    x: torch.Tensor,
    save_dir: Optional[Path],
    name: str,
    iteration: int,
    save_checkpoint: List[int],
    total_iter: int,
) -> None:
    """Persist iterate to disk if iteration is a checkpoint or the final step."""
    if save_dir is None:
        return
    if iteration in save_checkpoint or iteration == total_iter:
        torchvision.utils.save_image(
            x, save_dir / f"reconstructed_{name}_iteration_{iteration}.png"
        )


class MetricsTracker:
    def __init__(self, name: str):
        self.name = name
        self._records: List[dict] = []

    def record(self, recorded: list = ["name"], **kwargs) -> None:
        rec = {
            "name": self.name,
            **{
                key: float(val)
                for key, val in kwargs.items()
                if key in recorded and val is not None
            },
        }
        self._records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


def pgd(
    # pnp parameter
    x: torch.Tensor,
    denoiser: Denoiser,
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 5_000,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100, 500, 1_000],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # pgd specific parameter
    gamma: float = 1.0,
    # stopping condition
    min_step_norm: float = None,
    max_time: float = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    PnP Proximal Gradient Descent.
        x^{k+1} = D( x^k - gamma * grad_h(x^k) )
    """
    sigma = sigma_v * v_noise
    lam = (gamma + 2) / (gamma + 1)
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser._initialize_grad_func(sigma=sigma, lam=gamma)
    name, x_true, y, x, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise
    )
    step_0 = torch.linalg.norm(x.flatten()).item()

    total_time = 0
    for k in range(1, total_iter + 1):
        start_time = time.perf_counter()

        x_prev = x.clone()

        u = x_prev - lam * F.grad(x_prev, y, physics)
        x = denoiser(u, sigma)

        end_time = time.perf_counter()

        psnr = psnr_real(to_image(x), to_image(x_true))
        step_norm = torch.linalg.norm(x.flatten() - x_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(x, x_true) if k == total_iter else None

        metrics.record(
            recorded=["psnr", "step_norm", "time", "time_per_round"],
            psnr=psnr,
            step_norm=step_norm,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(y, save_dir, name, k, save_checkpoint, total_iter)
        if min_step_norm is not None and step_norm < min_step_norm:
            break

        if max_time is not None and total_time >= max_time:
            break

    return metrics.to_dataframe(), torch.clamp(x, 0.0, 1.0)


def drs(
    # pnp parameter
    x: torch.Tensor,
    denoiser: "GradientStepDenoiser",
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 5_000,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100, 500, 1_000, 5_000],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # drs specific parameter
    lam: float = 1.0,
    beta: float = 0.25,
    gamma: float = 0.45,
    # stopping condition
    min_step_norm: float = None,
    max_time: float = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    PnP Douglas-Rachford Splitting (Hurault et al. 2022, Theorem 4.4, eq. 14).

    Variant for non-differentiable f (denoiser comes first):
        y_{k+1} = D_sigma(x_k)
        z_{k+1} = prox_{lam * f}(2 y_{k+1} - x_k)
        x_{k+1} = x_k + (z_{k+1} - y_{k+1})
    """
    sigma = sigma_v * v_noise
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser._initialize_grad_func(sigma=sigma, lam=gamma)
    name, x_true, y, x_init, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise
    )
    step_0 = torch.linalg.norm(x.flatten()).item()

    x = x_init.clone()
    u = x_init.clone()

    total_time = 0
    for k in range(1, total_iter + 1):
        start_time = time.perf_counter()

        u_prev = u.clone()
        x_prev = x.clone()

        x = denoiser.forward(u_prev, sigma)
        z_in = 2 * x - u_prev
        z = F.prox(z_in, y, physics, gamma=lam)
        u = u_prev + (2 * beta) * (z - x)

        end_time = time.perf_counter()

        psnr = psnr_real(to_image(x), to_image(x_true))
        step_norm = torch.linalg.norm(x.flatten() - x_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(x, x_true) if k == total_iter else None

        metrics.record(
            recorded=["psnr", "step_norm", "time", "time_per_round"],
            psnr=psnr,
            step_norm=step_norm,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(x, save_dir, name, k, save_checkpoint, total_iter)
        if min_step_norm is not None and step_norm < min_step_norm:
            break

        if max_time is not None and total_time >= max_time:
            break

    return metrics.to_dataframe(), torch.clamp(x, 0.0, 1.0)


def alpha_pgd(
    # pnp parameter
    x: torch.Tensor,
    denoiser: Denoiser,
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 5_000,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100, 500, 1_000, 5_000],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # pgd specific parameter
    alpha: float = 1.0,
    L_f: float = 1.0,
    # stopping condition
    min_step_norm: float = None,
    max_time: float = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Relaxed PnP-PGD (Hurault, Chambolle, Leclaire, Papadakis 2023, eq. 19/20).
        q_{k+1} = (1 - alpha) y_k + alpha x_k
        x_{k+1} = D_sigma(x_k - lam * grad_f(q_{k+1}))
        y_{k+1} = (1 - alpha) y_k + alpha x_{k+1}
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    sigma = sigma_v * v_noise
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser.alpha = alpha
        denoiser._initialize_grad_func(sigma=sigma, lam=1.0)

    name, x_true, y_obs, x, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise
    )
    step_0 = torch.linalg.norm(x.flatten()).item()

    lam = (alpha + 1) / (alpha * L_f)
    alpha_hat = 1 / (lam * L_f)

    y = x.clone()  # y_0 = x_0
    total_time = 0
    for k in range(1, total_iter + 1):
        start_time = time.perf_counter()
        y_prev = y.clone()

        q = (1.0 - alpha_hat) * y + alpha_hat * x
        u = x - lam * F.grad(q, y_obs, physics)

        x_new = denoiser(u, sigma)
        y = (1.0 - alpha_hat) * y + alpha_hat * x_new
        x = x_new

        end_time = time.perf_counter()

        psnr = psnr_real(to_image(y), to_image(x_true))
        step_norm = torch.linalg.norm(y.flatten() - y_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(y, x_true) if k == total_iter else None

        metrics.record(
            recorded=["psnr", "step_norm", "time", "time_per_round"],
            psnr=psnr,
            step_norm=step_norm,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(y, save_dir, name, k, save_checkpoint, total_iter)
        if min_step_norm is not None and step_norm < min_step_norm:
            break

        if max_time is not None and total_time >= max_time:
            break

    return metrics.to_dataframe(), torch.clamp(y, 0.0, 1.0)


def dpir(
    # pnp parameter
    x: torch.Tensor,
    denoiser: RelaxedDenoiser,
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 5_000,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100, 500, 1_000, 5_000],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # dpir specific parameter
    lam: float = 0.33,
    K_sigma_max: int = 8,
    # stopping condition
    min_step_norm: float = None,
    max_time: float = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    DPIR https://arxiv.org/pdf/2008.13751
    """
    sigma_start = 49 / 255
    sigma_target = sigma_v * v_noise

    sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_target), K_sigma_max)

    name, x_true, y_obs, x, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise, weighted=True
    )
    step_0 = torch.linalg.norm(x.flatten()).item()

    y = x.clone()  # y_0 = x_0
    total_time = 0
    for k in range(1, total_iter + 1):
        start_time = time.perf_counter()
        y_prev = y.clone()

        if k <= K_sigma_max:
            sigma_k = sigmas[k - 1]
        else:
            sigma_k = sigma_target
        x = F.prox(y_prev, y_obs, physics, gamma=(sigma_k**2) / lam)
        y = denoiser(x, sigma_k)

        end_time = time.perf_counter()

        psnr = psnr_real(to_image(y), to_image(x_true))
        step_norm = torch.linalg.norm(y.flatten() - y_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(y, x_true) if k == total_iter else None

        metrics.record(
            recorded=["psnr", "step_norm", "time", "time_per_round"],
            psnr=psnr,
            step_norm=step_norm,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(y, save_dir, name, k, save_checkpoint, total_iter)
        if min_step_norm is not None and step_norm < min_step_norm:
            break

        if max_time is not None and total_time >= max_time:
            break

    return metrics.to_dataframe(), torch.clamp(y, 0.0, 1.0)


def q_newton_prox_tan_2024(
    # PnP parameter
    x: torch.Tensor,
    denoiser: "GradientStepDenoiser",
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 100,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # LBFGS-specific hyperparameters
    alpha: float = 1.0,
    lam: float = 1.0,
    gamma: float = 1.0,
    beta: float = 0.01,
    max_ls_iter: int = 10,
    m_history: int = 10,
    # stopping condition
    min_step_norm: float = None,
    grad_tol: float = None,
    max_time: float = None,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    PnP-LBFGS (Tan, Mukherjee, Tang, Schonlieb 2024, SIAM J. Imag. Sci.).
    arXiv:2303.07271. Adapted from: https://github.com/hyt35/Prox-qN/blob/main/PnP_restoration/prox_PnP_restoration.py

    Required denoiser: Prox-DRUNet (Hurault, 2022. Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization).
    Pretrained weights are at:
        https://plmbox.math.cnrs.fr/f/faf7d62213e449fa9c8a/?dl=1.
    """
    sigma = sigma_v * v_noise
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser.alpha = alpha
        denoiser._initialize_grad_func(sigma=sigma, lam=lam * gamma)
    name, x_true, y_obs, x, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise
    )
    step_0 = torch.linalg.norm(x.flatten()).item()

    x = x.clone()
    total_time = 0

    second_order = SecondOrderFBE(y_obs, physics, denoiser, sigma, lam, gamma, F)
    lbfgs = LBFGSSolver(m=m_history)

    start_time = time.perf_counter()
    fbe_val, grad_fbe, _ = second_order._phi_and_grad(x)
    end_time = time.perf_counter()

    grad_norm = torch.linalg.vector_norm(grad_fbe)
    grad_0 = max(grad_norm, 1e-8)

    total_time = total_time + (end_time - start_time)

    for k in range(1, total_iter + 1):
        start_time = time.perf_counter()

        x_prev = x.clone()
        d = lbfgs.solve(grad_fbe)
        d_flat = d.flatten()

        # armijo line search (Algorithm 2.1 MINFBE)
        tau = 1.0
        w = x + tau * d
        fbe_w, grad_fbe_w, T_w = second_order._phi_and_grad(w)

        expected_red = beta * tau * torch.dot(grad_fbe.flatten(), d_flat)
        ls_count = 0
        while fbe_w - fbe_val > expected_red and ls_count < max_ls_iter:
            tau *= 0.5
            w = x + tau * d
            fbe_w, grad_fbe_w, T_w = second_order._phi_and_grad(w)
            expected_red = beta * tau * torch.dot(grad_fbe.flatten(), d_flat)
            ls_count += 1

        # update BGFS history
        s = w - x
        y = grad_fbe_w - grad_fbe
        lbfgs.update(s, y)

        x = T_w
        fbe_val, grad_fbe, _ = second_order._phi_and_grad(x)
        grad_norm = torch.linalg.vector_norm(grad_fbe)

        end_time = time.perf_counter()

        psnr = psnr_real(to_image(x), to_image(x_true))
        step_norm = torch.linalg.norm(x.flatten() - x_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(x, x_true) if k == total_iter else None

        metrics.record(
            recorded=["psnr", "step_norm", "grad_norm", "time", "time_per_round"],
            psnr=psnr,
            step_norm=step_norm,
            grad_norm=grad_norm / grad_0,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(x, save_dir, name, k, save_checkpoint, total_iter)
        if min_step_norm is not None and step_norm < min_step_norm:
            break

        if (
            min_step_norm is None
            and grad_tol is not None
            and grad_norm < grad_tol * grad_0
        ):
            break

        if max_time is not None and total_time >= max_time:
            break

    return metrics.to_dataframe(), torch.clamp(x, 0.0, 1.0)


def trust_region_dre(
    # PnP parameter
    x: torch.Tensor,
    denoiser: Union[RelaxedDenoiser, GradientStepDenoiser],
    physics: Physics,
    v_noise: float = 0.01,
    sigma_v: float = 1.0,
    total_iter: int = 5_000,
    name: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: List[int] = [0, 50, 100, 500, 1_000, 5_000],
    device: torch.device = DEVICE,
    y_observed: Optional[torch.Tensor] = None,
    on_iteration: Optional[Callable[[int, Dict], None]] = None,
    # TR parameter
    lam: float = 1.0,
    gamma: float = 1.0,
    use_potential: bool = False,
    eta_1: float = 0.25,
    eta_2: float = 0.75,
    cg_max_iter: int = 2,
    L_phi_estimate: Optional[float] = None,
    # stopping criterion
    min_step_norm: float = None,
    grad_tol: float = None,
    max_time: float = None,
    test_save: bool = False,
    # experiment parameters
    objective_relative_error: bool = False,
    export_series_x: bool = False,
    max_exported_x: int = 8,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    sigma = sigma_v * v_noise
    if isinstance(denoiser, GradientStepDenoiser):
        denoiser._initialize_grad_func(sigma=sigma, lam=lam * gamma)
    if objective_relative_error:
        assert isinstance(
            denoiser, GradientStepDenoiser
        ), "require Gradient-Step structure"
        use_potential = True

    name, x_true, y_obs, x_init, denoiser, save_dir, F, metrics = _init_run(
        x, physics, denoiser, device, save_dir, name, y_observed, v_noise
    )
    denoiser._use_potential = use_potential

    if test_save:
        prev_psnr = 0
        patience = 10
        drop_iterate = 0

    u = x_init.clone()
    step_0 = torch.linalg.norm(u.flatten()).item()

    total_time = 0
    start_time = time.perf_counter()
    second_order = SecondOrderDRE(y_obs, physics, denoiser, sigma, lam, gamma, F)
    steihaug = Steihaug(denoiser, physics, cg_max_iter, second_order)

    # point_k: (u_k, x_k), u_k latent, x_k target
    phi_val, grad_phi, point_k = second_order._phi_and_grad(u)

    end_time = time.perf_counter()
    total_time = total_time + (end_time - start_time)

    grad_norm = torch.linalg.vector_norm(grad_phi)
    grad_0 = max(grad_norm, 1e-8)
    if L_phi_estimate is None:
        Delta = grad_norm
    else:
        Delta = grad_norm / L_phi_estimate

    Delta_max = 10.0 * Delta
    eta_k = min(0.5, torch.sqrt(grad_norm))
    accepted = False

    if export_series_x:
        step_x = [point_k]

    for k in range(1, total_iter + 1):
        u_prev = u.clone()

        start_time = time.perf_counter()

        d, Hd = steihaug.find_d(
            gradient=grad_phi.detach(),
            point=point_k,
            Delta=Delta,
            forcing_tol=eta_k,
            last_accept=accepted,
        )

        u_trial = (u + d).detach()

        d_flat = d.flatten()
        pred_red = -torch.dot((grad_phi + 0.5 * Hd).flatten(), d_flat)

        phi_trial, grad_phi_trial, point_trial = second_order._phi_and_grad(u_trial)

        if denoiser.use_potential() and phi_trial is not None:
            actual_red = phi_val - phi_trial
        else:
            actual_red = -0.5 * torch.dot((grad_phi + grad_phi_trial).flatten(), d_flat)

        rho = actual_red / (pred_red + 1e-30)

        if objective_relative_error:
            estimated_red = -0.5 * torch.dot(
                (grad_phi + grad_phi_trial).flatten(), d_flat
            )
            estimated_rho = estimated_red / (pred_red + 1e-8)
            rho_rel_err = (estimated_rho - rho) ** 2 / (rho) ** 2
            red_rel_err = (estimated_red - actual_red) ** 2 / (actual_red) ** 2

        # TR radius update
        if rho < eta_1:
            Delta = 0.25 * Delta
        elif rho > eta_2 and steihaug._hit_boundary:
            Delta = min(2.0 * Delta, Delta_max)

        # Accept / reject step
        if rho > eta_1:
            u = u_trial.clone()
            accepted = True
        else:
            accepted = False

        end_time = time.perf_counter()

        # Evaluate PSNR and SSIM on x_k
        x_out = point_k[1]

        psnr = psnr_real(to_image(x_out), to_image(x_true))
        step_norm = torch.linalg.norm(u.flatten() - u_prev.flatten()).item() / step_0
        total_time = total_time + (end_time - start_time)
        ssim_val = ssim_real(x_out, x_true) if k == total_iter else None

        if accepted:
            lipschitz = (
                torch.linalg.norm(grad_phi_trial - grad_phi)
            ) / torch.linalg.norm(d).item()
        else:
            lipschitz = 0
        metrics.record(
            recorded=(
                [
                    "psnr",
                    "step_norm",
                    "time",
                    "time_per_round",
                    "grad_norm",
                    "tr_radius",
                    "tr_accepted",
                    "tr_ratio",
                    "phi",
                    "neg_curv",
                    "lipschitz",
                    "pred_red",
                ]
                + (["rho_rel_err", "red_rel_err"] if objective_relative_error else [])
            ),
            psnr=psnr,
            step_norm=step_norm,
            ssim=ssim_val,
            time=total_time,
            time_per_round=(end_time - start_time),
            grad_norm=grad_norm / grad_0,
            tr_radius=Delta,
            tr_accepted=int(accepted),
            tr_ratio=rho,
            neg_curv=int(steihaug._steihaug_last_exit == 1),
            lipschitz=lipschitz,
            pred_red=pred_red,
            phi=float(phi_val) if phi_val is not None else None,
            rho_rel_err=rho_rel_err if objective_relative_error else None,
            red_rel_rro=red_rel_err if objective_relative_error else None,
        )

        if on_iteration is not None:
            on_iteration(k, metrics._records[-1])

        _save_iterate(x_out, save_dir, name, k, save_checkpoint, total_iter)

        # variable update
        if accepted:
            grad_phi = grad_phi_trial
            point_k = point_trial

            phi_val = phi_trial
            grad_norm = torch.linalg.vector_norm(grad_phi)
            eta_k = max(1e-4, min(0.5, torch.sqrt(grad_norm)))

            if export_series_x:
                step_x.append(point_k)
                if len(step_x) >= max_exported_x:
                    return step_x

        if min_step_norm is not None and step_norm < min_step_norm and accepted:
            break

        if (
            min_step_norm is None
            and grad_tol is not None
            and grad_norm < grad_tol * grad_0
        ):
            break

        if max_time is not None and total_time >= max_time:
            break

        if test_save:
            if prev_psnr >= psnr:
                drop_iterate += 1
            else:
                drop_iterate = 0
            if drop_iterate > patience:
                break

        prev_psnr = psnr

    if export_series_x:
        return step_x
    return metrics.to_dataframe(), torch.clamp(point_k[1], 0.0, 1.0)
