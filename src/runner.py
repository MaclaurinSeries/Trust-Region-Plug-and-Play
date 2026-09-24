from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, Callable

import numpy as np
import torch
import deepinv as dinv

from skimage.metrics import structural_similarity as ssim_metric

from .dataset import build_dataloader
from .denoiser import Denoiser, GradientStepDenoiser
from .kernels import get_kernels_for_problem
from .pnp import pgd, drs, alpha_pgd, q_newton_prox_tan_2024, trust_region_dre, dpir
from .utils import DEVICE, to_image, _build_result_dataframe, _save_image_pair

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


SOLVER_REGISTRY: Dict[str, Callable] = {
    "PGD": pgd,
    "DRS": drs,
    "aPGD": alpha_pgd,
    "LBFGS": q_newton_prox_tan_2024,
    "TR-DRE": trust_region_dre,
    "DPIR": dpir,
}

psnr_real = dinv.metric.PSNR(max_pixel=1, min_pixel=0)
ssim_real = dinv.metric.SSIM(max_pixel=1, min_pixel=0)
psnr_cplx = dinv.metric.PSNR(max_pixel=1, min_pixel=0, complex_abs=True)
ssim_cplx = dinv.metric.PSNR(max_pixel=1, min_pixel=0, complex_abs=True)


def _compute_psnr_ssim(
    x_restored: torch.Tensor, x_true: torch.Tensor
) -> Tuple[float, float]:
    """PSNR/SSIM on a [B=1, C, H, W] pair, both in [0, 1]."""
    psnr_val = psnr_real(x_restored, x_true)
    ssim_val = ssim_real(x_restored, x_true)

    return float(psnr_val), float(ssim_val)


def make_callback(label: str, solver: str, denoiser_name: str):
    def _cb(k: int, rec: Dict) -> None:
        payload: Dict[str, Any] = {
            f"{solver}/{denoiser_name}/{metric}": rec.get(metric)
            for metric in rec.keys()
        }
        wandb.log(payload, step=k)

    return _cb


def run_experiment(
    data_path: Union[str, Path],
    problem: str,
    solvers: Tuple[str, ...],
    denoiser_name: str,
    denoiser: Denoiser,
    v_noise: float,
    sigma_v: float,
    solver_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    result_root: Union[str, Path] = "result",
    n_sample_images: int = 10,
    motion_kernel_path: Union[str, Path] = "kernels/Levin09.mat",
    device: torch.device = DEVICE,
    seed: int = 42,
    wandb_api_key: Optional[str] = None,
    wandb_project: str = "pnp-newton-tr",
    wandb_entity: Optional[str] = None,
) -> None:
    data_path = Path(data_path)
    result_root = Path(result_root)
    dataset_name = data_path.name
    solver_kwargs = solver_kwargs or {}

    denoiser = denoiser.eval()

    use_wandb = wandb_api_key is not None and _HAS_WANDB
    if wandb_api_key is not None and not _HAS_WANDB:
        print(
            "WARNING: wandb_api_key was provided but the wandb library is "
            "not installed. Install with `pip install wandb`. Continuing "
            "without wandb logging."
        )
    if use_wandb:
        wandb.login(key=wandb_api_key)

    if problem == "sr3":
        loader, image_names = build_dataloader(
            data_path, batch_size=1, num_workers=0, sr=3
        )
    else:
        loader, image_names = build_dataloader(data_path, batch_size=1, num_workers=0)

    rng = np.random.default_rng(seed)
    sampled = set(
        rng.choice(
            image_names,
            size=min(n_sample_images, len(image_names)),
            replace=False,
        ).tolist()
    )

    print(
        f"\n=== Experiment: {dataset_name} | denoiser={denoiser_name} | "
        f"sigma_noise {v_noise} x {sigma_v} ==="
    )
    print(
        f"Sample-image snapshots ({len(sampled)}): {sorted(list(sampled))[:5]}"
        f"{'...' if len(sampled) > 5 else ''}\n"
    )

    print(f"\n--- Problem: {problem} ---")
    kernels = get_kernels_for_problem(problem, motion_mat_path=motion_kernel_path)
    kernel_aliases = tuple(kernels.keys())
    print(f"Kernels ({len(kernel_aliases)}): {list(kernel_aliases)}")

    problem_dir = result_root / problem
    problem_dir.mkdir(parents=True, exist_ok=True)

    # {solver: {image: {kernel: (psnr, ssim)}}}
    all_metrics: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
        s: {} for s in solvers
    }

    for kernel_alias, factory in kernels.items():
        measurements: Dict[str, Tuple[torch.Tensor, torch.Tensor, Any]] = {}
        for batch in loader:
            img_name, x_true = batch
            if isinstance(img_name, (list, tuple)):
                img_name = img_name[0]
            x_true = x_true.to(device)

            physics = factory(
                img_size=tuple(x_true.shape[1:]),
                device=device,
                sigma_noise=v_noise,
            )
            torch.manual_seed(hash((seed, problem, kernel_alias, img_name)) % (2**31))
            with torch.no_grad():
                y = physics(x_true)
            measurements[img_name] = (x_true, y, physics)

        # One wandb run per (problem, kernel, solver, denoiser).
        for solver in solvers:
            if solver not in SOLVER_REGISTRY:
                raise ValueError(f"Unknown solver: {solver!r}")
            solver_fn = SOLVER_REGISTRY[solver]
            kwargs = dict(solver_kwargs.get(solver, {}))

            wb_run = None

            for img_name, (x_true, y, physics) in measurements.items():
                series_label = f"{dataset_name}_{img_name}"

                on_iteration = None
                if use_wandb:
                    wb_run = wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=f"{img_name}__{problem}__{kernel_alias}",
                        group=f"{problem}__{kernel_alias}__{solver}__{denoiser_name}",
                        job_type=f"{solver}__{denoiser_name}",
                        config={
                            "dataset": dataset_name,
                            "problem": problem,
                            "kernel_alias": kernel_alias,
                            "solver": solver,
                            "denoiser": denoiser_name,
                            "v_noise": v_noise,
                            "sigma_d/sigma": sigma_v,
                            **{f"solver/{k}": v for k, v in kwargs.items()},
                        },
                        settings=wandb.Settings(_disable_stats=True),
                    )

                    on_iteration = make_callback(
                        series_label, solver=solver, denoiser_name=denoiser_name
                    )

                try:
                    _, x_final = solver_fn(
                        x=x_true,
                        denoiser=denoiser,
                        physics=physics,
                        v_noise=v_noise,
                        sigma_v=sigma_v,
                        save_dir=None,
                        device=device,
                        y_observed=y,
                        on_iteration=on_iteration,
                        **kwargs,
                    )
                except NotImplementedError as e:
                    if img_name == image_names[0] and kernel_alias == kernel_aliases[0]:
                        print(f"  [skip] {solver}: not implemented ({e})")
                    continue

                psnr, ssim = _compute_psnr_ssim(x_final, x_true)
                all_metrics[solver].setdefault(img_name, {})[kernel_alias] = (
                    psnr,
                    ssim,
                )

                if img_name in sampled:
                    sample_dir = problem_dir / f"{solver}-{denoiser_name}"
                    base = (
                        f"{dataset_name}-{img_name}-{solver}-"
                        f"{denoiser_name}-{kernel_alias}"
                    )
                    _save_image_pair(y, x_final, sample_dir, base)

                print(
                    f"  [{kernel_alias}][{solver}] {img_name}: "
                    f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}"
                )

                if wb_run is not None:
                    wb_run.finish()
                    wb_run = None

    # Per-solver xlsx
    for solver in solvers:
        if not all_metrics[solver]:
            continue
        df = _build_result_dataframe(all_metrics[solver], kernel_aliases)
        xlsx_path = problem_dir / f"{dataset_name}-{solver}-{denoiser_name}.xlsx"
        df.to_excel(xlsx_path)
        print(f"Saved: {xlsx_path}")

    print("\nDone.")
