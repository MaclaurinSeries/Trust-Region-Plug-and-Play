from typing import Dict, Tuple, Any
import torch
import argparse
import configparser
from pathlib import Path
from src.runner import run_experiment
from src.kernels import get_kernels_for_problem
from src.dataset import build_dataloader
from src.denoiser import (
    RelaxedDenoiser,
    GradientStepDenoiser,
    load_hurault_potential_network,
)
from src.pnp import trust_region_dre
from src.second_order import SecondOrderDRE, SecondOrderFBE
from src.utils import DEVICE
import deepinv as dinv
from enum import Enum
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepinv.optim.data_fidelity import L2
import os

os.environ["WANDB_SILENT"] = "true"

config = configparser.ConfigParser()
config.read("config.ini")

WANDB_API_KEY = config["wandb"]["key"]
WANDB_ENTITY = config["wandb"]["entity"]
WANDB_PROJECT = config["wandb"]["project"]

default_hyperparam = {
    "deblurring": {
        0.01: {
            "PGD": {"gamma": 0.6},
            "DRS": {"lam": 5},
            "aPGD": {"alpha": 0.6, "L_f": 1},
            "LBFGS": {"alpha": 0.5, "lam": 1},
        },
        0.03: {
            "PGD": {"gamma": 1},
            "DRS": {"lam": 1.5},
            "aPGD": {"alpha": 0.8, "L_f": 1},
            "LBFGS": {"alpha": 0.5, "lam": 1},
        },
        0.05: {
            "PGD": {"gamma": 1},
            "DRS": {"lam": 0.75},
            "aPGD": {"alpha": 0.85, "L_f": 1},
            "LBFGS": {"alpha": 0.7, "lam": 1},
        },
    },
    "sr": {
        0.01: {
            "PGD": {"gamma": 0.6},
            "DRS": {"lam": 5},
            "aPGD": {"alpha": 1, "L_f": 0.25},
            "LBFGS": {"alpha": 0.5, "lam": 4},
        },
        0.03: {
            "PGD": {"gamma": 1},
            "DRS": {"lam": 1.5},
            "aPGD": {"alpha": 1, "L_f": 0.25},
            "LBFGS": {"alpha": 0.5, "lam": 1.5},
        },
        0.05: {
            "PGD": {"gamma": 1},
            "DRS": {"lam": 0.75},
            "aPGD": {"alpha": 1, "L_f": 0.25},
            "LBFGS": {"alpha": 0.5, "lam": 1},
        },
    },
}
default_sigma_v = {
    "deblurring": {
        0.01: {
            "PGD": 1.25,
            "DRS": 2,
            "aPGD": 1.5,
            "LBFGS": 1,
        },
        0.03: {
            "PGD": 0.75,
            "DRS": 1,
            "aPGD": 1,
            "LBFGS": 0.75,
        },
        0.05: {
            "PGD": 0.5,
            "DRS": 0.75,
            "aPGD": 1,
            "LBFGS": 0.75,
        },
    },
    "sr": {
        0.01: {
            "PGD": 1.25,
            "DRS": 2,
            "aPGD": 2,
            "LBFGS": 2,
        },
        0.03: {
            "PGD": 0.75,
            "DRS": 1,
            "aPGD": 2,
            "LBFGS": 1,
        },
        0.05: {
            "PGD": 0.5,
            "DRS": 0.75,
            "aPGD": 2,
            "LBFGS": 0.75,
        },
    },
}


def _resolve_denoiser(denoiser_name):
    denoiser = None
    if denoiser_name == "DRUNet":
        denoiser = RelaxedDenoiser(
            dinv.models.DRUNet(pretrained="models/drunet_deepinv_color.pth").to(DEVICE)
        )
    elif denoiser_name == "DnCNN":
        denoiser = RelaxedDenoiser(
            dinv.models.DnCNN(pretrained="models/dncnn_sigma2_color.pth").to(DEVICE)
        )
    elif denoiser_name == "lipDnCNN":
        denoiser = RelaxedDenoiser(
            dinv.models.DnCNN(pretrained="models/dncnn_sigma2_lipschitz_color.pth").to(
                DEVICE
            )
        )
    elif denoiser_name == "Prox-DRUNet":
        denoiser = GradientStepDenoiser(
            load_hurault_potential_network(
                pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
            )
        )
    elif denoiser_name == "GS-DRUNet":
        denoiser = GradientStepDenoiser(
            load_hurault_potential_network(
                pretrained="models/GSDRUNet.ckpt", device=DEVICE
            )
        )
    assert denoiser is not None, "Invalid name"
    return denoiser


def power_iter_norm(so, point, tol=1e-3, max_it=100):
    v = torch.randn_like(point[1])
    v = v / torch.norm(v)
    lam_prev = None
    for it in range(max_it):
        hvp = so._hvp(point, v)
        lam = torch.norm(hvp)  # tighter lower bound on |lambda_1| than |v.Hv|
        if lam <= 1e-9:
            return 0.0, it
        v = hvp / lam
        if lam_prev is not None and abs(lam - lam_prev) <= tol * lam:
            return lam.item(), it
        lam_prev = lam
    return lam.item(), max_it


def _estimate_L_phi():
    denoiser = GradientStepDenoiser(
        load_hurault_potential_network(
            pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
        )
    )

    loader, image_names = build_dataloader(
        "datasets/CBSD10", batch_size=1, num_workers=0
    )

    collected_L = {
        "problem": [],
        "sigma_noise": [],
        "kernel": [],
        "image": [],
        "lipschitz": [],
    }

    for problem in ["super_resolution", "gaussian_deblur", "motion_deblur"]:
        for sigma_noise in [0.01, 0.03, 0.05]:
            denoiser._initialize_grad_func(sigma=sigma_noise, lam=1.0)
            kernels = get_kernels_for_problem(problem=problem)
            for kernel_alias, factory in kernels.items():
                measurements: Dict[str, Tuple[torch.Tensor, torch.Tensor, Any]] = {}
                for batch in loader:
                    img_name, x_true = batch
                    if isinstance(img_name, (list, tuple)):
                        img_name = img_name[0]
                    x_true = x_true.to(DEVICE)

                    physics = factory(
                        img_size=tuple(x_true.shape[1:]),
                        device=DEVICE,
                        sigma_noise=sigma_noise,
                    )
                    torch.manual_seed(
                        hash((42, problem, kernel_alias, img_name)) % (2**31)
                    )
                    with torch.no_grad():
                        y = physics(x_true)
                    measurements[img_name] = (x_true, y, physics)

                gamma = 1.0
                lam = 1.0
                L_history = []

                for img_name, (x_true, y_obs, physics) in tqdm(measurements.items()):

                    step_x = trust_region_dre(
                        x=x_true,
                        denoiser=denoiser,
                        physics=physics,
                        sigma=sigma_noise,
                        save_dir=None,
                        device=DEVICE,
                        y_observed=y_obs,
                        gamma=1.0,
                        lam=1.0,
                        export_series_x=True,
                        max_exported_x=4,
                        total_iter=6,
                    )

                    second_order = SecondOrderDRE(
                        y_obs=y_obs,
                        physics=physics,
                        denoiser=denoiser,
                        sigma_val=sigma_noise,
                        lam=lam,
                        gamma=gamma,
                        F=L2(),
                    )
                    L_phi = -1
                    for point in step_x:
                        L_current, _ = power_iter_norm(second_order, point=point)
                        L_phi = max(L_phi, L_current)
                    L_history.append(L_phi)

                    collected_L["image"].append(img_name)
                    collected_L["kernel"].append(kernel_alias)
                    collected_L["problem"].append(problem)
                    collected_L["sigma_noise"].append(sigma_noise)
                    collected_L["lipschitz"].append(L_phi)

                L_history = np.array(L_history)
                safe_L = np.percentile(L_history, 90)  # The "Safe Spot"

                print(f"Results for {kernel_alias} | {sigma_noise} | {problem} :")
                print(f"  Mean L: {np.mean(L_history):.4f}")
                print(f"  Max L:  {np.max(L_history):.4f}")
                print(f"  Min L:  {np.min(L_history):.4f}")
                print(f"  90th Percentile (Recommended L): {safe_L:.4f}")

    experiment_dir = Path("experiments") / "lipschitz"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    lipschitz_df = pd.DataFrame(collected_L)
    lipschitz_df.to_excel(experiment_dir / f"DRE-Lipschitz.xlsx")


def _estimate_L_phi_from_lambda_gamma():
    denoiser = GradientStepDenoiser(
        load_hurault_potential_network(
            pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
        )
    )

    loader, image_names = build_dataloader(
        "datasets/CBSD10", batch_size=1, num_workers=0
    )

    collected_L = {
        "kernel": [],
        "image": [],
        "gamma": [],
        "lambda": [],
        "lipschitz": [],
        "estimated_L": [],
    }
    problem = "deblurring"
    sigma_noise = 0.03
    kernels = get_kernels_for_problem(problem=problem)

    for kernel_alias, factory in kernels.items():
        for lam in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                denoiser._initialize_grad_func(sigma=sigma_noise, lam=lam * gamma)
                measurements: Dict[str, Tuple[torch.Tensor, torch.Tensor, Any]] = {}
                for batch in loader:
                    img_name, x_true = batch
                    if isinstance(img_name, (list, tuple)):
                        img_name = img_name[0]
                    x_true = x_true.to(DEVICE)

                    physics = factory(
                        img_size=tuple(x_true.shape[1:]),
                        device=DEVICE,
                        sigma_noise=sigma_noise,
                    )
                    torch.manual_seed(
                        hash((42, problem, kernel_alias, img_name)) % (2**31)
                    )
                    with torch.no_grad():
                        y = physics(x_true)
                    measurements[img_name] = (x_true, y, physics)

                L_history = []
                estimation_L = max(1.0 / (gamma + 1), 0.8 * lam)

                for img_name, (x_true, y_obs, physics) in tqdm(measurements.items()):
                    step_x = trust_region_dre(
                        x=x_true,
                        denoiser=denoiser,
                        physics=physics,
                        sigma=sigma_noise,
                        save_dir=None,
                        device=DEVICE,
                        y_observed=y_obs,
                        gamma=gamma,
                        lam=lam,
                        export_series_x=True,
                        max_exported_x=4,
                        total_iter=6,
                    )

                    second_order = SecondOrderDRE(
                        y_obs=y_obs,
                        physics=physics,
                        denoiser=denoiser,
                        sigma_val=sigma_noise,
                        lam=lam,
                        gamma=gamma,
                        F=L2(),
                    )
                    L_phi = -1
                    for point in step_x:
                        L_current, _ = power_iter_norm(second_order, point=point)
                        L_phi = max(L_phi, L_current)
                    L_history.append(L_phi)

                    collected_L["kernel"].append(kernel_alias)
                    collected_L["image"].append(img_name)
                    collected_L["gamma"].append(gamma)
                    collected_L["lambda"].append(lam)
                    collected_L["lipschitz"].append(L_phi)
                    collected_L["estimated_L"].append(estimation_L)

                L_history = np.array(L_history)
                safe_L = np.percentile(L_history, 90)

                print(
                    f"Results for {kernel_alias} | {sigma_noise} | {problem} | gamma {gamma} | lambda {lam}:"
                )
                print(f"  Estimated: {estimation_L:.4f}")
                print(f"  Mean L: {np.mean(L_history):.4f}")
                print(f"  Max L:  {np.max(L_history):.4f}")
                print(f"  Min L:  {np.min(L_history):.4f}")
                print(f"  90th Percentile (Recommended L): {safe_L:.4f}")

    experiment_dir = Path("experiments") / "lipschitz"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    lipschitz_df = pd.DataFrame(collected_L)
    lipschitz_df.to_excel(experiment_dir / f"DRE-Lipschitz-gamma-lambda.xlsx")


def _ablation_too_high_or_low_L_phi_estimation():
    denoiser = GradientStepDenoiser(
        load_hurault_potential_network(
            pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
        )
    )

    sigma_noise = 0.01

    lam = 0.8
    gamma = 0.5
    # lam = 1.0
    # gamma = 1.0
    denoiser._initialize_grad_func(sigma=sigma_noise, lam=lam * gamma)

    estimated_L_phi_dre = max(1.0 / (8 * gamma), 0.9 * lam)

    print(f"Lambda : {lam:.4f}")
    print(f"Gamma  : {gamma:.4f}")
    print(f"Est. L : {estimated_L_phi_dre:.4f}")

    for ratio in [0.05, 0.5, 0.8, 1.0, 1.25, 2.0, 20.0]:
        print(f"Usd. L : {ratio * estimated_L_phi_dre:.4f}")
        run_experiment(
            data_path=Path("datasets") / "Set3c",
            problems="deblurring",
            # solvers=( "TR-FBE", "TR-DRE",),
            solvers=("TR-DRE",),
            denoiser_name="Prox-DRUNet",
            denoiser=denoiser,
            sigma_noise=sigma_noise,
            solver_kwargs={
                "TR-DRE": dict(
                    total_iter=200,
                    lam=lam,
                    gamma=gamma,
                    use_potential=False,
                    L_phi_estimate=ratio * estimated_L_phi_dre,
                ),
            },
            result_root="experiments/lipschitz/L_phi_estimation",
            n_sample_images=10,
            wandb_api_key=(None if WANDB_API_KEY == "" else WANDB_API_KEY),
            wandb_project=f"{WANDB_PROJECT} [Set3c | L phi Ablation]",
            wandb_entity=WANDB_ENTITY,
            motion_kernel_path="kernels/Levin09.mat",
        )


def _ablation_trapezoidal():
    denoiser = GradientStepDenoiser(
        load_hurault_potential_network(
            pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
        )
    )

    sigma_noise = 0.03

    lam = 1.0
    gamma = 1.0
    denoiser._initialize_grad_func(sigma=sigma_noise, lam=lam)

    estimated_L_phi = max(1.0 / (8 * gamma), 0.9 * lam)

    run_experiment(
        data_path=Path("datasets") / "CBSD68",
        problems=("super_resolution",),
        solvers=("TR-DRE",),
        denoiser_name="Prox-DRUNet",
        denoiser=denoiser,
        sigma_noise=sigma_noise,
        solver_kwargs={
            "TR-DRE": dict(
                total_iter=100,
                lam=lam,
                gamma=gamma,
                use_potential=True,
                L_phi_estimate=estimated_L_phi,
            ),
        },
        result_root="experiments/trapezoidal",
        n_sample_images=10,
        wandb_api_key=(None if WANDB_API_KEY == "" else WANDB_API_KEY),
        wandb_project=f"{WANDB_PROJECT} [CBSD68 | Trapezoidal Ablation]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
    )

    run_experiment(
        data_path=Path("datasets") / "CBSD68",
        problems=("super_resolution",),
        solvers=("TR-DRE",),
        denoiser_name="Prox-DRUNet",
        denoiser=denoiser,
        sigma_noise=sigma_noise,
        solver_kwargs={
            "TR-DRE": dict(
                total_iter=100,
                lam=lam,
                gamma=gamma,
                use_potential=False,
                L_phi_estimate=estimated_L_phi,
            ),
        },
        result_root="experiments/trapezoidal",
        n_sample_images=10,
        wandb_api_key=(None if WANDB_API_KEY == "" else WANDB_API_KEY),
        wandb_project=f"{WANDB_PROJECT} [CBSD68 | Trapezoidal Ablation]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
    )


def _run_grid_search(denoiser_name="Prox-DRUNet", L_g=1):
    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    for problem in ["deblurring", "sr2", "sr3"]:
        for sigma_noise in [0.01, 0.03, 0.05]:
            for lam in [0.2, 0.25, 0.35, 0.5, 0.7, 1.0]:
                for sigma_v in [0.5, 0.75, 1.0, 1.5, 2.0]:
                    estimated_L_phi = max(1.0 / 8.0, L_g * lam)
                    run_experiment(
                        data_path=Path("datasets") / "Set3c",
                        problem=problem,
                        solvers=("TR-DRE",),
                        denoiser_name=denoiser_name,
                        denoiser=denoiser,
                        v_noise=sigma_noise,
                        sigma_v=sigma_v,
                        solver_kwargs={
                            "TR-DRE": dict(
                                total_iter=100,
                                lam=lam,
                                gamma=1.0,
                                use_potential=False,
                                L_phi_estimate=estimated_L_phi,
                                min_step_norm=1e-4,
                                cg_max_iter=5,
                            ),
                        },
                        n_sample_images=10,
                        wandb_api_key=(
                            None if WANDB_API_KEY == "" else WANDB_API_KEY
                        ),  # set to None to disable
                        wandb_project=f"{WANDB_PROJECT} [Set3c | Grid Search | TR | {problem}]",
                        wandb_entity=WANDB_ENTITY,
                        motion_kernel_path="kernels/Levin09.mat",
                    )


def _run_experiments(denoiser_name="Prox-DRUNet", L_g=1):
    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    # for problem in ["deblurring", "sr2", "sr3"]:
    for problem in ["deblurring", "sr2", "sr3"]:
        for sigma_noise in [0.01, 0.03, 0.05]:

            if problem == "deblurring":
                if sigma_noise == 0.01:
                    lam, sigma_v = 0.2, 1.5
                elif sigma_noise == 0.03:
                    lam, sigma_v = 0.5, 0.75
                elif sigma_noise == 0.05:
                    lam, sigma_v = 0.7, 0.75
            elif problem == "sr2":
                lam, sigma_v = 0.2, 2.0
                if sigma_noise == 0.03:
                    sigma_v = 1.5
            elif problem == "sr3":
                lam, sigma_v = 0.2, 2.0
                if sigma_noise == 0.05:
                    lam, sigma_v = 0.35, 1.0

            estimated_L_phi = max(1.0 / 8.0, L_g * lam)
            run_experiment(
                data_path=Path("datasets") / "CBSD10",
                problem=problem,
                solvers=("TR-DRE",),
                denoiser_name=denoiser_name,
                denoiser=denoiser,
                v_noise=sigma_noise,
                sigma_v=sigma_v,
                solver_kwargs={
                    "TR-DRE": dict(
                        total_iter=300 if problem == "deblurring" else 2000,
                        lam=lam,
                        gamma=1.0,
                        use_potential=False,
                        L_phi_estimate=estimated_L_phi,
                        min_step_norm=1e-5,
                        cg_max_iter=5,
                    ),
                },
                n_sample_images=10,
                wandb_api_key=(
                    None if WANDB_API_KEY == "" else WANDB_API_KEY
                ),  # set to None to disable
                wandb_project=f"{WANDB_PROJECT} [CBSD10 | new | TR-float16 | 5CG | {problem}]",
                wandb_entity=WANDB_ENTITY,
                motion_kernel_path="kernels/Levin09.mat",
                result_root=Path("experiments")
                / "Full-CBSD10"
                / f"{problem}-{sigma_noise * 100}",
            )


def _run_others(denoiser_name="Prox-DRUNet", dataset="CBSD10"):
    denoiser = _resolve_denoiser(denoiser_name="DRUNet")
    problem = "deblurring"
    solver = "DPIR"
    problem_alias = "deblurring"

    for sigma_noise in [0.01, 0.03, 0.05]:
        run_experiment(
            data_path=Path("datasets") / dataset,
            problem=problem,
            solvers=(solver,),
            denoiser_name="DRUNet",
            denoiser=denoiser,
            v_noise=sigma_noise,
            sigma_v=1.0,
            solver_kwargs={
                solver: dict(
                    total_iter=300 if solver == "LBFGS" else 3000,
                    min_step_norm=1e-5,
                    K_sigma_max=8,
                ),
            },
            n_sample_images=10,
            wandb_api_key=(
                None if WANDB_API_KEY == "" else WANDB_API_KEY
            ),  # set to None to disable
            wandb_project=f"{WANDB_PROJECT} [{dataset} | Others | {problem}]",
            wandb_entity=WANDB_ENTITY,
            motion_kernel_path="kernels/Levin09.mat",
            result_root=Path("experiments")
            / "Full-CBSD10"
            / f"{problem}-{sigma_noise * 100}",
        )

    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    problem = "deblurring"
    solver = "DRS"
    problem_alias = "deblurring"
    if "sr" in problem:
        problem_alias = "sr"
    for sigma_noise in [0.01, 0.03, 0.05]:
        run_experiment(
            data_path=Path("datasets") / dataset,
            problem=problem,
            solvers=(solver,),
            denoiser_name=denoiser_name,
            denoiser=denoiser,
            v_noise=sigma_noise,
            sigma_v=default_sigma_v[problem_alias][sigma_noise][solver],
            solver_kwargs={
                solver: dict(
                    total_iter=300 if solver == "LBFGS" else 3000,
                    min_step_norm=1e-5,
                    **default_hyperparam[problem_alias][sigma_noise][solver],
                ),
            },
            n_sample_images=10,
            wandb_api_key=(
                None if WANDB_API_KEY == "" else WANDB_API_KEY
            ),  # set to None to disable
            wandb_project=f"{WANDB_PROJECT} [{dataset} | Others | {problem}]",
            wandb_entity=WANDB_ENTITY,
            motion_kernel_path="kernels/Levin09.mat",
            result_root=Path("experiments")
            / "Full-CBSD10"
            / f"{problem}-{sigma_noise * 100}",
        )

    return

    for problem in ["sr2", "sr3"]:
        for sigma_noise in [0.01, 0.03, 0.05]:
            for solver in ["PGD", "DRS", "aPGD", "LBFGS"]:
                problem_alias = "deblurring"
                if "sr" in problem:
                    problem_alias = "sr"
                run_experiment(
                    data_path=Path("datasets") / dataset,
                    problem=problem,
                    solvers=(solver,),
                    denoiser_name=denoiser_name,
                    denoiser=denoiser,
                    v_noise=sigma_noise,
                    sigma_v=default_sigma_v[problem_alias][sigma_noise][solver],
                    solver_kwargs={
                        solver: dict(
                            total_iter=300 if solver == "LBFGS" else 3000,
                            min_step_norm=1e-5,
                            **default_hyperparam[problem_alias][sigma_noise][solver],
                        ),
                    },
                    n_sample_images=10,
                    wandb_api_key=(
                        None if WANDB_API_KEY == "" else WANDB_API_KEY
                    ),  # set to None to disable
                    wandb_project=f"{WANDB_PROJECT} [{dataset} | Others | {problem}]",
                    wandb_entity=WANDB_ENTITY,
                    motion_kernel_path="kernels/Levin09.mat",
                    result_root=Path("experiments")
                    / "Full-CBSD10"
                    / f"{problem}-{sigma_noise * 100}",
                )


def build_args():
    p = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset directory.",
    )
    p.add_argument(
        "--estimate-lipschitz",
        action="store_true",
        help="Lipschitz Estimation",
    )
    p.add_argument(
        "--estimate-lipschitz-hyperparams",
        action="store_true",
        help="Lipschitz Estimation from hyperparams",
    )
    p.add_argument(
        "--estimate-lipschitz-ablation",
        action="store_true",
        help="Lipschitz Estimation ablation",
    )
    p.add_argument(
        "--trapezoidal-ablation",
        action="store_true",
        help="Lipschitz Estimation ablation",
    )
    p.add_argument(
        "--grid-search-tr",
        action="store_true",
        help="Hyperparameter Sensitivity Analysis",
    )
    p.add_argument(
        "--grid-search-others",
        action="store_true",
        help="Hyperparameter search for other method",
    )
    p.add_argument(
        "--cbsd-exp",
        action="store_true",
        help="CBSD68 Experiments",
    )
    p.add_argument(
        "--others",
        action="store_true",
        help="CBSD68 Experiments",
    )
    p.add_argument(
        "--solver",
        type=str,
        default="tr",
        help="Solver Algorithm",
    )
    p.add_argument(
        "--denoiser",
        type=str,
        default="DRUNet",
        help="Denoiser",
    )
    p.add_argument(
        "--sigma-noise",
        type=float,
        default=0.01,
        help="Noise Intensity",
    )
    p.add_argument(
        "--iter",
        type=int,
        default=1000,
        help="Iteration count",
    )
    p.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="Step size",
    )
    return p.parse_args()


def main():
    args = build_args()

    if args.estimate_lipschitz:
        _estimate_L_phi()
    elif args.estimate_lipschitz_hyperparams:
        _estimate_L_phi_from_lambda_gamma()
    elif args.estimate_lipschitz_ablation:
        _ablation_too_high_or_low_L_phi_estimation()
    elif args.trapezoidal_ablation:
        _ablation_trapezoidal()
    elif args.grid_search_tr:
        _run_grid_search(denoiser_name="Prox-DRUNet", L_g=0.9)
    elif args.cbsd_exp:
        _run_experiments(L_g=0.9)
    elif args.others:
        _run_others()
    elif args.grid_search_others:
        pass


if __name__ == "__main__":
    main()
