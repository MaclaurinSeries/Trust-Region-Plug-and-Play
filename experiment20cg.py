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


def _run_experiments(denoiser_name="Prox-DRUNet", L_g=1):
    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    for problem in ["deblurring"]:
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
                        cg_max_iter=20,
                    ),
                },
                n_sample_images=10,
                wandb_api_key=(
                    None if WANDB_API_KEY == "" else WANDB_API_KEY
                ),  # set to None to disable
                wandb_project=f"{WANDB_PROJECT} [CBSD10 | time | TR-float32 | 20CG | {problem}]",
                wandb_entity=WANDB_ENTITY,
                motion_kernel_path="kernels/Levin09.mat",
                result_root=Path("experiments")
                / "Full-CBSD10"
                / f"{problem}-{sigma_noise * 100}",
            )


def _run_experiments68(denoiser_name="Prox-DRUNet", L_g=1):
    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    for problem in ["sr3"]:
        for sigma_noise in [0.03, 0.05]:

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
                data_path=Path("datasets") / "CBSD68",
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
                        cg_max_iter=20,
                        test_save=True,
                    ),
                },
                n_sample_images=10,
                wandb_api_key=(
                    None if WANDB_API_KEY == "" else WANDB_API_KEY
                ),  # set to None to disable
                wandb_project=f"{WANDB_PROJECT} [CBSD68 | new | TR-float16 | 20CG | {problem}]",
                wandb_entity=WANDB_ENTITY,
                motion_kernel_path="kernels/Levin09.mat",
                result_root=Path("experiments")
                / "Full-CBSD68"
                / f"{problem}-{sigma_noise * 100}",
            )


def run_sample():
    denoiser = _resolve_denoiser(denoiser_name="DRUNet")
    problem = "deblurring"
    solver = "DPIR"
    problem_alias = "deblurring"

    run_experiment(
        data_path=Path("datasets") / "Sample",
        problem=problem,
        solvers=(solver,),
        denoiser_name="DRUNet",
        denoiser=denoiser,
        v_noise=0.01,
        sigma_v=1.0,
        solver_kwargs={
            solver: dict(
                total_iter=8,
                min_step_norm=1e-5,
                K_sigma_max=8,
            ),
        },
        n_sample_images=10,
        wandb_api_key=(
            None if WANDB_API_KEY == "" else WANDB_API_KEY
        ),  # set to None to disable
        wandb_project=f"{WANDB_PROJECT} [Sample]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
        result_root=Path("experiments") / "Sample" / f"DPIR8",
    )

    run_experiment(
        data_path=Path("datasets") / "Sample",
        problem=problem,
        solvers=(solver,),
        denoiser_name="DRUNet",
        denoiser=denoiser,
        v_noise=0.01,
        sigma_v=1.0,
        solver_kwargs={
            solver: dict(
                total_iter=100,
                min_step_norm=1e-5,
                K_sigma_max=8,
            ),
        },
        n_sample_images=10,
        wandb_api_key=(
            None if WANDB_API_KEY == "" else WANDB_API_KEY
        ),  # set to None to disable
        wandb_project=f"{WANDB_PROJECT} [Sample | Others | {problem}]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
        result_root=Path("experiments") / "Sample" / f"DPIR100",
    )

    denoiser = _resolve_denoiser(denoiser_name="Prox-DRUNet")

    lam, sigma_v = 0.2, 1.5

    estimated_L_phi = max(1.0 / 8.0, 0.9 * lam)
    run_experiment(
        data_path=Path("datasets") / "Sample",
        problem=problem,
        solvers=("TR-DRE",),
        denoiser_name="Prox-DRUNet",
        denoiser=denoiser,
        v_noise=0.01,
        sigma_v=sigma_v,
        solver_kwargs={
            "TR-DRE": dict(
                total_iter=300,
                lam=lam,
                gamma=1.0,
                use_potential=False,
                L_phi_estimate=estimated_L_phi,
                min_step_norm=1e-5,
                cg_max_iter=20,
                test_save=True,
            ),
        },
        n_sample_images=10,
        wandb_api_key=(
            None if WANDB_API_KEY == "" else WANDB_API_KEY
        ),  # set to None to disable
        wandb_project=f"{WANDB_PROJECT} [Sample]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
        result_root=Path("experiments") / "Sample" / f"TR-DRE",
    )

    for solver in ["PGD", "DRS", "aPGD", "LBFGS"]:
        problem_alias = "deblurring"
        denoiser._use_potential = True
        run_experiment(
            data_path=Path("datasets") / "Sample",
            problem=problem,
            solvers=(solver,),
            denoiser_name="Prox-DRUNet",
            denoiser=denoiser,
            v_noise=0.01,
            sigma_v=default_sigma_v[problem_alias][0.01][solver],
            solver_kwargs={
                solver: dict(
                    total_iter=300 if solver == "LBFGS" else 3000,
                    min_step_norm=1e-5,
                    **default_hyperparam[problem_alias][0.01][solver],
                ),
            },
            n_sample_images=10,
            wandb_api_key=(
                None if WANDB_API_KEY == "" else WANDB_API_KEY
            ),  # set to None to disable
            wandb_project=f"{WANDB_PROJECT} [Sample]",
            wandb_entity=WANDB_ENTITY,
            motion_kernel_path="kernels/Levin09.mat",
            result_root=Path("experiments") / "Sample" / f"{solver}",
        )


def _run_experiments_proof(denoiser_name="Prox-DRUNet", L_g=1):
    denoiser = _resolve_denoiser(denoiser_name=denoiser_name)

    for problem in ["deblurring"]:
        for sigma_noise in [0.01]:

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

            gamma = 0.5

            estimated_L_phi = max(1.0 / (8.0 * gamma), L_g * lam)
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
                        total_iter=100,
                        lam=lam,
                        gamma=gamma,
                        use_potential=False,
                        L_phi_estimate=estimated_L_phi,
                        cg_max_iter=20,
                    ),
                },
                n_sample_images=10,
                wandb_api_key=(
                    None if WANDB_API_KEY == "" else WANDB_API_KEY
                ),  # set to None to disable
                wandb_project=f"{WANDB_PROJECT} [CBSD10 | evidence2 | 20CG | {problem}]",
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
        "--cbsd-exp",
        action="store_true",
        help="CBSD68 Experiments",
    )
    p.add_argument(
        "--cbsd68-exp",
        action="store_true",
        help="CBSD68 Experiments",
    )
    p.add_argument(
        "--run-sample",
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

    if args.cbsd_exp:
        _run_experiments_proof(L_g=0.9)
    if args.cbsd68_exp:
        _run_experiments68(L_g=0.9)
    if args.run_sample:
        run_sample()
    elif args.grid_search_others:
        pass


if __name__ == "__main__":
    main()
