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
                        cg_max_iter=10,
                    ),
                },
                n_sample_images=10,
                wandb_api_key=(
                    None if WANDB_API_KEY == "" else WANDB_API_KEY
                ),  # set to None to disable
                wandb_project=f"{WANDB_PROJECT} [CBSD10 | new | TR-float16 | 10CG | {problem}]",
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
        _run_experiments(L_g=0.9)
    elif args.grid_search_others:
        pass


if __name__ == "__main__":
    main()
