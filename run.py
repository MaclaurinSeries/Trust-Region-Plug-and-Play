import argparse
import configparser
from pathlib import Path
from src.runner import run_experiment
from src.denoiser import (
    RelaxedDenoiser,
    GradientStepDenoiser,
    load_hurault_potential_network,
)
from src.utils import DEVICE
import deepinv as dinv
from enum import Enum
import os

os.environ["WANDB_SILENT"] = "true"

config = configparser.ConfigParser()
config.read("config.ini")

WANDB_API_KEY = config["wandb"]["key"]
WANDB_ENTITY = config["wandb"]["entity"]
WANDB_PROJECT = config["wandb"]["project"]


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
        "--problem",
        default=None,
        help="Degradation",
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
        "--max-time",
        type=int,
        default=None,
        help="Maximum Time (Second)",
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
    return p.parse_known_args()


def translate_args(args, unknown):
    denoiser = None
    if args.denoiser == "DRUNet":
        denoiser = RelaxedDenoiser(
            dinv.models.DRUNet(pretrained="models/drunet_deepinv_color.pth").to(DEVICE)
        )
    elif args.denoiser == "DnCNN":
        denoiser = RelaxedDenoiser(
            dinv.models.DnCNN(pretrained="models/dncnn_sigma2_color.pth").to(DEVICE)
        )
    elif args.denoiser == "lipDnCNN":
        denoiser = RelaxedDenoiser(
            dinv.models.DnCNN(pretrained="models/dncnn_sigma2_lipschitz_color.pth").to(
                DEVICE
            )
        )
    elif args.denoiser == "Prox-DRUNet":
        denoiser = GradientStepDenoiser(
            load_hurault_potential_network(
                pretrained="models/Prox-DRUNet.ckpt", device=DEVICE
            )
        )
    elif args.denoiser == "GS-DRUNet":
        denoiser = GradientStepDenoiser(
            load_hurault_potential_network(
                pretrained="models/GSDRUNet.ckpt", device=DEVICE
            )
        )

    solver_kwargs = None
    if args.solver == "PGD":
        solver_kwargs = {
            "gamma": args.step_size,
            "total_iter": args.iter,
            "max_time": args.max_time,
        }
    elif args.solver == "RED":
        eta = 0.5
        for i in range(len(unknown)):
            if unknown[i].endswith("eta") and i + 1 < len(unknown):
                eta = float(unknown[i + 1])
        solver_kwargs = {"gamma": args.step_size, "total_iter": args.iter, "eta": eta}
    elif args.solver == "DRS":
        lam = 1.0
        for i in range(len(unknown)):
            if (
                unknown[i].endswith("lambda") or unknown[i].endswith("lam")
            ) and i + 1 < len(unknown):
                lam = float(unknown[i + 1])
        solver_kwargs = {"total_iter": args.iter, "lam": lam}
    elif args.solver == "aPGD":
        lam = 1.0
        eta = 0.5
        for i in range(len(unknown)):
            if (
                unknown[i].endswith("lambda") or unknown[i].endswith("lam")
            ) and i + 1 < len(unknown):
                lam = float(unknown[i + 1])
            if unknown[i].endswith("eta") and i + 1 < len(unknown):
                eta = float(unknown[i + 1])
        solver_kwargs = {
            "total_iter": args.iter,
            "lam": lam,
            "alpha": eta,
        }
    elif args.solver == "LBFGS":
        solver_kwargs = {"gamma": args.step_size, "total_iter": args.iter}
    elif args.solver == "TR-DRE":
        lam = 1.0
        for i in range(len(unknown)):
            if (
                unknown[i].endswith("lambda") or unknown[i].endswith("lam")
            ) and i + 1 < len(unknown):
                lam = float(unknown[i + 1])
            if unknown[i].endswith("beta") and i + 1 < len(unknown):
                beta = float(unknown[i + 1])
        solver_kwargs = {"total_iter": args.iter, "lam": lam, "max_time": args.max_time}
    elif args.solver == "TR-FBE":
        beta = 1.0
        lam = 0.5
        for i in range(len(unknown)):
            if (
                unknown[i].endswith("lambda") or unknown[i].endswith("lam")
            ) and i + 1 < len(unknown):
                lam = float(unknown[i + 1])
            if unknown[i].endswith("beta") and i + 1 < len(unknown):
                beta = float(unknown[i + 1])
        solver_kwargs = {
            "total_iter": args.iter,
            "lam": lam,
        }

    return denoiser, solver_kwargs


def main():
    args, unknown = build_args()
    denoiser, solver_kwargs = translate_args(args, unknown)

    run_experiment(
        data_path=Path("datasets") / args.dataset,
        problems=(args.problem,),
        solvers=(args.solver,),
        denoiser_name=args.denoiser,
        denoiser=denoiser,
        sigma_noise=args.sigma_noise,
        solver_kwargs={
            args.solver: solver_kwargs,
        },
        n_sample_images=10,
        wandb_api_key=(
            None if WANDB_API_KEY == "" else WANDB_API_KEY
        ),  # set to None to disable
        wandb_project=f"{WANDB_PROJECT} [{args.dataset}]",
        wandb_entity=WANDB_ENTITY,
        motion_kernel_path="kernels/Levin09.mat",
    )


if __name__ == "__main__":
    main()
