from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, Dict
import time

import numpy as np
import torch
import torch.nn as nn

import deepinv as dinv

from .utils import rademacher
from .GS_models.network_unet import UNetRes
from .GS_models.test_utils import test_mode


class Denoiser(ABC, nn.Module):
    def __init__(self, denoiser: nn.Module, name: str = ""):
        super().__init__()
        self.denoiser = denoiser
        self._name = name
        self._has_potential = False
        self._use_potential = True

    def use_potential(self):
        return self._has_potential and self._use_potential

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        no_grad: bool = True,
        lam: float = 1.0,
    ) -> torch.Tensor:
        pass

    @property
    def name(self) -> str:
        return self._name


class RelaxedDenoiser(Denoiser):
    def __init__(self, denoiser: nn.Module):
        super().__init__(denoiser=denoiser, name="relaxed")
        self._has_potential = False

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        lam: float = 1.0,
        no_grad: bool = True,
    ) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                return (1 - lam) * x + lam * self.denoiser(x, sigma)
        return (1 - lam) * x + lam * self.denoiser(x, sigma)

    def _calculate_grad(
        self, x: torch.Tensor, sigma: float | torch.Tensor, lam: float = 1.0
    ) -> torch.Tensor:
        N = self.forward(x, sigma, lam=lam)
        return (x - N).detach()

    @torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    def _jvp_D(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma: float | torch.Tensor,
        lam: float = 1.0,
    ) -> torch.Tensor:
        x = x.detach()
        v = v.detach()

        _, jvp = torch.func.jvp(
            lambda x_in: self.forward(x_in, sigma, no_grad=False, lam=lam), (x,), (v,)
        )
        return jvp.detach()


class GradientStepDenoiser(Denoiser):
    def __init__(self, denoiser: nn.Module, alpha: float = 1.0):
        super().__init__(denoiser=denoiser, name="gradient_step")
        self.alpha = alpha
        self.grad_func = None
        self._has_potential = True
        self._last_Dg = None

    def _pad_image(self, x: torch.Tensor, multiple: int = 16):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        x_padded = torch.nn.functional.pad(x, padding, mode="reflect")
        return x_padded, (pad_h, pad_w)

    def _crop_image(self, x: torch.Tensor, pad_info: tuple):
        pad_h, pad_w = pad_info
        h, w = x.shape[-2], x.shape[-1]

        return x[..., : h - pad_h, : w - pad_w]

    def _model_forward(self, v: torch.Tensor, sigma: float | torch.Tensor):
        noise_level_map = (
            torch.FloatTensor(v.size(0), 1, v.size(2), v.size(3))
            .fill_(sigma)
            .to(v.device)
        )
        v = torch.cat((v, noise_level_map), 1)
        return self.denoiser.forward(v)

    def _potential(
        self, x: torch.Tensor, sigma: float | torch.Tensor, lam: float = 1.0
    ) -> torch.Tensor:
        """
        from https://github.com/samuro95/GSPnP/blob/master/GS_denoising/lightning_GSDRUNet.py
        GradMatch
        """
        x_padded, pad_info = self._pad_image(x, multiple=16)
        N = self._model_forward(x_padded, sigma)
        N = self._crop_image(N, pad_info)
        u = (x - N).reshape((x.shape[0], -1)).flatten()
        g = 0.5 * torch.dot(u, u)
        return self.alpha * lam * g

    def _initialize_grad_func(self, sigma: float | torch.Tensor, lam: float = 1.0):
        # only valid for constant sigma schedule
        self.grad_func = torch.func.grad(
            lambda x_in: self._potential(x_in, sigma, lam=lam)
        )

    def _calculate_grad(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        lam: float = 1.0,
        no_grad: bool = True,
    ) -> torch.Tensor:
        if self.grad_func is None:
            self._initialize_grad_func(sigma=sigma, lam=lam)

        Dg = self.grad_func(x.detach())

        return Dg.detach()

    @torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    def _jvp_D(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma: float | torch.Tensor,
        lam: float = 1.0,
    ) -> torch.Tensor:
        x = x.detach()
        v = v.detach()

        _, hvp = torch.func.jvp(self.grad_func, (x,), (v,))

        return (v - hvp).detach()

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        lam: float = 1.0,
        no_grad=True,
    ) -> torch.Tensor:
        """
        from https://github.com/samuro95/GSPnP/blob/master/GS_denoising/lightning_GSDRUNet.py
        GradMatch
        """
        Dg = self._calculate_grad(x, sigma, lam=lam, no_grad=no_grad)
        return (x - Dg).detach()


def load_hurault_potential_network(
    pretrained: str = "ckpt_path",
    device: torch.device = None,
) -> nn.Module:
    ckpt_path = pretrained
    checkpoint = torch.load(
        ckpt_path, map_location="cpu" if device is None else device, weights_only=False
    )

    nb = checkpoint["hyper_parameters"]["DRUNET_nb"]
    act_mode = checkpoint["hyper_parameters"]["act_mode"]

    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=nb,
        act_mode=act_mode,
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )

    state_dict = {}
    skip = len("student_grad.model.")
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("student_grad.model."):
            state_dict[key[skip:]] = value

    model.load_state_dict(state_dict)
    model.eval()

    return model.to(device)
