from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import scipy.io as sio
import h5py

import deepinv as dinv
from deepinv.physics import GaussianNoise, Physics

SR_CONFIGS: Dict[str, Tuple[float, int]] = {
    "0.7x2": (0.7, 2),
    "1.2x2": (1.2, 2),
    "1.6x2": (1.6, 2),
    "2.0x2": (2.0, 2),
    "0.7x3": (0.7, 3),
    "1.2x3": (1.2, 3),
    "1.6x3": (1.6, 3),
    "2.0x3": (2.0, 3),
}


class PhysicsFactory:
    def __init__(self, kernel: torch.Tensor, factor: Optional[int] = None):
        self.kernel = kernel
        self.factor = factor

    def __call__(
        self,
        img_size: Tuple[int, int, int],
        device: torch.device,
        sigma_noise: float,
    ) -> Physics:
        noise = GaussianNoise(sigma=sigma_noise)
        kernel_on_dev = self.kernel.to(device)

        if self.factor is None:
            return dinv.physics.BlurFFT(
                img_size=img_size,
                filter=kernel_on_dev,
                device=device,
                noise_model=noise,
            )

        return dinv.physics.Downsampling(
            img_size=img_size,
            filter=kernel_on_dev,
            factor=self.factor,
            device=device,
            noise_model=noise,
        )

    def __repr__(self) -> str:
        h, w = self.kernel.shape[-2:]
        if self.factor is None:
            return f"PhysicsFactory(blur, kernel={h}x{w})"
        return f"PhysicsFactory(downsample x{self.factor}, kernel={h}x{w})"


def _gaussian_kernel(sigma: float, size: Optional[int] = None) -> torch.Tensor:
    if size is None:
        return dinv.physics.blur.gaussian_blur(sigma=(sigma, sigma), angle=0)

    half = (size - 1) / 2.0
    coords = np.arange(size, dtype=np.float32) - half
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k /= k.sum()
    return torch.from_numpy(k).unsqueeze(0).unsqueeze(0)


def _uniform_kernel(size: int) -> torch.Tensor:
    k = np.ones((size, size), dtype=np.float32) / float(size * size)
    return torch.from_numpy(k).unsqueeze(0).unsqueeze(0)


def _parse_motion_mat(mat_path: Path) -> Dict[str, torch.Tensor]:
    payload = None
    try:
        raw = sio.loadmat(str(mat_path))
        payload_key = next((k for k in raw.keys() if not k.startswith("__")), None)
        if payload_key is None:
            raise ValueError(f"No usable payload in {mat_path}")
        payload = raw[payload_key]
    except NotImplementedError as e:
        with h5py.File(str(mat_path), "r") as f:
            dset = f["kernels"]
            payload = [np.array(f[ref[0]][()]).T for ref in dset]

    kernel_list = []
    if isinstance(payload, list):
        for entry in payload:
            kernel_list.append(np.asarray(entry, dtype=np.float32))
    elif payload.dtype == object:
        for entry in payload.flat:
            kernel_list.append(np.asarray(entry, dtype=np.float32))
    elif payload.ndim == 3:
        stack_axis = int(np.argmin(payload.shape))
        for i in range(payload.shape[stack_axis]):
            slc = [slice(None)] * 3
            slc[stack_axis] = i
            kernel_list.append(payload[tuple(slc)].astype(np.float32))
    elif payload.ndim == 2:
        kernel_list.append(payload.astype(np.float32))
    else:
        raise ValueError(f"Unexpected payload shape {payload.shape} in {mat_path}")

    out = {}
    for i, k in enumerate(kernel_list):
        k = k / (k.sum() + 1e-12)
        out[f"k{i + 1:02d}"] = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)
    return out


def load_deblur_kernels(
    motion_mat_path: Union[str, Path] = "kernels/Levin09.mat",
) -> Dict[str, PhysicsFactory]:
    mat_path = Path(motion_mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Motion kernel file not found: {mat_path}")

    motion = _parse_motion_mat(mat_path)
    out: Dict[str, PhysicsFactory] = {}
    for alias, k in motion.items():
        out[f"motion_{alias[1:]}"] = PhysicsFactory(kernel=k)

    out["uniform"] = PhysicsFactory(kernel=_uniform_kernel(9))
    out["gaussian"] = PhysicsFactory(kernel=_gaussian_kernel(sigma=1.6, size=25))
    return out


def load_sr_kernels(sr_factor) -> Dict[str, PhysicsFactory]:
    out = {}
    for name, (sigma, factor) in SR_CONFIGS.items():
        if name.endswith(f"x{sr_factor}"):
            out[name] = PhysicsFactory(kernel=_gaussian_kernel(sigma), factor=factor)
    return out


def get_kernels_for_problem(
    problem: str,
    motion_mat_path: Union[str, Path] = "kernels/Levin09.mat",
) -> Dict[str, PhysicsFactory]:
    if problem == "deblurring":
        return load_deblur_kernels(motion_mat_path)
    if problem == "sr2":
        return load_sr_kernels(2)
    if problem == "sr3":
        return load_sr_kernels(3)
    raise ValueError(f"Unknown problem: {problem!r}")
