from typing import Literal, Optional, Tuple, Dict

from pathlib import Path
import numpy as np
import torch
import pandas as pd
import torchvision

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_complex(x):
    return torch.view_as_complex(torch.moveaxis(x, 1, -1).contiguous())


def to_image(x, clamp=True, rescale=False):
    if x.shape[1] == 2:
        x_complex = to_complex(x).contiguous()
        out = torch.abs(x_complex)
    elif x.shape[1] == 1:
        out = x[:, 0, ...]
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
    else:
        out = torch.moveaxis(x, 1, -1).contiguous()
        out = torch.nan_to_num(out)
        if clamp:
            out = torch.clamp(out, 0, 1)
        if rescale:
            out = out - out.min()
            out = out / out.max()
    return out


def rademacher(shape, dtype=torch.float32, device=DEVICE):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)


def _build_result_dataframe(
    metrics: Dict[str, Dict[str, Tuple[float, float]]],
    kernel_aliases: Tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    image_names = list(metrics.keys())
    for img in image_names:
        row = {}
        for k in kernel_aliases:
            psnr, ssim = metrics[img].get(k, (np.nan, np.nan))
            row[(k, "PSNR")] = psnr
            row[(k, "SSIM")] = ssim
        rows.append(row)

    df = pd.DataFrame(rows, index=image_names)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["kernel", "metric"])
    df.index.name = "image"

    # Append a final aggregate row
    if len(df) > 0:
        df.loc["MEAN"] = df.mean(numeric_only=True)
    return df


def _save_image_pair(
    x_degraded: torch.Tensor,
    x_restored: torch.Tensor,
    target_dir: Path,
    base_name: str,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        x_degraded.clamp(0, 1), target_dir / f"{base_name}-degraded.png"
    )
    torchvision.utils.save_image(
        x_restored.clamp(0, 1), target_dir / f"{base_name}-restored.png"
    )
