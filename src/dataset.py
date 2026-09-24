"""Flat-folder image dataset for restoration experiments."""

from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMG_SIZE = 256


class ImageFolderDataset(Dataset):
    """
    Dataset reading a flat folder of images (no nested subdirectories).

    Yields (image_name, image_tensor) where image_tensor is [C, H, W] in [0, 1].
    """

    def __init__(self, root: Union[str, Path], sr_factor: int = None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if sr_factor is not None:
            self.sr_factor = sr_factor
        else:
            self.sr_factor = 2

        self.files = sorted(
            p
            for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if len(self.files) == 0:
            raise ValueError(f"No image files in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        tensor = TF.to_tensor(img)  # [C, H, W] in [0, 1]

        tensor = TF.center_crop(tensor, output_size=[IMG_SIZE, IMG_SIZE])[
            ...,
            : IMG_SIZE - IMG_SIZE % self.sr_factor,
            : IMG_SIZE - IMG_SIZE % self.sr_factor,
        ]

        return path.stem, tensor


def _summarize_dataset(dataset: ImageFolderDataset) -> None:
    """Print a quick statistical summary of image shapes."""
    sample = dataset.files[: min(50, len(dataset.files))]
    sizes = []
    for p in sample:
        with Image.open(p) as img:
            sizes.append(img.size)  # (W, H)
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]

    print(f"Dataset: {dataset.root.name}")
    print(f"  Images   : {len(dataset)}")
    print(
        f"  Width    : min={min(widths)}, max={max(widths)}, "
        f"mean={sum(widths) / len(widths):.0f}"
    )
    print(
        f"  Height   : min={min(heights)}, max={max(heights)}, "
        f"mean={sum(heights) / len(heights):.0f}"
    )


def build_dataloader(
    root: Union[str, Path],
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = True,
    sr: int = None,
) -> Tuple[DataLoader, List[str]]:
    """
    Build a DataLoader over a flat image folder.

    Note
    ----
    Default batch_size is 1 because Set14/DIV2K images have heterogeneous
    shapes that cannot be batched without cropping or padding. Setting
    batch_size > 1 will fail at collation time unless every image has the
    same dimensions.

    Returns
    -------
    loader : DataLoader
    image_names : list[str]
        Filename stems in dataset order (matches DataLoader iteration).
    """
    dataset = ImageFolderDataset(root, sr)

    if verbose:
        _summarize_dataset(dataset)

    if batch_size != 1:
        print(
            f"  WARNING: batch_size={batch_size} > 1 with heterogeneous image "
            "sizes will fail in DataLoader collation."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    image_names = [p.stem for p in dataset.files]
    return loader, image_names
