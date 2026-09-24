# Newton Trust-Region Plug-and-Play for Image Restoration

PyTorch implementation of **Newton Trust-Region PnP** using the Douglas-Rachford Envelope (DRE), a true second-order optimization method for Plug-and-Play image restoration.

> **Paper:** _Trust Region Plug-and-Play with Provable Convergence for Image Restoration_<br>
> IEEE Access, 2026
> [DOI: 10.1109/ACCESS.2026.3737001](https://doi.org/10.1109/ACCESS.2026.3737001)<br>

## Method

This repository implements a Newton trust-region method [[1](#references)] that operates on the Douglas-Rachford Envelope (DRE) [[2](#references)] of the composite objective $\varphi=h+\lambda g_\theta$, where $h$ is a data fidelity term and $g_\theta$ is a learned prior from a proximal denoiser (Prox-DRUNet), with controlled strength $\lambda>0$.

The variational objective:

$$
\begin{aligned}
\mathrm{F}_\gamma(u)&= h(x) + \lambda g_\theta(z) + \tfrac{1}{2\gamma}\lVert x - z\rVert^2 - \tfrac{1}{\gamma}\langle x - u, x - z\rangle\\
\nabla \mathrm{F}_\gamma(u)&= \frac{1}{\gamma}(2J_x-I)(x - z)\\
\nabla^2 \mathrm{F}_\gamma(u) \cdot v&= \frac{1}{\gamma}(2J_x-I)\left[J_x v-J_z(2J_x v-v)\right]
\end{aligned}
$$

Where:

$$
\begin{aligned}
J_x&=\nabla_u\mathrm{prox}_{\gamma h}(u)\\
J_z v&=J_{D_\theta}(x)\cdot v
\end{aligned}
$$

The inner trust-region subproblem is solved by Steihaug-CG with explicit negative curvature handling. Hessian-vector products are computed matrix-free via `torch.func.jvp`.

**Baselines included:** PnP-PGD (version using Prox-DRUNet [[3](#references)]), PnP-DRS [[4](#references)], PnP-αPGD [[5](#references)], PnP-LBFGS [[6](#references)], DPIR [[7](#references)].

## Repository Structure

```
├── src/
│   ├── pnp.py               # All PnP solvers (PGD, DRS, aPGD, DPIR, LBFGS, Newton-TR)
│   ├── dataset.py           # Torch dataset class and dataloader
│   ├── second_order.py      # SecondOrderDRE (gradient + HVP)
│   ├── steihaug.py          # Steihaug-CG implementation
│   ├── denoiser.py          # Denoiser class and model loading
│   ├── kernels.py           # Blur kernel loading (Levin09, Gaussian, uniform, super-resolution)
│   ├── LBFGS.py             # L-BFGS solver for Tan et al. baseline
│   ├── utils.py             # Metrics, device config, helpers
│   ├── runner.py            # Experiment runner with SOLVER_REGISTRY
│   ├── preconditioner.py    # Fourier preconditioner (experimental, not used)
│   └── scehduler.py         # Sigma scheduler (experimental, not used)
├── kernels/                 # Blur kernel .mat files
├── models/                  # Denoiser checkpoints (see below)
├── experiment.py            # Reproduce the experiments
├── run.py                   # Run algorithm for single instance
└── README.md
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with `torch.func` support)
- [deepinv](https://github.com/deepinv/deepinv)
- numpy, scipy, pandas, wandb (for logging)

```bash
pip install torch torchvision deepinv numpy scipy pandas wandb
```

## Denoiser Checkpoints

This code uses the **Prox-DRUNet** checkpoint from [Hurault et al. (ICML 2022)](https://github.com/samuro95/Prox-PnP). Download the Prox-DRUNet (Softplus, L_g < 1) pretrained checkpoint and place it in `models/Prox-DRUNet.ckpt`.

For the DPIR baseline, standard DRUNet color weights are downloaded from deepinv [here](https://deepinv.github.io/deepinv/user_guide/reconstruction/pretrained-models.html), then place it in `models/drunet_deepinv_color.pth`.

## Usage

### Setting Up Weights & Biases Tracker

If you want to track PSNR evolution (and other metric), set up W&B tracker by creating `config.ini` files in this directory with API information.

```
[wandb]
key = WANDB_API_KEY
entity = WANDB_ENTITY_OR_USERNAME
project = PROJECT_NAME
```

### Newton-TR on DRE (proposed method)

```python
from src.pnp import trust_region_dre
from src.denoiser import GradientStepDenoiser, load_hurault_potential_network

# Load denoiser
model = load_hurault_potential_network("models/Prox-DRUNet.ckpt", device="cuda")
denoiser = GradientStepDenoiser(model)

# Run Newton-TR
metrics, x_restored = trust_region_dre(
    x=x_init,
    denoiser=denoiser,
    physics=physics,
    v_noise=0.01,
    sigma_v=1.0,
    lam=1.0,
    gamma=1.0,
    total_iter=100,
)
```

### Baselines

```python
from src.pnp import pgd, drs, alpha_pgd, dpir, q_newton_prox_tan_2024

# PnP-PGD
metrics, x_out = pgd(x, denoiser, physics, v_noise=0.01, gamma=1.5)

# PnP-DRS
metrics, x_out = drs(x, denoiser, physics, v_noise=0.01, lam=1.5, beta=0.25, gamma=0.45)

# PnP-αPGD
metrics, x_out = alpha_pgd(x, denoiser, physics, v_noise=0.01, alpha=1.0)

# PnP-LBFGS (Tan et al. 2024)
metrics, x_out = q_newton_prox_tan_2024(x, denoiser, physics, v_noise=0.01)

# DPIR (requires standard DRUNet, not Prox-DRUNet)
metrics, x_out = dpir(x, dpir_denoiser, physics, v_noise=0.01)
```

## Tasks

Supported inverse problems:

- **Gaussian deblurring** — 10 kernels (8 motion from Levin et al. 2009, 1 uniform 9×9, 1 Gaussian 25×25)
- **Super-resolution** — scales s ∈ {2, 3} with antialiasing kernels

Noise levels: ν ∈ {0.01, 0.03, 0.05} (corresponding to σ ∈ {2.55, 7.65, 12.75}/255).

## Citation

If you found our project helpful, please cite our paper.

```bibtex
@article{TODO,
  title={Trust Region Plug-and-Play with Provable Convergence for Image Restoration},
  author={TODO},
  journal={IEEE Access},
  year={2026}
}
```

## References

1. Conn at al. Trust region methods. Society for Industrial and Applied Mathematics, 2000.
2. Themelis et al. "Douglas--Rachford splitting and ADMM for nonconvex optimization: Tight convergence results." SIAM Journal on Optimization 30.1 (2020): 149-181.
3. Hurault at al. "Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization." International Conference on Machine Learning. PMLR, 2022.
4. Hurault et al. "Convergent plug-and-play with proximal denoiser and unconstrained regularization parameter." Journal of Mathematical Imaging and Vision 66.4 (2024): 616-638.
5. Hurault et al. "A relaxed proximal gradient descent algorithm for convergent plug-and-play with proximal denoiser." International Conference on Scale Space and Variational Methods in Computer Vision. Cham: Springer International Publishing, 2023.
6. Tan et al. "Provably convergent plug-and-play quasi-newton methods." SIAM Journal on Imaging Sciences 17.2 (2024): 785-819.
7. Zhang et al. "Plug-and-play image restoration with deep denoiser prior." IEEE Transactions on Pattern Analysis and Machine Intelligence 44.10 (2021): 6360-6376.
8. Levin et al. "Understanding and evaluating blind deconvolution algorithms." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.

Some part of this repository are copied from [Tan repo](https://github.com/hyt35/Prox-qN), and [Hurault Repo](https://github.com/samuro95/Prox-PnP).

## License

MIT
