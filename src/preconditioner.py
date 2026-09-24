import torch
from deepinv.physics import Physics


def get_blur_preconditioner(physics: Physics, x_shape, lam_damping=0.1):
    """
    Creates a preconditioner M ≈ (A^T A + lambda*I)^-1 using FFT.
    """
    H, W = x_shape[-2], x_shape[-1]

    kernel = physics.filter

    pad_h = H - kernel.shape[-2]
    pad_w = W - kernel.shape[-1]

    kernel_padded = torch.nn.functional.pad(kernel, (0, pad_w, 0, pad_h))
    kernel_padded = torch.roll(
        kernel_padded,
        shifts=(-(kernel.shape[-2] // 2), -(kernel.shape[-1] // 2)),
        dims=(-2, -1),
    )

    K_f = torch.fft.fftn(kernel_padded, dim=(-2, -1))
    M_f = 1.0 / (torch.abs(K_f) ** 2 + lam_damping)

    def M_apply(v):
        v_f = torch.fft.fftn(v, dim=(-2, -1))
        res_f = M_f * v_f
        return torch.real(torch.fft.ifftn(res_f, dim=(-2, -1)))

    return M_apply


def _M_inner(u: torch.Tensor, v: torch.Tensor, M_apply) -> float:
    """M-weighted inner product <u, v>_M = u^T M v."""
    return torch.dot(u.flatten(), v.flatten())
    return torch.dot(u.flatten(), M_apply(v).flatten())


def _M_norm_sq(u: torch.Tensor, M_apply) -> float:
    """||u||^2_M = u^T M u."""
    return _M_inner(u, u, M_apply)
