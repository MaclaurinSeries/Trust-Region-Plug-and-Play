import torch
from typing import Tuple, Union
from .denoiser import RelaxedDenoiser, GradientStepDenoiser
from deepinv.physics import Physics
from deepinv.optim.data_fidelity import L2


class SecondOrderFBE:
    def __init__(
        self,
        y_obs: torch.Tensor,
        physics: Physics,
        denoiser: "GradientStepDenoiser",
        sigma_val: float,
        lam: float,
        gamma: float,
        F: L2,
    ):
        self.observed = y_obs
        self.physics = physics
        self.denoiser = denoiser
        self.fidelity = F
        self._sigma = sigma_val
        self._lambda = lam
        self._gamma = gamma

    def _phi_and_grad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_h = self.fidelity.grad(x, self.observed, self.physics)

        z = x - self._gamma * self._lambda * grad_h

        grad_g_z = self.denoiser._calculate_grad(z, self._sigma)
        u = z - grad_g_z

        diff = (x - u) / self._gamma
        AtA_diff = self.physics.A_adjoint(self.physics.A(diff))
        grad_fbe = diff - self._gamma * self._lambda * AtA_diff

        if self.denoiser.use_potential():
            h_val = self._lambda * self.fidelity.fn(
                x, self.observed, physics=self.physics
            )

            xz = (x - z).flatten()
            zu = (z - u).flatten()
            term1 = torch.dot(xz, xz) / (2 * self._gamma)
            term2 = torch.dot(zu, zu) / (2 * self._gamma)

            g_val_u = self.denoiser._potential(u, self._sigma, lam=self._lambda)

            fbe_val = (h_val - term1 + term2 + g_val_u).detach()
        else:
            fbe_val = None

        return fbe_val, grad_fbe.detach(), u.detach()

    def _hvp(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Should be unused
        raise NotImplementedError

        AtA_v = self.physics.A_adjoint(self.physics.A(v))
        Mv = v - self._gamma * AtA_v

        J_Pg_v = self.denoiser._jvp_D(
            x, Mv, self._sigma, lam=self._gamma * self._lambda
        )
        inner = v - J_Pg_v

        AtA_inner = self.physics.A_adjoint(self.physics.A(inner))
        hvp_fbe = (inner / self._gamma) - AtA_inner

        return hvp_fbe.detach()


class SecondOrderDRE(torch.nn.Module):
    def __init__(
        self,
        y_obs: torch.Tensor,
        physics: Physics,
        denoiser: "GradientStepDenoiser",
        sigma_val: float,
        lam: float,
        gamma: float,
        F: L2,
    ):
        super().__init__()
        self.observed = y_obs
        self.physics = physics
        self.denoiser = denoiser
        self.fidelity = F
        self._sigma = sigma_val
        self._lambda = lam
        self._gamma = gamma
        self.register_buffer("_zero_y", torch.zeros_like(y_obs))

    def _Jx(self, v):
        # J_x = d/du prox_{gamma h}(u) = (I + gamma A^T A)^{-1}.
        # assuming prox is affine; prox(0; y=0)=0 => prox(v; y=0) == J_x v.
        return self.fidelity.prox(v, self._zero_y, self.physics, gamma=self._gamma)

    def _reflect(self, v):
        return 2.0 * self._Jx(v) - v

    def _phi_and_grad(
        self, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Douglas-Rachford splitting and ADMM for nonconvex optimization: tight convergence results
        Themelis et al. (3.6)

        langrangian DRE objective

        u = proxy variable
        x = prox_{gamma, f}(u)
        z = prox_{gamma, g}(2x - u)

        L = h(x) + lam * g(z) + 1/(gamma) * <u - x, z - x> + 1/(2gamma) * ||z - x||^2
        """
        x = self.fidelity.prox(u, self.observed, self.physics, gamma=self._gamma)

        z_in = 2 * x - u
        z = self.denoiser.forward(
            z_in,
            self._sigma,
            no_grad=False,
            lam=self._gamma * self._lambda,
        )

        grad_dre = self._reflect(x.detach() - z.detach()) / self._gamma

        if self.denoiser.use_potential():
            h_val = self.fidelity.fn(x, self.observed, physics=self.physics)

            a = (u - x).flatten()
            b = (z - x).flatten()
            term1 = torch.dot(a, b) / (self._gamma)
            term2 = torch.dot(b, b) / (2 * self._gamma)

            g_val_z = self.denoiser._potential(z, self._sigma, lam=self._lambda)
            dre_val = (h_val + term1 + term2 + g_val_z).detach()
        else:
            dre_val = None

        return dre_val, grad_dre, (u.detach(), x.detach())

    def _hvp(
        self, point: Tuple[torch.Tensor, torch.Tensor], v: torch.Tensor
    ) -> torch.Tensor:
        u, x = point

        v_x = self._Jx(v)

        v_in_g = 2 * v_x - v
        z_in = 2 * x - u

        J_Pg_v = self.denoiser._jvp_D(
            z_in, v_in_g, self._sigma, lam=self._gamma * self._lambda
        )
        raw = (v_x - J_Pg_v) / self._gamma

        return self._reflect(raw).detach()
