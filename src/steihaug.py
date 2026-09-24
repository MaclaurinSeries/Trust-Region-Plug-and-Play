import torch
from typing import Tuple, Union
from .denoiser import RelaxedDenoiser, GradientStepDenoiser
from deepinv.physics import Physics
from . import preconditioner as prec
from .second_order import SecondOrderDRE, SecondOrderFBE
from .utils import DEVICE


def _tr_boundary_step(d: torch.Tensor, p: torch.Tensor, Delta2: float) -> float:
    """
    Find tau >= 0 such that ||d + tau*p|| = Delta.
        tau^2 ||p||^2 + 2 tau <d, p> + ||d||^2 - Delta^2 = 0
    """
    a = torch.dot(p.flatten(), p.flatten())
    ba = torch.dot(d.flatten(), p.flatten()) / a
    ca = (torch.dot(d.flatten(), d.flatten()) - Delta2) / a

    disc = torch.clamp(ba * ba - ca, min=0.0)

    return -ba + torch.sqrt(disc)


class SteihaugCache:
    def __init__(self):
        self.is_empty = True
        self.recorded = []

    def empty(self):
        self.is_empty = True
        self.recorded = []

    def record_iteration(self, iter, **kwargs):
        self.is_empty = False
        if len(self.recorded) <= iter:
            self.recorded.append(kwargs)
        else:
            self.recorded[iter].update(kwargs)

    def replay_iteration(self, iter):
        return self.recorded[iter]


class Steihaug:
    """
    Steihaug-CG (Algorithm 2) with negative-curvature branch
    and elliptical TR ||d|| <= Delta.

    Init:
        d_0 = 0, r_0 = grad_b, p_0 = -r_0

    Loop (j = 0, 1, ...):
        kappa = p_j^T H p_j
        if kappa <= 0:
            tau = boundary step; return d_j + tau * p_j
        alpha_j = (r_j^T z_j) / kappa
        d_{j+1} = d_j + alpha_j p_j
        if ||d_{j+1}||_M >= Delta:
            tau = boundary step from d_j along p_j; return d_j + tau * p_j
        r_{j+1} = r_j + alpha_j H p_j
        if ||r_{j+1}|| <= eta ||grad_b||: return d_{j+1}
        z_{j+1} = M r_{j+1}
        gamma_j = (r_{j+1}^T z_{j+1}) / (r_j^T z_j)
        p_{j+1} = -z_{j+1} + gamma_j p_j
    """

    def __init__(
        self,
        denoiser: Union[RelaxedDenoiser, GradientStepDenoiser],
        physics: Physics,
        j_max: int,
        second_order: Union[SecondOrderDRE, SecondOrderFBE],
    ):
        self.denoiser = denoiser
        self.physics = physics
        self._max_iter = j_max
        self._second_order = second_order
        self._hit_boundary = False
        self._steihaug_last_exit = -1

        self.cache = SteihaugCache()

    def _replay_recorded(
        self,
        gradient: torch.Tensor,
        Delta2: torch.FloatType,
        eta2: torch.FloatType,
    ):
        Hd = torch.zeros(gradient.shape, device=DEVICE)
        for j in range(self._max_iter):
            cache = self.cache.replay_iteration(j)
            Hp, d, p = cache["Hp"], cache["d"], cache["p"]
            if cache["kappa"] <= 0:
                self._steihaug_last_exit = 1
                self._hit_boundary = True
                tau = _tr_boundary_step(d, p, Delta2)
                return d + tau * p, Hd + tau * Hp
            if cache["ddist"] >= Delta2:
                self._steihaug_last_exit = 2
                self._hit_boundary = True
                tau = _tr_boundary_step(d, p, Delta2)
                return d + tau * p, Hd + tau * Hp
            Hd = Hd + cache["alpha"] * Hp
            if cache["rdist"] <= eta2:
                self._steihaug_last_exit = 3
                self._hit_boundary = False
                return cache["d_new"], Hd

        self._steihaug_last_exit = 4
        cache = self.cache.replay_iteration(-1)
        self._hit_boundary = False
        return cache["d_new"], Hd

    def _run_steihaug(
        self,
        gradient: torch.Tensor,
        point: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        Delta2: torch.FloatType,
        eta2: torch.FloatType,
    ):
        d = torch.zeros(gradient.shape, device=DEVICE)
        Hd = torch.zeros(gradient.shape, device=DEVICE)
        r = gradient.clone()
        p = -r

        r2 = torch.dot(r.flatten(), r.flatten())

        for j in range(self._max_iter):
            Hp = self._second_order._hvp(point, p)
            kappa = torch.dot(p.flatten(), Hp.flatten())

            self.cache.record_iteration(iter=j, kappa=kappa, d=d, p=p, Hp=Hp)
            if kappa <= 0.0:
                self._steihaug_last_exit = 1
                self._hit_boundary = True
                tau = _tr_boundary_step(d, p, Delta2)
                return d + tau * p, Hd + tau * Hp

            alpha = r2 / kappa
            d_new = d + alpha * p
            Hd_new = Hd + alpha * Hp

            ddist = torch.dot(d_new.flatten(), d_new.flatten())
            self.cache.record_iteration(
                iter=j, ddist=ddist, d_new=d_new.detach(), alpha=alpha
            )
            if ddist >= Delta2:
                self._steihaug_last_exit = 2
                self._hit_boundary = True
                tau = _tr_boundary_step(d, p, Delta2)
                return d + tau * p, Hd + tau * Hp

            r_new = r + alpha * Hp

            rdist = torch.dot(r_new.flatten(), r_new.flatten())
            self.cache.record_iteration(iter=j, rdist=rdist)
            if rdist <= eta2:
                self._steihaug_last_exit = 3
                self._hit_boundary = False
                return d_new, Hd_new

            gamma_cg = rdist / r2

            p = -r_new + gamma_cg * p
            d, r, r2, Hd = d_new, r_new, rdist, Hd_new

        self._steihaug_last_exit = 4
        self._hit_boundary = False
        return d, Hd

    def find_d(
        self,
        gradient: torch.Tensor,
        point: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        Delta: float,
        forcing_tol: float,  # eta
        last_accept: bool = True,
    ):
        grad_norm = torch.dot(gradient.flatten(), gradient.flatten())
        eta2 = forcing_tol * forcing_tol * grad_norm
        Delta2 = Delta**2

        if last_accept or self.cache.is_empty:
            self.cache.empty()
            return self._run_steihaug(gradient, point, Delta2, eta2)
        else:
            return self._replay_recorded(gradient, Delta2, eta2)
