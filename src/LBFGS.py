import torch


class LBFGSSolver:
    def __init__(self, m=10):
        self.m = m
        self.history = []

    def update(self, s, y):
        if torch.dot(s.flatten(), y.flatten()) > 1e-10:  # Curvature condition
            if len(self.history) >= self.m:
                self.history.pop(0)
            self.history.append((s, y))

    def solve(self, grad):
        if not self.history:
            return -grad

        q = grad.clone()
        alphas = []

        # Backward pass
        for s, y in reversed(self.history):
            rho = 1.0 / torch.dot(y.flatten(), s.flatten())
            alpha = rho * torch.dot(s.flatten(), q.flatten())
            q -= alpha * y
            alphas.append(alpha)

        # Scaling
        s_last, y_last = self.history[-1]
        gamma = torch.dot(s_last.flatten(), y_last.flatten()) / torch.dot(
            y_last.flatten(), y_last.flatten()
        )
        z = q * gamma

        # Forward pass
        for i, (s, y) in enumerate(self.history):
            rho = 1.0 / torch.dot(y.flatten(), s.flatten())
            beta = rho * torch.dot(y.flatten(), z.flatten())
            z += s * (alphas[-(i + 1)] - beta)

        return -z
