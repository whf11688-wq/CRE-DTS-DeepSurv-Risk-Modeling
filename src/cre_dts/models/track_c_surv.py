"""Track C: DeepSurv time-to-event model (Technical Design Specification, Section 5.3).

h(t | x) = h0(t) * exp(g_theta(x)); g_theta is a neural network trained by minimizing the negative Cox partial
log-likelihood with Breslow ties. The baseline cumulative hazard is estimated with the Breslow estimator, giving a
survival curve S(t | x) for every loan. Censored loans (paid off or still active) enter the risk sets.
Time is measured in quarters.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DeepSurvNet(nn.Module):
    """Log-risk network g_theta: 2 x 64 SELU (initial design, Section 5.3)."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden, hidden), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def cox_ph_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood, Breslow ties, averaged over events.

    For each loan i with an observed default, compare its risk with the risk set
    {j : t_j >= t_i} of loans still at risk at that time.
    """
    order = torch.argsort(time, descending=True)
    lr, t, e = log_risk[order], time[order], event[order]
    log_cum = torch.logcumsumexp(lr, dim=0)
    # Breslow ties: every tied loan shares the full risk set, which ends at the last index of its tie block
    _, inverse, counts = torch.unique_consecutive(t, return_inverse=True, return_counts=True)
    last_idx = torch.cumsum(counts, 0) - 1
    log_den = log_cum[last_idx][inverse]
    n_events = e.sum().clamp(min=1.0)
    return -((lr - log_den) * e).sum() / n_events


def fit_deepsurv(X: np.ndarray, time: np.ndarray, event: np.ndarray, epochs: int = 200, lr: float = 1e-3,
                 weight_decay: float = 1e-4, hidden: int = 64, dropout: float = 0.1, seed: int = 0) -> DeepSurvNet:
    """Full-batch Adam training of the DeepSurv network (sufficient for the synthetic examples and tests)."""
    torch.manual_seed(seed)
    x = torch.as_tensor(X, dtype=torch.float32)
    t = torch.as_tensor(time, dtype=torch.float32)
    e = torch.as_tensor(event, dtype=torch.float32)
    model = DeepSurvNet(x.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = cox_ph_loss(model(x), t, e)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def log_risk(model: DeepSurvNet, X: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy()


def breslow_baseline(log_risk_train: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Breslow estimate of the baseline cumulative hazard H0(t) at each distinct event time."""
    risk = np.exp(np.asarray(log_risk_train, dtype=float))
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    times = np.unique(time[event == 1])
    dH = np.array([event[time == s].sum() / risk[time >= s].sum() for s in times])
    return times, np.cumsum(dH)


def survival_curve(log_risk_x, times, H0, grid) -> np.ndarray:
    """S(t | x) = exp(-H0(t) * exp(g(x))) evaluated on `grid` (step function, right-continuous)."""
    grid = np.asarray(grid, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    H = np.where(idx >= 0, H0[np.clip(idx, 0, None)], 0.0)
    lr = np.atleast_1d(np.asarray(log_risk_x, dtype=float))
    return np.exp(-np.outer(np.exp(lr), H))


def concordance_index(time, event, score) -> float:
    """Harrell's C-index: share of comparable pairs in which the higher-risk loan defaults first."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    score = np.asarray(score, dtype=float)
    num = den = 0.0
    for i in np.flatnonzero(event == 1):
        later = time > time[i]
        den += later.sum()
        num += (score[i] > score[later]).sum() + 0.5 * (score[i] == score[later]).sum()
    return float(num / den) if den else float("nan")
