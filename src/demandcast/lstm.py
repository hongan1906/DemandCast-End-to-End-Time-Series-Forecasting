"""V2: a small LSTM on the *same* windows (N, L, C). Requires `pip install torch`."""
from __future__ import annotations

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class LSTMForecaster:
    """LSTM over the window, concatenated with known target-day covariates -> MLP -> y/scale."""

    def __init__(self, hidden: int = 32, epochs: int = 25, lr: float = 2e-3, batch: int = 256, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        self.net = None

    def _build(self, c: int, k: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, hidden):
                super().__init__()
                s.lstm = nn.LSTM(c, hidden, batch_first=True)
                s.head = nn.Sequential(nn.Linear(hidden + k, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(s, x, kn):
                _, (h, _) = s.lstm(x)
                return s.head(__import__("torch").cat([h[-1], kn], dim=1)).squeeze(-1)

        return Net(self.hidden)

    def fit(self, X_seq: np.ndarray, known: np.ndarray, y: np.ndarray):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = self._build(X_seq.shape[2], known.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.L1Loss()  # aligned with MAE / WAPE
        X = torch.tensor(X_seq, dtype=torch.float32)
        K = torch.tensor(known, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        n = len(X)
        self.net.train()
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.batch):
                b = torch.tensor(idx[i : i + self.batch])
                opt.zero_grad()
                loss_fn(self.net(X[b], K[b]), Y[b]).backward()
                opt.step()
        return self

    def predict(self, X_seq: np.ndarray, known: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(known, dtype=torch.float32))
        return out.numpy()
