from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class Config:
    history: int = 28
    horizon: int = 7
    train_fraction: float = 0.8
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 1e-3


def make_series(n_days: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    t = np.arange(n_days, dtype=np.float32)

    promo = (rng.random(n_days) < 0.08).astype(np.float32)
    price_index = 1.0 + 0.05 * np.sin(2 * np.pi * t / 45.0) + rng.normal(0, 0.015, n_days)
    weekly = 8.0 * np.sin(2 * np.pi * t / 7.0)
    trend = 0.018 * t
    shock = rng.normal(0.0, 3.0, n_days)

    demand = (
        52.0
        + trend
        + weekly
        + 15.0 * promo
        - 30.0 * (price_index - 1.0)
        + shock
    )
    demand = np.maximum(demand, 1.0).astype(np.float32)

    features = np.column_stack(
        [
            demand,
            promo,
            price_index.astype(np.float32),
            np.sin(2 * np.pi * t / 7.0),
            np.cos(2 * np.pi * t / 7.0),
        ]
    ).astype(np.float32)
    return features, demand


def build_windows(
    features: np.ndarray,
    demand: np.ndarray,
    history: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for end in range(history, len(features) - horizon + 1):
        xs.append(features[end - history : end])
        ys.append(demand[end : end + horizon])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class GRUForecast(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecast(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.positional(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


def standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit(
    model: nn.Module,
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    cfg: Config,
) -> None:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train_scaled),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(x)).cpu().numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    over = np.maximum(error, 0.0)
    under = np.maximum(-error, 0.0)
    decision_cost = np.mean(1.0 * over + 4.0 * under)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "inventory_cost_proxy": float(decision_cost),
    }


def main() -> None:
    cfg = Config()
    features, demand = make_series()
    x, y = build_windows(features, demand, cfg.history, cfg.horizon)

    split = int(len(x) * cfg.train_fraction)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    x_train, x_test, _, _ = standardize(x_train, x_test)

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-6
    y_train_scaled = (y_train - y_mean) / y_std

    models = {
        "GRU": GRUForecast(x.shape[-1], cfg.horizon),
        "Transformer": TransformerForecast(x.shape[-1], cfg.horizon),
    }

    print("Model comparison")
    print("-" * 72)
    for name, model in models.items():
        fit(model, x_train, y_train_scaled.astype(np.float32), cfg)
        pred_scaled = predict(model, x_test)
        pred = pred_scaled * y_std + y_mean
        result = metrics(y_test, pred)
        print(
            f"{name:12s} | "
            f"MAE={result['MAE']:.3f} | "
            f"RMSE={result['RMSE']:.3f} | "
            f"inventory_cost_proxy={result['inventory_cost_proxy']:.3f}"
        )


if __name__ == "__main__":
    main()
