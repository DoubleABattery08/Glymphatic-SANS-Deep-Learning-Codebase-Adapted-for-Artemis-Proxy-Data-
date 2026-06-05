"""Models: the mechanism-constrained multi-task network and the cross-checks.

The tabular analog of the methodology's shared-representation network: a compact
shared trunk feeds one primary classification head (binary cross-entropy) and one
auxiliary regression head over several targets (masked mean squared error). The
combined objective is

    L_total = lambda_class * L_class + lambda_reg * L_reg,

so ``lambda_reg = 0`` recovers an ordinary single-task model and the auxiliary
contribution is measured directly by comparing the two. Auxiliary losses are
masked per element, so a target contributes only for observations where it was
actually measured; this is the faithful treatment of unequal multi-modal
coverage and avoids discarding subjects to a complete-case intersection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from artemis_proxy import config, targets


def to_supervised(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Split a modeling frame into model-ready arrays.

    Returns features ``X``, primary labels ``y``, auxiliary targets ``Z``, an
    observation-by-target boolean ``mask`` (True where the auxiliary is observed),
    and the ``subjects`` grouping vector.
    """

    feature_cols = targets.feature_columns()
    aux = frame[targets.AUXILIARY_TARGETS]
    return {
        "X": frame[feature_cols].to_numpy(dtype=np.float64),
        "y": frame["outcome"].to_numpy(dtype=np.float64),
        "y_continuous": frame["lv_mass_change"].to_numpy(dtype=np.float64),
        "Z": aux.to_numpy(dtype=np.float64),
        "mask": aux.notna().to_numpy(),
        "subjects": frame["Subject"].to_numpy(),
    }


class _MultiTaskNet(nn.Module):
    def __init__(self, input_dim: int, n_aux: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.MTL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.MTL_DROPOUT),
        )
        self.class_head = nn.Linear(config.MTL_HIDDEN_DIM, 1)
        self.aux_head = nn.Linear(config.MTL_HIDDEN_DIM, n_aux)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.class_head(hidden).squeeze(-1), self.aux_head(hidden)


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    observed = mask.sum()
    if observed == 0:
        return pred.new_zeros(())
    squared = (pred - target) ** 2 * mask
    return squared.sum() / observed


class MechanismConstrainedMTL:
    """Shared-trunk multi-task learner with masked auxiliary regression.

    The primary head is a binary classifier (``task="classification"``, the
    headline that mirrors the SANS framing) or a continuous regressor
    (``task="regression"``, the better-powered companion). The auxiliary heads are
    always masked regression. With ``lambda_reg = 0`` this is the single-task
    baseline. Feature, auxiliary-target, and (for regression) primary-target
    scalers are fit on the training data only, so no test-fold statistics leak.
    """

    def __init__(
        self,
        lambda_reg: float = config.MTL_LAMBDA_REG,
        task: str = "classification",
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task: {task}")
        self.lambda_reg = lambda_reg
        self.task = task
        self._feature_scaler = StandardScaler()
        self._net: _MultiTaskNet | None = None
        self._aux_mean: np.ndarray | None = None
        self._aux_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _scale_aux(self, Z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scaled = (Z - self._aux_mean) / self._aux_std
        # Unobserved entries are set to zero; the mask removes them from the loss.
        return np.where(mask, scaled, 0.0)

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray, mask: np.ndarray
    ) -> MechanismConstrainedMTL:
        # Deterministic initialization regardless of call order (folds, bootstrap).
        torch.manual_seed(config.SEED)
        x_scaled = self._feature_scaler.fit_transform(X)
        masked_Z = np.where(mask, Z, np.nan)
        self._aux_mean = np.nanmean(masked_Z, axis=0)
        self._aux_std = np.nanstd(masked_Z, axis=0)
        self._aux_std = np.where(self._aux_std > 0, self._aux_std, 1.0)
        z_scaled = self._scale_aux(Z, mask)

        if self.task == "regression":
            # Standardize the continuous primary target for stable optimization;
            # predictions are returned to the original scale at inference.
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y)) or 1.0
        y_fit = (y - self._y_mean) / self._y_std

        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_fit, dtype=torch.float32)
        z_t = torch.tensor(z_scaled, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)

        self._net = _MultiTaskNet(X.shape[1], Z.shape[1])
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=config.MTL_LEARNING_RATE,
            weight_decay=config.MTL_WEIGHT_DECAY,
        )
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        self._net.train()
        for _ in range(config.MTL_EPOCHS):
            optimizer.zero_grad()
            primary, aux_pred = self._net(x_t)
            if self.task == "classification":
                loss = config.MTL_LAMBDA_CLASS * bce(primary, y_t)
            else:
                loss = config.MTL_LAMBDA_CLASS * mse(primary, y_t)
            if self.lambda_reg > 0:
                loss = loss + self.lambda_reg * _masked_mse(aux_pred, z_t, mask_t)
            loss.backward()
            optimizer.step()
        return self

    def _primary_output(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model must be fit before prediction.")
        self._net.eval()
        x_t = torch.tensor(self._feature_scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            primary, _ = self._net(x_t)
        return primary.numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification.")
        return 1.0 / (1.0 + np.exp(-self._primary_output(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task != "regression":
            raise RuntimeError("predict is only defined for regression.")
        return self._primary_output(X) * self._y_std + self._y_mean


def build_elastic_net() -> Pipeline:
    """Transparent logistic elastic-net cross-check on the same feature set."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=config.ELASTICNET_L1_RATIO,
                    C=config.ELASTICNET_C,
                    max_iter=5000,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
