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
    """Shared-trunk multi-task classifier with masked auxiliary regression.

    With ``lambda_reg = 0`` this is the single-task baseline. Feature and
    auxiliary-target scalers are fit on the training data only, so no test-fold
    statistics leak into training.
    """

    def __init__(self, lambda_reg: float = config.MTL_LAMBDA_REG) -> None:
        self.lambda_reg = lambda_reg
        self._feature_scaler = StandardScaler()
        self._net: _MultiTaskNet | None = None
        self._aux_mean: np.ndarray | None = None
        self._aux_std: np.ndarray | None = None

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

        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        z_t = torch.tensor(z_scaled, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)

        self._net = _MultiTaskNet(X.shape[1], Z.shape[1])
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=config.MTL_LEARNING_RATE,
            weight_decay=config.MTL_WEIGHT_DECAY,
        )
        bce = nn.BCEWithLogitsLoss()
        self._net.train()
        for _ in range(config.MTL_EPOCHS):
            optimizer.zero_grad()
            logit, aux_pred = self._net(x_t)
            loss = config.MTL_LAMBDA_CLASS * bce(logit, y_t)
            if self.lambda_reg > 0:
                loss = loss + self.lambda_reg * _masked_mse(aux_pred, z_t, mask_t)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model must be fit before prediction.")
        self._net.eval()
        x_t = torch.tensor(self._feature_scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            logit, _ = self._net(x_t)
        return torch.sigmoid(logit).numpy()


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
