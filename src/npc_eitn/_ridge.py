"""Fold-local ridge/PCA prediction; importing this module never writes files.

The competing-risk estimand matches the existing project: a target event by
the horizon is 1; an observed competing event or event-free follow-up through
the horizon is 0. Early censoring is not a known binary outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


FIXED_EITN = "formal__ring15__q_hl"
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
CLINICAL_COLUMNS = ("age", "t_stage", "n_stage", "log_gtv_volume")
CENSORING_FLOOR = 0.05
RANDOM_SEED = 20261015


@dataclass(frozen=True)
class ModelSpec:
    name: str
    include_eitn: bool = False
    image_mode: str = "none"
    n_components: int = 3

    def __post_init__(self):
        if self.image_mode not in {"none", "pca", "ridge"}:
            raise ValueError("image_mode must be none, pca, or ridge")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")


def _times_codes(frame):
    times = pd.to_numeric(frame["time_after_landmark"], errors="coerce").to_numpy(float)
    raw_codes = pd.to_numeric(frame["endpoint_code"], errors="coerce").to_numpy(float)
    if (not np.isfinite(times).all() or np.any(times < 0)
            or not np.isfinite(raw_codes).all()
            or not np.isin(raw_codes, [0, 1, 2, 3]).all()):
        raise ValueError("Endpoint times/codes must be observed, nonnegative and valid")
    return times, raw_codes.astype(int)


def censoring_km(frame):
    """Product-limit censoring survival, including censoring at each saved time."""
    times, codes = _times_codes(frame)
    unique, counts = np.unique(times, return_counts=True)
    risk = len(times)
    survival = 1.0
    values = []
    for time, count in zip(unique, counts):
        censorings = int(np.sum((times == time) & (codes == 0)))
        survival *= max(0.0, 1.0 - censorings / risk)
        values.append(survival)
        risk -= int(count)
    return unique, np.asarray(values, float)


def horizon_labels_weights(frame, horizon, km_source=None):
    """Return binary labels and IPCW with G(t-), using training G if supplied."""
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive")
    source = frame if km_source is None else km_source
    if len(source) == 0:
        raise ValueError("Cannot estimate censoring KM from an empty sample")
    km_t, km_g = censoring_km(source)
    times, codes = _times_codes(frame)
    labels = np.full(len(frame), np.nan)
    observation_times = np.full(len(frame), np.nan)
    target = (codes == 1) & (times <= horizon)
    competing = np.isin(codes, [2, 3]) & (times <= horizon)
    through_horizon = times >= horizon
    labels[target] = 1.0
    labels[competing | (through_horizon & ~target)] = 0.0
    observation_times[target | competing] = times[target | competing]
    observation_times[through_horizon & ~target & ~competing] = horizon
    weights = np.zeros(len(frame), float)
    known = np.isfinite(labels)
    positions = np.searchsorted(km_t, observation_times[known], side="left") - 1
    g = np.ones(int(known.sum()), float)
    valid = positions >= 0
    g[valid] = km_g[positions[valid]]
    weights[known] = 1.0 / np.maximum(g, CENSORING_FLOOR)
    return labels, weights


def _numeric_frame(frame, columns):
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _clinical_frame(frame):
    columns = ["age", "t_stage", "n_stage"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing clinical columns: {missing}")
    result = _numeric_frame(frame, columns)
    result.loc[~result.age.between(0, 120), "age"] = np.nan
    result.loc[~result.t_stage.between(0, 4), "t_stage"] = np.nan
    result.loc[~result.n_stage.between(0, 3), "n_stage"] = np.nan
    if "gtv_volume_cm3" in frame:
        volume = pd.to_numeric(frame["gtv_volume_cm3"], errors="coerce")
        volume = volume.where(np.isfinite(volume) & (volume > 0))
        result["log_gtv_volume"] = np.log1p(volume)
    elif "log_gtv_volume" in frame:
        volume_log = pd.to_numeric(frame["log_gtv_volume"], errors="coerce")
        result["log_gtv_volume"] = volume_log.where(np.isfinite(volume_log) & (volume_log > 0))
    else:
        raise ValueError("Missing gtv_volume_cm3/log_gtv_volume")
    return result


def is_eitn_feature(column):
    """Keep all q_hl/EITN variables out of the non-EITN image block."""
    lowered = str(column).lower()
    return "q_hl" in lowered or "eitn" in lowered


class FoldPreprocessor:
    """Fit imputers, scales, unsupervised image filtering and PCA on train only."""

    def __init__(self, feature_cols, model_spec, eitn_col=FIXED_EITN,
                 max_image_missing_fraction=0.30, variance_threshold=1e-8):
        self.feature_cols = list(dict.fromkeys(feature_cols))
        self.spec = model_spec
        self.eitn_col = eitn_col
        self.max_image_missing_fraction = float(max_image_missing_fraction)
        self.variance_threshold = float(variance_threshold)

    def _fixed(self, frame):
        result = _clinical_frame(frame)
        if self.spec.include_eitn:
            if self.eitn_col not in frame:
                raise ValueError(f"Missing prespecified EITN column: {self.eitn_col}")
            result[self.eitn_col] = pd.to_numeric(frame[self.eitn_col], errors="coerce")
        return result.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame):
        fixed = self._fixed(frame)
        self.fixed_medians_ = fixed.median().fillna(0.0)
        self.fixed_all_missing_ = fixed.columns[fixed.isna().all()].tolist()
        fixed_values = fixed.fillna(self.fixed_medians_).to_numpy(float)
        self.image_cols_ = []
        self.image_medians_ = pd.Series(dtype=float)
        self.image_scaler_ = None
        self.pca_ = None
        self.effective_components_ = 0
        parts = [fixed_values]
        if self.spec.image_mode != "none":
            # Defensively remove EITN even if the caller supplies the full pool.
            candidates = [c for c in self.feature_cols if not is_eitn_feature(c)]
            absent = [c for c in candidates if c not in frame]
            if absent:
                raise ValueError(f"Missing image columns in training data: {absent[:5]}")
            images = _numeric_frame(frame, candidates)
            eligible = images.columns[
                (images.isna().mean() <= self.max_image_missing_fraction)
                & images.notna().any()
            ].tolist()
            medians = images[eligible].median()
            imputed = images[eligible].fillna(medians)
            self.image_cols_ = imputed.columns[
                imputed.var(ddof=0) > self.variance_threshold
            ].tolist()
            self.image_medians_ = medians.reindex(self.image_cols_)
            if self.image_cols_:
                self.image_scaler_ = StandardScaler().fit(imputed[self.image_cols_])
                image_values = self.image_scaler_.transform(imputed[self.image_cols_])
                if self.spec.image_mode == "pca":
                    self.effective_components_ = min(
                        self.spec.n_components, len(self.image_cols_), len(frame) - 1
                    )
                    if self.effective_components_ > 0:
                        self.pca_ = PCA(
                            n_components=self.effective_components_,
                            svd_solver="randomized", random_state=RANDOM_SEED,
                        ).fit(image_values)
                        parts.append(self.pca_.transform(image_values))
                else:
                    parts.append(image_values)
        combined = np.column_stack(parts)
        # Equalize clinical, EITN and PC units before a shared L2 penalty.
        self.output_scaler_ = StandardScaler().fit(combined)
        self.n_predictors_ = int(combined.shape[1])
        self.summary = {
            "model_spec": asdict(self.spec),
            "fixed_columns": list(fixed.columns),
            "fixed_medians": self.fixed_medians_.to_dict(),
            "all_missing_fixed_columns": self.fixed_all_missing_,
            "raw_image_candidates": len(self.feature_cols) if self.spec.image_mode != "none" else 0,
            "retained_image_columns": len(self.image_cols_),
            "effective_pca_components": self.effective_components_,
            "n_predictors": self.n_predictors_,
            "pca_explained_variance_ratio": (
                self.pca_.explained_variance_ratio_.tolist() if self.pca_ is not None else []
            ),
        }
        return self

    def transform(self, frame):
        fixed = self._fixed(frame).fillna(self.fixed_medians_).to_numpy(float)
        parts = [fixed]
        if self.image_scaler_ is not None:
            images = _numeric_frame(frame, self.image_cols_).fillna(self.image_medians_)
            image_values = self.image_scaler_.transform(images)
            if self.spec.image_mode == "pca":
                if self.pca_ is not None:
                    parts.append(self.pca_.transform(image_values))
            else:
                parts.append(image_values)
        result = self.output_scaler_.transform(np.column_stack(parts))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite values survived fold-local preprocessing")
        return result


def _fit_ridge(design, labels, weights, C):
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive")
    known = np.isfinite(labels)
    if len(np.unique(labels[known])) != 2:
        raise ValueError("Training partition needs at least one target event and one control")
    normalized_weights = weights[known] * (known.sum() / weights[known].sum())
    model = LogisticRegression(
        C=float(C), solver="lbfgs", max_iter=3000, tol=1e-8, random_state=RANDOM_SEED
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design[known], labels[known].astype(int), sample_weight=normalized_weights)
    return model


@dataclass
class FittedModel:
    preprocessor: FoldPreprocessor
    estimator: LogisticRegression
    horizon: float
    C: float
    summary: dict

    def predict_probability(self, frame):
        return self.estimator.predict_proba(self.preprocessor.transform(frame))[:, 1]

    def predict_proba(self, frame):
        probability = self.predict_probability(frame)
        return np.column_stack([1 - probability, probability])


def fit_model(train_df, feature_cols, model_spec, horizon, C,
              eitn_col=FIXED_EITN):
    prep = FoldPreprocessor(feature_cols, model_spec, eitn_col=eitn_col).fit(train_df)
    labels, weights = horizon_labels_weights(train_df, horizon, train_df)
    estimator = _fit_ridge(prep.transform(train_df), labels, weights, C)
    summary = dict(prep.summary)
    summary.update({
        "C": float(C), "horizon": float(horizon),
        "n_training_rows": len(train_df),
        "n_training_evaluable": int(np.isfinite(labels).sum()),
        "training_target_events": int(np.nansum(labels)),
        "solver_iterations": int(estimator.n_iter_[0]),
    })
    return FittedModel(prep, estimator, float(horizon), float(C), summary)


def fit_predict_model(train_df, test_df, feature_cols, model_spec, horizon, C,
                      eitn_col=FIXED_EITN):
    fitted = fit_model(train_df, feature_cols, model_spec, horizon, C, eitn_col)
    return fitted, fitted.predict_probability(test_df)


def tune_model(train_df, feature_cols, model_spec, horizon,
               C_grid=DEFAULT_C_GRID, inner_splits=(), eitn_col=FIXED_EITN,
               tie_tolerance=1e-4):
    """Choose C by mean inner IPCW log-loss; validation weights use training G.

    Returns (best_C, tuning_rows). Each row records losses for all inner folds.
    Preprocessing is fitted once per inner fold and reused across C values.
    """
    grid = sorted(set(float(C) for C in C_grid))
    if not grid or any(not np.isfinite(C) or C <= 0 for C in grid):
        raise ValueError("C_grid must contain finite positive values")
    prepared = []
    for train_index, validation_index in inner_splits:
        training = train_df.iloc[np.asarray(train_index)]
        validation = train_df.iloc[np.asarray(validation_index)]
        prep = FoldPreprocessor(feature_cols, model_spec, eitn_col=eitn_col).fit(training)
        train_labels, train_weights = horizon_labels_weights(training, horizon, training)
        val_labels, val_weights = horizon_labels_weights(validation, horizon, training)
        known = np.isfinite(val_labels)
        if not known.any():
            raise ValueError("Inner validation fold has no evaluable outcomes")
        prepared.append((prep.transform(training), train_labels, train_weights,
                         prep.transform(validation)[known],
                         val_labels[known].astype(int), val_weights[known]))
    if len(prepared) < 2:
        raise ValueError("Provide at least two inner train/validation splits")
    rows = []
    for C in grid:
        losses = []
        for train_x, train_y, train_w, val_x, val_y, val_w in prepared:
            model = _fit_ridge(train_x, train_y, train_w, C)
            probability = model.predict_proba(val_x)[:, 1]
            losses.append(float(log_loss(
                val_y, np.clip(probability, 1e-12, 1 - 1e-12),
                sample_weight=val_w, labels=[0, 1],
            )))
        rows.append({"C": C, "mean_loss": float(np.mean(losses)), "fold_losses": losses})
    minimum = min(row["mean_loss"] for row in rows)
    best_C = min(row["C"] for row in rows if row["mean_loss"] <= minimum + tie_tolerance)
    return best_C, rows


def weighted_metrics(frame, prediction, horizon, km_source=None):
    """Evaluation defaults to evaluation-set G; supplied predictions stay fixed."""
    prediction = np.asarray(prediction, float)
    if prediction.shape != (len(frame),):
        raise ValueError("prediction must have one probability per evaluation row")
    if np.any(np.isfinite(prediction) & ((prediction < 0) | (prediction > 1))):
        raise ValueError("Predictions must be probabilities in [0, 1]")
    labels, weights = horizon_labels_weights(frame, horizon, km_source)
    known = np.isfinite(labels) & np.isfinite(prediction)
    result = {
        "n_total": len(frame),
        "n_evaluable": int(known.sum()), "target_events": int(np.nansum(labels[known])),
        "ipcw_auc": np.nan, "ipcw_brier": np.nan, "ipcw_brier_normalized": np.nan,
        "calibration_slope": np.nan, "calibration_intercept": np.nan,
        "ipcw_log_loss": np.nan,
        "event_rate": np.nan,
        "mean_predicted_risk": (
            float(np.mean(prediction[np.isfinite(prediction)]))
            if np.isfinite(prediction).any() else np.nan
        ),
        "probability_min": (
            float(np.min(prediction[np.isfinite(prediction)]))
            if np.isfinite(prediction).any() else np.nan
        ),
        "probability_max": (
            float(np.max(prediction[np.isfinite(prediction)]))
            if np.isfinite(prediction).any() else np.nan
        ),
    }
    if not known.any():
        return result
    y, w, p = labels[known].astype(int), weights[known], prediction[known]
    result["ipcw_brier"] = float(np.sum(w * (y - p) ** 2) / len(frame))
    result["ipcw_brier_normalized"] = float(np.average((y - p) ** 2, weights=w))
    # IPCW cumulative-incidence estimate on the same full-cohort denominator.
    result["event_rate"] = float(np.sum(w * y) / len(frame))
    result["ipcw_log_loss"] = float(log_loss(
        y, np.clip(p, 1e-12, 1 - 1e-12), sample_weight=w, labels=[0, 1]
    ))
    if len(np.unique(y)) < 2:
        return result
    result["ipcw_auc"] = float(roc_auc_score(y, p, sample_weight=w))
    logits = logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None]
    if np.std(logits) > 1e-12:
        calibration = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=3000, tol=1e-9
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", ConvergenceWarning)
            calibration.fit(logits, y, sample_weight=w * len(w) / w.sum())
        if not any(isinstance(record.message, ConvergenceWarning) for record in records):
            result["calibration_slope"] = float(calibration.coef_[0, 0])
            # Joint calibration intercept (free slope), matching existing output.
            result["calibration_intercept"] = float(calibration.intercept_[0])
    return result
