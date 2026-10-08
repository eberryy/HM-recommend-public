"""P4.2 fixed, full-pool propensity fits and evaluation-only calibration.

The prediction API does not accept labels and never updates preprocessing. The
two model families call this same utility separately; no estimator is shared.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPSILON = 1e-6
MODEL_PARAMS = {
    "penalty": "l2", "solver": "lbfgs", "C": 1.0,
    "max_iter": 1000, "tol": 1e-7, "random_state": 20260909,
    "class_weight": None, "fit_intercept": True,
}
THREAD_LIMIT = 4


def _feature_names(feature_spec: dict) -> tuple[list[str], list[str], list[str]]:
    numeric = list(feature_spec["numeric"])
    binary = list(feature_spec["binary"])
    if not numeric or any(not isinstance(x, str) for x in numeric + binary):
        raise ValueError("At least one named numeric feature is required")
    columns = numeric + binary + [x + "_available" for x in numeric]
    if len(columns) != len(set(columns)):
        raise ValueError("Feature and generated availability names must be unique")
    return numeric, binary, columns


def _column(frame, name: str) -> np.ndarray:
    value = frame[name]
    if hasattr(value, "to_numpy"):
        return value.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(value, dtype=np.float64)


def _labels(value) -> np.ndarray:
    y = np.asarray(value, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a finite one-dimensional binary vector")
    return y.astype(np.int8)


def _design_matrix(frame, preprocessing: dict | None, feature_spec: dict | None):
    """Allocate one design matrix; only the fit path may compute medians/scales."""
    from sklearn.preprocessing import StandardScaler

    fitting = preprocessing is None
    if fitting:
        numeric, binary, columns = _feature_names(feature_spec)
        preprocessing = {
            "format_version": 1, "numeric": numeric, "binary": binary,
            "columns": columns, "numeric_availability": [x + "_available" for x in numeric],
            "training_rows": int(len(frame)), "median": {}, "missing_training_rows": {},
            "all_missing_numeric": [], "numeric_missing_rule": "nonfinite -> training median; all missing -> 0",
            "binary_missing_rule": "nonfinite -> 0; finite values must be 0 or 1",
            "scaling_rule": "training-only StandardScaler ddof=0 for numeric; binary and availability unchanged",
        }
    else:
        numeric, binary, columns = _feature_names(preprocessing)
        if columns != preprocessing["columns"]:
            raise ValueError("Serialized preprocessing feature order is inconsistent")
    n, p, b = len(frame), len(numeric), len(binary)
    x = np.empty((n, len(columns)), dtype=np.float64, order="C")
    for j, name in enumerate(numeric):
        raw = _column(frame, name)
        valid = np.isfinite(raw)
        if fitting:
            median = float(np.median(raw[valid])) if valid.any() else 0.0
            preprocessing["median"][name] = median
            preprocessing["missing_training_rows"][name] = int((~valid).sum())
            if not valid.any():
                preprocessing["all_missing_numeric"].append(name)
        x[:, j] = raw
        x[~valid, j] = preprocessing["median"][name]
        x[:, p + b + j] = valid
    for j, name in enumerate(binary):
        raw = _column(frame, name)
        valid = np.isfinite(raw)
        if not np.isin(raw[valid], [0, 1]).all():
            raise ValueError(f"Binary feature {name!r} has a value outside 0/1")
        x[:, p + j] = raw
        x[~valid, p + j] = 0.0
    if fitting:
        scaler = StandardScaler(copy=False)
        scaler.fit(x[:, :p])
        scaler.transform(x[:, :p], copy=False)
        preprocessing["scaler"] = {
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
        }
    else:
        x[:, :p] -= np.asarray(preprocessing["scaler"]["mean"], dtype=np.float64)
        x[:, :p] /= np.asarray(preprocessing["scaler"]["scale"], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite value in the transformed design matrix")
    return x, preprocessing


def fit_propensity(frame, feature_spec: dict, *, target_column: str = "target") -> dict:
    """Fit one preregistered model on every supplied historical candidate row.

    The caller owns temporal pool selection and preregistration. No model sweep,
    weighting, sampling, balancing, retries, or validation data are accepted.
    Returned coefficients and preprocessing are JSON-serializable.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    y = _labels(_column(frame, target_column))
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("model-data insufficiency: historical training pool needs both classes")
    numeric, binary, columns = _feature_names(feature_spec)
    if target_column in numeric + binary:
        raise ValueError("The target label cannot be an input feature")
    with threadpool_limits(limits=THREAD_LIMIT):
        x, preprocessing = _design_matrix(frame, None, feature_spec)
        estimator = LogisticRegression(**MODEL_PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(x, y)
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
        raise ValueError("Nonfinite fitted LogisticRegression parameters")
    model = {
        "format_version": 1, "type": "L2 LogisticRegression",
        "params": dict(MODEL_PARAMS), "classes": estimator.classes_.astype(int).tolist(),
        "coefficient": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
        "n_iter": estimator.n_iter_.astype(int).tolist(),
    }
    if model["classes"] != [0, 1]:
        raise ValueError("Unexpected fitted class order")
    return {
        "model": model, "preprocessing": preprocessing,
        "audit": {
            "training_rows": int(len(y)), "fitted_rows": int(len(y)),
            "positives": int(y.sum()), "base_rate": float(y.mean()),
            "input_features": len(numeric) + len(binary), "model_features": len(columns),
            "feature_order": columns, "negative_sampling": False, "oversampling": False,
            "class_weight": None, "sample_weight": None, "thread_limit": THREAD_LIMIT,
            "fit_seconds": time.perf_counter() - started,
            "design_matrix_bytes": int(x.nbytes),
            "status": "non_converged" if convergence_warnings else "converged",
            "convergence_warnings": convergence_warnings,
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        },
    }


def predict_propensity(fitted: dict, frame) -> np.ndarray:
    """Replay saved preprocessing/coefficients, returning unclipped propensity."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=THREAD_LIMIT):
        x, _ = _design_matrix(frame, fitted["preprocessing"], None)
        coef = np.asarray(fitted["model"]["coefficient"], dtype=np.float64)
        if len(coef) != x.shape[1] or fitted["model"]["classes"] != [0, 1]:
            raise ValueError("Serialized model and preprocessing dimensions/classes differ")
        score = x @ coef + fitted["model"]["intercept"]
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite propensity logit")
        return expit(score)


def clip_propensity(probability) -> np.ndarray:
    q = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Propensities must be finite probabilities in [0, 1]")
    return np.clip(q, EPSILON, 1 - EPSILON)


def _calibration_line(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """Unregularized outer diagnostic only; never used to modify predictions."""
    base = {"intercept": None, "slope": None, "success": False,
            "diagnostic_only": True, "initial_parameters": [0.0, 1.0], "max_iter": 200}
    if len(np.unique(y)) < 2:
        return {**base, "status": "undefined_single_class"}
    if float(np.ptp(logits)) <= 1e-12:
        return {**base, "status": "undefined_constant_prediction"}
    positive, negative = logits[y == 1], logits[y == 0]
    if float(positive.min()) >= float(negative.max()) or float(positive.max()) <= float(negative.min()):
        # At a complete/quasi-separating boundary the unregularized optimum is
        # approached at infinite coefficients: numerical convergence is not a
        # finite maximum-likelihood estimate of slope/intercept.
        return {**base, "status": "undefined_separation"}

    def objective(parameters):
        z = parameters[0] + parameters[1] * logits
        residual = expit(z) - y
        value = np.mean(np.logaddexp(0.0, z) - y * z)
        gradient = np.array([residual.mean(), np.mean(residual * logits)])
        return value, gradient

    # No penalty and no feedback path to the separately fitted qC/qW models.
    result = minimize(objective, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    valid = bool(result.success and np.isfinite(result.x).all())
    return {**base, "intercept": float(result.x[0]) if valid else None,
            "slope": float(result.x[1]) if valid else None, "success": valid,
            "status": "converged" if valid else "diagnostic_optimization_failed",
            "iterations": int(result.nit), "message": str(result.message),
            "objective": float(result.fun) if np.isfinite(result.fun) else None}


def propensity_calibration(label, probability) -> dict:
    """Full-row audit; ten equal-frequency bins use q then original row order.

    Equal probabilities may straddle adjacent bins. The membership depends on
    predictions and immutable row order only, never on labels. All metrics use
    the exact clipped probabilities used by admission; raw mean is also shown.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from threadpoolctl import threadpool_limits

    y = _labels(label)
    raw = np.asarray(probability, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != len(y):
        raise ValueError("Calibration labels and predictions must have equal one-dimensional shape")
    q = clip_propensity(raw)
    n, positives = len(y), int(y.sum())
    bins = []
    order = np.argsort(q, kind="stable")
    for number, idx in enumerate(np.array_split(order, 10), 1):
        bins.append({
            "bin": number, "rows": int(len(idx)), "positives": int(y[idx].sum()),
            "mean_probability": float(q[idx].mean()) if len(idx) else None,
            "observed_positive_rate": float(y[idx].mean()) if len(idx) else None,
            "min_probability": float(q[idx].min()) if len(idx) else None,
            "max_probability": float(q[idx].max()) if len(idx) else None,
        })
    observed, mean = (float(y.mean()), float(q.mean())) if n else (None, None)
    both_classes = 0 < positives < n
    ratio = mean / observed if observed else None
    with threadpool_limits(limits=THREAD_LIMIT):
        line = _calibration_line(y, np.log(q) - np.log1p(-q)) if n else {
            "intercept": None, "slope": None, "success": False,
            "diagnostic_only": True, "status": "undefined_empty_rows",
        }
        roc = float(roc_auc_score(y, q)) if both_classes else None
        pr = float(average_precision_score(y, q)) if both_classes else None
    warnings_out = []
    if not n:
        warnings_out.append("empty_evaluation_rows")
    if n and not both_classes:
        warnings_out.append("single_class_evaluation")
    if n and float(np.ptp(q)) <= 1e-12:
        warnings_out.append("constant_prediction")
    if ratio is not None and (ratio < .25 or ratio > 4):
        warnings_out.append("mean_probability_to_observed_rate_outside_0.25_4")
    if roc is not None and roc <= .5:
        warnings_out.append("roc_auc_not_above_chance")
    if line["slope"] is not None and line["slope"] <= 0:
        warnings_out.append("nonpositive_calibration_slope")
    if line["status"] == "diagnostic_optimization_failed":
        warnings_out.append("calibration_line_diagnostic_failed")
    if line["status"] == "undefined_separation":
        warnings_out.append("calibration_line_no_finite_mle_due_to_separation")
    return {
        "rows": n, "positives": positives, "observed_positive_rate": observed,
        "mean_predicted_probability": mean,
        "unclipped_mean_predicted_probability": float(raw.mean()) if n else None,
        "predicted_to_observed_rate_ratio": ratio,
        "brier_score": float(np.mean((q-y)**2)) if n else None,
        "roc_auc": roc, "pr_auc": pr, "pr_auc_definition": "non-interpolated average precision",
        "calibration_bins": bins,
        "bin_method": "stable sort by clipped probability then original row order; numpy.array_split into 10",
        "ece": float(sum(b["rows"] * abs(b["mean_probability"] - b["observed_positive_rate"])
                         for b in bins if b["rows"]) / n) if n else None,
        "calibration_intercept": line["intercept"], "calibration_slope": line["slope"],
        "calibration_line": line, "epsilon": EPSILON,
        "clipped_low_rows": int((raw < EPSILON).sum()),
        "clipped_high_rows": int((raw > 1-EPSILON).sum()),
        "warnings": warnings_out, "prediction_semantics": "next-week truth propensity conditional on the frozen candidate pool",
    }
