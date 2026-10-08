"""P4.2R fixed-budget qW repairs; no temporal selection or branch selection.

R1 removes training-constant and exact-duplicate logical input columns. R2
additionally transforms the preregistered count list. The caller must authorize
R2 only after the winter R1 fit fails. Neither repair uses labels for cleanup.
Deleting duplicate columns changes the effective fixed-L2 penalty; this module
does not claim prediction or optimization equivalence with the unrepaired fit.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.special import expit

from .p42_propensity import MODEL_PARAMS, THREAD_LIMIT, _column, _feature_names, _labels


R2_COUNT_FEATURES = (
    "source_count", "item2vec_seed_support", "item2vec_vocab_count",
    "user_history_events_12w", "user_unique_items_12w",
    "item_events_7d", "item_events_28d", "item_events_12w", "item_unique_customers_28d",
    "user_item_events_12w", "user_item_events_28d",
    "user_product_code_events_28d", "user_product_code_events_12w",
    "user_product_type_events_28d", "user_product_type_events_12w",
    "user_department_events_28d", "user_department_events_12w",
    "user_garment_events_28d", "user_garment_events_12w",
)


def _logical_matrix(frame, feature_spec, *, preprocessing=None, repair="R1"):
    """Impute, append availability, and validate, without any scaling/cleanup."""
    if repair not in ("R1", "R2"):
        raise ValueError("repair must be R1 or R2")
    numeric, binary, columns = _feature_names(feature_spec)
    fitting = preprocessing is None
    if fitting:
        preprocessing = {
            "format_version": 1, "repair": repair,
            "numeric": numeric, "binary": binary, "input_columns": columns,
            "numeric_availability": [name + "_available" for name in numeric],
            "training_rows": int(len(frame)), "median": {},
            "missing_training_rows": {}, "all_missing_numeric": [],
            "numeric_missing_rule": "nonfinite -> training median; all missing -> 0",
            "binary_missing_rule": "nonfinite -> 0; finite values must be 0 or 1",
            "pipeline": ["training-only median imputation and availability generation",
                         "training-only constant then exact-duplicate logical-column cleanup",
                         "log1p on retained preregistered counts" if repair == "R2" else "identity transform",
                         "training-only StandardScaler ddof=0 on retained numeric columns"],
            "scaling_rule": "retained numeric only; binary and availability remain unscaled",
            "r2_count_features": list(R2_COUNT_FEATURES) if repair == "R2" else [],
        }
    elif columns != preprocessing["input_columns"] or repair != preprocessing["repair"]:
        raise ValueError("Serialized preprocessing feature order or repair is inconsistent")
    p, b = len(numeric), len(binary)
    x = np.empty((len(frame), len(columns)), dtype=np.float64, order="C")
    for j, name in enumerate(numeric):
        raw = _column(frame, name)
        valid = np.isfinite(raw)
        # Validate before projection: even a dropped constant/duplicate count
        # must not hide a negative observed value in either train or prediction.
        if repair == "R2" and name in R2_COUNT_FEATURES and (raw[valid] < 0).any():
            raise ValueError(f"negative finite count before log1p: {name}")
        if fitting:
            preprocessing["median"][name] = float(np.median(raw[valid])) if valid.any() else 0.0
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
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite logical feature matrix")
    return x, preprocessing


def _cleanup_training_matrix(x, columns):
    """Label-free exact comparison with deterministic survivor priority.

    Range and sum only narrow possible duplicate comparisons; np.array_equal
    is always the acceptance criterion, never a correlation or sketch test.
    """
    if not len(x):
        raise ValueError("model-data insufficiency: empty training matrix")
    low, high, sums = x.min(axis=0), x.max(axis=0), x.sum(axis=0)
    constants = [{"feature": name, "value": float(low[j]),
                  "reason": "constant_on_training_matrix"}
                 for j, name in enumerate(columns) if low[j] == high[j]]
    active = [j for j in range(len(columns)) if low[j] != high[j]]
    ordered = sorted(active, key=lambda j: (columns[j].endswith("_available"), columns[j]))
    representatives = {}
    duplicates, survivors = [], []
    for j in ordered:
        signature = (float(low[j]), float(high[j]), float(sums[j]))
        bucket = representatives.setdefault(signature, [])
        survivor = next((i for i in bucket if np.array_equal(x[:, i], x[:, j])), None)
        if survivor is None:
            bucket.append(j)
            survivors.append(j)
        else:
            duplicates.append({"feature": columns[j], "survivor": columns[survivor],
                               "reason": "exact_duplicate_on_imputed_training_logical_input"})
    kept_indices = sorted(survivors)
    if not kept_indices:
        raise ValueError("model-data insufficiency: no varying features after deterministic cleanup")
    return {
        "dims_before": len(columns), "constant_dropped": constants,
        "constant_dropped_count": len(constants), "exact_duplicate_dropped": duplicates,
        "exact_duplicate_dropped_count": len(duplicates), "dims_after": len(kept_indices),
        "kept_indices": kept_indices, "kept_columns": [columns[j] for j in kept_indices],
        "survivor_priority": "non *_available first; then lexicographically smaller name",
        "surviving_column_order": "original feature order",
        "comparison_layer": "training-median-imputed logical input before log1p and scaling",
        "duplicate_comparison": "np.array_equal after range/sum candidate narrowing",
        "high_correlation_columns_dropped": 0, "labels_used": False,
        "regularization_boundary": "duplicate deletion changes effective fixed-L2 penalty; not equivalent to old model",
    }


def repaired_design_matrix(frame, feature_spec=None, *, repair="R1", preprocessing=None):
    """Public deterministic preprocessing replay for independent verification.

    Returns (matrix, preprocessing, cleanup). On replay cleanup is the saved
    training decision and is never recomputed on evaluation rows.
    """
    from sklearn.preprocessing import StandardScaler

    fitting = preprocessing is None
    if not fitting:
        repair = preprocessing["repair"]
        feature_spec = {"numeric": preprocessing["numeric"], "binary": preprocessing["binary"]}
    logical, prep = _logical_matrix(frame, feature_spec, preprocessing=preprocessing, repair=repair)
    if fitting:
        cleanup = _cleanup_training_matrix(logical, prep["input_columns"])
        prep["cleanup"] = cleanup
        prep["columns"] = cleanup["kept_columns"]
        prep["kept_indices"] = cleanup["kept_indices"]
        prep["scaled_numeric"] = [name for name in prep["columns"] if name in prep["numeric"]]
        prep["log1p_applied_features"] = [name for name in prep["columns"]
                                           if repair == "R2" and name in R2_COUNT_FEATURES]
        prep["log1p_count_features_dropped_by_cleanup"] = [name for name in R2_COUNT_FEATURES
                                                          if name in prep["numeric"] and name not in prep["columns"]]
    else:
        cleanup = prep["cleanup"]
        expected = [prep["input_columns"][j] for j in prep["kept_indices"]]
        if expected != prep["columns"] or expected != cleanup["kept_columns"]:
            raise ValueError("Serialized cleanup projection is inconsistent")
    x = np.ascontiguousarray(logical[:, prep["kept_indices"]])
    del logical
    for name in prep["log1p_applied_features"]:
        j = prep["columns"].index(name)
        if (x[:, j] < 0).any():
            raise ValueError(f"negative imputed count before log1p: {name}")
        np.log1p(x[:, j], out=x[:, j])
    numeric_indices = [prep["columns"].index(name) for name in prep["scaled_numeric"]]
    if fitting:
        if numeric_indices:
            scaler = StandardScaler()
            # Fancy-index input is a copy: write transformed values back explicitly.
            x[:, numeric_indices] = scaler.fit_transform(x[:, numeric_indices])
            prep["scaler"] = {"columns": prep["scaled_numeric"], "mean": scaler.mean_.tolist(),
                              "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
                              "n_samples_seen": int(scaler.n_samples_seen_)}
        else:
            prep["scaler"] = {"columns": [], "mean": [], "scale": [], "var": [],
                              "n_samples_seen": int(len(frame))}
    else:
        if prep["scaler"]["columns"] != prep["scaled_numeric"]:
            raise ValueError("Serialized scaler column order is inconsistent")
        x[:, numeric_indices] = ((x[:, numeric_indices] - np.asarray(prep["scaler"]["mean"])) /
                                 np.asarray(prep["scaler"]["scale"]))
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite repaired design matrix")
    return x, prep, cleanup


def _numerical_diagnostics(x, y, model, preprocessing):
    """Fixed-parameter train-only gradient and local curvature, no optimizer."""
    started = time.perf_counter()
    n, p = x.shape
    coef = np.asarray(model["coefficient"], dtype=np.float64)
    logits = x @ coef + model["intercept"]
    q = expit(logits)
    strength = 1.0 / (model["params"]["C"] * n)
    residual = q - y
    gradient = np.concatenate([x.T @ residual / n + strength * coef, [residual.mean()]])
    weights = q * (1 - q)
    hessian = np.zeros((p + 1, p + 1), dtype=np.float64)
    # Bounded temporary allocation: never materialize a second full n-by-p matrix.
    for start in range(0, n, 32768):
        block, w = x[start:start + 32768], weights[start:start + 32768]
        hessian[:p, :p] += block.T @ (block * w[:, None]) / n
    hessian[np.arange(p), np.arange(p)] += strength
    hessian[:p, p] = hessian[p, :p] = x.T @ weights / n
    hessian[p, p] = weights.mean()
    eigenvalues = np.linalg.eigvalsh(hessian)
    columns = preprocessing["columns"]
    numeric = preprocessing["scaled_numeric"]
    maxima = np.maximum(np.abs(x.min(axis=0)), np.abs(x.max(axis=0)))
    numeric_maxima = [(name, float(maxima[columns.index(name)])) for name in numeric]
    numeric_maxima.sort(key=lambda row: (-row[1], row[0]))
    names = columns + ["intercept"]
    order = np.argsort(-np.abs(gradient), kind="stable")[:15]
    return {
        "kind": "read_only_fitted_parameter_diagnostic", "training_rows": n, "features": p,
        "optimizer_calls": 0, "outer_rows_read": 0,
        "objective_definition": "mean logistic loss + ||beta||^2/(2*C*n); intercept unpenalized",
        "fixed_objective": float(np.mean(np.logaddexp(0, logits) - y * logits) + .5 * strength * np.dot(coef, coef)),
        "l2_reg_strength": strength, "gradient_infinity_norm": float(np.abs(gradient).max()),
        "gradient_top15": [{"feature": names[j], "gradient": float(gradient[j]),
                            "absolute": float(abs(gradient[j]))} for j in order],
        "frozen_gtol": model["params"]["tol"], "lbfgs_ftol": float(64 * np.finfo(float).eps),
        "local_regularized_hessian_min_eigenvalue": float(eigenvalues[0]),
        "local_regularized_hessian_max_eigenvalue": float(eigenvalues[-1]),
        "local_regularized_hessian_condition_number": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None,
        "max_standardized_absolute_value": numeric_maxima[0][1] if numeric_maxima else None,
        "max_standardized_absolute_feature": numeric_maxima[0][0] if numeric_maxima else None,
        "standardized_numeric_abs_max_top15": [{"feature": name, "abs_max": value} for name, value in numeric_maxima[:15]],
        "mean_training_prediction": float(q.mean()), "observed_training_positive_rate": float(y.mean()),
        "interpretation_limit": "Local numerical geometry is descriptive; cleanup also changes fixed-L2 effective penalty and is not a unique causal root proof.",
        "seconds": time.perf_counter() - started,
    }


def fit_repaired_propensity(frame, feature_spec, *, repair="R1", target_column="target"):
    """Fit every supplied training row once under the unchanged P4.2 budget."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    numeric, binary, _ = _feature_names(feature_spec)
    if target_column in numeric + binary:
        raise ValueError("The target label cannot be an input feature")
    y = _labels(_column(frame, target_column))
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("model-data insufficiency: historical training pool needs both classes")
    with threadpool_limits(limits=THREAD_LIMIT):
        x, prep, cleanup = repaired_design_matrix(frame, feature_spec, repair=repair)
        preprocessing_seconds = time.perf_counter() - started
        estimator = LogisticRegression(**MODEL_PARAMS)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(x, y)
        optimizer_fit_seconds = time.perf_counter() - fit_started
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise ValueError("Nonfinite fitted LogisticRegression parameters")
        model = {"format_version": 1, "type": "L2 LogisticRegression", "params": dict(MODEL_PARAMS),
                 "classes": estimator.classes_.astype(int).tolist(), "coefficient": estimator.coef_[0].tolist(),
                 "intercept": float(estimator.intercept_[0]), "n_iter": estimator.n_iter_.astype(int).tolist()}
        if model["classes"] != [0, 1]:
            raise ValueError("Unexpected fitted class order")
        fit_seconds = time.perf_counter() - started
        diagnostics = _numerical_diagnostics(x, y, model, prep)
    return {
        "model": model, "preprocessing": prep, "cleanup": cleanup, "diagnostics": diagnostics,
        "audit": {
            "repair": repair, "training_rows": int(len(y)), "fitted_rows": int(len(y)),
            "positives": int(y.sum()), "base_rate": float(y.mean()),
            "input_features": len(numeric) + len(binary), "model_features": x.shape[1],
            "feature_order": prep["columns"], "negative_sampling": False, "oversampling": False,
            "class_weight": None, "sample_weight": None, "thread_limit": THREAD_LIMIT,
            "preprocessing_seconds": preprocessing_seconds, "optimizer_fit_seconds": optimizer_fit_seconds,
            "fit_seconds": fit_seconds, "design_matrix_bytes": int(x.nbytes),
            "status": "non_converged" if convergence_warnings else "converged",
            "converged": not convergence_warnings, "convergence_warnings": convergence_warnings,
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "r2_finite_observed_count_validation": "passed" if repair == "R2" else "not_run",
        },
    }


def predict_repaired_propensity(fitted, frame):
    """Replay the training-only projection, transforms, scaler and coefficients."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=THREAD_LIMIT):
        x, _, _ = repaired_design_matrix(frame, preprocessing=fitted["preprocessing"])
        coef = np.asarray(fitted["model"]["coefficient"], dtype=np.float64)
        if len(coef) != x.shape[1] or fitted["model"]["classes"] != [0, 1]:
            raise ValueError("Serialized model and preprocessing dimensions/classes differ")
        score = x @ coef + fitted["model"]["intercept"]
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite propensity logit")
        return expit(score)
