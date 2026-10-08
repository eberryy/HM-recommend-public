"""P4.2R3 single-variable qW solver-budget repair with actual exit evidence.

Only max_iter changes from 1000 to 1200. This module does not mutate any prior
stage constants. A scoped read-only observer records SciPy's actual result at
sklearn's existing post-solver check; the original checker is called unchanged.
The calling workflow must run these fits serially, not concurrently with other
LogisticRegression fits in this process while the observer is installed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.metadata
import threading
import time
import warnings

import numpy as np
from scipy.special import expit

from .p42_propensity import MODEL_PARAMS, THREAD_LIMIT, _column, _feature_names, _labels
from .p42r_propensity import repaired_design_matrix, _numerical_diagnostics


QW_MODEL_PARAMS = {**MODEL_PARAMS, "max_iter": 1200}
_OBSERVER_LOCK = threading.Lock()


@contextmanager
def capture_solver_result():
    """Observe actual OptimizeResult only, then call the original checker.

    No callback is injected into the optimizer, and no initial values, solver
    options, objective values, derivatives or returned status are modified.
    Restoration happens even when fitting or the original checker raises.
    """
    import sklearn.linear_model._logistic as logistic

    records = []
    with _OBSERVER_LOCK:
        original = logistic._check_optimize_result

        def observer(solver, result, *args, **kwargs):
            max_iter = args[0] if args else kwargs.get("max_iter")
            # Every value comes from the actual result object before sklearn's
            # warning interpretation; neither success nor message is inferred.
            records.append({
                "capture_kind": "actual_scipy_OptimizeResult_before_original_sklearn_check",
                "solver": str(solver), "success": bool(result.success), "status": int(result.status),
                "message": str(result.message), "nit": int(result.nit),
                "nfev": int(result.nfev), "njev": int(result.njev), "fun": float(result.fun),
                "jac_infinity_norm": float(np.abs(np.asarray(result.jac)).max()),
                "parameter_vector": np.asarray(result.x, dtype=np.float64).tolist(),
                "max_iter_argument": int(max_iter) if max_iter is not None else None,
                "observer_changes_solver_options": False,
                "observer_changes_result": False,
                "original_check_called_unchanged": True,
                "source_hook": "sklearn.linear_model._logistic._check_optimize_result",
                "scikit_learn_version": importlib.metadata.version("scikit-learn"),
                "scipy_version": importlib.metadata.version("scipy"),
            })
            return original(solver, result, *args, **kwargs)

        logistic._check_optimize_result = observer
        try:
            yield records
        finally:
            logistic._check_optimize_result = original


def fit_boundary_propensity(frame, feature_spec, *, target_column="target"):
    """One full-row R2 fit, fixed 1200 budget, actual solver-success gate."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    numeric, binary, _ = _feature_names(feature_spec)
    if target_column in numeric+binary:
        raise ValueError("The target label cannot be an input feature")
    y = _labels(_column(frame, target_column))
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("model-data insufficiency: historical training pool needs both classes")
    with threadpool_limits(limits=THREAD_LIMIT):
        x, prep, cleanup = repaired_design_matrix(frame, feature_spec, repair="R2")
        preprocessing_seconds = time.perf_counter()-started
        estimator = LogisticRegression(**QW_MODEL_PARAMS)
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with capture_solver_result() as actual_results:
                estimator.fit(x, y)
        optimizer_fit_seconds = time.perf_counter()-fit_start
        if len(actual_results) != 1:
            raise ValueError(f"Expected exactly one actual binary LBFGS result, observed {len(actual_results)}")
        solver_result = actual_results[0]
        if solver_result["solver"] != "lbfgs" or solver_result["max_iter_argument"] != 1200:
            raise ValueError("Actual solver or iteration budget differs from P4.2R3 contract")
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise ValueError("Nonfinite fitted LogisticRegression parameters")
        model = {"format_version": 1, "type": "L2 LogisticRegression", "params": dict(QW_MODEL_PARAMS),
            "classes": estimator.classes_.astype(int).tolist(), "coefficient": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]), "n_iter": estimator.n_iter_.astype(int).tolist()}
        if model["classes"] != [0, 1]:
            raise ValueError("Unexpected fitted class order")
        np.testing.assert_array_equal(np.r_[estimator.coef_[0], estimator.intercept_[0]], solver_result["parameter_vector"])
        if model["n_iter"] != [min(solver_result["nit"], 1200)]:
            raise ValueError("Recorded actual iteration count does not match sklearn's reported iteration count")
        solver_result["returned_parameters_match_estimator"] = True
        converged = bool(solver_result["success"] and solver_result["status"] == 0 and
                         solver_result["nit"] < 1200 and not convergence_warnings)
        fit_seconds = time.perf_counter()-started
        diagnostics = _numerical_diagnostics(x, y, model, prep)
    return {"model": model, "preprocessing": prep, "cleanup": cleanup, "diagnostics": diagnostics,
        "audit": {"repair": "R2", "solver_budget_stage": "P4.2R3", "training_rows": int(len(y)),
            "fitted_rows": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
            "input_features": len(numeric)+len(binary), "model_features": x.shape[1], "feature_order": prep["columns"],
            "negative_sampling": False, "oversampling": False, "class_weight": None, "sample_weight": None,
            "thread_limit": THREAD_LIMIT, "preprocessing_seconds": preprocessing_seconds,
            "optimizer_fit_seconds": optimizer_fit_seconds, "fit_seconds": fit_seconds,
            "design_matrix_bytes": int(x.nbytes), "status": "converged" if converged else "non_converged",
            "converged": converged, "convergence_warnings": convergence_warnings,
            "convergence_warning_count": len(convergence_warnings),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "solver_result": solver_result,
            "convergence_gate_definition": "actual solver success and status0 and actual nit<1200 and zero ConvergenceWarning; independent gradient never overrides status",
            "r2_finite_observed_count_validation": "passed"}}


def parameter_delta(frame, old_fitted, new_fitted, *, target_column="target"):
    """Full same-training-row diagnostics, no fit and no model selection.

    Training log loss is mean unregularized binary logistic loss evaluated
    stably from logits, not clipped prediction loss. Predictions are unclipped.
    Only the outer driver's metadata lineage is ignored in preprocessing parity.
    """
    from threadpoolctl import threadpool_limits

    old_prep, new_prep = [copy.deepcopy(fitted["preprocessing"]) for fitted in (old_fitted, new_fitted)]
    for prep in (old_prep, new_prep):
        prep.pop("lineage", None)
    if old_prep != new_prep:
        raise ValueError("P4.2R2/P4.2R3 preprocessing drift: only solver max_iter may change")
    old_model, new_model = old_fitted["model"], new_fitted["model"]
    if old_model["params"] != MODEL_PARAMS or new_model["params"] != QW_MODEL_PARAMS:
        raise ValueError("Expected exact old1000/new1200 model parameters")
    if old_model["classes"] != [0, 1] or new_model["classes"] != [0, 1]:
        raise ValueError("Unexpected fitted class order")
    old_beta, new_beta = [np.asarray(m["coefficient"], dtype=np.float64) for m in (old_model, new_model)]
    if old_beta.shape != new_beta.shape or old_beta.size != len(old_prep["columns"]):
        raise ValueError("Coefficient feature order/dimension drift")
    y = _labels(_column(frame, target_column))
    if len(y) != old_prep["training_rows"]:
        raise ValueError("Parameter delta must use the complete original training rows")
    with threadpool_limits(limits=THREAD_LIMIT):
        x, _, _ = repaired_design_matrix(frame, preprocessing=old_prep)
        old_z = x@old_beta+old_model["intercept"]
        new_z = x@new_beta+new_model["intercept"]
        old_q, new_q = expit(old_z), expit(new_z)
        old_loss = float(np.mean(np.logaddexp(0, old_z)-y*old_z))
        new_loss = float(np.mean(np.logaddexp(0, new_z)-y*new_z))
    absolute = np.abs(new_q-old_q)
    coefficient_delta = new_beta-old_beta
    old_nit, new_nit = int(old_model["n_iter"][0]), int(new_model["n_iter"][0])
    old_gradient = old_fitted.get("diagnostics", old_fitted.get("audit", {}).get("diagnostics", {})).get("gradient_infinity_norm")
    new_gradient = new_fitted.get("diagnostics", new_fitted.get("audit", {}).get("diagnostics", {})).get("gradient_infinity_norm")
    return {"kind": "read_only_same_training_population_parameter_comparison", "rows": len(frame),
        "preprocessing_exact_except_lineage": True, "coefficient_order_exact": True,
        "old_iterations": old_nit, "new_iterations": new_nit,
        "additional_iterations_beyond_old_cap": max(0, new_nit-1000), "iteration_delta_vs_old": new_nit-old_nit,
        "coefficient_delta_l2": float(np.linalg.norm(coefficient_delta)),
        "coefficient_delta_max_abs": float(np.abs(coefficient_delta).max()),
        "intercept_delta": float(new_model["intercept"]-old_model["intercept"]),
        "old_training_log_loss": old_loss, "new_training_log_loss": new_loss, "training_log_loss_delta": new_loss-old_loss,
        "log_loss_definition": "mean unregularized binary logistic loss from raw logits; no probability clipping",
        "old_gradient_infinity_norm": old_gradient, "new_gradient_infinity_norm": new_gradient,
        "gradient_infinity_norm_delta": new_gradient-old_gradient if old_gradient is not None and new_gradient is not None else None,
        "training_prediction_abs_delta": {"mean": float(absolute.mean()), "p95": float(np.quantile(absolute, .95)),
            "max": float(absolute.max()), "quantile_method": "numpy linear"},
        "exact_coefficient_intercept_parity": bool(np.array_equal(new_beta, old_beta) and new_model["intercept"] == old_model["intercept"]),
        "exact_training_prediction_parity": bool(np.array_equal(new_q, old_q)),
        "diagnostic_only": True, "new_optimizer_calls": 0}
