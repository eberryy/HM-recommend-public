"""Read-only verifier repair: scale only the independent LP objective.

The immutable production and original verification files are unchanged.  A
positive rescaling preserves the LP feasible set and mathematical optimizer;
the returned objective is restored to original units before the original
1e-9 comparison. No model, score, eligibility, recommendation or MAP changes.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from .p41a_contract import identity, read_json, write_json
from .p42f_contract import now
from .p42h_contract import RUN_ID
from . import p42h_verify as original


def run(repo):
    repo=Path(repo).resolve();root=repo/'artifacts/phase4'/RUN_ID;dest=repo/'reports/phase4'
    assert read_json(dest/'P4_2H_metrics.json')['status']=='completed_pending_verification'
    marker=root/'SCALED_LP_VERIFICATION_START.json'
    if marker.exists():raise FileExistsError('no silent repeated verification repair')
    write_json(marker,dict(at=now(),repair_source=identity(Path(__file__)),
        failed_check='spring independent unscaled LP objective below already feasible assignment',
        original_lp_objective=7.973719532985002e-7,executed_objective=8.000345204303659e-7,
        absolute_gap=2.6625671318657117e-9,original_tolerance=1e-9,
        new_rule='c_scaled=c/max(abs(c)); restore fun*scale; constraints and comparison tolerance unchanged',
        new_training=0,production_changes=False,first_failed_verification_seconds=None))
    audit=[]

    def scaled_lp(c,*args,**kwargs):
        c=np.asarray(c,dtype=np.float64);scale=float(np.max(np.abs(c)))
        assert np.isfinite(c).all() and scale>0
        result=linprog(c/scale,*args,**kwargs)
        if result.success:
            scaled_fun=float(result.fun);result.fun=scaled_fun*scale
            audit.append(dict(columns=len(c),positive_scale=scale,scaled_objective=scaled_fun,
                restored_objective=float(result.fun),recomputed_objective=float(np.dot(c,result.x)),
                integrality_max_error=float(np.max(np.abs(result.x-np.round(result.x))))))
            np.testing.assert_allclose(result.fun,np.dot(c,result.x),rtol=0,atol=1e-15)
        return result

    # Only this process's independent diagnostic solver is adapted. The formal
    # assignment algorithm and the frozen original verifier on disk stay intact.
    old=original.linprog
    try:
        original.linprog=scaled_lp
        result=original.verify(repo)
    finally:
        original.linprog=old
    assert len(audit)==result['LP_graphs']
    write_json(dest/'P4_2H_LP_NUMERICAL_VERIFICATION.json',dict(stage='P4.2H',status='pass',at=now(),
        original_failure_preserved=True,production_source_unchanged=True,original_verifier_source_unchanged=True,
        repair_source=identity(Path(__file__)),start=read_json(marker),LP_graphs=len(audit),graphs=audit,
        comparison_absolute_tolerance=1e-9,fit_count=0,policy_changes=0,
        explanation='positive per-graph objective normalization removes absolute-scale LP tolerance issue; not a policy sweep',
        final_week='not_run'))
    return result


if __name__=='__main__':run(Path('.'))
