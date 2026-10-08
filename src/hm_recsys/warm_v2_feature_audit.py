"""Post-experiment read-only checks that real feature bundles are not inert."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, now, read, write
from .warm_v2_engine import Engine, load_parquet
from .warm_v2_features import attach_features, columns_for


def audit():
    c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
    engine=Engine(c)
    cutoff='2019-11-27'
    path,meta=engine.cached_data(cutoff,'sample')
    base=load_parquet(path)
    families=['item_demand','user_rhythm','repeat_affinity','source_agreement','category_competition','price_context','fine_hierarchy','candidate_context']
    distributions={}
    for family in families:
        frame=attach_features(base,cutoff,[family],engine)
        stats={}
        for name in columns_for([family]):
            v=frame[name]
            finite=v.dropna()
            stats[name]={'nonmissing_rows':int(v.notna().sum()),'nonmissing_fraction':float(v.notna().mean()),
                'distinct_nonmissing_values':int(v.nunique()),'nonzero_rows':int((v.fillna(0)!=0).sum()),
                'min':float(finite.min()) if len(finite) else None,'max':float(finite.max()) if len(finite) else None}
            assert not np.isinf(v.to_numpy()).any()
        distributions[family]={'rows':len(frame),'unit':'sampled training user-item candidate rows at 2019-11-27, not users/items',
            'variable_feature_count':sum(s['distinct_nonmissing_values']>1 for s in stats.values()),'features':stats}
        assert distributions[family]['variable_feature_count']>0
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    models={}
    for trial in registry['trials']:
        trial_id=trial['experiment_id']
        models[trial_id]={}
        for window in c['rolling_protocol']:
            model=lgb.Booster(model_file=str(ARTIFACT/trial_id/window/'inner_model.txt'))
            names=model.feature_name();splits=model.feature_importance('split')
            dump=model.dump_model()
            models[trial_id][window]={'num_trees':model.num_trees(),'max_realized_leaves':max(t['num_leaves'] for t in dump['tree_info']),
                'mean_realized_leaves':float(np.mean([t['num_leaves'] for t in dump['tree_info']])),
                'new_features_used':[n for n,s in zip(names,splits) if n.startswith('wv2_') and s>0],
                'new_feature_splits':int(sum(s for n,s in zip(names,splits) if n.startswith('wv2_')))}
    result={'audited_at':now(),'role':'posthoc_read_only_mechanics_not_selection','cutoff':cutoff,'sample':meta,
        'distributions':distributions,'models':models,
        'conclusion':'All eight bundles contain varying real training features. Nonzero splits mean the model used them, not that they add causal value. Capacity variants must be interpreted using realized leaves and historical outcomes, not nominal parameters alone.',
        'no_new_training':True,'no_outer_selection':True,'final_week':'not_run'}
    write(REPORT/'WARM_FEATURE_REALIZATION_AUDIT.json',result)
    print({f:v['variable_feature_count'] for f,v in distributions.items()})


if __name__=='__main__':
    audit()
