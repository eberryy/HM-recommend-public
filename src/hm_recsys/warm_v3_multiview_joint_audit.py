"""Read-only INNER audit for BPR, sequence, and genuine LightGCN correction overlap."""
from . import warm_v3_joint_representation_audit as audit


def run():
    audit.TRIALS = {
        'sequence': 'WV3-301',
        'graph': 'WV3-201',
        'userknn': 'WV3-221',
        'lightgcn': 'WV3-211',
    }
    audit.COLUMNS = {
        'bpr': 'wv2_bpr_user_item_score',
        'sequence': 'wv3_sequence_score',
        'graph': 'wv3_graph_user_item_score',
        'userknn': 'wv3_userknn_neighbor_score',
        'lightgcn': 'wv3_lightgcn_score',
    }
    return audit.run('MULTIVIEW_JOINT_REPRESENTATION_AUDIT.json')


if __name__ == '__main__':
    run()
