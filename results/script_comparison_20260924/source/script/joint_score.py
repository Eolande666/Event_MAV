"""Single normalized decision axis; no repetition-only acceptance branch."""


def score_candidate(features, rotation, cfg, *, deferred=False):
    rotation_score = rotation['rotation_q'] if rotation.get('observable', False) else 0.0
    score = (cfg.rotation_weight*rotation_score
             + cfg.repetition_weight*features['repetition_score']
             + cfg.structure_weight*features['spatial_score']) / (
                 cfg.rotation_weight+cfg.repetition_weight+cfg.structure_weight)
    threshold = cfg.joint_threshold
    # Budget exhaustion cannot be treated as a measured failure of observability.
    passed = not deferred and score >= threshold
    return {'joint_score': float(score), 'joint_threshold': float(threshold),
            'joint_pass': bool(passed),
            'state': 'budget_deferred' if deferred else 'joint_candidate' if passed else 'joint_reject'}
