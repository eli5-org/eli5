import numpy as np

from eli5.base import (
    Explanation, TargetExplanation, FeatureWeights, FeatureWeight)
from eli5.formatters.as_dict import format_as_dict


# format_as_dict is called in eli5.tests.utils.format_as_all


def test_format_as_dict():
    assert format_as_dict(Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'y', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('a', np.float32(13.0))],
                    neg=[])),
        ],
    )) == {'estimator': 'some estimator',
           'targets': [
               {'target': 'y',
                'feature_weights': {
                    'pos': [{
                        'feature': 'a',
                        'weight': 13.0,
                        'std': None,
                        'value': None}],
                    'pos_remaining': 0,
                    'neg': [],
                    'neg_remaining': 0,
                },
                'score': None,
                'proba': None,
                'weighted_spans': None,
                'heatmap': None,
                },
           ],
           'decision_tree': None,
           'description': None,
           'error': None,
           'feature_importances': None,
           'highlight_spaces': None,
           'is_regression': False,
           'method': None,
           'transition_features': None,
           'image': None,
           }
