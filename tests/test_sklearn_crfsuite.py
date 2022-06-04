import re

import pytest

pytest.importorskip('sklearn_crfsuite')
import numpy as np
from sklearn_crfsuite import CRF

from eli5 import explain_weights

from .utils import format_as_all


@pytest.fixture()
def xseq():
    return [
        {'walk': 1, 'shop': 0.5},
        {'walk': 1},
        {'walk': 1, 'clean': 0.5},
        {'shop': 0.5, 'clean': 0.5},
        {'walk': 0.5, 'clean': 1},
        {'clean': 1, 'shop': 0.1},
        {'walk': 1, 'shop': 0.5},
        {},
        {'clean': 1},
        {'солнце': 'не светит'.encode('utf8'), 'clean': 1},
    ]


@pytest.fixture()
def yseq():
    return ['sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'rainy',
            'sunny', 'sunny', 'rainy', 'rainy']


def test_sklearn_crfsuite(xseq, yseq):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])

    expl = explain_weights(crf)
    text, html = format_as_all(expl, crf)

    assert "y='sunny' top features" in text
    assert "y='rainy' top features" in text
    assert "Transition features" in text
    assert "sunny   -0.130    0.696" in text
    assert '+0.124  солнце:не светит' in text

    html_nospaces = html.replace(' ', '').replace("\n", '')
    assert 'солнце:не светит' in html
    assert '<th>rainy</th><th>sunny</th>' in html_nospaces

    try:
        from eli5 import format_as_dataframes
    except ImportError:
        pass
    else:
        from .test_formatters_as_dataframe import check_targets_dataframe
        df_dict = format_as_dataframes(expl)
        check_targets_dataframe(df_dict['targets'], expl)
        df_transition = df_dict['transition_features']
        assert list(df_transition.columns) == ['from', 'to', 'coef']
        df_indexed = df_transition.groupby(['from', 'to']).agg(lambda x: x)
        transition = expl.transition_features
        print(df_indexed)
        assert list(transition.class_names) == ['rainy', 'sunny']
        assert np.isclose(df_indexed.loc['rainy', 'rainy'].coef,
                          transition.coef[0, 0])
        assert np.isclose(df_indexed.loc['rainy', 'sunny'].coef,
                          transition.coef[0, 1])
        assert np.isclose(df_indexed.loc['sunny', 'rainy'].coef,
                          transition.coef[1, 0])


def test_sklearn_crfsuite_feature_re(xseq, yseq):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])

    expl = explain_weights(
        crf, feature_re=re.compile('(солн|clean)', re.U))
    for expl in format_as_all(expl, crf):
        assert 'солн' in expl
        assert 'clean' in expl
        assert 'walk' not in expl


@pytest.mark.parametrize(['targets'], [
    [['rainy', 'sunny']],
    [['sunny', 'rainy']],
])
def test_sklearn_targets(xseq, yseq, targets):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])

    res = explain_weights(crf,
                          target_names={'sunny': '☀'},
                          targets=targets)
    for expl in format_as_all(res, crf):
        assert '☀' in expl
        if targets[0] == 'rainy':
            assert expl.index('rainy') < expl.index('☀')
        else:
            assert expl.index('rainy') > expl.index('☀')

