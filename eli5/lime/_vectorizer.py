from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from eli5.base import DocWeightedSpans, FeatureWeights
from eli5.sklearn.text import _get_feature_weights_dict
from .textutils import TokenizedText


class SingleDocumentVectorizer(BaseEstimator, TransformerMixin):
    """ Fake vectorizer which converts document just to a vector of ones """

    def __init__(self, token_pattern):
        # type: (str) -> None
        self.token_pattern = token_pattern

    def fit(self, X, y=None):
        self.text_ = X[0]
        if not isinstance(self.text_, TokenizedText):
            self.text_ = TokenizedText(self.text_,
                                       token_pattern=self.token_pattern)
        return self

    def transform(self, X):
        # assert X[0] == self.doc_
        return np.ones(len(self.text_.tokens)).reshape((1, -1))

    def get_doc_weighted_spans(self,
                               doc: str,
                               feature_weights: FeatureWeights,
                               feature_fn: Callable[[str], str],
                               ) -> tuple[dict[tuple[str, int], float], DocWeightedSpans]:
        feature_weights_dict = _get_feature_weights_dict(feature_weights,
                                                         feature_fn)
        spans = []
        found_features = {}
        for idx, (span, feature) in enumerate(self.text_.spans_and_tokens):
            featname = self._featname(idx, feature)
            if featname not in feature_weights_dict:
                continue
            weight, key = feature_weights_dict[featname]
            spans.append((feature, [span], weight))
            found_features[key] = weight

        doc_weighted_spans = DocWeightedSpans(
            document=doc,
            spans=spans,
            preserve_density=False,
        )
        return found_features, doc_weighted_spans

    def _featname(self, idx: int, token: str) -> str:
        return "[{}] {}".format(idx, token)

    def get_feature_names_out(self) -> list[str]:
        return [self._featname(idx, token)
                for idx, token in enumerate(self.text_.tokens)]
