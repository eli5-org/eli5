from collections import Counter
from typing import Optional

import numpy as np

from eli5.base import TargetExplanation, DocWeightedSpans
from eli5.base_utils import attrs
from eli5.utils import max_or_0


def get_char_weights(
        doc_weighted_spans: DocWeightedSpans, preserve_density: Optional[bool] = None,
    ) -> np.ndarray:
    """ Return character weights for a text document with highlighted features.
    If preserve_density is True, then color for longer fragments will be
    less intensive than for shorter fragments, so that "sum" of intensities
    will correspond to feature weight.
    If preserve_density is None, then it's value is taken from
    the corresponding attribute of doc_weighted_spans.
    """
    if preserve_density is None:
        preserve_density = doc_weighted_spans.preserve_density
    char_weights = np.zeros(len(doc_weighted_spans.document))
    feature_counts = Counter(f for f, _, __ in doc_weighted_spans.spans)
    for feature, spans, weight in doc_weighted_spans.spans:
        for start, end in spans:
            # start can be -1 for char_wb at the start of the document.
            start = max(0, start)
            if preserve_density:
                weight /= (end - start)
            weight /= feature_counts[feature]
            char_weights[start:end] += weight
    return char_weights


@attrs
class PreparedWeightedSpans(object):
    def __init__(self,
                 doc_weighted_spans: DocWeightedSpans,
                 char_weights: np.ndarray,
                 weight_range: float,
                 ):
        self.doc_weighted_spans = doc_weighted_spans
        self.char_weights = char_weights
        self.weight_range = weight_range

    def __eq__(self, other):
        # Working around __eq__ on numpy arrays
        if self.__class__ == other.__class__:
            return (
                (self.doc_weighted_spans, self.weight_range) ==
                (other.doc_weighted_spans, other.weight_range) and
                self.char_weights.shape == other.char_weights.shape and
                np.allclose(self.char_weights, other.char_weights))
        return False


def prepare_weighted_spans(targets: list[TargetExplanation],
                           preserve_density: Optional[bool] = None,
                           ) -> list[Optional[list[PreparedWeightedSpans]]]:
    """ Return weighted spans prepared for rendering.
    Calculate a separate weight range for each different weighted
    span (for each different index): each target has the same number
    of weighted spans.
    """
    targets_char_weights: list[Optional[list[np.ndarray]]] = [
        [get_char_weights(ws, preserve_density=preserve_density)
         for ws in t.weighted_spans.docs_weighted_spans]
         if t.weighted_spans else None
         for t in targets]
    max_idx = max_or_0(len(ch_w or []) for ch_w in targets_char_weights)

    targets_char_weights_not_None: list[list[np.ndarray]] = [
        cw for cw in targets_char_weights
        if cw is not None]

    spans_weight_ranges = [
        max_or_0(
            abs(x) for char_weights in targets_char_weights_not_None
            for x in char_weights[idx])
        for idx in range(max_idx)]
    return [
        [PreparedWeightedSpans(ws, char_weights, weight_range)
         for ws, char_weights, weight_range in zip(
            t.weighted_spans.docs_weighted_spans,  # type: ignore
            t_char_weights,
            spans_weight_ranges)]
        if t_char_weights is not None else None
        for t, t_char_weights in zip(targets, targets_char_weights)]
