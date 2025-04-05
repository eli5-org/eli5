from typing import Union, Optional, Sequence

import numpy as np

from .base_utils import attrs
from .formatters.features import FormattedFeatureName


# @attrs decorator used in this file calls @attr.s(slots=True),
# creating attr.ib entries based on the signature of __init__.


@attrs
class Explanation:
    """ An explanation for classifier or regressor,
    it can either explain weights or a single prediction.
    """
    def __init__(self,
                 estimator: str,
                 description: Optional[str] = None,
                 error: Optional[str] = None,
                 method: Optional[str] = None,
                 is_regression: bool = False,
                 targets: Optional[list['TargetExplanation']] = None,
                 feature_importances: Optional['FeatureImportances'] = None,
                 decision_tree: Optional['TreeInfo'] = None,
                 highlight_spaces: Optional[bool] = None,
                 transition_features: Optional['TransitionFeatureWeights'] = None,
                 image=None,
                 ):
        self.estimator = estimator
        self.description = description
        self.error = error
        self.method = method
        self.is_regression = is_regression
        self.targets = targets
        self.feature_importances = feature_importances
        self.decision_tree = decision_tree
        self.highlight_spaces = highlight_spaces
        self.transition_features = transition_features
        self.image = image # if arg is not None, assume we are working with images

    def _repr_html_(self):
        """ HTML formatting for the notebook.
        """
        from eli5.formatters import fields
        from eli5.formatters.html import format_as_html
        return format_as_html(self, force_weights=False, show=fields.WEIGHTS)


@attrs
class FeatureImportances:
    """ Feature importances with number of remaining non-zero features.
    """
    def __init__(self, importances, remaining):
        self.importances: list[FeatureWeight] = importances
        self.remaining: int = remaining

    @classmethod
    def from_names_values(cls, names, values, std=None, **kwargs):
        params = zip(names, values) if std is None else zip(names, values, std)
        importances = [FeatureWeight(*x) for x in params]  # type: ignore
        return cls(importances, **kwargs)


@attrs
class TargetExplanation:
    """ Explanation for a single target or class.
    Feature weights are stored in the :feature_weights: attribute,
    and features highlighted in text in the :weighted_spans: attribute.

    Spatial values are stored in the :heatmap: attribute.
    """
    def __init__(self,
                 target: Union[str, int],
                 feature_weights: Optional['FeatureWeights'] = None,
                 proba: Optional[float] = None,
                 score: Optional[float] = None,
                 weighted_spans: Optional['WeightedSpans'] = None,
                 heatmap: Optional[np.ndarray] = None,
                 ):
        self.target = target
        self.feature_weights = feature_weights
        self.proba = proba
        self.score = score
        self.weighted_spans = weighted_spans
        self.heatmap = heatmap


# List is currently used for unhashed features
Feature = Union[str, list, FormattedFeatureName]


@attrs
class FeatureWeights:
    """ Weights for top features, :pos: for positive and :neg: for negative,
    sorted by descending absolute value.
    Number of remaining positive and negative features are stored in
    :pos_remaining: and :neg_remaining: attributes.
    """
    def __init__(self,
                 pos: list['FeatureWeight'],
                 neg: list['FeatureWeight'],
                 pos_remaining: int = 0,
                 neg_remaining: int = 0,
                 ):
        self.pos = pos
        self.neg = neg
        self.pos_remaining = pos_remaining
        self.neg_remaining = neg_remaining


@attrs
class FeatureWeight:
    def __init__(self, feature: Feature, weight: float, std: Optional[float] = None, value=None):
        self.feature = feature
        self.weight = weight
        self.std = std
        self.value = value


@attrs
class WeightedSpans:
    """ Holds highlighted spans for parts of document - a DocWeightedSpans
    object for each vectorizer, and other features not highlighted anywhere.
    """
    def __init__(self,
                 docs_weighted_spans: list['DocWeightedSpans'],
                 other: Optional[FeatureWeights] = None,
                 ):
        self.docs_weighted_spans = docs_weighted_spans
        self.other = other


WeightedSpan = tuple[
    Feature,
    list[tuple[int, int]],  # list of spans (start, end) for this feature
    float,  # feature weight or probability
]


@attrs
class DocWeightedSpans:
    """ Features highlighted in text. :document: is a pre-processed document
    before applying the analyzer. :weighted_spans: holds a list of spans
    for features found in text (span indices correspond to
    :document:). :preserve_density: determines how features are colored
    when doing formatting - it is better set to True for char features
    and to False for word features.
    :with_probabilities: would interpret weights as probabilities from 0 to 1,
    using a more suitable color scheme.
    """
    def __init__(self,
                 document: str,
                 spans: Sequence[WeightedSpan],
                 preserve_density: Optional[bool] = None,
                 with_probabilities: Optional[bool] = None,
                 vec_name: Optional[str] = None,
                 ):
        self.document = document
        self.spans = spans
        self.preserve_density = preserve_density
        self.with_probabilities = with_probabilities
        self.vec_name = vec_name


@attrs
class TransitionFeatureWeights:
    """ Weights matrix for transition features. """
    def __init__(self, class_names: list[str], coef):
        self.class_names = class_names
        self.coef = coef


@attrs
class TreeInfo:
    """ Information about the decision tree. :criterion: is the name of
    the function to measure the quality of a split, :tree: holds all nodes
    of the tree, and :graphviz: is the tree rendered in graphviz .dot format.
    """
    def __init__(self, criterion: str, tree: 'NodeInfo', graphviz: str, is_classification: bool):
        self.criterion = criterion
        self.tree = tree
        self.graphviz = graphviz
        self.is_classification = is_classification


@attrs
class NodeInfo:
    """ A node in a binary tree.
    Pointers to left and right children are in :left: and :right: attributes.
    """
    def __init__(self,
                 id: int,
                 is_leaf: bool,
                 value,
                 value_ratio,
                 impurity: float,
                 samples: int,
                 sample_ratio: float,
                 feature_name: Optional[str] = None,
                 feature_id: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['NodeInfo'] = None,
                 right: Optional['NodeInfo'] = None,
                 ):
        self.id = id
        self.is_leaf = is_leaf
        self.value = value
        self.value_ratio = value_ratio
        self.impurity = impurity
        self.samples = samples
        self.sample_ratio = sample_ratio
        self.feature_name = feature_name
        self.feature_id = feature_id
        self.threshold = threshold
        self.left = left
        self.right = right
