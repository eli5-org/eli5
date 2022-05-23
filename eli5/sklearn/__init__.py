from eli5.sklearn import transform as _
from eli5.sklearn.explain_prediction import (
    explain_prediction_linear_classifier,
    explain_prediction_linear_regressor,
    explain_prediction_sklearn,
)
from eli5.sklearn.explain_weights import (
    explain_decision_tree,
    explain_linear_classifier_weights,
    explain_linear_regressor_weights,
    explain_rf_feature_importance,
    explain_weights_sklearn,
)
from eli5.sklearn.permutation_importance import PermutationImportance
from eli5.sklearn.unhashing import (
    FeatureUnhasher,
    InvertableHashingVectorizer,
    invert_hashing_and_fit,
)
