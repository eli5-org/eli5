__version__ = '0.13.0'

from .explain import (
    explain_prediction,
    explain_weights,
)
from .formatters import (
    format_as_dict,
    format_as_html,
    format_as_text,
    format_html_styles,
)
from .sklearn import (
    explain_prediction_sklearn,
    explain_weights_sklearn,
)
from .transform import transform_feature_names

try:
    from .ipython import (
        show_prediction,
        show_weights,
    )
except ImportError:
    pass  # IPython is not installed

try:
    from .formatters.as_dataframe import (
        explain_prediction_df,
        explain_prediction_dfs,
        explain_weights_df,
        explain_weights_dfs,
        format_as_dataframe,
        format_as_dataframes,
    )
except ImportError:
    pass  # pandas not available

try:
    from .formatters import format_as_image
except ImportError:
    # Pillow or matplotlib not available
    pass

try:
    from .lightning import (
        explain_prediction_lightning,
        explain_weights_lightning,
    )
except ImportError as e:
    # lightning is not available
    pass

try:
    from .sklearn_crfsuite import explain_weights_sklearn_crfsuite
except ImportError as e:
    # sklearn-crfsuite is not available
    pass

try:
    from .xgboost import (
        explain_prediction_xgboost,
        explain_weights_xgboost,
    )
except ImportError:
    # xgboost is not available
    pass
except Exception as e:
    if e.__class__.__name__ == 'XGBoostLibraryNotFound':
        # improperly installed xgboost
        pass
    else:
        raise

try:
    from .lightgbm import (
        explain_prediction_lightgbm,
        explain_weights_lightgbm,
    )
except ImportError:
    # lightgbm is not available
    pass
except OSError:
    # improperly installed lightgbm
    pass

try:
    from .catboost import explain_weights_catboost
except ImportError:
    # catboost is not available
    pass


try:
    from .keras import explain_prediction_keras
except ImportError:
    # keras is not available
    pass
