"""
Functions to convert explanations to human-digestible formats.

TODO: IPython integration, customizability.
"""

from .html import (
    format_as_html,
    format_html_styles,
)
from .text import format_as_text

try:
    from .as_dataframe import (
        explain_prediction_df,
        explain_prediction_dfs,
        explain_weights_df,
        explain_weights_dfs,
        format_as_dataframe,
        format_as_dataframes,
    )
except ImportError:
    pass  # pandas not available
from . import fields
from .as_dict import format_as_dict
from .features import FormattedFeatureName

try:
    from .image import format_as_image
except ImportError:
    # Pillow or matplotlib not available
    pass