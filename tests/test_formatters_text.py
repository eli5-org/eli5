from eli5.formatters import FormattedFeatureName
from eli5.formatters.text import (
    _SPACE,
    _format_feature,
)


def format_feature(feature, hl_spaces=True):
    return _format_feature(feature, hl_spaces=hl_spaces)


def test_format_formatted_feature():
    assert format_feature(FormattedFeatureName('a b')) == 'a b'
    assert format_feature('a b') == 'a{}b'.format(_SPACE)


def test_format_unhashed_feature():
    assert format_feature([]) == ''
    assert format_feature([{'name': 'foo', 'sign': 1}]) == 'foo'
    assert format_feature([{'name': ' foo', 'sign': -1}]) == \
        '(-){}foo'.format(_SPACE)
    assert format_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1}
        ]) == 'foo | (-)bar'
