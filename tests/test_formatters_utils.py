import json

import numpy as np
import pytest

from eli5.formatters.utils import tabulate, format_value, numpy_to_python


def test_tabulate():
    assert tabulate([]) == []
    assert tabulate([], header=['foo', 'bar']) == [
        'foo  bar',
        '---  ---',
    ]
    assert tabulate([[1, 'oneee'], [2.3, 'two']]) == [
        '1    oneee',
        '2.3  two  ',
    ]
    assert tabulate([[1, 'oneee'], [2.3, 'two']], header=['Digit', '']) == [
        'Digit       ',
        '-----  -----',
        '1      oneee',
        '2.3    two  ',
    ]
    assert tabulate([[1, 'oneee'], [2.3, 'two']], header=['Digit', 'Word'],
                    col_align='lr') == [
        'Digit   Word',
        '-----  -----',
        '1      oneee',
        '2.3      two',
    ]
    assert tabulate([[1, 'oneee'], [2.3, 'two']], header=['Digit', 'Word'],
                    col_align='r') == [
        'Digit   Word',
        '-----  -----',
        '    1  oneee',
        '  2.3    two',
    ]
    assert tabulate([[1, 'oneee'], [2.3, 'two']], header=['Digit', 'Word'],
                    col_align=['c', 'r']) == [
        'Digit   Word',
        '-----  -----',
        '  1    oneee',
        ' 2.3     two',
    ]
    with pytest.raises(ValueError):
        assert tabulate([[1, 'oneee'], [2.3]])
    with pytest.raises(ValueError):
        assert tabulate([[1], [2.3]], header=['Digit', 'Word'])
    with pytest.raises(ValueError):
        assert tabulate([[1], [2.3]], header=['Digit'], col_align='rr')


def test_format_value():
    assert format_value(None) == ''
    assert format_value(float('nan')) == 'Missing'
    assert format_value(12.23333334) == '12.233'
    assert format_value(-12.23333334) == '-12.233'


def test_numpy_to_python():
    x = numpy_to_python({
        'x': np.int32(12),
        'y': [np.ones(2)],
        'z': {'inner': np.bool_(False)},
    })
    assert x == {
        'x': 12,
        'y': [[1.0, 1.0]],
        'z': {'inner': False},
    }
    json.dumps(x)
