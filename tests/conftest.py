from os.path import dirname
from os.path import join
import csv
import pytest
import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_iris
from sklearn.utils import shuffle

NEWSGROUPS_CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'sci.space',
    'talk.religion.misc',
]
NEWSGROUPS_CATEGORIES_BINARY = [
    'alt.atheism',
    'comp.graphics',
]
SIZE = 100


def _get_newsgroups(binary=False, remove_chrome=False, test=False, size=SIZE):
    remove = ('headers', 'footers', 'quotes') if remove_chrome else []
    categories = (
        NEWSGROUPS_CATEGORIES_BINARY if binary else NEWSGROUPS_CATEGORIES)
    subset = 'test' if test else 'train'
    data = fetch_20newsgroups(subset=subset, categories=categories,
                              shuffle=True, random_state=42,
                              remove=remove, n_retries=5, delay=5.0)
    assert data.target_names == categories
    return data.data[:size], data.target[:size], data.target_names


@pytest.fixture(scope="session")
def newsgroups_train():
    return _get_newsgroups(remove_chrome=True)


@pytest.fixture(scope="session")
def newsgroups_train_binary():
    return _get_newsgroups(binary=True, remove_chrome=True)


@pytest.fixture(scope="session")
def newsgroups_train_big():
    return _get_newsgroups(remove_chrome=True, size=1000)


@pytest.fixture(scope="session")
def newsgroups_train_binary_big():
    return _get_newsgroups(binary=True, remove_chrome=True, size=1000)


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_boston():
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', 'boston_house_prices.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype=float)

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1])

@pytest.fixture(scope="session")
def boston_train(size=SIZE):
    data = load_boston()
    X, y = shuffle(data.data, data.target, random_state=13)
    X = X.astype(np.float64)
    return X[:size], y[:size], data.feature_names


@pytest.fixture(scope="session")
def iris_train():
    data = load_iris()
    X, y = shuffle(data.data, data.target, random_state=13)
    return X, y, data.feature_names, data.target_names


@pytest.fixture(scope="session")
def iris_train_binary():
    data = load_iris()
    X, y = shuffle(data.data, data.target, random_state=13)
    flt = y < 2
    X, y = X[flt], y[flt]
    return X, y, data.feature_names
