"""
Utilities for text generation.
"""
import re
import math
from typing import List, Tuple, Union, Optional

import numpy as np
from sklearn.utils import check_random_state

from eli5.utils import indices_to_bool_mask, vstack


# the same as scikit-learn token pattern, but allows single-char tokens
DEFAULT_TOKEN_PATTERN = r'\b\w+\b'

# non-whitespace chars
CHAR_TOKEN_PATTERN = r'[^\s]'


def generate_samples(text: 'TokenizedText',
                     n_samples=500,
                     bow=True,
                     random_state=None,
                     replacement='',
                     min_replace=1.0,
                     max_replace=1.0,
                     group_size=1,
                     ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Return ``n_samples`` changed versions of text (with some words removed),
    along with distances between the original text and a generated
    examples. If ``bow=False``, all tokens are considered unique
    (i.e. token position matters).
    """
    kwargs = dict(
        n_samples=n_samples,
        replacement=replacement,
        random_state=random_state,
        min_replace=min_replace,
        max_replace=max_replace,
    )
    if bow:
        num_tokens = len(text.vocab)
        res = text.replace_random_tokens_bow(**kwargs)
    else:
        num_tokens = len(text.tokens)
        res = text.replace_random_tokens(group_size=group_size, **kwargs)

    texts, num_removed_vec, masks = zip(*res)
    similarity = cosine_similarity_vec(num_tokens, num_removed_vec)
    return texts, similarity, vstack(masks)


def cosine_similarity_vec(num_tokens, num_removed_vec):
    """
    Return cosine similarity between a binary vector with all ones
    of length ``num_tokens`` and vectors of the same length with
    ``num_removed_vec`` elements set to zero.
    """
    remaining = -np.array(num_removed_vec) + num_tokens
    return remaining / (np.sqrt(num_tokens + 1e-6) * np.sqrt(remaining + 1e-6))


class TokenizedText(object):
    def __init__(self, text: str, token_pattern=DEFAULT_TOKEN_PATTERN):
        self.text = text
        self.split = SplitResult.fromtext(text, token_pattern)
        self._vocab: Optional[list[str]] = None

    def replace_random_tokens(self,
                              n_samples: int,
                              replacement='',
                              random_state=None,
                              min_replace=1.0,
                              max_replace=1.0,
                              group_size=1,
                              ) -> list[tuple[str, int, np.ndarray]]:
        """ 
        Return a list of ``(text, replaced_count, mask)``
        tuples with n_samples versions of text with some words replaced.
        By default words are replaced with '', i.e. removed.
        """
        n_tokens = len(self.tokens)
        indices = np.arange(n_tokens)
        if not n_tokens:
            nomask = np.array([], dtype=int)
            return [('', 0, nomask)] * n_samples

        min_replace, max_replace = self._get_min_max(min_replace, max_replace,
                                                     n_tokens)
        rng = check_random_state(random_state)
        replace_sizes = rng.randint(low=min_replace, high=max_replace + 1,
                                    size=n_samples)
        res = []
        for num_to_replace in replace_sizes:
            idx_to_replace = rng.choice(indices, num_to_replace, replace=False)
            idx_to_replace = np.array([idx_to_replace] + [
                idx_to_replace + shift for shift in range(1, group_size)
            ]).ravel()
            padded_size = n_tokens + group_size - 1
            mask = indices_to_bool_mask(idx_to_replace, padded_size)[:n_tokens]
            s = self.split.masked(mask, replacement)
            res.append((s.text, num_to_replace, mask))
        return res
    
    def replace_random_tokens_bow(self,
                                  n_samples: int,
                                  replacement='',
                                  random_state=None,
                                  min_replace=1.0,
                                  max_replace=1.0,
                                  ) -> list[tuple[str, int, np.ndarray]]:
        """
        Return a list of ``(text, replaced_words_count, mask)`` tuples with
        n_samples versions of text with some words replaced.
        If a word is replaced, all duplicate words are also replaced
        from the text. By default words are replaced with '', i.e. removed.
        """
        if not self.vocab:
            nomask = np.array([], dtype=int)
            return [('', 0, nomask)] * n_samples

        min_replace, max_replace = self._get_min_max(min_replace, max_replace,
                                                     len(self.vocab))
        rng = check_random_state(random_state)
        replace_sizes = rng.randint(low=min_replace, high=max_replace + 1,
                                    size=n_samples)
        res = []
        for num_to_replace in replace_sizes:
            tokens_to_replace = set(rng.choice(self.vocab, num_to_replace,
                                               replace=False))
            idx_to_replace = [idx for idx, token in enumerate(self.tokens)
                              if token in tokens_to_replace]
            mask = indices_to_bool_mask(idx_to_replace, len(self.tokens))
            s = self.split.masked(idx_to_replace, replacement)
            res.append((s.text, num_to_replace, mask))
        return res

    def _get_min_max(self,
                     min_replace: Union[int, float],
                     max_replace: Union[int, float],
                     hard_maximum: int,
                     ) -> tuple[int, int]:
        if isinstance(min_replace, float):
            min_replace = int(math.floor(hard_maximum * min_replace)) or 1
        if isinstance(max_replace, float):
            max_replace = int(math.ceil(hard_maximum * max_replace))
        else:
            max_replace = min(max_replace, hard_maximum)
        return min_replace, max_replace

    @property
    def vocab(self) -> list[str]:
        if self._vocab is None:
            self._vocab = sorted(set(self.tokens))
        return self._vocab

    @property
    def tokens(self):
        return self.split.tokens

    @property
    def spans_and_tokens(self):
        return list(zip(self.split.token_spans, self.split.tokens))


class SplitResult(object):
    def __init__(self, parts):
        self.parts = np.array(parts, ndmin=1)
        self.lenghts = np.array([len(p) for p in parts])
        self.starts = self.lenghts.cumsum()

    @classmethod
    def fromtext(cls, text: str, token_pattern=DEFAULT_TOKEN_PATTERN) -> 'SplitResult':
        token_pattern = u"(%s)" % token_pattern
        parts = re.split(token_pattern, text)
        return cls(parts)

    @property
    def separators(self):
        return self.parts[::2]

    @property
    def tokens(self):
        return self.parts[1::2]

    @property
    def token_spans(self) -> list[tuple[int, int]]:
        return list(zip(self.starts[::2], self.starts[1::2]))

    def copy(self) -> 'SplitResult':
        return self.__class__(self.parts.copy())

    def masked(self, invmask: Union[np.ndarray, list[int]], replacement='') -> 'SplitResult':
        s = self.copy()
        s.tokens[invmask] = replacement
        return s

    @property
    def text(self) -> str:
        return "".join(self.parts)
