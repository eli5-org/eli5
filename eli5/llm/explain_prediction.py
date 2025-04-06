import math
import warnings
from typing import Optional, Union

import openai
from openai.types.chat.chat_completion import (
    ChatCompletion, ChatCompletionTokenLogprob, ChoiceLogprobs)

from eli5.base import Explanation, TargetExplanation, WeightedSpans, DocWeightedSpans
from eli5.explain import explain_prediction


LOGPROBS_ESTIMATOR = 'llm_logprobs'


@explain_prediction.register(ChoiceLogprobs)
def explain_prediction_openai_logprobs(logprobs: ChoiceLogprobs, doc=None):
    """ Creates an explanation of the logprobs
    (available as ``.choices[idx].logprobs`` on a ChatCompletion object),
    highlighting them proportionally to the log probability.
    More likely tokens are highlighted in green,
    while unlikely tokens are highlighted in red.
    ``doc`` argument is ignored.
    """
    if logprobs.content is None:
        raise ValueError('Predictions must be obtained with logprobs enabled')
    text = ''.join(x.token for x in logprobs.content)
    spans = []
    idx = 0
    for lp in logprobs.content:
        token_len = len(lp.token)
        spans.append((
            f'{idx}-{lp.token}',  # each token is a unique feature with it's own weight
            [(idx, idx + token_len)],
            math.exp(lp.logprob)))
        idx += token_len
    weighted_spans = WeightedSpans([
        DocWeightedSpans(
            document=text,
            spans=spans,
            preserve_density=False,
            with_probabilities=True,
        )
    ])
    target_explanation = TargetExplanation(target=text, weighted_spans=weighted_spans)
    return Explanation(
        estimator=LOGPROBS_ESTIMATOR,
        targets=[target_explanation],
    )


@explain_prediction.register(ChatCompletion)
def explain_prediction_openai_completion(
        completion: ChatCompletion, doc=None):
    """ Creates an explanation of the ChatCompletion's logprobs
    highlighting them proportionally to the log probability.
    More likely tokens are highlighted in green,
    while unlikely tokens are highlighted in red.
    ``doc`` argument is ignored.
    """
    targets = []
    for choice in completion.choices:
        if choice.logprobs is None:
            raise ValueError('Predictions must be obtained with logprobs enabled')
        target, = explain_prediction_openai_logprobs(choice.logprobs).targets
        target.target = choice
        targets.append(target)
    explanation = Explanation(
        estimator=LOGPROBS_ESTIMATOR,
        targets=targets,
    )
    return explanation


@explain_prediction.register(openai.Client)
def explain_prediction_openai_client(
        client: openai.Client,
        doc: Union[str, list[dict]],
        *,
        model: str,
        **kwargs,
        ):
    """
    Calls OpenAI client, obtaining response for ``doc`` (a string, or a list of messages),
    with logprobs enabled, and explains the prediction,
    highlighting tokens proportionally to the log probability.
    More likely tokens are highlighted in green,
    while unlikely tokens are highlighted in red.
    . Other keyword arguments are passed to OpenAI client, with
    ``model`` keyword argument required.
    """
    if isinstance(doc, str):
        messages = [{"role": "user", "content": doc}]
    else:
        messages = doc
    kwargs['logprobs'] = True
    completion = client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model,
        **kwargs)
    for choice in completion.choices:
        _recover_logprobs(choice.logprobs, model)
        if choice.logprobs is None:
            raise ValueError('logprobs not found, likely API does not support them')
        if choice.logprobs.content is None:
            raise ValueError(f'logprobs.content is empty: {choice.logprobs}')
    return explain_prediction_openai_completion(completion)


def _recover_logprobs(logprobs: Optional[ChoiceLogprobs], model: str):
    """ Some servers don't populate logprobs.content, try to recover it.
    """
    if logprobs is None:
        return
    if logprobs.content is not None:
        return
    if not (
            getattr(logprobs, 'token_logprobs', None) and
            getattr(logprobs, 'tokens', None)):
        return
    assert hasattr(logprobs, 'token_logprobs')  # for mypy
    assert hasattr(logprobs, 'tokens')  # for mypy
    try:
        import tokenizers
    except ImportError:
        warnings.warn('tokenizers library required to recover logprobs.content')
        return
    try:
        tokenizer = tokenizers.Tokenizer.from_pretrained(model)
    except Exception:
        warnings.warn(f'could not load tokenizer for {model} with tokenizers library')
        return
    assert len(logprobs.token_logprobs) == len(logprobs.tokens)
    # get tokens as strings with spaces, is there any better way?
    text = tokenizer.decode(logprobs.tokens)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    text_tokens = [text[start:end] for (start, end) in encoded.offsets]
    logprobs.content = []
    for logprob, token in zip(logprobs.token_logprobs, text_tokens):
        logprobs.content.append(
            ChatCompletionTokenLogprob(
                token=token,
                bytes=list(map(int, token.encode('utf8'))),
                logprob=logprob,
                top_logprobs=[],  # we could recover that too
            )
        )
