import math

import openai
from openai.types.chat.chat_completion import ChoiceLogprobs, ChatCompletion

from eli5.base import Explanation, TargetExplanation, WeightedSpans, DocWeightedSpans
from eli5.explain import explain_prediction


LOGPROBS_ESTIMATOR = 'llm_logprobs'


@explain_prediction.register(ChoiceLogprobs)
def explain_prediction_openai_logprobs(logprobs: ChoiceLogprobs, doc=None):
    """ Creates an explanation of the logprobs
    (available as ``.choices[idx].logprobs`` on a ChatCompletion object),
    highlighting them proportionally to the log probability.
    More likely tokens are highligted in green,
    while unlikely tokens are highlighted in red.
    ``doc`` argument is ignored.
    """
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
        chat_completion: ChoiceLogprobs, doc=None):
    """ Creates an explanation of the ChatCompletion's logprobs
    highlighting them proportionally to the log probability.
    More likely tokens are highligted in green,
    while unlikely tokens are highlighted in red.
    ``doc`` argument is ignored.
    """
    targets = []
    for choice in chat_completion.choices:
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
        doc: str | list[dict],
        *,
        model: str,
        **kwargs,
        ):
    """
    Calls OpenAI client, obtaining response for ``doc`` (a string, or a list of messages),
    with logprobs enabled, and explains the prediction,
    highlighting tokens proportionally to the log probability.
    More likely tokens are highligted in green,
    while unlikely tokens are highlighted in red.
    . Other keyword arguments are passed to OpenAI client, with
    ``model`` keyword argument required.
    """
    if isinstance(doc, str):
        messages = [{"role": "user", "content": doc}]
    else:
        messages = doc
    kwargs['logprobs'] = True
    chat_completion = client.chat.completions.create(
        messages=messages, model=model, **kwargs)
    return explain_prediction_openai_completion(chat_completion)
