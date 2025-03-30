import openai
from openai.types.chat.chat_completion import ChoiceLogprobs

from eli5.explain import explain_prediction


@explain_prediction.register(ChoiceLogprobs)
def explain_prediction_logprobs(logprobs: ChoiceLogprobs, doc=None):
    """ Creates an explanation of the logprobs
    (available as ``.choices[idx].logprobs`` on a ChatCompletion object),
    highlighting them proportionally to the log probability.
    More likely tokens are highligted in green, while unlikely tokens are highlighted in red.
    ``doc`` argument is ignored.
    """
    ...


@explain_prediction.register(ChoiceLogprobs)
def explain_prediction_client(
        client: openai.Client,
        doc: str | list[dict] = None,
        *,
        model: str,
        **kwargs,
        ):
    """
    Calls OpenAI client, obtaining response for ``doc`` (a string, or a list of messages),
    with logprobs enabled. Other keyword arguments are passed to OpenAI client, with
    ``model`` keyword argument required.
    """
    ...
