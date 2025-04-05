import math
import pytest
from unittest.mock import Mock

import eli5
from eli5.base import Explanation
from eli5.formatters.html import format_as_html

from openai.types.chat.chat_completion import (
    ChoiceLogprobs,
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionTokenLogprob,
    Choice,
)
from openai import Client


@pytest.fixture
def example_logprobs():
    return ChoiceLogprobs(content=[
        ChatCompletionTokenLogprob(
            token="Hello",
            logprob=math.log(0.9),
            top_logprobs=[],
        ),
        ChatCompletionTokenLogprob(
            token=" world",
            logprob=math.log(0.2),
            top_logprobs=[],
        ),
        ChatCompletionTokenLogprob(
            token=" world",
            logprob=math.log(0.4),
            top_logprobs=[],
        ),
    ])


@pytest.fixture
def example_completion(example_logprobs):
    return ChatCompletion(
        id='chatcmpl-x',
        created=1743590849,
        model='gpt-4o-2024-08-06',
        object='chat.completion',
        choices=[
            Choice(
                logprobs=example_logprobs,
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    content=''.join(x.token for x in example_logprobs.content),
                    role='assistant',
                ),
            )
        ],
    )


def _assert_explanation_structure_and_html(explanation: Explanation):
    assert isinstance(explanation, Explanation)
    assert explanation.targets
    target = explanation.targets[0]
    html = format_as_html(explanation)

    spans = target.weighted_spans.docs_weighted_spans[0].spans
    assert len(spans) == 3
    assert spans[0][1:] == ([(0, 5)], 0.9)
    assert spans[1][1:] == ([(5, 11)], 0.2)
    assert spans[2][1:] == ([(11, 17)], 0.4)

    assert isinstance(target.target, (str, Choice))

    assert "Hello" in html
    assert "world" in html
    assert "0.900" in html
    assert "0.200" in html
    assert "0.400" in html


def test_explain_prediction_choice_logprobs(example_logprobs):
    explanation = eli5.explain_prediction(example_logprobs)
    _assert_explanation_structure_and_html(explanation)


def test_explain_prediction_chat_completion(example_completion):
    explanation = eli5.explain_prediction(example_completion)
    _assert_explanation_structure_and_html(explanation)


class MockClient(Client):
    def __init__(self, chat_return_value):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock(return_value=chat_return_value)


def test_explain_prediction_openai_client(monkeypatch, example_completion):
    client = MockClient(example_completion)

    explanation = eli5.explain_prediction(client, doc="Hello world", model="gpt-4o")
    _assert_explanation_structure_and_html(explanation)

    client.chat.completions.create.assert_called_once()
