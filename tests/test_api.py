import pytest

try:
    from src.api.server import app, ChatRequest
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from pydantic import ValidationError


@pytest.mark.skipif(not HAS_DEPS, reason="transformers not installed")
class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(prompt="Hello", max_new_tokens=100, temperature=0.5, top_p=0.9)
        assert req.prompt == "Hello"
        assert req.max_new_tokens == 100

    def test_default_values(self):
        req = ChatRequest(prompt="Test")
        assert req.max_new_tokens == 256
        assert req.temperature == 0.7

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(prompt="")