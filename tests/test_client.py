"""Direct-API client behaviour, tested offline against a mock HTTP transport.

These pin the request shapes and failure handling for the OpenAI and
Anthropic backends, which the paper's runs (all on Bedrock) never exercised.
"""
from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
import pytest

from ndcb.client import ChatClient


def _attach(client: ChatClient, handler):
    """Swap the client's HTTP transport for a mock, keeping its headers."""
    client._http = httpx.Client(transport=httpx.MockTransport(handler),
                                headers=client._http.headers)
    return client


def _openai_ok(text="<candidate_a>x</candidate_a><candidate_b>x</candidate_b>",
               finish="stop"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}, "finish_reason": finish}]})


def _anthropic_ok(text="ok", stop="end_turn"):
    return httpx.Response(200, json={
        "content": [{"type": "thinking", "thinking": ""},
                    {"type": "text", "text": text}],
        "stop_reason": stop})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("NDCB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "NDCB_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


# --- request shapes -------------------------------------------------------- #
@pytest.mark.parametrize("api", ["bedrock", "openai", "anthropic"])
@pytest.mark.parametrize("max_tokens", [None, 500])
def test_generation_token_limit_on_every_backend(api, max_tokens, monkeypatch):
    """Defaults and explicit paper-budget overrides must reach the provider."""
    kwargs = {} if max_tokens is None else {"max_tokens": max_tokens}
    expected = 2000 if max_tokens is None else max_tokens
    if api == "bedrock":
        session = Mock()
        runtime = session.client.return_value
        runtime.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
        }
        monkeypatch.setattr(boto3, "Session", Mock(return_value=session))
        with ChatClient("m", api=api, region="us-east-1", **kwargs) as client:
            assert client.complete("hi").ok
        request = runtime.converse.call_args.kwargs
        assert request["inferenceConfig"]["maxTokens"] == expected
    else:
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return _openai_ok() if api == "openai" else _anthropic_ok()

        with _attach(ChatClient("m", api=api, api_key="test", **kwargs),
                     handler) as client:
            assert client.complete("hi").ok
        field = "max_completion_tokens" if api == "openai" else "max_tokens"
        assert seen[field] == expected


def test_openai_request_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _openai_ok()

    c = _attach(ChatClient("gpt-5.5", api="openai", api_key="dummy-openai-key",
                           max_tokens=500,
                           extra_body={"reasoning_effort": "medium"}), handler)
    assert c.complete("hi").ok
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer dummy-openai-key"
    assert seen["body"]["max_completion_tokens"] == 500
    assert "max_tokens" not in seen["body"]
    assert seen["body"]["reasoning_effort"] == "medium"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_request_shape_and_text_extraction():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        seen["body"] = json.loads(request.content)
        return _anthropic_ok("verdict")

    c = _attach(ChatClient("claude-opus-4-8", api="anthropic",
                           api_key="dummy-anthropic-key", temperature=None,
                           max_tokens=400), handler)
    out = c.complete("hi")
    assert out.ok and out.text == "verdict"      # thinking block skipped
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["key"] == "dummy-anthropic-key"
    assert seen["version"] == "2023-06-01"
    assert seen["body"]["max_tokens"] == 400
    assert "temperature" not in seen["body"]


# --- rejected temperature -------------------------------------------------- #
@pytest.mark.parametrize("api,ok", [("openai", _openai_ok),
                                    ("anthropic", _anthropic_ok)])
def test_rejected_temperature_is_dropped_and_retried(api, ok):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unsupported value: 'temperature' does not support "
                           "0 with this model."}})
        return ok()

    c = _attach(ChatClient("m", api=api, api_key="k", temperature=0.0), handler)
    out = c.complete("hi")
    assert out.ok
    assert len(bodies) == 2
    assert "temperature" in bodies[0] and "temperature" not in bodies[1]
    assert c.temperature is None and c.dropped_temperature


def test_unrelated_400_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad model id"}})

    c = _attach(ChatClient("m", api="openai", api_key="k"), handler)
    out = c.complete("hi")
    assert not out.ok and "HTTP 400" in out.error
    assert len(calls) == 1
    assert c.temperature == 0.0


# --- truncation and empty output ------------------------------------------- #
def test_openai_length_finish_is_reported():
    c = _attach(ChatClient("m", api="openai", api_key="k"),
                lambda r: _openai_ok(text="cut off mid", finish="length"))
    out = c.complete("hi")
    assert out.ok and out.stop_reason == "length"


def test_reasoning_model_with_empty_content_is_a_failure_not_a_verdict():
    """All budget spent on hidden reasoning -> empty content -> ok=False."""
    c = _attach(ChatClient("m", api="openai", api_key="k"),
                lambda r: _openai_ok(text="", finish="length"))
    out = c.complete("hi")
    assert not out.ok and out.error == "empty completion"
    assert out.stop_reason == "length"


def test_anthropic_stop_reason_is_reported():
    c = _attach(ChatClient("m", api="anthropic", api_key="k"),
                lambda r: _anthropic_ok(stop="max_tokens"))
    assert c.complete("hi").stop_reason == "max_tokens"


# --- environment handling -------------------------------------------------- #
def test_provider_key_beats_generic_key(monkeypatch):
    """GPT under test + Claude judge in one shell must not cross keys."""
    monkeypatch.setenv("NDCB_API_KEY", "generic")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic")
    assert ChatClient("m", api="openai").api_key == "dummy-openai"
    assert ChatClient("m", api="anthropic").api_key == "dummy-anthropic"


def test_generic_key_is_the_fallback(monkeypatch):
    monkeypatch.setenv("NDCB_API_KEY", "generic")
    assert ChatClient("m", api="openai").api_key == "generic"


def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env")
    assert ChatClient("m", api="openai", api_key="flag").api_key == "flag"


def test_missing_key_raises():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ChatClient("m", api="openai")


def test_openai_base_url_does_not_leak_into_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")
    a = ChatClient("m", api="anthropic", api_key="k")
    o = ChatClient("m", api="openai", api_key="k")
    assert a.base_url == "https://api.anthropic.com/v1"
    assert o.base_url == "https://api.together.xyz/v1"


def test_explicit_base_url_wins_and_is_normalised():
    c = ChatClient("m", api="openai", api_key="k",
                   base_url="http://localhost:8000/v1/")
    assert c.base_url == "http://localhost:8000/v1"


def test_rejects_unknown_api():
    with pytest.raises(ValueError):
        ChatClient("m", api="vertex", api_key="k")


# --- Bedrock region: always the caller's, never a built-in default --------- #
boto3 = pytest.importorskip("boto3")


@pytest.fixture
def no_aws_region(monkeypatch, tmp_path):
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))


def test_bedrock_without_any_region_raises(no_aws_region):
    with pytest.raises(ValueError, match="No AWS region set"):
        ChatClient("us.anthropic.claude-opus-4-8", api="bedrock")


def test_bedrock_region_from_env(no_aws_region, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert ChatClient("m", api="bedrock").region == "eu-central-1"


def test_bedrock_region_from_default_region_env(no_aws_region, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    assert ChatClient("m", api="bedrock").region == "ap-northeast-1"


def test_bedrock_region_from_profile_config(no_aws_region, monkeypatch, tmp_path):
    cfg = tmp_path / "config_with_region"
    cfg.write_text("[profile research]\nregion = us-east-2\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg))
    assert ChatClient("m", api="bedrock", profile="research").region == "us-east-2"


def test_bedrock_explicit_region_wins(no_aws_region, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert ChatClient("m", api="bedrock", region="us-west-2").region == "us-west-2"
