"""Minimal multi-backend chat client.

Three backends, one interface:

``openai``
    Anything speaking the OpenAI ``/chat/completions`` contract -- OpenAI,
    Together, Fireworks, Groq, vLLM, Ollama, OpenRouter, LiteLLM proxies.
``anthropic``
    Anthropic's native Messages API.
``bedrock``
    Amazon Bedrock ``converse``. This is what produced the paper's numbers,
    including the Claude Opus 4.8 judge, so it is the backend to use for a
    faithful reproduction with an explicit Opus 4.8 judge and 500-token
    generation limit.

The knobs are `model`, `base_url`/`region`, and `api_key`.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .prompts import MODEL_MAX_TOKENS

DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 180.0
# Retried: rate limits, transient upstream failures, gateway errors.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# Bedrock botocore error codes worth retrying.
RETRY_BEDROCK = {
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "InternalServerException", "ModelTimeoutException", "ModelNotReadyException",
}


@dataclass
class Completion:
    """Result of one model call. `ok=False` carries `error` and empty `text`."""
    ok: bool
    text: str = ""
    error: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)
    #: Normalised across backends: Bedrock `stopReason`, Anthropic
    #: `stop_reason`, OpenAI `finish_reason`. "max_tokens" / "length" means
    #: the answer was cut off -- check this before trusting a verdict.
    stop_reason: Optional[str] = None


def _stop_reason(api: str, data: Optional[Dict]) -> Optional[str]:
    if not data:
        return None
    if api == "bedrock":
        return data.get("stopReason")
    if api == "anthropic":
        return data.get("stop_reason")
    choices = data.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


class ChatClient:
    """One model behind one endpoint.

    Args:
        model: provider model id, e.g. ``us.anthropic.claude-fable-5-2``,
            ``gpt-5.5``, ``accounts/fireworks/models/qwen3-32b``.
        api: ``"openai"``, ``"anthropic"`` or ``"bedrock"``.
        temperature: 0.0 reproduces the paper. Some reasoning models reject an
            explicit temperature -- pass ``None`` to omit the field entirely.
        max_tokens: 2000 by default for new generations. The harness uses 400
            for the judge; pass 500 explicitly for the paper's generation limit.
        region: Bedrock region. There is no built-in default: the region is
            the caller's, because model access is granted per account and
            region. Resolved from this argument, then ``$AWS_REGION`` /
            ``$AWS_DEFAULT_REGION``, then the AWS profile's configured region.
        profile: Bedrock AWS profile name.
        extra_body: provider passthrough, e.g. ``{"reasoning_effort": "medium"}``.
    """

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, api: str = "openai",
                 temperature: Optional[float] = 0.0,
                 max_tokens: Optional[int] = MODEL_MAX_TOKENS,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 region: Optional[str] = None, profile: Optional[str] = None,
                 extra_body: Optional[Dict] = None):
        if api not in ("openai", "anthropic", "bedrock"):
            raise ValueError(
                f"api must be 'openai', 'anthropic' or 'bedrock', got {api!r}")
        self.model = model
        self.api = api
        self.temperature = temperature
        if max_tokens is None and api == "anthropic":
            raise ValueError("the Anthropic API requires a token limit")
        #: None sends no limit (the paper ran GPT-5.5 that way).
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self._http = None
        self._bedrock = None
        #: Set when the provider rejected `temperature` and it was dropped.
        self.dropped_temperature = False

        if api == "bedrock":
            self.profile = profile or os.environ.get("AWS_PROFILE")
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:                     # pragma: no cover
                raise ImportError(
                    "The bedrock backend needs boto3: pip install 'ndcb[bedrock]'"
                ) from exc
            session = (boto3.Session(profile_name=self.profile) if self.profile
                       else boto3.Session())
            # boto3's session already reads $AWS_DEFAULT_REGION and the
            # profile's `region` setting; $AWS_REGION is checked explicitly
            # because older botocore releases ignore it.
            self.region = (region or os.environ.get("AWS_REGION")
                           or session.region_name)
            if not self.region:
                raise ValueError(
                    "No AWS region set. Pass --region, set $AWS_REGION, or "
                    "configure a region in your AWS profile. Use a region "
                    "where your account has been granted access to "
                    f"{model!r} on Bedrock.")
            self._bedrock = session.client(
                "bedrock-runtime", region_name=self.region,
                config=Config(read_timeout=timeout, connect_timeout=30,
                              retries={"max_attempts": 0}),
            )
            return

        # The provider-specific variable wins over the generic $NDCB_API_KEY,
        # so a GPT model under test and a Claude judge can run from the same
        # shell without one silently receiving the other's key.
        env_key = "OPENAI_API_KEY" if api == "openai" else "ANTHROPIC_API_KEY"
        self.api_key = (api_key or os.environ.get(env_key)
                        or os.environ.get("NDCB_API_KEY"))
        if not self.api_key:
            raise ValueError(
                f"No API key. Set --api-key, ${env_key}, or $NDCB_API_KEY.")

        # $OPENAI_BASE_URL applies to the openai backend only; it must never
        # redirect Anthropic-native calls to an OpenAI-compatible host.
        if api == "openai":
            default_base = (os.environ.get("OPENAI_BASE_URL")
                            or "https://api.openai.com/v1")
        else:
            default_base = (os.environ.get("ANTHROPIC_BASE_URL")
                            or "https://api.anthropic.com/v1")
        self.base_url = (base_url or os.environ.get("NDCB_BASE_URL")
                         or default_base).rstrip("/")

        import httpx
        headers = {"content-type": "application/json"}
        if api == "openai":
            headers["authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        self._http = httpx.Client(timeout=timeout, headers=headers)

    # -- request shaping ---------------------------------------------------- #
    def _payload(self, prompt: str) -> Dict:
        body: Dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            body["max_tokens" if self.api == "anthropic"
                 else "max_completion_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    @staticmethod
    def _extract(api: str, data: Dict) -> str:
        if api == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text")
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        # Some gateways return content as a list of typed parts.
        content = msg.get("content")
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content
                           if isinstance(p, dict))
        return content or ""

    # -- bedrock ------------------------------------------------------------ #
    def _complete_bedrock(self, prompt: str) -> Completion:
        inference: Dict = {}
        if self.max_tokens is not None:
            inference["maxTokens"] = self.max_tokens
        if self.temperature is not None:
            inference["temperature"] = self.temperature
        kwargs: Dict = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference,
        }
        kwargs.update(self.extra_body)

        last = "no attempt made"
        for attempt in range(self.max_retries):
            try:
                resp = self._bedrock.converse(**kwargs)
                blocks = resp.get("output", {}).get("message", {}).get("content", [])
                text = "".join(b.get("text", "") for b in blocks if "text" in b)
                stop = _stop_reason("bedrock", resp)
                if not text.strip():
                    return Completion(False, error="empty completion", raw=resp,
                                      stop_reason=stop)
                return Completion(True, text=text, raw=resp, stop_reason=stop)
            except Exception as exc:
                code = getattr(exc, "response", {}).get(
                    "Error", {}).get("Code", type(exc).__name__)
                last = f"{code}: {exc}"
                # A rejected sampling parameter is not transient. Drop
                # temperature once and retry: that is how the paper's
                # reasoning models were run (API defaults), and newer models
                # reject the field outright ("`temperature` is deprecated for
                # this model").
                if (self.temperature is not None
                        and "temperature" in str(exc).lower()):
                    self.temperature = None
                    inference.pop("temperature", None)
                    self.dropped_temperature = True
                    continue
                if code not in RETRY_BEDROCK:
                    return Completion(False, error=last)
            if attempt < self.max_retries - 1:
                time.sleep(min(2 ** attempt + random.random(), 60))
        return Completion(
            False, error=f"exhausted {self.max_retries} retries; last: {last}")

    # -- public ------------------------------------------------------------- #
    def complete(self, prompt: str) -> Completion:
        """Call the model once, retrying transient failures with jittered backoff."""
        if self.api == "bedrock":
            return self._complete_bedrock(prompt)

        path = "/chat/completions" if self.api == "openai" else "/messages"
        url = self.base_url + path
        body = self._payload(prompt)
        last = "no attempt made"

        for attempt in range(self.max_retries):
            try:
                resp = self._http.post(url, json=body)
                if resp.status_code == 200:
                    data = resp.json()
                    text = self._extract(self.api, data)
                    stop = _stop_reason(self.api, data)
                    if not text.strip():
                        # Empty completion: usually a reasoning model spending
                        # the whole token budget on hidden reasoning. Surface it
                        # rather than scoring it as a verdict.
                        return Completion(False, error="empty completion",
                                          raw=data, stop_reason=stop)
                    return Completion(True, text=text, raw=data, stop_reason=stop)
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                # Same rule as Bedrock: a rejected sampling parameter is not
                # transient. OpenAI reasoning models (GPT-5.x, o-series) and
                # Claude Opus 4.7+ reject temperature, so drop it once and
                # retry at API defaults -- how the paper ran such models.
                if (resp.status_code == 400 and self.temperature is not None
                        and "temperature" in resp.text.lower()):
                    self.temperature = None
                    body.pop("temperature", None)
                    self.dropped_temperature = True
                    continue
                if resp.status_code not in RETRY_STATUS:
                    return Completion(False, error=last)
            except Exception as exc:                       # network / timeout
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries - 1:
                # Full jitter; honours the common 1-2 min rate-limit windows.
                time.sleep(min(2 ** attempt + random.random(), 60))
        return Completion(
            False, error=f"exhausted {self.max_retries} retries; last: {last}")

    def close(self):
        if self._http is not None:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def map_threaded(fn, items: List, workers: int = 8, desc: str = ""):
    """Run `fn` over `items` concurrently, preserving input order.

    An exception in a worker becomes `None` in the output rather than killing
    the run -- a 4,182-call sweep should not be lost to one bad response.
    """
    from concurrent.futures import ThreadPoolExecutor
    results: List = [None] * len(items)
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(items), desc=desc)
    except ImportError:
        bar = None

    def work(pair):
        i, item = pair
        try:
            results[i] = fn(item)
        except Exception as exc:
            results[i] = None
            print(f"  [warn] item {i} raised {type(exc).__name__}: {exc}")
        finally:
            if bar is not None:
                bar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, enumerate(items)))
    if bar is not None:
        bar.close()
    return results
