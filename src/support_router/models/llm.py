"""Zero- and few-shot classification through an OpenAI-compatible endpoint.

Invalid responses are counted and mapped to the four-label compatibility fallback. A
production deployment should route invalid output and timeouts to human review.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import LABELS

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8001/v1"

#: The routing policy given to the model. This is a prompt, but it is also the clearest
#: written statement of the label definitions in the repo, so it doubles as documentation
#: of what the four routes mean and where the boundary between the confusable pair sits.
SYSTEM_PROMPT = """You are a support-ticket router for a crypto/fintech product.
Classify the user's message into exactly one of four routes:

- account-access: cannot log in, locked out, password/2FA/verification problems, \
account suspended or restricted.
- transaction-dispute: a transaction the user initiated went wrong - wrong price, \
stuck pending, never arrived, cancelled but still charged, unexpected fee.
- fraud-report: activity the user did NOT authorise, or an attempt to defraud them - \
unauthorised withdrawal, account takeover, phishing, scam, stolen funds.
- general: everything else - how-to questions, fees, limits, availability, tax \
documents, timelines, product information.

The critical distinction: transaction-dispute means the USER made the transaction and \
it went wrong. fraud-report means SOMEONE ELSE acted on their account, or someone tried \
to trick them.

Respond with the route name and nothing else. No punctuation, no explanation."""

_NORMALISE = {
    "account-access": "account-access",
    "account_access": "account-access",
    "accountaccess": "account-access",
    "access": "account-access",
    "transaction-dispute": "transaction-dispute",
    "transaction_dispute": "transaction-dispute",
    "transactiondispute": "transaction-dispute",
    "dispute": "transaction-dispute",
    "fraud-report": "fraud-report",
    "fraud_report": "fraud-report",
    "fraudreport": "fraud-report",
    "fraud": "fraud-report",
    "general": "general",
    "other": "general",
}

#: Compatibility fallback for the supplied four-label taxonomy. See module docstring.
FALLBACK_LABEL = "general"

# Keyed by the full prompt, so zero-shot and few-shot (whose examples change per fold)
# never share an entry. Bounded by the dataset, which is 400 rows.
_RESPONSE_CACHE: dict[tuple, str] = {}


def parse_label(raw: str) -> str | None:
    """Extract a route from a model response, or None if nothing valid is present."""
    if not raw:
        return None
    text = raw.strip().strip('"').strip("'").lower()
    text = re.sub(r"^(route|label|answer|classification)\s*[:=-]\s*", "", text).strip()

    direct = _NORMALISE.get(text.replace(" ", "-"))
    if direct:
        return direct

    # The model may wrap the answer in prose; take the first route mentioned.
    for canonical in LABELS:
        if re.search(rf"\b{re.escape(canonical)}\b", text):
            return canonical
    for alias, canonical in _NORMALISE.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return None


class LLMClassifier:
    """Zero-shot or few-shot classification via a chat-completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        few_shot: int = 0,
        temperature: float = 0.0,
        max_concurrency: int = 8,
        request_timeout: float = 30.0,
        seed: int = 0,
        **_: object,
    ) -> None:
        # Env wins over config so the same run can be pointed at whichever
        # OpenAI-compatible server is up locally without editing params.yaml.
        self.model = os.environ.get("LLM_MODEL") or model or DEFAULT_MODEL
        self.base_url = os.environ.get("LLM_BASE_URL") or base_url or DEFAULT_BASE_URL
        self.api_key_env = api_key_env
        self.few_shot = few_shot
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.request_timeout = request_timeout
        self.seed = seed
        self._examples: list[tuple[str, str]] = []
        self.parse_failures_ = 0
        self.last_error_: str | None = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    'openai is required. Install with: uv pip install -e ".[llm]"'
                ) from exc
            self._client = OpenAI(
                base_url=self.base_url,
                # vLLM ignores the key but the client requires a non-empty one.
                api_key=os.environ.get(self.api_key_env, "not-needed"),
                timeout=self.request_timeout,
                max_retries=2,
            )
        return self._client

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> LLMClassifier:
        """Select in-context examples. No weights are updated.

        Examples are drawn round-robin across the four routes so the prompt does not
        inherit the training set's class skew - showing the model 40% `general`
        exemplars would bias it toward `general` for exactly the reason a trained model
        would be biased.
        """
        self.parse_failures_ = 0
        self._examples = []
        if self.few_shot <= 0:
            return self

        rng = np.random.default_rng(self.seed)
        by_label: dict[str, list[str]] = {lab: [] for lab in LABELS}
        for text, label in zip(texts, labels, strict=True):
            if label in by_label:
                by_label[label].append(text)
        for pool in by_label.values():
            rng.shuffle(pool)

        per_label = max(1, self.few_shot // len(LABELS))
        for i in range(per_label):
            for label in LABELS:
                if i < len(by_label[label]):
                    self._examples.append((by_label[label][i], label))
        return self

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example_text, example_label in self._examples:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": example_label})
        messages.append({"role": "user", "content": text})
        return messages

    def _cache_key(self, text: str) -> tuple:
        return (self.base_url, self.model, self.temperature, tuple(self._examples), text)

    def _classify_one(self, text: str) -> str:
        # Decoding is greedy and the prompt is fixed, so an identical request has an
        # identical answer. Repeated CV asks for the same 400 messages once per repeat
        # per scheme; without this the sweep pays for the same tokens eight times.
        key = self._cache_key(text)
        cached = _RESPONSE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=self._build_messages(text),
                temperature=self.temperature,
                # Route names are short; a tight budget stops the model rambling into
                # an explanation we would then have to parse out.
                max_tokens=12,
                seed=self.seed,
            )
            parsed = parse_label(response.choices[0].message.content or "")
        except Exception as exc:
            # A dead endpoint or a timeout must not abort a 20-fold sweep; it is
            # recorded as a failure and the fallback is used. The message is kept so
            # a wholly unreachable server is distinguishable from a model that merely
            # answers badly - both otherwise look like a column of `general`.
            self.last_error_ = f"{type(exc).__name__}: {exc}"
            parsed = None

        if parsed is None:
            # Not cached: a timeout or a dead endpoint is transient, and caching the
            # fallback would turn one blip into a permanently wrong row.
            self.parse_failures_ += 1
            return FALLBACK_LABEL
        _RESPONSE_CACHE[key] = parsed
        return parsed

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.array([], dtype=object)
        # Threads, not processes: these are network-bound calls and vLLM batches
        # concurrent requests server-side.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return np.array(list(pool.map(self._classify_one, texts)), dtype=object)

    def health_check(self) -> dict:
        """Verify the endpoint answers before committing to a full sweep."""
        try:
            models = self._get_client().models.list()
            return {"ok": True, "models": [m.id for m in models.data]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "base_url": self.base_url}


def build_llm_zero_shot(seed: int = 0, **kw: object) -> LLMClassifier:
    kw.setdefault("few_shot", 0)
    return LLMClassifier(seed=seed, **kw)


def build_llm_few_shot(seed: int = 0, **kw: object) -> LLMClassifier:
    kw.setdefault("few_shot", 8)
    return LLMClassifier(seed=seed, **kw)


def load_llm_config(params_llm: dict) -> dict:
    """Translate the `llm:` block of params.yaml into constructor kwargs."""
    return {
        "model": os.environ.get(
            "LLM_MODEL", params_llm.get("generative_model", DEFAULT_MODEL)
        ),
        "base_url": os.environ.get(
            "LLM_BASE_URL", params_llm.get("base_url", DEFAULT_BASE_URL)
        ),
        "api_key_env": params_llm.get("api_key_env", "LLM_API_KEY"),
        "temperature": params_llm.get("temperature", 0.0),
        "max_concurrency": params_llm.get("max_concurrency", 8),
        "request_timeout": params_llm.get("request_timeout", 30),
    }


def dump_prompt(path: str) -> None:
    """Write the routing policy to disk so it can be versioned and diffed in review."""
    with open(path, "w") as fh:
        json.dump({"system_prompt": SYSTEM_PROMPT, "labels": list(LABELS)}, fh, indent=2)
