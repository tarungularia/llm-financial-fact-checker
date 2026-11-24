"""Provider-agnostic LLM client with a disk cache and cost/latency logging.

Every LLM call in this project goes through `LLMClient`, which:

* reads the provider and model from config.yaml (Anthropic by default, OpenAI optional),
* caches each response on disk, keyed by a hash of the prompt, model and settings,
  so re-running an experiment costs nothing and returns the same output,
* parses the response as JSON and retries once if it is not valid JSON,
* logs tokens, cost, latency and cache hits to a JSONL file for evaluation.

API keys are read from the environment (loaded from `.env`) and are never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. "
    "Reply again with ONLY a single valid JSON object and no other text."
)

# A provider call takes (system, prompt) and returns (text, input_tokens, output_tokens).
ProviderCall = Callable[[str, str], tuple[str, int, int]]


@dataclass
class LLMResult:
    """The outcome of one logical LLM request (including any JSON retry)."""

    data: dict[str, Any] | None  # parsed JSON, or None if invalid after the retry
    raw_text: str
    valid_json: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0  # API time only; 0 when every attempt was served from cache
    cached: bool = False  # True only if every attempt was served from cache
    attempts: int = 1
    model: str = ""
    errors: list[str] = field(default_factory=list)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding text.

    Returns None if no JSON object can be parsed.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for dry-run cost forecasts."""
    return max(1, len(text) // 4)


class LLMClient:
    """Thin wrapper around the configured LLM provider."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        provider_call: ProviderCall | None = None,
    ) -> None:
        """Create a client.

        Args:
            config: Parsed config.yaml. Loaded from the project root if omitted.
            provider_call: Optional replacement for the real API call (used in tests).
        """
        load_dotenv(PROJECT_ROOT / ".env")
        self.config = config or load_config()
        llm_cfg = self.config["llm"]
        self.provider: str = llm_cfg["provider"]
        if self.provider not in llm_cfg["providers"]:
            raise ValueError(f"Unknown provider '{self.provider}' in config.yaml")
        self.provider_cfg: dict[str, Any] = llm_cfg["providers"][self.provider]
        self.model: str = self.provider_cfg["model"]
        self.max_tokens: int = llm_cfg["max_tokens"]
        self.temperature: float = llm_cfg["temperature"]
        self.timeout: float = llm_cfg.get("timeout_seconds", 60)
        self.cache_dir = PROJECT_ROOT / llm_cfg["cache_dir"]
        self.log_path = PROJECT_ROOT / llm_cfg["log_path"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._provider_call = provider_call or self._build_provider_call()

    # ----- public API -------------------------------------------------------

    def complete_json(self, prompt: str, system: str = "", tag: str = "") -> LLMResult:
        """Send a prompt and return parsed JSON, retrying once if the reply is not valid JSON.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            tag: Free-text label written to the log (e.g. "checker_v1:case_017").
        """
        result = LLMResult(data=None, raw_text="", valid_json=False, model=self.model, cached=True)
        current_prompt = prompt
        for attempt in (1, 2):
            text, in_tok, out_tok, latency, cached = self._call_with_cache(system, current_prompt)
            result.attempts = attempt
            result.raw_text = text
            result.input_tokens += in_tok
            result.output_tokens += out_tok
            result.latency_s += latency
            result.cached = result.cached and cached
            result.cost_usd += self.cost(in_tok, out_tok) if not cached else 0.0
            result.data = parse_json(text)
            result.valid_json = result.data is not None
            if result.valid_json:
                break
            result.errors.append(f"attempt {attempt}: invalid JSON")
            current_prompt = prompt + RETRY_SUFFIX
        self._log(result, tag)
        return result

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of a call, from the per-million-token prices in config.yaml."""
        price = self.provider_cfg["price_per_mtok"]
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    def estimate_cost(self, prompt_chars: int, expected_output_tokens: int, n_calls: int) -> float:
        """Forecast the USD cost of `n_calls` similar calls (used for dry runs)."""
        in_tok = max(1, prompt_chars // 4)
        return self.cost(in_tok, expected_output_tokens) * n_calls

    # ----- caching ------------------------------------------------------------

    def cache_key(self, system: str, prompt: str) -> str:
        """Hash of everything that can change the response: provider, model, settings, prompts."""
        payload = {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if self.provider_cfg.get("send_temperature") else None,
            "thinking": bool(self.provider_cfg.get("thinking", False)),
            "system": system,
            "prompt": prompt,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _call_with_cache(self, system: str, prompt: str) -> tuple[str, int, int, float, bool]:
        """Return (text, input_tokens, output_tokens, latency_s, cached)."""
        key = self.cache_key(system, prompt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            return entry["text"], entry["input_tokens"], entry["output_tokens"], 0.0, True

        start = time.perf_counter()
        text, in_tok, out_tok = self._provider_call(system, prompt)
        latency = time.perf_counter() - start
        entry = {
            "provider": self.provider,
            "model": self.model,
            "text": text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": round(latency, 3),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        return text, in_tok, out_tok, latency, False

    def cached_latency(self, system: str, prompt: str) -> float | None:
        """Latency recorded when a cached response was first fetched (for reporting on re-runs)."""
        path = self.cache_dir / f"{self.cache_key(system, prompt)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("latency_s")

    # ----- logging ------------------------------------------------------------

    def _log(self, result: LLMResult, tag: str) -> None:
        """Append one line per logical request to the JSONL call log."""
        record = asdict(result)
        record.pop("data")
        record.pop("raw_text")
        record.update({"tag": tag, "provider": self.provider, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ----- providers ----------------------------------------------------------

    def _api_key(self) -> str:
        """Read the provider's API key from the environment without ever printing it."""
        env_var = self.provider_cfg["api_key_env"]
        key = os.getenv(env_var)
        if not key:
            raise RuntimeError(
                f"{env_var} is not set. Copy .env.example to .env and add your key."
            )
        return key

    def _build_provider_call(self) -> ProviderCall:
        """Return a function that calls the configured provider's API."""
        if self.provider == "anthropic":
            return self._anthropic_call()
        if self.provider == "openai":
            return self._openai_call()
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _anthropic_call(self) -> ProviderCall:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key(), timeout=self.timeout)

        def call(system: str, prompt: str) -> tuple[str, int, int]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            if self.provider_cfg.get("send_temperature"):
                kwargs["temperature"] = self.temperature
            if not self.provider_cfg.get("thinking", False):
                kwargs["thinking"] = {"type": "disabled"}
            response = client.messages.create(**kwargs)
            if response.stop_reason == "refusal":
                text = ""  # treated as invalid output by the caller
            else:
                text = "".join(b.text for b in response.content if b.type == "text")
            return text, response.usage.input_tokens, response.usage.output_tokens

        return call

    def _openai_call(self) -> ProviderCall:
        import openai

        client = openai.OpenAI(api_key=self._api_key(), timeout=self.timeout)

        def call(system: str, prompt: str) -> tuple[str, int, int]:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.provider_cfg.get("send_temperature"):
                kwargs["temperature"] = self.temperature
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            usage = response.usage
            return text, usage.prompt_tokens, usage.completion_tokens

        return call


def summarize_log(log_path: str | Path | None = None, tag_prefix: str = "") -> dict[str, float]:
    """Total calls, tokens, cost and latency from the JSONL log, filtered by tag prefix."""
    path = Path(log_path) if log_path else PROJECT_ROOT / load_config()["llm"]["log_path"]
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
              "latency_s": 0.0, "cache_hits": 0, "invalid": 0}
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("tag", "").startswith(tag_prefix):
            continue
        totals["calls"] += 1
        totals["input_tokens"] += rec["input_tokens"]
        totals["output_tokens"] += rec["output_tokens"]
        totals["cost_usd"] += rec["cost_usd"]
        totals["latency_s"] += rec["latency_s"]
        totals["cache_hits"] += int(rec["cached"])
        totals["invalid"] += int(not rec["valid_json"])
    return totals
