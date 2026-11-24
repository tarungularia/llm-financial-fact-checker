"""Shared helpers for scripts that call the LLM: dry-run forecasts and confirmation.

Project rule: before any full run that calls the API, run on a small sample first,
print the estimated cost and time for the full run, and ask for confirmation.
"""

from __future__ import annotations

import sys

from src.llm_client import LLMResult


def forecast(sample: list[LLMResult], n_total: int) -> dict[str, float]:
    """Extrapolate cost and time for the full run from a dry-run sample.

    Results already cached cost nothing on the full run, so only the remaining
    items are counted.
    """
    if not sample:
        return {"n_remaining": n_total, "cost_usd": 0.0, "minutes": 0.0}
    paid = [r for r in sample if not r.cached] or sample
    avg_cost = sum(r.cost_usd for r in paid) / len(paid)
    avg_latency = sum(r.latency_s for r in paid) / len(paid)
    n_remaining = max(0, n_total - len(sample))
    return {
        "n_remaining": n_remaining,
        "avg_cost_usd": avg_cost,
        "avg_latency_s": avg_latency,
        "cost_usd": avg_cost * n_remaining,
        "minutes": avg_latency * n_remaining / 60,
    }


def print_forecast(label: str, sample: list[LLMResult], n_total: int) -> None:
    """Print the dry-run results and the full-run estimate."""
    f = forecast(sample, n_total)
    spent = sum(r.cost_usd for r in sample)
    invalid = sum(not r.valid_json for r in sample)
    print(f"\n=== Dry run: {label} ===")
    print(f"Sample size:          {len(sample)} (invalid JSON: {invalid})")
    print(f"Dry-run cost:         ${spent:.4f}")
    if sample:
        print(f"Avg cost per item:    ${f['avg_cost_usd']:.5f}")
        print(f"Avg latency per item: {f['avg_latency_s']:.1f}s")
    print(f"Full run: {f['n_remaining']} remaining items -> "
          f"est. ${f['cost_usd']:.4f}, ~{f['minutes']:.1f} min")


def confirm(question: str, assume_yes: bool) -> bool:
    """Ask a yes/no question on the terminal unless --yes was passed."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> not confirmed (no terminal). Re-run with --yes to proceed.")
        return False
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
