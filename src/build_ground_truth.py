"""Pre-fill a DRAFT ground-truth table from extracted factsheet text using the LLM.

The draft is only a starting point. A human must check every row against the PDF,
correct mistakes, set verified=TRUE and save it as data/processed/ground_truth.csv.
Later phases call `load_verified_ground_truth()`, which refuses to continue if any
row is unverified.

For each fund the LLM also returns a short quote from the source for every field,
saved to data/processed/ground_truth_evidence.json to speed up manual checking.

Usage:
    python -m src.build_ground_truth          # dry run on 10 funds, then asks to continue
    python -m src.build_ground_truth --yes    # run on all funds without asking
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_client import PROJECT_ROOT, LLMClient, LLMResult, load_config
from src.run_utils import confirm, print_forecast

NUMERIC_FIELDS = ["aum_cr", "nav", "expense_ratio", "return_1y", "return_3y"]
TEXT_FIELDS = ["fund_name", "benchmark", "fund_manager"]
DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

SYSTEM_PROMPT = (
    "You extract facts from mutual fund factsheets. Use ONLY the factsheet text you are "
    "given. Never guess or use outside knowledge. If a value is not in the text, use null."
)

PROMPT_TEMPLATE = """Extract these fields from the factsheet text below.

Rules:
- Where the factsheet lists several plans, use the {plan} for nav, expense_ratio, return_1y and return_3y.
- fund_name: the scheme's full name as printed.
- factsheet_date: the month the data is "as on", formatted YYYY-MM.
- aum_cr: month-end AUM in INR crore, as a number (not average AUM, unless that is the only AUM shown).
- nav: NAV in INR, as a number.
- expense_ratio: total expense ratio in %, as a number (e.g. 1.65).
- return_1y, return_3y: the fund's own 1-year and 3-year returns in % (CAGR for 3y), as numbers. Not the benchmark's.
- benchmark: the benchmark index name as printed.
- fund_manager: the fund manager name(s) as printed; separate several names with "; ".
- has_risk_disclaimer: true if the text contains a market-risk disclaimer such as
  "Mutual Fund investments are subject to market risks", otherwise false.
- For every field, also give "evidence": the shortest exact quote from the text that supports it (or null).

Return ONLY this JSON object:
{{
  "fund_name": str|null, "factsheet_date": "YYYY-MM"|null, "aum_cr": number|null,
  "nav": number|null, "expense_ratio": number|null, "return_1y": number|null,
  "return_3y": number|null, "benchmark": str|null, "fund_manager": str|null,
  "has_risk_disclaimer": true|false,
  "evidence": {{"<field name>": "<quote>"|null, ...}}
}}

FACTSHEET TEXT:
<<<
{text}
>>>"""


def to_number(value: Any) -> float | None:
    """Convert '₹1,234.5 Cr', '12.3%' or 12.3 to a float; None if not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalise_row(fund_id: str, data: dict[str, Any] | None, fields: list[str]) -> dict[str, Any]:
    """Turn the LLM's JSON into one ground-truth row, with notes on anything to check."""
    row: dict[str, Any] = {"fund_id": fund_id}
    notes: list[str] = []
    if data is None:
        row.update({f: None for f in fields})
        notes.append("LLM output invalid - fill this row by hand")
    else:
        for f in fields:
            value = data.get(f)
            if f in NUMERIC_FIELDS:
                number = to_number(value)
                if value is not None and number is None:
                    notes.append(f"{f} not numeric: {value!r}")
                value = number
            elif f == "factsheet_date" and value is not None and not DATE_RE.match(str(value)):
                notes.append(f"factsheet_date not YYYY-MM: {value!r}")
            elif f == "has_risk_disclaimer":
                value = bool(value)
            row[f] = value
            if value is None:
                notes.append(f"{f} missing")
    row["verified"] = False
    row["notes"] = "; ".join(notes)
    return row


def extract_fund(client: LLMClient, fund_id: str, text: str, plan: str) -> LLMResult:
    """One LLM call to pre-fill the fields for a single fund."""
    prompt = PROMPT_TEMPLATE.format(plan=plan, text=text)
    return client.complete_json(prompt, system=SYSTEM_PROMPT, tag=f"ground_truth:{fund_id}")


def load_verified_ground_truth(path: str | Path | None = None) -> pd.DataFrame:
    """Load ground_truth.csv, refusing to continue unless every row is human-verified."""
    cfg = load_config()
    path = Path(path) if path else PROJECT_ROOT / cfg["paths"]["ground_truth"]
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found. Verify ground_truth_draft.csv by hand, "
            "set verified=TRUE on every row, and save it as ground_truth.csv."
        )
    df = pd.read_csv(path)
    verified = df["verified"].astype(str).str.strip().str.upper() == "TRUE"
    if not verified.all():
        unverified = df.loc[~verified, "fund_id"].tolist()
        raise SystemExit(
            f"Refusing to run: {len(unverified)} row(s) in {path.name} have verified != TRUE: "
            f"{', '.join(map(str, unverified))}"
        )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="skip the confirmation after the dry run")
    args = parser.parse_args()

    cfg = load_config()
    text_dir = PROJECT_ROOT / cfg["paths"]["extracted_text"]
    draft_path = PROJECT_ROOT / cfg["paths"]["ground_truth_draft"]
    evidence_path = draft_path.with_name("ground_truth_evidence.json")
    fields: list[str] = cfg["fields"]
    plan: str = cfg["plan_convention"]

    files = sorted(text_dir.glob("*.txt")) if text_dir.exists() else []
    if not files:
        print(f"No extracted text in {text_dir}. Run `python -m src.extract_text` first.")
        return 1

    client = LLMClient(cfg)
    n_dry = min(cfg["dry_run"]["n_cases"], len(files))
    results: dict[str, LLMResult] = {}

    for path in files[:n_dry]:
        results[path.stem] = extract_fund(client, path.stem, path.read_text(encoding="utf-8"), plan)
    print_forecast("ground-truth pre-fill", list(results.values()), len(files))
    preview = pd.DataFrame([normalise_row(k, r.data, fields) for k, r in results.items()])
    print("\nPreview of dry-run rows:")
    print(preview.drop(columns=["notes"]).to_string(index=False))

    if len(files) > n_dry and not confirm(f"Run the remaining {len(files) - n_dry} funds?", args.yes):
        print("Stopped after the dry run. Nothing written.")
        return 0

    for path in files[n_dry:]:
        results[path.stem] = extract_fund(client, path.stem, path.read_text(encoding="utf-8"), plan)

    rows = [normalise_row(k, r.data, fields) for k, r in results.items()]
    pd.DataFrame(rows).to_csv(draft_path, index=False)
    evidence = {k: (r.data or {}).get("evidence") for k, r in results.items()}
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")

    total_cost = sum(r.cost_usd for r in results.values())
    flagged = sum(bool(r["notes"]) for r in rows)
    print(f"\nWrote {len(rows)} rows -> {draft_path.relative_to(PROJECT_ROOT)} (verified=FALSE)")
    print(f"Evidence quotes -> {evidence_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows with notes to check: {flagged}. Cost of this run: ${total_cost:.4f}")
    print("\nNEXT: check every row against its PDF, fix mistakes, set verified=TRUE, "
          "and save as data/processed/ground_truth.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
