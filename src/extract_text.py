"""Extract text from factsheet PDFs in data/raw/ into data/processed/text/{fund_id}.txt.

Each PDF also gets a row in data/processed/extraction_log.csv with simple quality
checks, so badly extracted PDFs (scanned pages, garbled fonts, very short text)
can be reviewed by hand before ground truth is built.

Usage:
    python -m src.extract_text
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

from src.llm_client import PROJECT_ROOT, load_config

MIN_CHARS = 800          # a single-fund factsheet with less text than this is suspicious
MIN_DIGIT_RATIO = 0.03   # factsheets are number-heavy; very few digits suggests a bad extraction
MAX_CID_TOKENS = 20      # "(cid:123)" tokens mean the font could not be decoded
MIN_COMMON_WORD_RATIO = 0.08  # readable English text has many of these words
COMMON_WORDS = {
    "the", "of", "and", "to", "in", "fund", "is", "for", "on", "as", "by", "with",
    "are", "be", "or", "an", "at", "this", "that", "from", "investment", "returns",
    "scheme", "risk", "date", "nav", "plan", "growth", "total", "market", "index",
}


def fund_id_from_filename(path: Path) -> str:
    """Turn 'HDFC Flexi Cap Fund - Oct 2026.pdf' into 'hdfc_flexi_cap_fund_oct_2026'."""
    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return slug or "fund"


def extract_pdf(path: Path) -> tuple[str, int, int]:
    """Return (text, page_count, empty_page_count) for one PDF."""
    pages_text: list[str] = []
    empty_pages = 0
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages += 1
            pages_text.append(f"--- page {i} ---\n{text}")
        page_count = len(pdf.pages)
    return "\n\n".join(pages_text), page_count, empty_pages


def quality_flags(text: str, page_count: int, empty_pages: int) -> list[str]:
    """Simple checks that point to a badly extracted PDF."""
    flags: list[str] = []
    body = re.sub(r"--- page \d+ ---", "", text)
    chars = len(body.strip())
    if chars < MIN_CHARS:
        flags.append(f"short_text({chars} chars)")
    if empty_pages:
        flags.append(f"empty_pages({empty_pages}/{page_count}) - possibly scanned")
    digits = sum(c.isdigit() for c in body)
    if chars and digits / chars < MIN_DIGIT_RATIO:
        flags.append(f"few_digits({digits / chars:.1%})")
    cid_count = len(re.findall(r"\(cid:\d+\)", body))
    if cid_count > MAX_CID_TOKENS:
        flags.append(f"undecoded_font({cid_count} cid tokens)")
    words = re.findall(r"[a-z]+", body.lower())
    if words:
        common = sum(w in COMMON_WORDS for w in words) / len(words)
        if common < MIN_COMMON_WORD_RATIO:
            flags.append(f"garbled_text({common:.1%} common words) - font encoding not readable")
    if page_count > 4:
        flags.append(f"many_pages({page_count}) - may not be a single-fund factsheet")
    return flags


def main() -> int:
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_pdfs"]
    out_dir = PROJECT_ROOT / cfg["paths"]["extracted_text"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found in {raw_dir}. Add public factsheet PDFs and re-run.")
        return 1

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for path in pdfs:
        fund_id = fund_id_from_filename(path)
        if fund_id in seen_ids:
            print(f"ERROR: two PDFs map to fund_id '{fund_id}'. Rename one of them.")
            return 1
        seen_ids.add(fund_id)
        try:
            text, page_count, empty_pages = extract_pdf(path)
            flags = quality_flags(text, page_count, empty_pages)
        except Exception as exc:  # corrupt or encrypted PDFs should not stop the batch
            text, page_count, empty_pages, flags = "", 0, 0, [f"failed({type(exc).__name__})"]
        (out_dir / f"{fund_id}.txt").write_text(text, encoding="utf-8")
        rows.append({
            "fund_id": fund_id,
            "source_file": path.name,
            "pages": page_count,
            "chars": len(text),
            "status": "CHECK" if flags else "ok",
            "flags": "; ".join(flags),
        })

    log = pd.DataFrame(rows)
    log_path = PROJECT_ROOT / "data/processed/extraction_log.csv"
    log.to_csv(log_path, index=False)

    bad = log[log["status"] != "ok"]
    print(f"Extracted {len(log)} PDFs -> {out_dir.relative_to(PROJECT_ROOT)}/")
    print(f"Log: {log_path.relative_to(PROJECT_ROOT)}")
    if bad.empty:
        print("All PDFs passed the quality checks.")
    else:
        print(f"\n{len(bad)} PDF(s) need a manual look:")
        print(bad[["fund_id", "flags"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
