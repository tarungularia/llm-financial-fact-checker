# PRD: LLM Fact-Checker for AI-Written Fund Summaries

**Status:** Draft v1 · **Type:** Portfolio project (public data only)

## 1. User
A **marketing or compliance reviewer at a global asset manager** (funds sold in several markets, e.g. UCITS funds in Europe and Asia). They approve fund summaries for factsheets, websites and client emails before anything goes out. More and more of these summaries are drafted by an LLM.

## 2. Problem
AI-written fund summaries read well but can be wrong in ways that are easy to miss:
- **Wrong numbers:** NAV, fund size (AUM), ongoing charges or returns (e.g. 45.2% becomes 54.2%), or the wrong currency
- **Wrong dates:** the factsheet month or year
- **Wrong names:** fund manager or benchmark
- **Missing risk disclaimer:** required risk language such as "past performance is not a guide to future returns" is left out
- **Unsupported claims:** e.g. "guaranteed returns", which the source does not say

Checking each figure against the source by hand is slow and repetitive, so mistakes get through. A wrong figure in client material is a regulatory and reputational risk.

## 3. Solution
A checker that takes (a) the source factsheet text and (b) the AI summary, and returns:
- **verdict:** pass / fail
- **errors:** list of `{field, summary_value, source_value, type}`
- **confidence:** high / medium / low

On top of that, a **human-review rule** sends uncertain documents to a person, so the reviewer only spends time where the checker is unsure.

Two designs are built and compared:
- **v1, single prompt:** one LLM call checks the whole summary against the source.
- **v2, hybrid:** the LLM pulls the fields out of the summary, Python compares numbers and dates against verified ground truth using set tolerances, and the LLM is used only for language judgements (disclaimer present? unsupported claims?).

## 4. Success metrics
| Metric | Definition | Target | Why |
|---|---|---|---|
| **Missed-error rate** (primary) | missed errors ÷ cases that contain an error | **< 5%** | A wrong figure reaching clients is the main risk |
| False-alarm rate | false alarms ÷ clean cases | < 15% | Too many false alarms and reviewers stop trusting the tool |
| Cost per document | USD per summary checked | track only | Has to be cheap at volume |
| Latency per document | seconds per summary checked | track only | Has to fit into a review workflow |
| Invalid-output rate | unparseable responses ÷ total | track only | Reliability |
| % routed to human | share of documents sent for review | track only | Measures how much reviewer time is saved |

## 5. Risks
- **The checker hallucinates:** it flags values that are correct or invents "source values". *Mitigation:* check against verified ground truth; v2 does numeric comparisons in code.
- **Ground truth is wrong:** *Mitigation:* the LLM pre-fills it, then every row is checked by a human (`verified=TRUE`). The pipeline refuses to run on unverified rows.
- **Synthetic errors are too easy:** the injected errors may not look like real LLM mistakes. *Mitigation:* error analysis on real failures, stated openly in Limitations.
- **PDF extraction errors:** tables in factsheets can extract badly. *Mitigation:* bad extractions are logged and checked manually.
- **Overfitting to a small dataset** (~30 funds, ~200 cases). *Mitigation:* results are reported with that caveat and no claims of general accuracy.

## 6. Out of scope
- Legal or regulatory advice. The tool helps a reviewer; it does not replace one.
- Live or real-time market data. Only the factsheet itself is the source of truth.
- Non-English documents.
- Converting between currencies. Each fund is checked in the currency its factsheet reports.
- Charts and images inside PDFs (text only).
- Any employer or client data. Public factsheets only.

## 7. Evaluation plan
1. **Data:** ~30 public single-fund factsheets → extracted text → human-verified ground truth for 11 fields (including currency, since funds report in USD, EUR, GBP, etc.). Only factsheets whose text extracts cleanly are used; garbled extractions are flagged and excluded.
2. **Test set (~200 cases):** ~100 clean summaries (checked against ground truth) + ~100 with exactly one injected error, balanced across the 5 error types, generated deterministically with a fixed seed.
3. **Runs:** v1 and v2 on the same test set. Temperature 0 where the model accepts it, and every response cached, so results can be reproduced at no extra cost.
4. **Metrics:** confusion matrix, accuracy, missed-error rate, false-alarm rate, results by error type, invalid-output rate, cost and latency per document.
5. **Error analysis:** read every wrong prediction and group the failures into 3–5 patterns.
6. **Review policy:** compare at least 3 routing settings on % sent to humans vs. the missed-error rate left after review, and recommend one.

Every number reported comes from saved runs in `/results`.
