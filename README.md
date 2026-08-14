# Probabilistic Data Structures in the Apache Software Foundation

An automated **Mining Software Repositories (MSR)** pipeline that extracts,
parses, and analyzes the adoption of **Probabilistic Data Structures (PDS)** —
Bloom Filters, HyperLogLog, Count-Min Sketch, Cuckoo Filters, and related
sketches — across the official Java repositories of the Apache Software
Foundation (ASF).

The pipeline parses Abstract Syntax Trees (ASTs) with `tree-sitter`, classifies
each finding against a literature-based taxonomy, and answers four research
questions on adoption, architectural context, implementation origin, and
technology generation. An additional module applies unsupervised and supervised
machine learning to the enriched dataset.

---

## System Architecture

The pipeline runs in four automated phases, orchestrated by `main.py`:

| Phase | Script | Output |
|-------|--------|--------|
| 1. Extraction | `fetch_apache_repos.py` | `repos.txt` |
| 2. Mining | `miner.py` | `output/dataset.csv`, `output/mining_provenance.json`, `output/parse_failures.json` |
| 3. Analytics | `analytics.py` | `plots/RQ*.png`, `output/dataset_enriched.json`, `output/classification_funnel.json`, coverage JSONs |
| 4. Advanced / ML | `ml_analytics.py` | `plots/ml/*.png`, `output/ml_analysis_summary.json` |

**Extraction** triangulates the official ASF project registry with GitHub
metadata to isolate active Java repositories. **Mining** clones each repo
(`--depth 1`), walks every Java file's AST, and records imports and
instantiations of the target structures defined in `pds_targets.json`.
**Analytics** normalizes each finding to a concrete algorithm and functional
category, infers a usage domain and an implementation origin, and renders the
research-question figures. **Advanced/ML** runs PCA, K-Means and hierarchical
clustering, Isolation-Forest anomaly detection, and a grouped-cross-validated
Random Forest classifier.

---

## Research Questions

- **RQ1 / RQ1b** — Which PDS categories are adopted, by raw usage intensity
  (RQ1) and by adoption breadth across distinct repositories (RQ1b)?
- **RQ2** — In which architectural domains do these structures appear?
  *(heuristic, inferred from file-path tokens — see Limitations.)*
- **RQ3** — Are implementations custom-built or drawn from third-party
  libraries? *(best-effort — see Limitations.)*
- **RQ4** — How does usage split between "legacy" and "modern" algorithms?

---

## Prerequisites and Setup

Requires **Python 3.9+** and **git** on the PATH.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

After a successful install, freeze an exact lock for the paper artifact:

```bash
pip freeze > requirements.lock.txt
```

### Environment configuration

Create a `.env` file in the project root with a GitHub token (raises the API
rate limit during extraction):

```env
GITHUB_TOKEN=your_token_here
```

---

## How to Run

Run the whole pipeline:

```bash
python main.py
```

Because cloning and analyzing large repositories (Hadoop, Cassandra, ...) can
take **hours**, the orchestrator supports skipping phases so you can iterate on
analysis without re-mining:

```bash
python main.py --skip-fetch --skip-mine     # reuse repos.txt + dataset.csv, re-run analytics + ML
python main.py --skip-ml                     # everything except the ML phase
```

Individual phases can also be run directly, e.g. `python analytics.py`.

---

## Configuration: `pds_targets.json`

The mining vocabulary is centralized in `pds_targets.json`. `miner.py` searches
these tokens as case-insensitive substrings; `analytics.py` normalizes each hit
to an algorithm and category.

> **Keep the two aligned.** Every token in `pds_targets.json` should map to an
> algorithm in `analytics.PDS_ALGORITHM_PATTERNS`, and any algorithm you want to
> report must be reachable from a token here. Short/ambiguous tokens (`hll`,
> `cms`, `cpc`, `xor`, `shards`, `loglog`, `ifilter`) are deliberately excluded
> to protect precision; add them only alongside manual validation.

---

## Reproducibility

- **`repos.txt`** — the frozen list of mined repositories.
- **`output/mining_provenance.json`** — per repository, the exact `commit_sha`
  cloned and the UTC timestamp, so the precise snapshot can be reconstructed
  (clones use `--depth 1`, i.e. HEAD at clone time).
- **`output/classification_funnel.json`** — the extraction→classification
  funnel (raw findings, unknown/dropped, retained), the number reviewers expect.
- Pinned `requirements.txt` (+ a frozen `requirements.lock.txt`).

### Extractor validation (recommended before publishing)

Detector accuracy is not yet reported. Scaffolding is included:

```python
import pandas as pd, analytics
df = analytics.enrich_dataset(pd.read_csv("output/dataset.csv"))
analytics.build_gold_sample_template(df)          # -> output/gold_sample_template.csv
# ...label the 'true_algorithm' column by hand, then:
metrics = analytics.compute_extractor_metrics("output/gold_sample_template.csv")
print(metrics["report"])
```

Report the resulting precision / recall / F1 and confusion matrix in the paper.

---

## Output Files

```
repos.txt                              validated ASF Java repositories
output/
  dataset.csv                          raw AST findings
  dataset_enriched.json                cleaned + enriched dataset
  classification_funnel.json           extraction -> classification funnel
  coverage_by_algorithm.json           occurrences / files / repos per algorithm
  coverage_by_category.json            occurrences / files / repos per category
  mining_provenance.json               commit SHA + timestamp per repo
  parse_failures.json                  files that failed to parse
  ml_analysis_summary.json             PCA / clustering / classifier results
plots/
  RQ1_Adoption.png  RQ1b_Adoption_Breadth.png
  RQ2_Usage_Domain.png  RQ3_Implementation_Origin.png  RQ4_Tech_Debt.png
  ml/                                  PCA, K-selection, feature-importance figures
```

---

## Limitations (Threats to Validity)

These are inherent to the heuristics and should be stated in the paper:

- **RQ2 usage domain** is inferred from file-path tokens (e.g. `net`, `cache`,
  `io`), not a validated architectural analysis. Matching is on whole path
  tokens, not raw substrings, to avoid false positives.
- **RQ3 implementation origin** is best-effort. `INSTANTIATION` findings are
  labeled `Unresolved` because origin cannot be determined without resolving
  the corresponding import; the miner does not yet perform that cross-reference.
- **Mining recall** is bounded by the tokens in `pds_targets.json`; structures
  named outside that vocabulary are not captured.
- **Detector precision/recall** against a manually labeled gold set is not yet
  reported (see the validation scaffolding above).
- **Snapshot drift** — `--depth 1` clones capture HEAD at clone time; use
  `mining_provenance.json` to pin the exact commits.
