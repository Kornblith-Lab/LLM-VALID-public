# pecarn_llm_extraction

Analysis code for *Evaluating human–large language model interaction in clinical prediction rule calculation: Development and Validation Study*.

The study evaluated GPT-5-mini for extracting predictor variables and calculating risk categories for four PECARN pediatric trauma clinical prediction rules (TBI in children younger than 2 years, TBI in children 2 years and older, cervical spine injury, and intra-abdominal injury) from deidentified emergency department physician notes.

## Data availability

This repository contains **code only**. No patient-level data, clinical notes, LLM extraction outputs, or review workbooks are included. The data come from the University of California, San Francisco (UCSF) Clinical Data Warehouse and are not publicly available because of institutional data governance restrictions. Researchers may request access by contacting the corresponding author.

The scripts expect these inputs, which are not distributed:

| Path | Contents |
|---|---|
| `cohort/` | Cohort pulls, exclusion lists, and full-cohort LLM eligibility and extraction outputs |
| `outputs/` | LLM inference caches and aggregated extraction CSVs written by the pipeline scripts |
| `review_documents/` | Validation workbooks: LLM extractions, each expert reviewer's extractions, consensus, physician-adjudicated consensus, reference standard, and the concordance audit sample |
| `data/development/` | Development-cohort eligibility labels used for Supplemental Table 1 (`tbi_inclusion_younger.csv`, `tbi_inclusion_older.csv`, `iai_inclusion.csv`, `cspine_inclusion.csv`) |
| `local/manual_encounter_lists.json` | Encounter keys excluded or reclassified after manual chart review. The structure is given in `local/manual_encounter_lists.example.json`. |

## Layout

```
prompts/extraction/         Final predictor extraction prompts (Supplemental Data 1)
prompts/inclusion/          Injury-specific eligibility prompts
march/helpers.py            Note trimming and function-calling JSON schemas
march/rule_calc.py          PECARN rule calculation from extracted predictors
valid/versa.py              LLM inference client (3 replicates per note, majority vote)
march/                      Cohort preparation, eligibility screening, extraction, ablation
march/publication/          Validation analyses for the main text and supplement
revision/                   Analyses added during peer review
```

## Results and the code that produces them

| Manuscript item | Script |
|---|---|
| Rule calculation (all analyses) | `march/rule_calc.py` |
| Table 2: weighted rule-level performance and noninferiority (7.5% margin) | `march/publication/comparative_performance_adj.py` |
| Table 2, footnote c: conservative zero-event lower bound for IAI sensitivity | `revision/reviewer_ci/design_analysis.py` |
| Figure 2: confusion matrices | `march/publication/comparative_performance_adj.py` |
| Table 3: predictor-level accuracy and noninferiority; LLM sensitivity for substantial head injury | `march/publication/comparative_performance_adj.py` |
| Table 1 and Table 4: disagreement and correction counts; construction of the review workbooks | `march/08_discordance.ipynb` |
| Concordance audit sample (135 extractions) | `march/publication/agreement_review.py` |
| Ablation analysis | `march/11_counterfactual.py` |
| Supplemental Table 1: eligibility screening | `march/publication/inclusion.py` |
| Supplemental Table 2: Cohen's κ with 95% CIs; reference-standard prevalence | `revision/analyze_table3_prevalence_kappa.py` |
| Supplemental Table 4: sensitivity analysis with the physician-adjudicated consensus as reference | `revision/build_reference_standard_supp_table.py` |
| Supplemental Figure 1: analytic cohorts and sampling weights | `march/publication/flowchart.py` |
| Supplemental Data 1: prompts and schemas | `prompts/`, `march/helpers.py` |

## Pipeline

1. `sql prompt` (available by request): pull pediatric ED visits with injury-related chief complaints, the longest ED physician note per visit, ICD-10 groups, and exclusion diagnoses.
2. `march/00_prepare.py`: apply disposition exclusions and trim notes for the development cohort.
3. `march/05_full_inclusion.py`: run the eligibility prompts on the full cohort.
4. `march/06_full_extraction.py`: run the extraction prompts (gpt-5-mini-2025-08-07, three replicates per note).
5. `march/publication/06_consolidate.py`: aggregate the three replicates by majority vote.
6. `march/publication/flowchart.py` (`create_final_dfs`): apply rule-specific exclusions and calculate LLM risk categories, which define the sampling strata.
7. Validation sampling: 50 visits per LLM-predicted risk stratum (low and high for TBI and IAI; low, moderate, and high for CSI), excluding development-cohort visits.
8. `march/08_discordance.ipynb`, then the analysis scripts listed above.

## Running

Install the package from the repository root so that `march` and `valid` are importable:

```bash
pip install -e .
```

Dependencies: pandas, numpy, scikit-learn, statsmodels, matplotlib, openpyxl, python-docx, openai, pydantic, python-dotenv, sqlitedict, tqdm, typed-argument-parser.

LLM inference uses an Azure OpenAI–compatible endpoint. Set the credentials in a `.env` file at the repository root (it is ignored by git):

```
API_KEY=...
RESOURCE_ENDPOINT=...
```

Run the Python scripts from the repository root, for example `python march/publication/comparative_performance_adj.py`. `comparative_performance_adj.py` imports `rule_calc` directly, so add `march/` to the path when running it (`PYTHONPATH=march`). Run `march/08_discordance.ipynb` with `march/` as the working directory; its paths are relative to that folder.
