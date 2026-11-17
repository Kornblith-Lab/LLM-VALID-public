from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COHORT_DIR = ROOT / "cohort"
REVIEW_DIR = ROOT / "review_documents"
ARTIFACTS_DIR = ROOT / "artifacts"
PUBLICATION_OUTPUT_DIR = ROOT / "march" / "publication" / "outputs"


def read_csv(*parts: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ROOT.joinpath(*parts), **kwargs)


def read_excel(*parts: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(ROOT.joinpath(*parts), **kwargs)


def review_workbook_path(filename: str) -> Path:
    return REVIEW_DIR / filename


def cohort_path(filename: str) -> Path:
    return COHORT_DIR / filename


def exclusion_path(filename: str) -> Path:
    return COHORT_DIR / "exclusion" / filename


def inclusion_extraction_path(filename: str) -> Path:
    return COHORT_DIR / "inclusion_extraction" / filename


def load_tbi_older_combined() -> pd.DataFrame:
    p1 = inclusion_extraction_path("extraction_p1_TBI_OLDER_apr_23.csv")
    p2 = inclusion_extraction_path("extraction_p2_TBI_OLDER_apr_23.csv")
    return pd.concat([pd.read_csv(p1), pd.read_csv(p2)], ignore_index=True)


def revision_output_dir() -> Path:
    return ARTIFACTS_DIR / "revisions"

