from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision" / "table3_prevalence_kappa.json"
N_BOOT = 2_000
RNG = np.random.default_rng(20261026)

COHORTS = [
    ("Traumatic brain injury in younger (n=100)", "TBI YOUNGER", [
        ("Loss of consciousness", "risk_loc", "Yes"),
        ("Palpable skull fracture", "risk_skullfx", "Yes"),
        ("Altered mental status", "risk_altered", "Yes"),
        ("Non-frontal hematoma", "risk_hematoma", "Yes"),
        ("Severe mechanism of injury", "risk_mechanism", "Yes"),
        ("Acting abnormally per caregiver", "risk_baseline", "No"),
    ]),
    ("Traumatic brain injury in older (n=100)", "TBI OLDER", [
        ("Severe mechanism of injury", "risk_mechanism", "Yes"),
        ("Severe headache", "risk_headache", "Yes"),
        ("Vomiting", "risk_vomit", "Yes"),
        ("Loss of consciousness", "risk_loc", ("Yes", "Suspected")),
        ("Signs of basilar skull fracture", "risk_sf", "Yes"),
        ("Altered mental status", "risk_altered", "Yes"),
    ]),
    ("Cervical spine injury (n=150)", "CSPINE", [
        ("Glasgow Coma Scale score", "risk_gcs", lambda s: s < 15),
        ("Mental status (AVPU)", "risk_mentalstatus", lambda s: s != "Alert"),
        ("ABC abnormalities", "risk_abc", "Yes"),
        ("Focal neurological deficits", "risk_focal", "Yes"),
        ("Midline neck tenderness", "risk_tenderness", "Yes"),
        ("Midline neck pain", "risk_pain", "Yes"),
        ("Substantial head injury", "risk_headinj", "Yes"),
        ("Substantial torso injury", "risk_torsoinj", "Yes"),
    ]),
    ("Intra-abdominal injury (n=100)", "IAI", [
        ("Glasgow Coma Scale score", "risk_gcs", lambda s: s < 14),
        ("Vomiting", "risk_vomit", "Yes"),
        ("Decreased breath sounds", "risk_breathing", "Yes"),
        ("Abdominal pain", "risk_abdpain", "Yes"),
        ("Abdominal wall trauma", "risk_walltrauma", "Yes"),
        ("Chest wall trauma", "risk_chestwall", "Yes"),
        ("Abdominal tenderness", "risk_tenderness", "Yes"),
    ]),
]


def load(path: str, suffix: str = ""):
    book = ROOT / "review_documents" / path
    result = {}
    for _label, sheet, variables in COHORTS:
        raw = pd.read_excel(book, sheet_name=sheet + suffix)
        df = pd.read_excel(book, sheet_name=sheet + suffix, skiprows=1 if "EncounterKey" not in raw.columns else None)
        if "exclude" in df.columns:
            df = df[df.exclude.isna()]
        cols = ["EncounterKey"] + [col for _display, col, _positive in variables]
        result[sheet] = df[cols].copy()
    return result


def kappa_ci(reference, method, paired=False):
    reference, method = canonical_labels(reference, method)
    point = kappa(reference, method)
    # Re-sample within reference categories. This preserves rare categories so
    # bootstrap kappa is defined even for predictors with only one positive case.
    groups = [np.flatnonzero(reference == category) for category in np.unique(reference)]
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([RNG.choice(group, len(group), replace=True) for group in groups])
        boot[i] = kappa(reference[idx], method[idx])
    lo, hi = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    if np.isclose(lo, hi):
        lo, hi = transformed_wilson_kappa_ci(reference, method)
    return [point, lo, hi]


def kappa_ci_human(reference, method, keys):
    reference, method = canonical_labels(reference, method)
    keys = np.asarray(keys)
    point = kappa(reference, method)
    unique = np.unique(keys)
    # Clustered, reference-stratified re-sampling retains both reviewers for a
    # visit and avoids degenerate kappa samples for rare reference categories.
    key_to_index = {key: np.flatnonzero(keys == key) for key in unique}
    key_ref = {key: reference[key_to_index[key][0]] for key in unique}
    groups = [np.array([key for key in unique if key_ref[key] == category]) for category in np.unique(list(key_ref.values()))]
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_keys = np.concatenate([RNG.choice(group, len(group), replace=True) for group in groups])
        idx = np.concatenate([key_to_index[key] for key in sampled_keys])
        boot[i] = kappa(reference[idx], method[idx])
    lo, hi = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    if np.isclose(lo, hi):
        lo, hi = transformed_wilson_kappa_ci(reference, method)
    return [point, lo, hi]


def count_positive(series, definition):
    if callable(definition):
        return int(definition(series).sum())
    if isinstance(definition, tuple):
        return int(series.isin(definition).sum())
    return int((series == definition).sum())


def kappa(reference, method):
    """Unweighted Cohen kappa, matching sklearn's nominal-category calculation."""
    labels = np.union1d(reference, method)
    agreement = float(np.mean(reference == method))
    expected = sum(float(np.mean(reference == label) * np.mean(method == label)) for label in labels)
    if np.isclose(1 - expected, 0):
        return 1.0 if np.isclose(agreement, 1) else float("nan")
    return (agreement - expected) / (1 - expected)


def transformed_wilson_kappa_ci(reference, method):
    """Wilson CI for observed agreement, transformed to kappa with fixed marginals."""
    reference, method = canonical_labels(reference, method)
    n = len(reference)
    observed = float(np.mean(reference == method))
    labels = np.union1d(reference, method)
    expected = sum(float(np.mean(reference == label) * np.mean(method == label)) for label in labels)
    z = 1.95996398454
    den = 1 + z * z / n
    center = (observed + z * z / (2 * n)) / den
    half = z * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / den
    if np.isclose(1 - expected, 0):
        return -1.0, 1.0
    lo = max(-1.0, min(1.0, (center - half - expected) / (1 - expected)))
    hi = max(-1.0, min(1.0, (center + half - expected) / (1 - expected)))
    return float(lo), float(hi)


def canonical_labels(reference, method):
    """Use a shared numeric representation when the variable is numeric."""
    ref = pd.Series(reference)
    pred = pd.Series(method)
    ref_num = pd.to_numeric(ref, errors="coerce")
    pred_num = pd.to_numeric(pred, errors="coerce")
    if ref_num.notna().all() and pred_num.notna().all():
        return ref_num.to_numpy(dtype=float), pred_num.to_numpy(dtype=float)
    return ref.astype(str).to_numpy(), pred.astype(str).to_numpy()


def main():
    reference = load("PECARN HUMAN-LLM COOPERATION.xlsx")
    llm = load("PECARN LLM ALONE.xlsx")
    reviewer_1 = load("PECARN MULTIPLE VALIDATION.xlsx")
    reviewer_2 = load("PECARN MULTIPLE VALIDATION.xlsx", " LEISHA")
    physician = load("PECARN THREE HUMAN CONSENSUS.xlsx", " HUMAN CONSENSUS")
    all_rows = []
    for cohort, sheet, variables in COHORTS:
        ref = reference[sheet]
        for display, variable, definition in variables:
            llm_m = ref[["EncounterKey", variable]].merge(llm[sheet][["EncounterKey", variable]], on="EncounterKey", suffixes=("_ref", "_method"))
            physician_m = ref[["EncounterKey", variable]].merge(physician[sheet][["EncounterKey", variable]], on="EncounterKey", suffixes=("_ref", "_method"))
            h1 = ref[["EncounterKey", variable]].merge(reviewer_1[sheet][["EncounterKey", variable]], on="EncounterKey", suffixes=("_ref", "_method"))
            h2 = ref[["EncounterKey", variable]].merge(reviewer_2[sheet][["EncounterKey", variable]], on="EncounterKey", suffixes=("_ref", "_method"))
            human_m = pd.concat([h1, h2], ignore_index=True)
            positive = count_positive(ref[variable], definition)
            row = {
                "cohort": cohort,
                "variable": display,
                "source_variable": variable,
                "n": int(len(ref)),
                "positive": positive,
                "prevalence": positive / len(ref),
                "llm": kappa_ci(llm_m[f"{variable}_ref"], llm_m[f"{variable}_method"]),
                "human": kappa_ci_human(human_m[f"{variable}_ref"], human_m[f"{variable}_method"], human_m.EncounterKey),
                "physician": kappa_ci(physician_m[f"{variable}_ref"], physician_m[f"{variable}_method"]),
            }
            if sheet == "TBI OLDER" and variable == "risk_headache":
                row["prevalence_note"] = "Severe: 5/100 (5%); non-severe: 25/100 (25%)"
            elif sheet == "TBI OLDER" and variable == "risk_loc":
                row["prevalence_note"] = "Yes: 11/100 (11%); suspected: 8/100 (8%)"
            elif sheet == "CSPINE" and variable == "risk_gcs":
                row["prevalence_note"] = "GCS <15: 27/150 (18%); GCS <=8: 10/150 (7%)"
            elif sheet == "CSPINE" and variable == "risk_mentalstatus":
                row["prevalence_note"] = "Non-alert: 16/150 (11%)"
            else:
                row["prevalence_note"] = f"{positive}/{len(ref)} ({positive / len(ref):.0%})"
            all_rows.append(row)
    OUT.write_text(json.dumps(all_rows, indent=2))
    for row in all_rows:
        print(row["cohort"], row["variable"], row["prevalence_note"], row["llm"], row["human"], row["physician"])


if __name__ == "__main__":
    main()
