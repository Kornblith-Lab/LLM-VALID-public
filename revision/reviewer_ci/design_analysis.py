"""Reproduce design-weighted CPR metrics and zero-event bounds for reviewer response."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "march"))
from rule_calc import run_cspine_rule, run_tbi_older_rule, run_tbi_younger_rule, run_iai_rule, CSI

RULES = {
    "TBI in younger children": ("tbi_younger", run_tbi_younger_rule),
    "TBI in older children": ("tbi_older", run_tbi_older_rule),
    "Cervical spine injury": ("cspine", run_cspine_rule),
    "Intra-abdominal injury": ("iai", run_iai_rule),
}

def load_book(path, suffix=""):
    out = {}
    xl = pd.ExcelFile(path)
    for short in ["iai", "cspine", "tbi_older", "tbi_younger"]:
        name = {"iai":"IAI", "cspine":"CSPINE", "tbi_older":"TBI OLDER", "tbi_younger":"TBI YOUNGER"}[short] + suffix
        raw = pd.read_excel(xl, name)
        if "EncounterKey" not in raw.columns:
            raw = pd.read_excel(xl, name, skiprows=1)
        raw = raw[raw.exclude.isna()] if "exclude" in raw.columns else raw
        keep = ["EncounterKey"] + [c for c in raw if c.startswith("risk_") and c != "risk_rom"]
        out[short] = raw[keep].copy()
    return out

def binary_rules(book, rule, fun, three_level=False):
    d = book[rule].replace({"Yes (non-severe)":"No", "Suspected":"Yes"}).copy()
    val = d.apply(fun, axis=1)
    if rule == "cspine":
        val = val.replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    return pd.DataFrame({"EncounterKey":d.EncounterKey, "rule":(val == "High risk").astype(int)})

ref = load_book(ROOT / "review_documents/PECARN HUMAN-LLM COOPERATION.xlsx")
llm = load_book(ROOT / "review_documents/PECARN LLM ALONE.xlsx")
multi = ROOT / "review_documents/PECARN MULTIPLE VALIDATION.xlsx"
expert1 = load_book(multi, "")
expert2 = load_book(multi, " LEISHA")

# Population sizes implied by Table 2's published IPSW footnote. CSI's positive
# population combines the LLM moderate- and high-risk strata (100 sampled visits).
N_BY_STRATUM = {
    "tbi_younger": {0: 2489, 1: 2709},
    "tbi_older": {0: 11633, 1: 4139},
    "cspine": {0: 3152, 1: 2199},
    "iai": {0: 9995, 1: 2297},
}

def calc_metrics(df, pred):
    y, p, w = df.ref.to_numpy(), df[pred].to_numpy(), df.w.to_numpy()
    def ratio(num, den): return float(np.sum(w * num) / np.sum(w * den))
    return {
        "accuracy": ratio((y == p), np.ones(len(y))),
        "sensitivity": ratio((y == 1) & (p == 1), y == 1),
        "specificity": ratio((y == 0) & (p == 0), y == 0),
    }

def bootstrap(df, pred, reps=20000, seed=20261027):
    rng = np.random.default_rng(seed)
    pieces = [g.index.to_numpy() for _, g in df.groupby("stratum", sort=True)]
    vals = {m: [] for m in ["accuracy", "sensitivity", "specificity"]}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(x, len(x), replace=True) for x in pieces])
        res = calc_metrics(df.loc[idx].reset_index(drop=True), pred)
        for m,v in res.items(): vals[m].append(v)
    return {m: [float(np.quantile(vals[m], .025)), float(np.quantile(vals[m], .975))] for m in vals}

def expert_mean_metrics(df):
    a, b = calc_metrics(df, "expert_1"), calc_metrics(df, "expert_2")
    return {m: (a[m] + b[m]) / 2 for m in a}

def bootstrap_expert_mean(df, reps=20000, seed=20261027):
    rng = np.random.default_rng(seed)
    pieces = [g.index.to_numpy() for _, g in df.groupby("stratum", sort=True)]
    vals = {m: [] for m in ["accuracy", "sensitivity", "specificity"]}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(x, len(x), replace=True) for x in pieces])
        res = expert_mean_metrics(df.loc[idx].reset_index(drop=True))
        for m, v in res.items(): vals[m].append(v)
    return {m: [float(np.quantile(vals[m], .025)), float(np.quantile(vals[m], .975))] for m in vals}

results, counts, zero_bounds = {}, [], []
for label, (rule, fun) in RULES.items():
    yref = binary_rules(ref, rule, fun)
    yllm = binary_rules(llm, rule, fun).rename(columns={"rule":"llm"})
    y1 = binary_rules(expert1, rule, fun).rename(columns={"rule":"expert_1"})
    y2 = binary_rules(expert2, rule, fun).rename(columns={"rule":"expert_2"})
    d = yref.rename(columns={"rule":"ref"}).merge(yllm).merge(y1).merge(y2)
    d["expert_mean"] = (d.expert_1 + d.expert_2) / 2
    # Sampling strata are the LLM's original risk categories. For CSI, low is
    # stratum 0 and the sampled moderate/high visits form the single positive stratum.
    d["stratum"] = d.llm
    ns = d.groupby("stratum").size().to_dict()
    assert ns == ({0:50, 1:100} if rule == "cspine" else {0:50, 1:50}), (label, ns)
    d["w"] = d.stratum.map({k: N_BY_STRATUM[rule][k]/v for k,v in ns.items()})
    assert d.w.notna().all()
    counts.append({"Rule":label, "Reference positive":int(d.ref.sum()), "Reference negative":int((1-d.ref).sum()), "LLM missed positives":int(((d.ref == 1)&(d.llm == 0)).sum())})
    results[label] = {
        "llm": {"estimate":calc_metrics(d, "llm"), "ci":bootstrap(d, "llm")},
        "expert_mean": {"estimate":expert_mean_metrics(d), "ci":bootstrap_expert_mean(d)},
    }
    zero = d[(d.stratum == 0) & (d.ref == 1)]
    if len(zero) == 0:
        n = int((d.stratum == 0).sum()); N=N_BY_STRATUM[rule][0]
        # Exact one-sided 95% binomial upper confidence bound for an unknown
        # reference-positive prevalence after observing 0/ n.
        upper = float(1 - .05 ** (1 / n))
        estimated_observed_positives = float((d[(d.stratum==1)&(d.ref==1)].w).sum())
        max_missed = upper*N
        lower_sens = estimated_observed_positives/(estimated_observed_positives + max_missed)
        zero_bounds.append({"Rule":label, "sampled LLM-low visits":n, "observed reference positives":0, "one-sided 95% upper prevalence":upper, "eligible LLM-low visits":N, "upper bound missed positives":max_missed, "conservative sensitivity lower bound":lower_sens})

output = {"design_bootstrap":results, "unweighted_counts":counts, "zero_event_bounds":zero_bounds}
(ROOT / "revision/reviewer_ci/design_analysis.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
