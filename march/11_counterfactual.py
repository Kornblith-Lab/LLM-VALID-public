import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from march.helpers import PECARN_TBI_OLDER, PECARN_TBI_YOUNGER, PECARN_IAI, PECARN_CSPINE
from march.rule_calc import run_cspine_rule, run_tbi_older_rule, run_tbi_younger_rule, run_iai_rule, CSI
import valid.versa as versa

EXCEL_PATH = "review_documents/PECARN HUMAN-LLM COOPERATION.xlsx"

SHEET_CONFIG = {
    "TBI YOUNGER": {
        "prompt": "prompts/extraction_trimmed/tbi_younger.txt",
        "schema": PECARN_TBI_YOUNGER,
        "rule_fn": run_tbi_younger_rule,
        "risk_cols": ["risk_loc", "risk_skullfx", "risk_altered", "risk_hematoma", "risk_mechanism", "risk_baseline"],
    },
    "TBI OLDER": {
        "prompt": "prompts/extraction_trimmed/tbi_older.txt",
        "schema": PECARN_TBI_OLDER,
        "rule_fn": run_tbi_older_rule,
        "risk_cols": ["risk_mechanism", "risk_headache", "risk_vomit", "risk_loc", "risk_sf", "risk_altered"],
    },
    "CSPINE": {
        "prompt": "prompts/extraction_trimmed/cspine.txt",
        "schema": PECARN_CSPINE,
        "rule_fn": run_cspine_rule,
        "risk_cols": ["risk_gcs", "risk_mentalstatus", "risk_abc", "risk_focal", "risk_pain", "risk_tenderness", "risk_headinj", "risk_torsoinj"],
    },
    "IAI": {
        "prompt": "prompts/extraction_trimmed/iai.txt",
        "schema": PECARN_IAI,
        "rule_fn": run_iai_rule,
        "risk_cols": ["risk_gcs", "risk_vomit", "risk_breathing", "risk_abdpain", "risk_walltrauma", "risk_chestwall", "risk_tenderness"],
    },
}


def load_sheet(sheet_name):
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
    if "exclude" in df.columns:
        df = df[df.exclude.isna()]
    return df


def rename_gt_columns(df, risk_cols):
    rename_map = {col: f"gt_{col}" for col in risk_cols if col in df.columns}
    return df.rename(columns=rename_map)


def compute_rule(df, rule_fn, risk_cols, prefix=""):
    src_cols = [f"{prefix}{col}" for col in risk_cols]
    temp = df[src_cols].copy()
    temp.columns = risk_cols
    temp = temp.replace("Yes (non-severe)", "No").replace("Suspected", "Yes")
    temp["rule"] = temp.apply(rule_fn, axis=1)
    if rule_fn == run_cspine_rule:
        temp["rule"] = temp["rule"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    return (temp["rule"] == "High risk").astype(int)


def run_extraction(df, prompt, schema, sheet_name):
    output_name = sheet_name.replace(" ", "_").upper()
    cache_path = f"outputs/counterfactual_{output_name}.csv"

    import os
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        return pd.read_csv(cache_path)

    config = versa.ExtractionConfig(
        df=df,
        prompt_file=prompt,
        tool_calling=True,
        llm="gpt-5-mini-2025-08-07",
        response_schema=schema,
        n_replicates=3,
        n_workers=250,
    )
    modu = versa.Extraction(config)
    df_result = modu.extract(f"outputs/counterfactual_{output_name}")
    return df_result


results = {}

for sheet_name, cfg in SHEET_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"Processing: {sheet_name}")
    print(f"{'='*60}")

    df = load_sheet(sheet_name)
    risk_cols = cfg["risk_cols"]

    df_for_extraction = rename_gt_columns(df, risk_cols)

    df_result = run_extraction(df_for_extraction, cfg["prompt"], cfg["schema"], sheet_name)

    gt_rule = compute_rule(df_result, cfg["rule_fn"], risk_cols, prefix="gt_")
    predicted_rule = compute_rule(df_result, cfg["rule_fn"], risk_cols, prefix="")

    df_result["gt_rule"] = gt_rule
    df_result["predicted_rule"] = predicted_rule

    results[sheet_name] = df_result

# --- Confusion matrices ---

print(f"\n{'='*60}")
print("CONFUSION MATRICES & METRICS")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, (sheet_name, cfg) in enumerate(SHEET_CONFIG.items()):
    df_result = results[sheet_name]
    y_true = df_result["gt_rule"].values
    y_pred = df_result["predicted_rule"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax = axes[i]
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for (ri, ci), val in np.ndenumerate(cm):
        threshold = cm.max() / 2
        color = "white" if cm[ri, ci] > threshold else "black"
        ax.text(ci, ri, str(val), ha="center", va="center", fontsize=16, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low risk", "High risk"])
    ax.set_yticklabels(["Low risk", "High risk"])
    ax.set_title(sheet_name)
    ax.set_xlabel("Predicted")
    if i == 0:
        ax.set_ylabel("Reference")

    # Metrics
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    print(f"\n{sheet_name}:")
    print(f"  Accuracy:    {accuracy*100:.0f}%")
    print(f"  Sensitivity: {sensitivity*100:.0f}%")

plt.tight_layout()
plt.savefig("artifacts/counterfactual_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix figure saved to artifacts/counterfactual_confusion_matrices.png")

# --- Variable-level accuracy CSV and comparison ---

print(f"\n{'='*60}")
print("VARIABLE-LEVEL ACCURACY")
print(f"{'='*60}")

variable_rows = []

for sheet_name, cfg in SHEET_CONFIG.items():
    variable_rows.append({"Variable": sheet_name, "LLM trimmed": ""})
    df_result = results[sheet_name]
    risk_cols = cfg["risk_cols"]

    for col in risk_cols:
        gt_col = f"gt_{col}"
        gt_vals = df_result[gt_col]
        pred_vals = df_result[col]
        if pd.api.types.is_numeric_dtype(gt_vals) or pd.api.types.is_numeric_dtype(pred_vals):
            gt_vals = pd.to_numeric(gt_vals, errors="coerce")
            pred_vals = pd.to_numeric(pred_vals, errors="coerce")
        else:
            gt_vals = gt_vals.astype(str)
            pred_vals = pred_vals.astype(str)
        acc = (gt_vals == pred_vals).mean()
        variable_rows.append({"Variable": col, "LLM trimmed": f"{acc:.2f}"})

variable_table = pd.DataFrame(variable_rows)
variable_table.to_csv("artifacts/outputs/counterfactual_variable_accuracy.csv", index=False)
print("Saved to artifacts/outputs/counterfactual_variable_accuracy.csv")

# --- Comparison with original LLM only accuracy ---

print(f"\n{'='*60}")
print("COMPARISON: LLM only vs LLM trimmed")
print(f"{'='*60}")

original = pd.read_csv("artifacts/outputs/variable_table_accuracy.csv")

comparison_rows = []
for _, row in variable_table.iterrows():
    var = row["Variable"]
    trimmed = row["LLM trimmed"]
    if trimmed == "":
        comparison_rows.append({"Variable": var, "LLM only": "", "LLM trimmed": "", "Difference": ""})
        continue

    match = original[original["Variable"] == var]
    if match.empty:
        comparison_rows.append({"Variable": var, "LLM only": "", "LLM trimmed": trimmed, "Difference": ""})
        continue

    llm_only_str = match.iloc[0]["LLM only"]
    llm_only_point = float(llm_only_str.split(" ")[0]) if llm_only_str else float("nan")
    trimmed_point = float(trimmed)
    diff = trimmed_point - llm_only_point
    comparison_rows.append({
        "Variable": var,
        "LLM only": f"{llm_only_point:.2f}",
        "LLM trimmed": trimmed,
        "Difference": f"{diff:+.2f}",
    })

comparison_table = pd.DataFrame(comparison_rows)
comparison_table.to_csv("artifacts/outputs/counterfactual_comparison.csv", index=False)
print(comparison_table.to_string(index=False))

diffs = comparison_table["Difference"][comparison_table["Difference"] != ""].apply(float)
n_decrease = (diffs <= -0.05).sum()
n_increase = (diffs >= 0.05).sum()
print(f"\nVariables with >=5% decrease in accuracy: {n_decrease}")
print(f"Variables with >=5% increase in accuracy: {n_increase}")

print("\nSaved to artifacts/outputs/counterfactual_comparison.csv")

# --- Rule-level IPW-weighted accuracy difference with bootstrap CIs ---

print(f"\n{'='*60}")
print("RULE-LEVEL ACCURACY DIFFERENCE (IPW-weighted): LLM only vs LLM trimmed")
print(f"{'='*60}")

LLM_ALONE_PATH = "review_documents/PECARN LLM ALONE.xlsx"
N_BOOT = 10_000
RNG = np.random.default_rng(42)

FULL_EXTRACTION_PATHS = {
    "TBI YOUNGER": "outputs/extraction_TBI_YOUNGER_apr_23.csv",
    "TBI OLDER": "outputs/extraction_combined_TBI_OLDER_apr_23.csv",
    "CSPINE": "outputs/extraction_v2_CSPINE_apr_23.csv",
    "IAI": "outputs/extraction_IAI_apr_23.csv",
}


def compute_ipw_weights(sheet_name, rule_fn, llm_strata):
    """Compute IPW weights from population rule counts vs sample strata."""
    pop_df = pd.read_csv(FULL_EXTRACTION_PATHS[sheet_name])
    pop_df = pop_df.replace("Suspected", "Yes").replace("Yes (non-severe)", "No")
    pop_df["rule"] = pop_df.apply(rule_fn, axis=1)
    if rule_fn == run_cspine_rule:
        pop_df["rule"] = pop_df["rule"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    pop_df = pop_df.dropna(subset=["rule"])
    pop_counts = pop_df["rule"].value_counts()
    pop_high = pop_counts.get("High risk", 0)
    pop_low = pop_counts.get("Low risk", 0)

    llm_strata = np.asarray(llm_strata)
    sampled_high = (llm_strata == 1).sum()
    sampled_low = (llm_strata == 0).sum()

    w_high = pop_high / sampled_high
    w_low = pop_low / sampled_low

    print(f"  IPW: w_high={w_high:.2f} (pop={pop_high}, sampled={sampled_high}), w_low={w_low:.2f} (pop={pop_low}, sampled={sampled_low})")
    return np.where(llm_strata == 1, w_high, w_low)


def weighted_accuracy(y_true, y_pred, weights):
    return np.average((y_true == y_pred).astype(float), weights=weights)


llm_alone_xl = pd.ExcelFile(LLM_ALONE_PATH)

for sheet_name, cfg in SHEET_CONFIG.items():
    risk_cols = cfg["risk_cols"]
    rule_fn = cfg["rule_fn"]

    # LLM only predictions
    llm_df = pd.read_excel(llm_alone_xl, sheet_name=sheet_name)
    llm_df = llm_df[["EncounterKey"] + [c for c in risk_cols if c in llm_df.columns]]
    llm_df = llm_df.replace("Yes (non-severe)", "No").replace("Suspected", "Yes")
    llm_df["rule_llm_only"] = llm_df[risk_cols].apply(rule_fn, axis=1)
    if rule_fn == run_cspine_rule:
        llm_df["rule_llm_only"] = llm_df["rule_llm_only"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    llm_df["rule_llm_only"] = (llm_df["rule_llm_only"] == "High risk").astype(int)

    # LLM trimmed predictions
    df_result = results[sheet_name]
    trimmed_df = df_result[["EncounterKey"] + risk_cols].copy()
    trimmed_df = trimmed_df.replace("Yes (non-severe)", "No").replace("Suspected", "Yes")
    trimmed_df["rule_trimmed"] = trimmed_df[risk_cols].apply(rule_fn, axis=1)
    if rule_fn == run_cspine_rule:
        trimmed_df["rule_trimmed"] = trimmed_df["rule_trimmed"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    trimmed_df["rule_trimmed"] = (trimmed_df["rule_trimmed"] == "High risk").astype(int)

    # GT from the reference
    gt_df = df_result[["EncounterKey"] + [f"gt_{col}" for col in risk_cols]].copy()
    gt_df.columns = ["EncounterKey"] + risk_cols
    gt_df = gt_df.replace("Yes (non-severe)", "No").replace("Suspected", "Yes")
    gt_df["rule_gt"] = gt_df[risk_cols].apply(rule_fn, axis=1)
    if rule_fn == run_cspine_rule:
        gt_df["rule_gt"] = gt_df["rule_gt"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    gt_df["rule_gt"] = (gt_df["rule_gt"] == "High risk").astype(int)

    # Merge on EncounterKey to align rows
    merged = gt_df[["EncounterKey", "rule_gt"]].merge(
        llm_df[["EncounterKey", "rule_llm_only"]], on="EncounterKey"
    ).merge(
        trimmed_df[["EncounterKey", "rule_trimmed"]], on="EncounterKey"
    )

    gt_arr = merged["rule_gt"].values
    llm_arr = merged["rule_llm_only"].values
    trim_arr = merged["rule_trimmed"].values

    # IPW weights based on LLM-only rule (the sampling stratum)
    weights = compute_ipw_weights(sheet_name, rule_fn, llm_arr)

    acc_llm = weighted_accuracy(gt_arr, llm_arr, weights)
    acc_trim = weighted_accuracy(gt_arr, trim_arr, weights)
    point_diff = acc_trim - acc_llm

    # Stratified bootstrap CI on difference
    strata = llm_arr
    unique_strata = np.unique(strata)
    idx_by_stratum = {s: np.where(strata == s)[0] for s in unique_strata}

    boot_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot_idx = np.concatenate([
            RNG.choice(s_idx, size=len(s_idx), replace=True)
            for s_idx in idx_by_stratum.values()
        ])
        boot_acc_llm = weighted_accuracy(gt_arr[boot_idx], llm_arr[boot_idx], weights[boot_idx])
        boot_acc_trim = weighted_accuracy(gt_arr[boot_idx], trim_arr[boot_idx], weights[boot_idx])
        boot_diffs[i] = boot_acc_trim - boot_acc_llm

    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)

    print(f"\n{sheet_name}:")
    print(f"  LLM only accuracy (IPW):   {acc_llm*100:.0f}%")
    print(f"  LLM trimmed accuracy (IPW): {acc_trim*100:.0f}%")
    print(f"  Difference (trimmed - only): {point_diff*100:+.0f}% (95% CI: {ci_lo*100:+.0f}% to {ci_hi*100:+.0f}%)")
