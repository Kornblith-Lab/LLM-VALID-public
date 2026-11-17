from dataclasses import dataclass
from rule_calc import run_cspine_rule, run_tbi_older_rule, run_tbi_younger_rule, run_iai_rule, CSI

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

@dataclass
class Sheets:
    iai: pd.DataFrame
    tbi_older: pd.DataFrame
    tbi_younger: pd.DataFrame
    cspine: pd.DataFrame

    @classmethod
    def from_excel(cls, excel_path, suffix_override=None):
        xl = pd.ExcelFile(excel_path)

        extra_name = " HUMAN CONSENSUS" if " HUMAN CONSENSUS" in str(xl.sheet_names[0]) else ""

        if type(suffix_override) == str:
            extra_name = "" if suffix_override == "kush" else " LEISHA"

        skip_row = 1 if "EncounterKey" not in pd.read_excel(xl, f"IAI{extra_name}").columns else None

        df_iai = pd.read_excel(xl, f"IAI{extra_name}", skiprows=skip_row)
        df_cspine = pd.read_excel(xl, f"CSPINE{extra_name}", skiprows=skip_row)
        df_tbi_older = pd.read_excel(xl, f"TBI OLDER{extra_name}", skiprows=skip_row)
        df_tbi_younger = pd.read_excel(xl, f"TBI YOUNGER{extra_name}", skiprows=skip_row)

        df_iai, df_cspine, df_tbi_older, df_tbi_younger = [
            (
                d[d.exclude.isna()] if "exclude" in d.columns else d
            )[["EncounterKey"] + [c for c in d.columns if c.startswith("risk_")]].drop(columns=["risk_rom"], errors="ignore")
            for d in [df_iai, df_cspine, df_tbi_older, df_tbi_younger]
        ]

        assert len(df_iai) == 100 and len(df_cspine) == 150 and len(df_tbi_older) == 100 and len(df_tbi_older) == 100

        return cls(
            iai=df_iai,
            tbi_older=df_tbi_older,
            tbi_younger=df_tbi_younger,
            cspine=df_cspine,
        )

reference_standard = Sheets.from_excel("review_documents/PECARN HUMAN-LLM COOPERATION.xlsx")
best_case_reference_standard = Sheets.from_excel("review_documents/PECARN HUMAN-LLM COOPERATION-BEST CASE.xlsx")

llm_alone = Sheets.from_excel("review_documents/PECARN LLM ALONE.xlsx")
two_humans = Sheets.from_excel("review_documents/PECARN HUMAN CONSENSUS.xlsx")
three_humans = Sheets.from_excel("review_documents/PECARN THREE HUMAN CONSENSUS.xlsx")

kush = Sheets.from_excel("review_documents/PECARN MULTIPLE VALIDATION.xlsx", suffix_override="kush")
leisha = Sheets.from_excel("review_documents/PECARN MULTIPLE VALIDATION.xlsx", suffix_override="leisha")

human_avg = Sheets(
    iai=pd.concat([kush.iai, leisha.iai], ignore_index=True),
    tbi_older=pd.concat([kush.tbi_older, leisha.tbi_older], ignore_index=True),
    tbi_younger=pd.concat([kush.tbi_younger, leisha.tbi_younger], ignore_index=True),
    cspine=pd.concat([kush.cspine, leisha.cspine], ignore_index=True),
)

# --- Configuration ---

os.makedirs("artifacts/outputs", exist_ok=True)

methods = {
    # "Kush": kush,
    # "Leisha": leisha,
    "LLM only": llm_alone,
    "Human avg": human_avg,
    "Two humans": two_humans,
    "Three humans": three_humans,
}

cohorts = [
    ("TBI younger", "tbi_younger",
     ["risk_loc", "risk_skullfx", "risk_altered", "risk_hematoma", "risk_mechanism", "risk_baseline"]),
    ("TBI older", "tbi_older",
     ["risk_mechanism", "risk_headache", "risk_vomit", "risk_loc", "risk_sf", "risk_altered"]),
    ("CSPINE", "cspine",
     ["risk_gcs", "risk_mentalstatus", "risk_abc", "risk_focal", "risk_tenderness", "risk_pain", "risk_headinj", "risk_torsoinj"]),
    ("IAI", "iai",
     ["risk_gcs", "risk_vomit", "risk_breathing", "risk_abdpain", "risk_walltrauma", "risk_chestwall", "risk_tenderness"]),
]

rule_fns = {
    "iai": run_iai_rule,
    "tbi_older": run_tbi_older_rule,
    "tbi_younger": run_tbi_younger_rule,
    "cspine": run_cspine_rule,
}

# --- Helper functions ---

N_BOOT = 10_000
RNG = np.random.default_rng(42)

def cluster_bootstrap_diff_ci(keys, y_true, y_pred_a, y_pred_b, metric_fn, alpha=0.05):
    """
    Cluster bootstrap CI for the DIFFERENCE between two methods (Method A - Method B).
    Method A is usually the new method (LLM). Method B is the reference (Human).
    """
    keys = np.asarray(keys)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    unique_keys = np.unique(keys)
    n_clusters = len(unique_keys)
    key_to_idx = {k: np.where(keys == k)[0] for k in unique_keys}

    # Calculate the point estimate of the difference
    point_diff = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    boots_diff = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_keys = RNG.choice(unique_keys, size=n_clusters, replace=True)
        idx = np.concatenate([key_to_idx[k] for k in sampled_keys])
        
        val_a = metric_fn(y_true[idx], y_pred_a[idx])
        val_b = metric_fn(y_true[idx], y_pred_b[idx])
        boots_diff[i] = val_a - val_b

    ci_lo = np.percentile(boots_diff, 100 * alpha / 2)
    ci_hi = np.percentile(boots_diff, 100 * (1 - alpha / 2))

    return point_diff, ci_lo, ci_hi

def cluster_bootstrap_ci(keys, y_true, y_pred, metric_fn, alpha=0.05):
    """Cluster bootstrap CI that resamples by EncounterKey.

    Same cases across reviewers produce correlated estimates; resampling
    individual rows independently understates uncertainty. Resampling
    whole cases (clusters) preserves the pairing structure.

    When each key appears exactly once, this reduces to ordinary bootstrap.
    Falls back to Wilson score interval if bootstrap CI has zero width.
    """
    keys = np.asarray(keys)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_keys = np.unique(keys)
    n_clusters = len(unique_keys)
    key_to_idx = {k: np.where(keys == k)[0] for k in unique_keys}

    point = metric_fn(y_true, y_pred)

    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_keys = RNG.choice(unique_keys, size=n_clusters, replace=True)
        idx = np.concatenate([key_to_idx[k] for k in sampled_keys])
        boots[i] = metric_fn(y_true[idx], y_pred[idx])

    ci_lo = np.percentile(boots, 100 * alpha / 2)
    ci_hi = np.percentile(boots, 100 * (1 - alpha / 2))

    if ci_lo == ci_hi:
        if metric_fn == sensitivity_metric:
            n_total = (y_true == 1).sum()
        elif metric_fn == specificity_metric:
            n_total = (y_true == 0).sum()
        elif metric_fn == accuracy_metric:
            n_total = len(y_true)
        else:
            raise Exception(f"Unknown metric_fn: {metric_fn}")
        n_success = int(round(point * n_total))
        ci_lo, ci_hi = proportion_confint(n_success, n_total, alpha=alpha, method='wilson')

    return point, ci_lo, ci_hi


def accuracy_metric(y_true, y_pred):
    return (y_true == y_pred).mean()


def sensitivity_metric(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def specificity_metric(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def fmt_ci(point, lo, hi):
    return f"{point:.2f} ({lo:.2f}-{hi:.2f})"


def preprocess_for_rules(sheets):
    """Copy sheets and apply value conversions for rule application."""
    return Sheets(
        iai=sheets.iai.copy().replace("Yes (non-severe)", "No").replace("Suspected", "Yes"),
        tbi_older=sheets.tbi_older.copy().replace("Yes (non-severe)", "No").replace("Suspected", "Yes"),
        tbi_younger=sheets.tbi_younger.copy().replace("Yes (non-severe)", "No").replace("Suspected", "Yes"),
        cspine=sheets.cspine.copy().replace("Yes (non-severe)", "No").replace("Suspected", "Yes"),
    )


def compute_rules(sheets_processed, attr_name):
    """Run rule function and return EncounterKey + binary rule column."""
    df = getattr(sheets_processed, attr_name).copy()
    df["rule"] = df.apply(rule_fns[attr_name], axis=1)
    if attr_name == "cspine":
        df["rule"] = df["rule"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    df["rule_binary"] = (df["rule"] == "High risk").astype(int)
    return df[["EncounterKey", "rule_binary"]]


# --- Table 1: Variable-by-variable performance ---


def build_variable_tables(ref_sheets):
    rows_kappa = []
    rows_accuracy = []

    for cohort_name, attr_name, risk_cols in cohorts:
        rows_kappa.append({"Variable": cohort_name, **{m: "" for m in methods}})
        rows_accuracy.append({"Variable": cohort_name, **{m: "" for m in methods}})

        ref_df = getattr(ref_sheets, attr_name)

        for col in risk_cols:
            row_kappa = {"Variable": f"{col}"}
            row_acc = {"Variable": f"{col}"}

            for method_name, method_sheets in methods.items():
                method_df = getattr(method_sheets, attr_name)
                merged = ref_df[["EncounterKey", col]].merge(
                    method_df[["EncounterKey", col]], on="EncounterKey", suffixes=("_ref", "_method")
                )
                assert not merged[f"{col}_ref"].isna().any(), f"NaN in ref {col} for {cohort_name}"
                assert not merged[f"{col}_method"].isna().any(), f"NaN in method {col} for {method_name}/{cohort_name}"

                ref_col = merged[f"{col}_ref"]
                method_col = merged[f"{col}_method"]
                if pd.api.types.is_numeric_dtype(ref_col) or pd.api.types.is_numeric_dtype(method_col):
                    ref_col = ref_col.astype(float)
                    method_col = method_col.astype(float)
                y_true = ref_col.astype(str).values
                y_pred = method_col.astype(str).values

                kappa = cohen_kappa_score(y_true, y_pred)
                row_kappa[method_name] = f"{kappa:.2f}"

                keys = merged["EncounterKey"].values
                match_true = (y_true == y_pred).astype(int)
                ones = np.ones(len(match_true), dtype=int)
                point, ci_lo, ci_hi = cluster_bootstrap_ci(keys, match_true, ones, accuracy_metric)
                row_acc[method_name] = fmt_ci(point, ci_lo, ci_hi)

            rows_kappa.append(row_kappa)
            rows_accuracy.append(row_acc)

    return pd.DataFrame(rows_kappa), pd.DataFrame(rows_accuracy)


print("Computing variable-by-variable tables...")

kappa_table, accuracy_table = build_variable_tables(reference_standard)
kappa_table.to_csv("artifacts/outputs/variable_table_kappa.csv", index=False)
accuracy_table.to_csv("artifacts/outputs/variable_table_accuracy.csv", index=False)

_, accuracy_table_best = build_variable_tables(best_case_reference_standard)
accuracy_table_best.to_csv("artifacts/outputs/variable_table_accuracy_best_case.csv", index=False)
print("  Saved variable tables.")


def build_variable_non_inferiority_table(ref_sheets, margin=0.075):
    rows = []
    n_total = 0
    n_non_inferior = 0

    for cohort_name, attr_name, risk_cols in cohorts:
        rows.append({"Variable": cohort_name, "Difference (LLM - Human)": "", "Non-Inferior?": ""})
        ref_df = getattr(ref_sheets, attr_name)

        for col in risk_cols:
            row = {"Variable": col}

            llm_df = getattr(llm_alone, attr_name)
            human_df = getattr(human_avg, attr_name)

            merged = ref_df[["EncounterKey", col]].merge(
                llm_df[["EncounterKey", col]], on="EncounterKey", suffixes=("_ref", "_llm")
            ).merge(
                human_df[["EncounterKey", col]], on="EncounterKey"
            ).rename(columns={col: f"{col}_human"})

            ref_col = merged[f"{col}_ref"]
            llm_col = merged[f"{col}_llm"]
            human_col = merged[f"{col}_human"]
            if pd.api.types.is_numeric_dtype(ref_col):
                ref_col = ref_col.astype(float)
                llm_col = llm_col.astype(float)
                human_col = human_col.astype(float)
            y_true = ref_col.astype(str).values
            y_pred_llm = llm_col.astype(str).values
            y_pred_human = human_col.astype(str).values

            keys = merged["EncounterKey"].values
            match_llm = (y_true == y_pred_llm).astype(int)
            match_human = (y_true == y_pred_human).astype(int)
            ones = np.ones(len(match_llm), dtype=int)

            pt, lo, hi = cluster_bootstrap_diff_ci(keys, ones, match_llm, match_human, accuracy_metric)

            row["Difference (LLM - Human)"] = f"{pt:+.2f} ({lo:.2f} to {hi:.2f})"

            if np.isnan(lo):
                row["Non-Inferior?"] = "N/A"
            else:
                is_ni = lo > -margin
                row["Non-Inferior?"] = "Yes" if is_ni else "No"
                n_total += 1
                if is_ni:
                    n_non_inferior += 1

            rows.append(row)

    rows.append({"Variable": "", "Difference (LLM - Human)": "", "Non-Inferior?": ""})
    rows.append({"Variable": "Summary", "Difference (LLM - Human)": f"{n_non_inferior}/{n_total} variables non-inferior", "Non-Inferior?": f"{n_non_inferior/n_total*100:.0f}%"})

    return pd.DataFrame(rows)


print("Computing variable-level non-inferiority...")
var_ni_table = build_variable_non_inferiority_table(reference_standard, margin=0.075)
var_ni_table.to_csv("artifacts/outputs/variable_non_inferiority_table.csv", index=False)

var_ni_table_best = build_variable_non_inferiority_table(best_case_reference_standard, margin=0.075)
var_ni_table_best.to_csv("artifacts/outputs/variable_non_inferiority_table_best_case.csv", index=False)
print("  Saved variable non-inferiority tables.")

# --- Table 2: Rule performance ---

print("Computing rule tables...")

rule_methods = {
    "LLM": llm_alone,
    "Reviewer 1": kush,
    "Reviewer 2": leisha,
    "Individual reviewers (mean)": human_avg,
    "Human consensus": two_humans,
    "Physician-adjudicated consensus": three_humans,
}


def build_rule_table(ref_sheets):
    rows = []
    for cohort_name, attr_name, _ in cohorts:
        rows.append({"Metric": cohort_name, **{m: "" for m in rule_methods}})

        ref_processed = preprocess_for_rules(ref_sheets)
        ref_rules = compute_rules(ref_processed, attr_name)

        for metric_name, metric_fn in [("Accuracy", accuracy_metric), ("Sensitivity", sensitivity_metric), ("Specificity", specificity_metric)]:
            row = {"Metric": f"{metric_name}"}

            for method_name, method_sheets in rule_methods.items():
                method_processed = preprocess_for_rules(method_sheets)
                method_rules = compute_rules(method_processed, attr_name)

                merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
                assert not merged["rule_binary_ref"].isna().any()
                assert not merged["rule_binary_method"].isna().any()

                keys = merged["EncounterKey"].values
                y_true = merged["rule_binary_ref"].values
                y_pred = merged["rule_binary_method"].values

                point, lo, hi = cluster_bootstrap_ci(keys, y_true, y_pred, metric_fn)
                row[method_name] = fmt_ci(point, lo, hi)

            rows.append(row)

    return pd.DataFrame(rows)

rule_table = build_rule_table(reference_standard)
rule_table.to_csv("artifacts/outputs/rule_table.csv", index=False)

rule_table_best = build_rule_table(best_case_reference_standard)
rule_table_best.to_csv("artifacts/outputs/rule_table_best_case.csv", index=False)
print("  Saved rule tables.")

# --- Confusion matrix figure ---

print("Generating confusion matrix figures...")


def build_confusion_figure(ref_sheets, output_path):
    ref_processed = preprocess_for_rules(ref_sheets)
    rule_methods_local = { k:v for k,v in rule_methods.items() if k != "Individual reviewers (mean)" }
    n_methods = len(rule_methods_local)
    n_cohorts = len(cohorts)
    fig, axes = plt.subplots(n_cohorts, n_methods, figsize=(3.5 * n_methods, 3.5 * n_cohorts))

    for i, (cohort_name, attr_name, _) in enumerate(cohorts):
        ref_rules = compute_rules(ref_processed, attr_name)

        for j, (method_name, method_sheets) in enumerate(rule_methods_local.items()):
            ax = axes[i, j]
            method_processed = preprocess_for_rules(method_sheets)
            method_rules = compute_rules(method_processed, attr_name)

            merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
            y_true = merged["rule_binary_ref"].values
            y_pred = merged["rule_binary_method"].values

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            ax.imshow(cm, cmap="Blues", interpolation="nearest")
            for (ri, ci), val in np.ndenumerate(cm):
                threshold = cm.max() / 2
                color = "white" if cm[ri, ci] > threshold else "black"
                txt = ax.text(ci, ri, str(val), ha='center', va='center', fontsize=16, color=color)
                

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Low risk", "High risk"])
            ax.set_yticklabels(["Low risk", "High risk"])

            if i == 0:
                ax.set_title(method_name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{cohort_name}\n(Reference)", fontsize=9)
            if i == n_cohorts - 1:
                ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


build_confusion_figure(reference_standard, "artifacts/outputs/confusion_matrices.png")
build_confusion_figure(best_case_reference_standard, "artifacts/outputs/confusion_matrices_best_case.png")
print("  Saved confusion matrix figures.")

# --- CSPINE 3-level confusion matrix figure ---

print("Generating CSPINE 3-level confusion matrix figure...")


def compute_cspine_rules_3level(sheets_processed):
    df = sheets_processed.cspine.copy()
    df["rule"] = df.apply(run_cspine_rule, axis=1)
    return df[["EncounterKey", "rule"]]


def build_cspine_3level_figure(ref_sheets, output_path, two_rows=False):
    ref_processed = preprocess_for_rules(ref_sheets)
    ref_rules = compute_cspine_rules_3level(ref_processed)

    cspine_methods = {
        "LLM": llm_alone,
        "Reviewer 1": kush,
        "Reviewer 2": leisha,
        "Human consensus": two_humans,
        "Physician-adjudicated consensus": three_humans,
    }

    labels = [CSI.LOW_RISK, CSI.MODERATE_RISK, CSI.HIGH_RISK]
    tick_labels = ["Low risk", "Moderate\nrisk", "High risk"]
    n_methods = len(cspine_methods)

    if two_rows:
        ncols = (n_methods + 1) // 2
        nrows = 2
    else:
        ncols = n_methods
        nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    for j, (method_name, method_sheets) in enumerate(cspine_methods.items()):
        ax = axes_flat[j]
        method_processed = preprocess_for_rules(method_sheets)
        method_rules = compute_cspine_rules_3level(method_processed)

        merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
        y_true = merged["rule_ref"].values
        y_pred = merged["rule_method"].values

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for (ri, ci), val in np.ndenumerate(cm):
            threshold = cm.max() / 2
            color = "white" if cm[ri, ci] > threshold else "black"
            ax.text(ci, ri, str(val), ha='center', va='center', fontsize=16, color=color)

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_title(method_name, fontsize=10)
        if j % ncols == 0:
            ax.set_ylabel("Reference")
        ax.set_xlabel("Predicted")

    for k in range(n_methods, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


build_cspine_3level_figure(reference_standard, "artifacts/outputs/cspine_3level_confusion.png", two_rows=True)
print("  Saved CSPINE 3-level confusion matrix figure.")

# --- Slim confusion matrix figure (no Human consensus column, CSPINE 3-level) ---

print("Generating slim confusion matrix figures...")


def build_confusion_figure_slim(ref_sheets, output_path):
    ref_processed = preprocess_for_rules(ref_sheets)
    slim_methods = {k: v for k, v in rule_methods.items() if k not in ("Individual reviewers (mean)", "Human consensus")}
    n_methods = len(slim_methods)
    n_cohorts = len(cohorts)
    fig, axes = plt.subplots(n_cohorts, n_methods, figsize=(4.2 * n_methods, 4.2 * n_cohorts))

    FONT = 20
    LABEL_FONT = 26
    cspine_labels = [CSI.LOW_RISK, CSI.MODERATE_RISK, CSI.HIGH_RISK]
    cspine_tick_labels = ["Low risk", "Moderate\nrisk", "High risk"]

    method_names_display = {
        "Physician-adjudicated consensus": "Physician-adjudicated\nconsensus"
    }

    for i, (cohort_name, attr_name, _) in enumerate(cohorts):
        is_cspine = attr_name == "cspine"

        if is_cspine:
            ref_rules = compute_cspine_rules_3level(ref_processed)
        else:
            ref_rules = compute_rules(ref_processed, attr_name)

        for j, (method_name, method_sheets) in enumerate(slim_methods.items()):
            ax = axes[i, j]
            method_processed = preprocess_for_rules(method_sheets)

            if is_cspine:
                method_rules = compute_cspine_rules_3level(method_processed)
                merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
                y_true = merged["rule_ref"].values
                y_pred = merged["rule_method"].values
                cm = confusion_matrix(y_true, y_pred, labels=cspine_labels)
                ax.imshow(cm, cmap="Blues", interpolation="nearest")
                for (ri, ci), val in np.ndenumerate(cm):
                    threshold = cm.max() / 2
                    color = "white" if cm[ri, ci] > threshold else "black"
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=FONT, color=color)
                ax.set_xticks([0, 1, 2])
                ax.set_yticks([0, 1, 2])
                if i == n_cohorts - 1:
                    ax.set_xticklabels(cspine_tick_labels, fontsize=FONT)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_yticklabels(cspine_tick_labels, fontsize=FONT)
                else:
                    ax.set_yticklabels([])
            else:
                method_rules = compute_rules(method_processed, attr_name)
                merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
                y_true = merged["rule_binary_ref"].values
                y_pred = merged["rule_binary_method"].values
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                ax.imshow(cm, cmap="Blues", interpolation="nearest")
                for (ri, ci), val in np.ndenumerate(cm):
                    threshold = cm.max() / 2
                    color = "white" if cm[ri, ci] > threshold else "black"
                    ax.text(ci, ri, str(val), ha='center', va='center', fontsize=FONT, color=color)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                if i == n_cohorts - 1:
                    ax.set_xticklabels(["Low risk", "High risk"], fontsize=FONT)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_yticklabels(["Low risk", "High risk"], fontsize=FONT)
                else:
                    ax.set_yticklabels([])

            if i == 0:
                display_name = method_names_display.get(method_name, method_name)
                ax.set_title(display_name, fontsize=FONT)
            if j == 0:
                ax.set_ylabel(cohort_name, fontsize=FONT)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    # fig.supxlabel("Predicted", fontsize=LABEL_FONT, y=-0.01)
    fig.supylabel("Reference", fontsize=LABEL_FONT)
    # plt.tight_layout(rect=[0.03, 0.01, 1, 1])
    fig.supxlabel("Predicted", fontsize=LABEL_FONT, y=0.02)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


build_confusion_figure_slim(reference_standard, "artifacts/outputs/confusion_matrices_slim.png")
build_confusion_figure_slim(best_case_reference_standard, "artifacts/outputs/confusion_matrices_slim_best_case.png")
print("  Saved slim confusion matrix figures.")
print("Done.")


def build_non_inferiority_table(ref_sheets, margin=0.075):
    rows = []

    for cohort_name, attr_name, _ in cohorts:
        rows.append({"Metric": cohort_name, "Difference (LLM - Human)": "", "Non-Inferior?": ""})

        ref_processed = preprocess_for_rules(ref_sheets)
        ref_rules = compute_rules(ref_processed, attr_name)

        llm_processed = preprocess_for_rules(llm_alone)
        llm_rules = compute_rules(llm_processed, attr_name)

        human_processed = preprocess_for_rules(human_avg)
        human_rules = compute_rules(human_processed, attr_name)

        llm_merged = ref_rules.merge(llm_rules, on="EncounterKey", suffixes=("_ref", "_llm"))
        merged = llm_merged.merge(human_rules, on="EncounterKey")
        merged.rename(columns={"rule_binary": "rule_binary_human"}, inplace=True)

        keys = merged["EncounterKey"].values
        y_true = merged["rule_binary_ref"].values
        y_pred_llm = merged["rule_binary_llm"].values
        y_pred_human = merged["rule_binary_human"].values

        for metric_name, metric_fn in [("Accuracy", accuracy_metric), ("Sensitivity", sensitivity_metric), ("Specificity", specificity_metric)]:
            row = {"Metric": f"{metric_name}"}

            pt, lo, hi = cluster_bootstrap_diff_ci(keys, y_true, y_pred_llm, y_pred_human, metric_fn)

            row["Difference (LLM - Human)"] = f"{pt:+.2f} ({lo:.2f} to {hi:.2f})"

            if np.isnan(lo):
                row["Non-Inferior?"] = "N/A"
            else:
                row["Non-Inferior?"] = "Yes" if lo > -margin else "No"

            rows.append(row)

    return pd.DataFrame(rows)

print("Computing non-inferiority comparisons (LLM vs. Individual Reviewers)...")
ni_table = build_non_inferiority_table(reference_standard, margin=0.075)
ni_table.to_csv("artifacts/outputs/non_inferiority_table.csv", index=False)
print("  Saved non-inferiority table.")

ni_table = build_non_inferiority_table(best_case_reference_standard, margin=0.075)
ni_table.to_csv("artifacts/outputs/non_inferiority_table_best_case.csv", index=False)
print("  Saved non-inferiority table.")

# --- LLM sensitivity for risk_headinj and risk_torsoinj ---

# --- Table 2b: Rule performance with IPW adjustment ---

print("Computing IPW-adjusted rule tables...")

FULL_EXTRACTION_PATHS = {
    "tbi_younger": "outputs/extraction_TBI_YOUNGER_apr_23.csv",
    "tbi_older": "outputs/extraction_combined_TBI_OLDER_apr_23.csv",
    "cspine": "outputs/extraction_v2_CSPINE_apr_23.csv",
    "iai": "outputs/extraction_IAI_apr_23.csv",
}


def compute_population_rule_counts(attr_name):
    """Load the full extraction CSV and count High/Low risk in the population."""
    df = pd.read_csv(FULL_EXTRACTION_PATHS[attr_name])
    df = df.replace("Suspected", "Yes").replace("Yes (non-severe)", "No")
    df["rule"] = df.apply(rule_fns[attr_name], axis=1)
    if attr_name == "cspine":
        df["rule"] = df["rule"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    df = df.dropna(subset=["rule"])
    counts = df["rule"].value_counts()
    return {"High risk": counts.get("High risk", 0), "Low risk": counts.get("Low risk", 0)}


_ipw_weights_printed = set()


def compute_ipw_weights(attr_name, sample_rules_binary):
    """
    Compute inverse probability weights for a sampled validation set.

    The validation sample was stratified by the LLM's predicted rule outcome:
    50 low-risk + 50 high-risk for TBI/IAI, 50 low + 100 high for CSPINE.
    Weights = (population_count / sampled_count) for each stratum, where
    strata are defined by the LLM's rule prediction (the sampling variable).
    """
    pop_counts = compute_population_rule_counts(attr_name)
    sample_rules_binary = np.asarray(sample_rules_binary)

    sampled_high = (sample_rules_binary == 1).sum()
    sampled_low = (sample_rules_binary == 0).sum()

    w_high = pop_counts["High risk"] / sampled_high
    w_low = pop_counts["Low risk"] / sampled_low

    if attr_name not in _ipw_weights_printed:
        print(f"  IPW weights for {attr_name}: w_high={w_high:.2f} (pop={pop_counts['High risk']}, sampled={sampled_high}), w_low={w_low:.2f} (pop={pop_counts['Low risk']}, sampled={sampled_low})")
        _ipw_weights_printed.add(attr_name)

    weights = np.where(sample_rules_binary == 1, w_high, w_low)
    return weights


def weighted_accuracy_metric(y_true, y_pred, weights):
    correct = (y_true == y_pred).astype(float)
    return np.average(correct, weights=weights)


def weighted_sensitivity_metric(y_true, y_pred, weights):
    pos_mask = y_true == 1
    if pos_mask.sum() == 0:
        return np.nan
    tp = np.sum(weights[pos_mask] * (y_pred[pos_mask] == 1))
    total_pos = np.sum(weights[pos_mask])
    return tp / total_pos


def weighted_specificity_metric(y_true, y_pred, weights):
    neg_mask = y_true == 0
    if neg_mask.sum() == 0:
        return np.nan
    tn = np.sum(weights[neg_mask] * (y_pred[neg_mask] == 0))
    total_neg = np.sum(weights[neg_mask])
    return tn / total_neg


def stratified_cluster_bootstrap_ci_ipw(keys, y_true, y_pred, weights, strata, weighted_metric_fn, alpha=0.05):
    """Stratified cluster bootstrap CI with IPW-adjusted metrics.

    Resamples encounter keys within each stratum separately, preserving the
    fixed stratum sizes from the original stratified sampling design.
    """
    keys = np.asarray(keys)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    strata = np.asarray(strata)

    unique_keys = np.unique(keys)
    key_to_idx = {k: np.where(keys == k)[0] for k in unique_keys}
    key_to_stratum = {k: strata[key_to_idx[k][0]] for k in unique_keys}

    keys_by_stratum = {}
    for k in unique_keys:
        s = key_to_stratum[k]
        keys_by_stratum.setdefault(s, []).append(k)
    keys_by_stratum = {s: np.array(ks) for s, ks in keys_by_stratum.items()}

    point = weighted_metric_fn(y_true, y_pred, weights)

    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_keys = np.concatenate([
            RNG.choice(stratum_keys, size=len(stratum_keys), replace=True)
            for stratum_keys in keys_by_stratum.values()
        ])
        idx = np.concatenate([key_to_idx[k] for k in sampled_keys])
        boots[i] = weighted_metric_fn(y_true[idx], y_pred[idx], weights[idx])

    ci_lo = np.percentile(boots, 100 * alpha / 2)
    ci_hi = np.percentile(boots, 100 * (1 - alpha / 2))

    if ci_lo == ci_hi:
        if weighted_metric_fn == weighted_sensitivity_metric:
            n_total = (y_true == 1).sum()
        elif weighted_metric_fn == weighted_specificity_metric:
            n_total = (y_true == 0).sum()
        elif weighted_metric_fn == weighted_accuracy_metric:
            n_total = len(y_true)
        else:
            raise Exception(f"Unknown weighted_metric_fn: {weighted_metric_fn}")
        n_success = int(round(point * n_total))
        ci_lo, ci_hi = proportion_confint(n_success, n_total, alpha=alpha, method='wilson')

    return point, ci_lo, ci_hi


def build_rule_table_ipw(ref_sheets):
    """Same structure as build_rule_table, but with IPW-adjusted metrics.

    Weights are derived from the LLM's rule prediction (the sampling stratum).
    """
    rows = []
    for cohort_name, attr_name, _ in cohorts:
        rows.append({"Metric": cohort_name, **{m: "" for m in rule_methods}})

        ref_processed = preprocess_for_rules(ref_sheets)
        ref_rules = compute_rules(ref_processed, attr_name)

        # Compute LLM rule predictions — these define the sampling strata
        llm_processed = preprocess_for_rules(llm_alone)
        llm_rules = compute_rules(llm_processed, attr_name)

        ipw_metrics = [
            ("Accuracy", weighted_accuracy_metric),
            ("Sensitivity", weighted_sensitivity_metric),
            ("Specificity", weighted_specificity_metric),
        ]

        for metric_name, metric_fn in ipw_metrics:
            row = {"Metric": f"{metric_name}"}

            for method_name, method_sheets in rule_methods.items():
                method_processed = preprocess_for_rules(method_sheets)
                method_rules = compute_rules(method_processed, attr_name)

                merged = ref_rules.merge(method_rules, on="EncounterKey", suffixes=("_ref", "_method"))
                merged = merged.merge(llm_rules.rename(columns={"rule_binary": "llm_rule_binary"}), on="EncounterKey")
                assert not merged["rule_binary_ref"].isna().any()
                assert not merged["rule_binary_method"].isna().any()

                keys = merged["EncounterKey"].values
                y_true = merged["rule_binary_ref"].values
                y_pred = merged["rule_binary_method"].values
                llm_strata = merged["llm_rule_binary"].values
                weights = compute_ipw_weights(attr_name, llm_strata)

                point, lo, hi = stratified_cluster_bootstrap_ci_ipw(
                    keys, y_true, y_pred, weights, llm_strata, metric_fn
                )
                row[method_name] = fmt_ci(point, lo, hi)

            rows.append(row)

    return pd.DataFrame(rows)


rule_table_ipw = build_rule_table_ipw(reference_standard)
rule_table_ipw.to_csv("artifacts/outputs/rule_table_ipw.csv", index=False)

rule_table_ipw_best = build_rule_table_ipw(best_case_reference_standard)
rule_table_ipw_best.to_csv("artifacts/outputs/rule_table_ipw_best_case.csv", index=False)
print("  Saved IPW-adjusted rule tables.")


# --- Non-inferiority with IPW adjustment ---

def stratified_cluster_bootstrap_diff_ci_ipw(keys, y_true, y_pred_a, y_pred_b, weights, strata, weighted_metric_fn, alpha=0.05):
    """Stratified cluster bootstrap CI for the DIFFERENCE (A - B) with IPW.

    Method A is the LLM. Method B is the human comparator.
    Resamples within strata to respect the stratified sampling design.
    """
    keys = np.asarray(keys)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    weights = np.asarray(weights)
    strata = np.asarray(strata)

    unique_keys = np.unique(keys)
    key_to_idx = {k: np.where(keys == k)[0] for k in unique_keys}
    key_to_stratum = {k: strata[key_to_idx[k][0]] for k in unique_keys}

    keys_by_stratum = {}
    for k in unique_keys:
        s = key_to_stratum[k]
        keys_by_stratum.setdefault(s, []).append(k)
    keys_by_stratum = {s: np.array(ks) for s, ks in keys_by_stratum.items()}

    point_diff = weighted_metric_fn(y_true, y_pred_a, weights) - weighted_metric_fn(y_true, y_pred_b, weights)

    boots_diff = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_keys = np.concatenate([
            RNG.choice(stratum_keys, size=len(stratum_keys), replace=True)
            for stratum_keys in keys_by_stratum.values()
        ])
        idx = np.concatenate([key_to_idx[k] for k in sampled_keys])
        val_a = weighted_metric_fn(y_true[idx], y_pred_a[idx], weights[idx])
        val_b = weighted_metric_fn(y_true[idx], y_pred_b[idx], weights[idx])
        boots_diff[i] = val_a - val_b

    ci_lo = np.percentile(boots_diff, 100 * alpha / 2)
    ci_hi = np.percentile(boots_diff, 100 * (1 - alpha / 2))

    return point_diff, ci_lo, ci_hi


def build_non_inferiority_table_ipw(ref_sheets, margin=0.075):
    """Non-inferiority table (LLM vs Human) with IPW-adjusted metrics."""
    rows = []

    for cohort_name, attr_name, _ in cohorts:
        rows.append({"Metric": cohort_name, "Difference (LLM - Human)": "", "Non-Inferior?": ""})

        ref_processed = preprocess_for_rules(ref_sheets)
        ref_rules = compute_rules(ref_processed, attr_name)

        llm_processed = preprocess_for_rules(llm_alone)
        llm_rules = compute_rules(llm_processed, attr_name)

        human_processed = preprocess_for_rules(human_avg)
        human_rules = compute_rules(human_processed, attr_name)

        llm_merged = ref_rules.merge(llm_rules, on="EncounterKey", suffixes=("_ref", "_llm"))
        merged = llm_merged.merge(human_rules, on="EncounterKey")
        merged.rename(columns={"rule_binary": "rule_binary_human"}, inplace=True)

        keys = merged["EncounterKey"].values
        y_true = merged["rule_binary_ref"].values
        y_pred_llm = merged["rule_binary_llm"].values
        y_pred_human = merged["rule_binary_human"].values
        llm_strata = merged["rule_binary_llm"].values
        weights = compute_ipw_weights(attr_name, llm_strata)

        ipw_metrics = [
            ("Accuracy", weighted_accuracy_metric),
            ("Sensitivity", weighted_sensitivity_metric),
            ("Specificity", weighted_specificity_metric),
        ]

        for metric_name, metric_fn in ipw_metrics:
            row = {"Metric": f"{metric_name}"}

            pt, lo, hi = stratified_cluster_bootstrap_diff_ci_ipw(
                keys, y_true, y_pred_llm, y_pred_human, weights, llm_strata, metric_fn
            )

            row["Difference (LLM - Human)"] = f"{pt:+.2f} ({lo:.2f} to {hi:.2f})"

            if np.isnan(lo):
                row["Non-Inferior?"] = "N/A"
            else:
                row["Non-Inferior?"] = "Yes" if lo > -margin else "No"

            rows.append(row)

    return pd.DataFrame(rows)


print("Computing IPW-adjusted non-inferiority comparisons...")
ni_table_ipw = build_non_inferiority_table_ipw(reference_standard, margin=0.075)
ni_table_ipw.to_csv("artifacts/outputs/non_inferiority_table_ipw.csv", index=False)

ni_table_ipw_best = build_non_inferiority_table_ipw(best_case_reference_standard, margin=0.075)
ni_table_ipw_best.to_csv("artifacts/outputs/non_inferiority_table_ipw_best_case.csv", index=False)
print("  Saved IPW-adjusted non-inferiority tables.")

# --- LLM sensitivity for risk_headinj and risk_torsoinj ---

print("\n--- LLM Sensitivity for CSPINE variables ---")
ref_df = reference_standard.cspine
llm_df = llm_alone.cspine

for col in ["risk_headinj", "risk_torsoinj"]:
    merged = ref_df[["EncounterKey", col]].merge(
        llm_df[["EncounterKey", col]], on="EncounterKey", suffixes=("_ref", "_llm")
    )
    y_true = (merged[f"{col}_ref"].astype(str) == "Yes").astype(int).values
    y_pred = (merged[f"{col}_llm"].astype(str) == "Yes").astype(int).values
    keys = merged["EncounterKey"].values

    point, lo, hi = cluster_bootstrap_ci(keys, y_true, y_pred, sensitivity_metric)
    print(f"  {col}: sensitivity = {point:.2f} ({lo:.2f}-{hi:.2f})")
