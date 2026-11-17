# python march/publication/inclusion.py --cohort TBI
# python march/publication/inclusion.py --cohort IAI
# python march/publication/inclusion.py --cohort CSPINE

import json

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
import valid.versa as versa
from pathlib import Path
from march.helpers import PECARN_CSPINE_INCLUSION, PECARN_IAI_INCLUSION, PECARN_TBI_INCLUSION
from tap import tapify
from dataclasses import dataclass

@dataclass
class Args:
    cohort: str

args = tapify(Args)
cohort = args.cohort

prompt = {
    "TBI": "prompts/inclusion/tbi-inclusion-exclusion.txt",
    "CSPINE": "prompts/inclusion/cspine-inclusion-exclusion.txt",
    "IAI": "prompts/inclusion/iai-inclusion-exclusion.txt"
}[cohort]

schema = {
    "TBI": PECARN_TBI_INCLUSION,
    "CSPINE": PECARN_CSPINE_INCLUSION,
    "IAI": PECARN_IAI_INCLUSION
}[cohort]

def get_df_for_cohort():
    if cohort == "TBI":
        df_younger = pd.read_csv("data/development/tbi_inclusion_younger.csv", skiprows=1)
        df_older = pd.read_csv("data/development/tbi_inclusion_older.csv", skiprows=1)
        df_younger = df_younger[df_younger.include == "yes"][["EncounterKey", "trimmed_note_text", "recent_injury"]]
        df_older = df_older[df_older.include == "yes"][["EncounterKey", "trimmed_note_text", "recent_injury"]]
        df = pd.concat([df_older, df_younger])
        df = df.rename(columns={"trimmed_note_text": "note_text", "recent_injury": "human_recent_injury"})
        assert len(df) == 100
        return df
    
    if cohort == "IAI":
        df_iai = pd.read_csv("data/development/iai_inclusion.csv", skiprows=1)
        df = df_iai[df_iai.include == "yes"][["EncounterKey", "note_text", "blunt_injury"]]
        df = df.rename(columns={"blunt_injury": "human_blunt_injury"})
        assert len(df) == 100
        return df

    if cohort == "CSPINE":
        df_cspine = pd.read_csv("data/development/cspine_inclusion.csv", skiprows=1)
        df = df_cspine[df_cspine.include == "yes"][["EncounterKey", "note_text", "blunt_injury", "evaluation_by_trauma_team", "collar"]]
        df = df.rename(columns={"blunt_injury": "human_blunt_injury", "evaluation_by_trauma_team": "human_evaluation_by_trauma_team", "collar": "human_collar"})
        assert len(df) == 100
        return df
    
    raise Exception(f"Unknown cohort: {cohort}")

from sklearn.metrics import accuracy_score, recall_score
from statsmodels.stats.proportion import proportion_confint
from sqlitedict import SqliteDict

def evaluate_reliability(db_path, llm_col):
    db = SqliteDict(db_path, tablename="inference")
    rows = []
    for key in db.keys():
        encounter_key, rep = key.split("_")
        rep = int(rep[-1])
        rows.append({
            "EncounterKey": encounter_key,
            "rep": rep,
            **json.loads(db[key])
        })
    
    df = pd.DataFrame(rows)

    for eval_col in [llm_col]:
        # Step 1: pivot to wide format
        pivot = df.pivot(index='EncounterKey', columns='rep', values=eval_col)

        # Step 2: convert to category counts per subject
        # ensure consistent category ordering
        categories = sorted(df[eval_col].unique())

        def row_to_counts(row):
            return [np.sum(row == cat) for cat in categories]

        rating_matrix = np.array([row_to_counts(row) for _, row in pivot.iterrows()])

        # Step 3: Fleiss' kappa
        kappa = fleiss_kappa(rating_matrix)

        # Step 4: Bootstrap CI
        n_boot = 1000
        boot_kappas = []

        rng = np.random.default_rng(42)

        for _ in range(n_boot):
            sample_idx = rng.choice(len(rating_matrix), size=len(rating_matrix), replace=True)
            sample = rating_matrix[sample_idx]
            boot_kappas.append(fleiss_kappa(sample))

        ci_lower = np.percentile(boot_kappas, 2.5)
        ci_upper = np.percentile(boot_kappas, 97.5)

        print(f"Column {eval_col}")
        print(f"Fleiss' kappa: {kappa:.2f} ({ci_lower:.2f}–{ci_upper:.2f})")

import json
import numpy as np
import pandas as pd
from sqlitedict import SqliteDict
from statsmodels.stats.proportion import proportion_confint

def evaluate_reliability_2(db_path, llm_col):
    db = SqliteDict(db_path, tablename="inference")
    rows = []
    for key in db.keys():
        encounter_key, rep = key.split("_")
        rep = int(rep[-1])
        rows.append({
            "EncounterKey": encounter_key,
            "rep": rep,
            **json.loads(db[key])
        })

    df = pd.DataFrame(rows)

    for eval_col in [llm_col]:
        # Step 1: pivot to wide format
        pivot = df.pivot(index='EncounterKey', columns='rep', values=eval_col)

        # Step 2: compute per-subject consistency (fraction of runs matching majority)
        def subject_consistency(row):
            row = row.dropna()
            if len(row) == 0:
                return np.nan
            majority_count = row.value_counts().iloc[0]
            return majority_count / len(row)

        consistencies = pivot.apply(subject_consistency, axis=1)
        valid_consistencies = consistencies.dropna()

        # Step 3: mean consistency across all subjects
        consistency = valid_consistencies.mean()

        # Step 4: Bootstrap CI for mean consistency (resampling subjects)
        n_boot = 10000
        boot_consistencies = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            sample_idx = rng.choice(len(valid_consistencies), size=len(valid_consistencies), replace=True)
            boot_consistencies.append(valid_consistencies.iloc[sample_idx].mean())
        ci_lower = np.percentile(boot_consistencies, 2.5)
        ci_upper = np.percentile(boot_consistencies, 97.5)

        # Step 5: full-agreement rate as a binary proportion
        full_agreement = (valid_consistencies == 1.0).astype(int)
        full_agreement_rate = full_agreement.mean()

        # Wilson CI for the full-agreement rate
        n = len(full_agreement)
        k = int(full_agreement.sum())
        fa_ci_lower, fa_ci_upper = proportion_confint(k, n, alpha=0.05, method="wilson")

        print(f"Column: {eval_col}")
        print(f"Mean consistency: {consistency:.2f} ({ci_lower:.2f}–{ci_upper:.2f})")
        print(f"Full agreement rate: {full_agreement_rate:.2f} ({fa_ci_lower:.2f}–{fa_ci_upper:.2f})")
        print("  (subjects where all runs matched)")

N_BOOT=10_000
RNG = np.random.default_rng(42)

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

def kappa_metric(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def bootstrap_ci(y_true, y_pred, metric_fn, alpha=0.05):
    """Bootstrap CI for a metric. Falls back to Wilson if zero-width."""
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.choice(n, size=n, replace=True)
        boots[i] = metric_fn(y_true[idx], y_pred[idx])

    if metric_fn == kappa_metric:
        boots = boots[~np.isnan(boots)]

    ci_lo = np.percentile(boots, 100 * alpha / 2)
    ci_hi = np.percentile(boots, 100 * (1 - alpha / 2))

    if ci_lo == ci_hi and metric_fn != kappa_metric:
        if metric_fn == sensitivity_metric:
            n_total = (y_true == 1).sum()
        elif metric_fn == specificity_metric:
            n_total = (y_true == 0).sum()
        elif metric_fn == accuracy_metric:
            n_total = len(y_true)
        else:
            raise Exception(f"Unknown metric_fn: {metric_fn}")
        n_success = int(round(point * n))
        ci_lo, ci_hi = proportion_confint(n_success, n_total, alpha=alpha, method='wilson')

    return point, ci_lo, ci_hi

def evaluate_column(df, human_col, llm_col):
    human = df[human_col].values
    llm = df[llm_col].values

    pos_label = "Yes"

    # --- Count + proportion for positive class ---
    n_total = len(human)
    n_pos = (human == pos_label).sum()
    prop_pos = n_pos / n_total if n_total > 0 else 0

    print(f"{pos_label}: {n_pos} ({prop_pos*100:.2f}%)")

    # --- Binary arrays for bootstrap ---
    human_bin = (human == pos_label).astype(int)
    llm_bin = (llm == pos_label).astype(int)

    # --- Accuracy (bootstrapped) ---
    acc, acc_low, acc_high = bootstrap_ci(human_bin, llm_bin, accuracy_metric)

    # --- Sensitivity (bootstrapped) ---
    sens, sens_low, sens_high = bootstrap_ci(human_bin, llm_bin, sensitivity_metric)

    # --- Cohen's Kappa (bootstrapped) ---
    kappa, kappa_low, kappa_high = bootstrap_ci(human, llm, kappa_metric)

    # --- Print results ---
    print(f"Evaluating column {llm_col} versus {human_col}")
    print(f"Accuracy: {acc:.2f} ({acc_low:.2f}–{acc_high:.2f})")
    print(f"Sensitivity: {sens:.2f} ({sens_low:.2f}–{sens_high:.2f})")
    print(f"Cohen's Kappa: {kappa:.2f} ({kappa_low:.2f}–{kappa_high:.2f})")

def main():
    df = get_df_for_cohort()
    print(df)

    save_path = Path(f"march/publication/outputs/inclusion/{cohort}")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_csv = save_path.with_suffix(".csv")

    if not save_csv.exists():
        config = versa.ExtractionConfig(df=df, 
                                prompt_file=prompt,
                                tool_calling=True,
                                llm="gpt-5-nano-2025-08-07",
                                response_schema=schema,
                                n_replicates=3)
            
        modu = versa.Extraction(config)
        df_inference = modu.extract(str(save_path))
    
    df_inference = pd.read_csv(save_csv)
    print(df_inference)

    cols_to_evaluate = [c for c in df_inference.columns if c.startswith("human_")]

    for human_col in cols_to_evaluate:
        llm_col = human_col.replace("human_", "")
        evaluate_column(df_inference, human_col, llm_col)
        evaluate_reliability_2(save_csv.with_suffix(".sqlite"), llm_col)
        evaluate_reliability(save_csv.with_suffix(".sqlite"), llm_col)
        print("=" * 40)
        

if __name__ == "__main__":
    main()