import pandas as pd
import numpy as np
from tap import tapify
from dataclasses import dataclass
from march.helpers import trim_note_before_initial_assessment

@dataclass
class Args:
    input_csv: str # = "data/tbi.csv"
    diagnosis_csv: str
    save_csv: str # = "derived/tbi.csv"

    # python march/00_prepare.py --input_csv data/iai.csv --diagnosis_csv data/iai_diagnosis.csv --save_csv derived/iai.csv

args = tapify(Args)

df = pd.read_csv(args.input_csv)
df = df.groupby("EncounterKey").nth(0)
original_len = len(df)

df = df[~df["DischargeDisposition"].isin(["Elopement OR Left ED after Medical Screening", "LWBS after Triage", "Left Against Medical Advice", "ED Dismiss - Never Arrived", "*Unspecified"])]

# Added "Eloped" as part of the exclusion criteria (TBI and CSPINE were already done)
df = df[~df["EdDisposition"].isin(["Expired", "Eloped or Left after MSE", "AMA", "*Unspecified", "Eloped"])]

if "cspine" in args.input_csv or "iai" in args.input_csv:
    y = input("Enter (y) to confirm that the note will NOT be trimmed:")
    if y.lower().strip() not in ["y", "yes"]: exit()
else:
    # This line is conditional, since we want to include the full note for IAI and CSPINE, since the inclusion criteria involves ordering any of a certain list of imaging.
    df = trim_note_before_initial_assessment(df)

diagnosis_df = pd.read_csv(args.diagnosis_csv)
diagnosis_encounters = set(diagnosis_df.EncounterKey)
df["has_diagnosis"] = df["EncounterKey"].apply(lambda e: e in diagnosis_encounters)

final_len = len(df)

print(f"Dataframe reduced from {original_len} to {final_len} after processing.")

df.to_csv(args.save_csv, index=False)
