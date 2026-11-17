import pandas as pd
import numpy as np
from openpyxl import load_workbook

INPUT_PATH = "review_documents/PECARN HUMAN-LLM COOPERATION.xlsx"
OUTPUT_PATH = "review_documents/PECARN HUMAN-LLM COOPERATION - REVIEW.xlsx"
SHEETS = ["TBI OLDER", "TBI YOUNGER", "CSPINE", "IAI"]
SAMPLE_N = 5

np.random.seed(42)

wb = load_workbook(INPUT_PATH)
xls = pd.ExcelFile(INPUT_PATH)


def is_colored(ws, row_idx, col_idx):
    """Check if a cell has a fill/highlight (patternType == 'solid')."""
    cell = ws.cell(row=row_idx, column=col_idx)
    return cell.fill.patternType is not None


def get_eligible_indices(ws, df, col_name):
    """Return DataFrame indices whose corresponding cell is not colored."""
    headers = {cell.value: cell.column for cell in ws[1]}
    col_idx = headers[col_name]
    eligible = []
    for i, df_idx in enumerate(df.index):
        excel_row = i + 2  # header is row 1
        if not is_colored(ws, excel_row, col_idx):
            eligible.append(df_idx)
    return eligible


with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    for sheet in SHEETS:
        ws = wb[sheet]
        df = pd.read_excel(xls, sheet_name=sheet)
        risk_cols = [c for c in df.columns if c.startswith("risk_")]

        rows = []
        for col in risk_cols:
            eligible = get_eligible_indices(ws, df, col)
            assert len(eligible) >= SAMPLE_N, (
                f"Not enough non-colored cells in {sheet}/{col}: "
                f"only {len(eligible)} available"
            )
            sampled_indices = np.random.choice(eligible, size=SAMPLE_N, replace=False)
            for idx in sampled_indices:
                row = df.loc[idx]
                rows.append({
                    "EncounterKey": row["EncounterKey"],
                    "note_text": row["note_text"],
                    "variable_name": col,
                    "current_value": row[col],
                })

        out_df = pd.DataFrame(rows)
        out_df.to_excel(writer, sheet_name=sheet, index=False)

print(f"Written to {OUTPUT_PATH}")

# --- Spot-check verification ---
print("\nSpot-check: verifying all sampled cells are non-colored...")
output_xls = pd.ExcelFile(OUTPUT_PATH)
for sheet in SHEETS:
    ws = wb[sheet]
    df_orig = pd.read_excel(xls, sheet_name=sheet)
    df_out = pd.read_excel(output_xls, sheet_name=sheet)

    headers = {cell.value: cell.column for cell in ws[1]}

    for _, out_row in df_out.iterrows():
        var_name = out_row["variable_name"]
        enc_key = out_row["EncounterKey"]
        col_idx = headers[var_name]

        matching = df_orig[df_orig["EncounterKey"] == enc_key]
        assert len(matching) > 0, f"EncounterKey {enc_key} not found in {sheet}"

        for orig_idx in matching.index:
            excel_row = orig_idx + 2
            assert not is_colored(ws, excel_row, col_idx), (
                f"FAIL: {sheet}, row {excel_row}, col {var_name} is colored!"
            )

    print(f"  {sheet}: {len(df_out)} rows verified OK")

print("All spot-checks passed.")
