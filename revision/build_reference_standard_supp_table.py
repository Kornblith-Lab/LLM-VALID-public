from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from march.rule_calc import CSI, run_cspine_rule, run_iai_rule, run_tbi_older_rule, run_tbi_younger_rule

OUT = ROOT / "output" / "docx" / "supplementary_table_reference_standard_sensitivity.docx"
N_BOOT = 10_000
RNG = np.random.default_rng(20261026)

RULES = {
    "Traumatic brain\ninjury in younger\n(n=100)": ("TBI YOUNGER", "tbi_younger", run_tbi_younger_rule,
        ["risk_loc", "risk_skullfx", "risk_altered", "risk_hematoma", "risk_mechanism", "risk_baseline"]),
    "Traumatic brain\ninjury in older\n(n=100)": ("TBI OLDER", "tbi_older", run_tbi_older_rule,
        ["risk_mechanism", "risk_headache", "risk_vomit", "risk_loc", "risk_sf", "risk_altered"]),
    "Cervical spine\ninjury (n=150)": ("CSPINE", "cspine", run_cspine_rule,
        ["risk_gcs", "risk_mentalstatus", "risk_abc", "risk_focal", "risk_tenderness", "risk_pain", "risk_headinj", "risk_torsoinj"]),
    "Intra-abdominal\ninjury (n=100)": ("IAI", "iai", run_iai_rule,
        ["risk_gcs", "risk_vomit", "risk_breathing", "risk_abdpain", "risk_walltrauma", "risk_chestwall", "risk_tenderness"]),
}

# Natural distribution of LLM-predicted low- and high-risk visits in the eligible cohort.
POPULATION = {
    "tbi_younger": (2489, 2709),
    "tbi_older": (11638, 4139),
    "cspine": (3152, 2199),
    "iai": (10001, 2291),
}


def read_workbook(filename: str, suffix: str = ""):
    path = ROOT / "review_documents" / filename
    result = {}
    for _label, (sheet, key, _fn, columns) in RULES.items():
        sheet_name = sheet + suffix
        raw = pd.read_excel(path, sheet_name=sheet_name)
        # The reviewer workbook carries a presentation row above its field names;
        # all other workbooks expose EncounterKey in the first row.
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1 if "EncounterKey" not in raw.columns else None)
        if "exclude" in df:
            df = df[df["exclude"].isna()]
        result[key] = df[["EncounterKey"] + columns].copy()
    return result


def normalize(df):
    return df.replace("Yes (non-severe)", "No").replace("Suspected", "Yes")


def rule_output(df, fn, key):
    work = normalize(df.copy())
    work["value"] = work.apply(fn, axis=1)
    if key == "cspine":
        work["value"] = work["value"].replace(CSI.MODERATE_RISK, CSI.HIGH_RISK)
    work["rule"] = (work["value"] == "High risk").astype(int)
    return work[["EncounterKey", "rule"]]


def metric(y, pred, weight, kind):
    y, pred, weight = np.asarray(y), np.asarray(pred), np.asarray(weight)
    if kind == "Accuracy":
        return float(np.average(y == pred, weights=weight))
    mask = y == 1 if kind == "Sensitivity" else y == 0
    target = 1 if kind == "Sensitivity" else 0
    return float(np.sum(weight[mask] * (pred[mask] == target)) / np.sum(weight[mask]))


def wilson(k, n):
    z = 1.95996398454
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def ci_or_wilson(point, samples, y, pred, kind):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    # When every bootstrap draw is identical (for example, observed sensitivity 1.00),
    # report an unweighted Wilson interval as in the main table.
    if lo == hi:
        mask = np.asarray(y) == (1 if kind == "Sensitivity" else 0)
        successes = int(np.sum(np.asarray(pred)[mask] == (1 if kind == "Sensitivity" else 0)))
        return wilson(successes, int(mask.sum()))
    return float(lo), float(hi)


def calculate(reference, llm, reviewer_1, reviewer_2):
    rows = []
    raw = {}
    for label, (_sheet, key, fn, _columns) in RULES.items():
        ref = rule_output(reference[key], fn, key)
        machine = rule_output(llm[key], fn, key)
        first = rule_output(reviewer_1[key], fn, key)
        second = rule_output(reviewer_2[key], fn, key)
        visit = ref.merge(machine, on="EncounterKey", suffixes=("_ref", "_llm"))
        visit = visit.merge(first, on="EncounterKey").rename(columns={"rule": "first"})
        visit = visit.merge(second, on="EncounterKey").rename(columns={"rule": "second"})
        long = pd.concat([
            visit[["EncounterKey", "rule_ref", "rule_llm", "first"]].rename(columns={"rule_ref": "ref", "rule_llm": "llm", "first": "human"}),
            visit[["EncounterKey", "rule_ref", "rule_llm", "second"]].rename(columns={"rule_ref": "ref", "rule_llm": "llm", "second": "human"}),
        ], ignore_index=True)
        low_n, high_n = POPULATION[key]
        sampled_low, sampled_high = int((visit.rule_llm == 0).sum()), int((visit.rule_llm == 1).sum())
        weight_map = dict(zip(visit.EncounterKey, np.where(visit.rule_llm.to_numpy() == 1, high_n / sampled_high, low_n / sampled_low)))
        long["weight"] = long.EncounterKey.map(weight_map)
        by_key = {k: np.flatnonzero(long.EncounterKey.to_numpy() == k) for k in visit.EncounterKey}
        groups = [visit.loc[visit.rule_llm == s, "EncounterKey"].to_numpy() for s in (0, 1)]
        y, m, h, w = (long.ref.to_numpy(), long.llm.to_numpy(), long.human.to_numpy(), long.weight.to_numpy())
        boot = {kind: {"llm": np.empty(N_BOOT), "human": np.empty(N_BOOT), "diff": np.empty(N_BOOT)} for kind in ("Accuracy", "Sensitivity", "Specificity")}
        for i in range(N_BOOT):
            keys = np.concatenate([RNG.choice(group, len(group), replace=True) for group in groups])
            ix = np.concatenate([by_key[k] for k in keys])
            for kind in boot:
                bm = metric(y[ix], m[ix], w[ix], kind)
                bh = metric(y[ix], h[ix], w[ix], kind)
                boot[kind]["llm"][i] = bm
                boot[kind]["human"][i] = bh
                boot[kind]["diff"][i] = bm - bh
        for kind in boot:
            p_m = metric(y, m, w, kind)
            p_h = metric(y, h, w, kind)
            d = p_m - p_h
            # The LLM has one extraction per encounter. Keep that denominator for
            # Wilson fallback intervals rather than the duplicated reviewer table.
            llo, lhi = ci_or_wilson(p_m, boot[kind]["llm"], visit.rule_ref.to_numpy(), visit.rule_llm.to_numpy(), kind)
            hlo, hhi = ci_or_wilson(p_h, boot[kind]["human"], y, h, kind)
            dlo, dhi = np.quantile(boot[kind]["diff"], [0.025, 0.975])
            rows.append({"rule": label, "metric": kind, "llm": [p_m, llo, lhi], "human": [p_h, hlo, hhi], "diff": [d, float(dlo), float(dhi)], "noninferior": bool(dlo > -0.075)})
            raw[(key, kind)] = rows[-1]
    return rows, raw


def set_cell_border(cell, color="808080", size="8"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = "w:" + edge
        elem = borders.find(qn(tag))
        if elem is None:
            elem = OxmlElement(tag)
            borders.append(elem)
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), size)
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def margins(cell, top=70, start=75, bottom=70, end=75):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn("w:" + name))
        if node is None:
            node = OxmlElement("w:" + name)
            mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_text(cell, text, *, italic=False, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=8.0):
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.name = "Arial"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    r.font.size = Pt(size)
    r.italic = italic
    r.bold = bold
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_border(cell)
    margins(cell)


def fmt(value, ci, star=False):
    text = f"{value:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
    return text + ("*" if star else "")


def fmt_diff(values):
    p, lo, hi = values
    return f"{p:+.2f} ({lo:+.2f} to {hi:+.2f})"


def build_doc(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.42)
    section.bottom_margin = Inches(0.42)
    section.left_margin = Inches(0.42)
    section.right_margin = Inches(0.42)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    styles["Normal"].font.size = Pt(9)

    caption = doc.add_paragraph()
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.line_spacing = 1.25
    run = caption.add_run("Supplementary Table 2. ")
    run.bold = True
    run.italic = True
    run.font.name = "Arial"; run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    run = caption.add_run("Reference-standard sensitivity analysis of adjusted clinical prediction rule calculation performance for different extraction methods, compared to the pre-LLM physician-adjudicated consensus. All performance metrics are adjusted by inverse probability sampling weights (IPSW) to match the natural distribution of high-risk and low-risk children in the entire eligible cohort. LLM: large language model.")
    run.italic = True
    run.font.name = "Arial"; run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")

    table = doc.add_table(rows=1, cols=4)
    table.autofit = False
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    widths = [1.62, 1.67, 1.79, 2.08]
    headers = ["Adjusted Metric", "Large language\nmodel", "Individual expert\nreviewers\n(mean)", "Difference (LLM -\nExperts)"]
    for index, (c, text, width) in enumerate(zip(table.rows[0].cells, headers, widths)):
        c.width = Inches(width)
        shade(c, "F2F2F2")
        add_text(c, text, italic=True, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT if text.startswith("Adjusted") else WD_ALIGN_PARAGRAPH.CENTER, size=8)
        if index == 0:
            note = c.paragraphs[0].add_run("a")
            note.font.name = "Arial"
            note._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
            note._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
            note.font.size = Pt(6)
            note.font.superscript = True
            note.italic = True
    for start in range(0, len(rows), 3):
        group = rows[start:start+3]
        heading = table.add_row().cells
        heading[0].width = Inches(widths[0])
        add_text(heading[0], group[0]["rule"], italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, size=8)
        for c, width in zip(heading[1:], widths[1:]):
            c.width = Inches(width)
            add_text(c, "", size=8)
        for item in group:
            cells = table.add_row().cells
            add_text(cells[0], item["metric"], align=WD_ALIGN_PARAGRAPH.LEFT, size=8)
            llm_italic = item["llm"][0] >= item["human"][0]
            add_text(cells[1], fmt(item["llm"][0], item["llm"][1:], item["noninferior"]), italic=llm_italic, size=8)
            add_text(cells[2], fmt(item["human"][0], item["human"][1:]), size=8)
            add_text(cells[3], fmt_diff(item["diff"]), size=8)
            for c, width in zip(cells, widths):
                c.width = Inches(width)

    notes = [
        "Values are italicized when the point estimate for the LLM matched or exceeded the point estimate for the individual expert reviewers.",
        "aIPSW weights for TBI in younger (positive class [pos], 54.18; negative class [neg], 49.78), TBI in older (pos, 82.78; neg, 232.66), CSI (pos, 21.99; neg, 63.04), and IAI (pos, 45.82; neg, 199.90).",
        "*Denotes LLM non-inferiority to individual experts (7.5% margin).",
    ]
    for text in notes:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        r.font.name = "Arial"; r.font.size = Pt(7.8)
        r._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        r._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    doc.core_properties.title = "Reference Standard Sensitivity Analysis"
    doc.core_properties.author = ""
    doc.save(OUT)


def main():
    reference = read_workbook("PECARN THREE HUMAN CONSENSUS.xlsx", " HUMAN CONSENSUS")
    llm = read_workbook("PECARN LLM ALONE.xlsx")
    r1 = read_workbook("PECARN MULTIPLE VALIDATION.xlsx", " LEISHA")
    r2 = read_workbook("PECARN MULTIPLE VALIDATION.xlsx")
    rows, _ = calculate(reference, llm, r1, r2)
    build_doc(rows)
    (ROOT / "revision" / "reference_standard_table_values.json").write_text(json.dumps(rows, indent=2))
    print(OUT)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
