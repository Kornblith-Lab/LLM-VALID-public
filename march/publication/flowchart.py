"""Figure 1"""

import pandas as pd
import os
from xml.sax.saxutils import escape

from march.publication.local_data import cohort_path, exclusion_path, inclusion_extraction_path, load_tbi_older_combined, ROOT

# Encounter keys from manual chart review are kept out of version control.
# See local/manual_encounter_lists.example.json for the expected structure.
import json
MANUAL_LISTS = json.loads((ROOT / "local" / "manual_encounter_lists.json").read_text())

def create_flowchart_legacy():
    """
    Creates a flowchart replicating the cohort selection process
    and saves it to the specified directory at 600 DPI.
    """
    # Ensure the output directory exists
    output_dir = "march/publication/outputs/descriptive"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowchart_v3.png")

    with schemdraw.Drawing(show=False) as d:
        # 1. Increased font size and removed invalid 'align' argument
        d.config(fontsize=16, font='sans-serif')

        # --- 1. Total ED Visits ---
        # Increased box dimensions for larger font
        b1 = flow.RoundBox(w=10, h=3).label(r'$\mathbf{1{,}041{,}437}$ total ED visits')
        d += b1

        # Arrow down (increased length to give vertical breathing room)
        a1 = flow.Arrow().down().length(3.5).at(b1.S)
        d += a1

        # Exclusion 1
        # Pushed further to the right to prevent any crowding
        d += flow.Arrow().right().length(4).at(a1.center)
        d += flow.RoundBox(w=8, h=2.5).anchor('W').label(r'$\mathbf{625{,}462}$ adults')

        # --- 2. Pediatric ED Visits ---
        b2 = flow.RoundBox(w=10, h=3).at(a1.end).anchor('N').label(r'$\mathbf{415{,}975}$' "\npediatric ED visits")
        d += b2

        # Arrow down
        a2 = flow.Arrow().down().length(3.5).at(b2.S)
        d += a2

        # Exclusion 2
        d += flow.Arrow().right().length(4).at(a2.center)
        d += flow.RoundBox(w=8, h=2.5).anchor('W').label(r"$\mathbf{65{,}957}$ missing ED notes" "\n" r"$\mathbf{320{,}443}$ non-trauma chief" "\n" "complaints")

        # --- 3. Pediatric trauma visits ---
        b3 = flow.RoundBox(w=10, h=3).at(a2.end).anchor('N').label(r"$\mathbf{29{,}575}$" "\npediatric trauma visits")
        d += b3

        # Split Main Vertical Line
        a_split = flow.Line().down().length(2.5).at(b3.S)
        d += a_split

        # ------------------- Left Branch (Cervical Spine) -------------------
        # Increased branch distance from 10 to 12.5 to prevent horizontal overlap
        L_line = flow.Line().left().length(12.5).at(a_split.end)
        d += L_line
        
        a_L1 = flow.Arrow().down().length(2).at(L_line.end)
        d += a_L1
        
        b_L1 = flow.RoundBox(w=10, h=6).at(a_L1.end).anchor('N').label(
            r'$\mathbf{5{,}351}$' '\n'
            'LLM-identified cervical spine injury visits\n\n'
            'Defined by any blunt trauma, and 1+ of:\n'
            '1. Brought in by EMS*\n'
            '2. Trauma or neurosurgery consult\n'
            '3. Use of C-collar'
        )
        d += b_L1
        
        a_L2 = flow.Arrow().down().length(2.5).at(b_L1.S)
        d += a_L2
        
        b_L2 = flow.RoundBox(w=10, h=5).at(a_L2.end).anchor('N').label(
            r"$\mathbf{4{,}655}$" "\n"
            "eligible cervical spine injury visits\n\n"
            "After excluding:\n"
            r"$\mathbf{22}$ penetrating trauma" "\n"
            r"$\mathbf{11}$ cardiac arrest" "\n"
            r"$\mathbf{663}$ transfers"
        )
        d += b_L2


        # ------------------- Middle Branch (TBI) -------------------
        a_M1 = flow.Arrow().down().length(2).at(a_split.end)
        d += a_M1
        
        b_M1 = flow.RoundBox(w=10, h=6).at(a_M1.end).anchor('N').label(
            r"$\mathbf{20{,}975}$" "\n"
            "LLM-identified traumatic brain\n"
            "injury visits\n\n"
            "Defined by recent blunt head\n"
            "trauma (<=24 hours old)\n" \
            "and GCS 14+"
        )
        d += b_M1
        
        a_M2 = flow.Arrow().down().length(2.5).at(b_M1.S)
        d += a_M2
        
        # Made this box notably taller to fit the long list of exclusions
        b_M2 = flow.RoundBox(w=10, h=9).at(a_M2.end).anchor('N').label(
            r"$\mathbf{5{,}028}$ aged <2 yo" "\n"
            r"$\mathbf{15{,}147}$ aged 2+ yo" "\n"
            "eligible traumatic brain injury visits\n\n"
            "After excluding:\n"
            r"$\mathbf{56}$ penetrating trauma" "\n"
            r"$\mathbf{1}$ cardiac arrest" "\n"
            r"$\mathbf{599}$ transfers" "\n"
            r"$\mathbf{87}$ bleeding disorders" "\n"
            r"$\mathbf{14}$ intellectual disability" "\n"
            r"$\mathbf{11}$ brain tumors" "\n"
            r"$\mathbf{25}$ VP shunts" "\n"
            r"$\mathbf{2}$ non-injury" "\n"
            r"$\mathbf{5}$ invalid LLM extractions"
        )
        d += b_M2


        # ------------------- Right Branch (Intra-abdominal) -------------------
        R_line = flow.Line().right().length(12.5).at(a_split.end)
        d += R_line
        
        a_R1 = flow.Arrow().down().length(2).at(R_line.end)
        d += a_R1
        
        b_R1 = flow.RoundBox(w=10, h=6).at(a_R1.end).anchor('N').label(
            r"$\mathbf{12{,}292}$" "\n"
            "LLM-identified intra-abdominal\n"
            "injury visits\n\n"
            "Defined by recent blunt torso\n"
            "trauma (<=24 hours old)\n"
        )
        d += b_R1
        
        a_R2 = flow.Arrow().down().length(2.5).at(b_R1.S)
        d += a_R2
        
        b_R2 = flow.RoundBox(w=10, h=5).at(a_R2.end).anchor('N').label(
            r"$\mathbf{11{,}769}$" "\n"
            "eligible intra-abdominal injury visits\n\n"
            "After excluding:\n"
            r"$\mathbf{18}$ penetrating trauma" "\n"
            r"$\mathbf{7}$ cardiac arrest" "\n"
            r"$\mathbf{487}$ transfers" "\n"
            r"$\mathbf{11}$ intellectual disability"
        )
        d += b_R2

        # Save the drawing
        d.save(output_path, dpi=600)
        print(f"Flowchart successfully saved to {output_path} at 600 DPI.")


def create_flowchart_matplotlib():
    """Create the Figure 1 flowchart using the analytic pools behind Table 2 IPSW."""
    output_dir = "march/publication/outputs/descriptive"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowchart_v4_ipw.png")
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, text, width=3.15, height=1.15, color="#EAF2F8"):
        patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                               boxstyle="round,pad=0.03,rounding_size=0.08",
                               facecolor=color, edgecolor="#2C3E50", linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.1, wrap=True)
        return x, y, width, height

    def arrow(source, target):
        sx, sy, sw, sh = source
        tx, ty, tw, th = target
        ax.annotate("", xy=(tx, ty + th / 2), xytext=(sx, sy - sh / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#2C3E50"))

    all_ed = box(8, 11.1, "1,041,437 total ED visits", 4.0, 0.8, "#D6EAF8")
    pediatric = box(8, 9.7, "415,975 pediatric ED visits\n(excluded: 625,462 adults)", 4.3, 0.95)
    trauma = box(8, 8.1, "29,575 pediatric trauma visits with an ED physician note\n(excluded: 65,957 without ED notes; 320,443 non-trauma chief complaints)", 5.5, 1.2, "#D6EAF8")
    arrow(all_ed, pediatric)
    arrow(pediatric, trauma)

    xs = [2.0, 6.0, 10.0, 14.0]
    pools = [
        box(xs[0], 5.9, "TBI younger analytic pool\n5,198 visits\nLLM-low: 2,489; LLM-high: 2,709"),
        box(xs[1], 5.9, "TBI older analytic pool\n15,772 visits\nLLM-low: 11,633; LLM-high: 4,139\n(5 invalid extractions removed)", height=1.4),
        box(xs[2], 5.9, "CSI analytic pool\n5,351 visits\nLLM-low: 3,152; moderate/high: 2,199"),
        box(xs[3], 5.9, "IAI analytic pool\n12,292 visits\nLLM-low: 9,995; LLM-high: 2,297"),
    ]
    samples = [
        box(xs[0], 3.8, "Random sample\n50 LLM-low + 50 LLM-high\n(n=100)", color="#FCF3CF"),
        box(xs[1], 3.8, "Random sample\n50 LLM-low + 50 LLM-high\n(n=100)", color="#FCF3CF"),
        box(xs[2], 3.8, "Random sample\n50 low + 50 moderate + 50 high\n(n=150)", color="#FCF3CF"),
        box(xs[3], 3.8, "Random sample\n50 LLM-low + 50 LLM-high\n(n=100)", color="#FCF3CF"),
    ]
    for p in pools:
        ax.annotate("", xy=(p[0], p[1] + p[3] / 2), xytext=(8, 7.45),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#2C3E50"))
    for p, s in zip(pools, samples):
        arrow(p, s)
    final = box(8, 1.55, "Validation cohort: 450 distinct ED visits", 4.7, 0.85, "#D6EAF8")
    for s in samples:
        ax.annotate("", xy=(8, 1.975), xytext=(s[0], s[1] - s[3] / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#2C3E50"))
    ax.text(8, 0.55, "Analytic pools are the denominators used for inverse-probability sampling weights.\nEligibility was identified from prespecified LLM screening prompts; risk strata were generated before manual validation.", ha="center", va="center", fontsize=8.5)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Flowchart successfully saved to {output_path}.")


def create_flowchart():
    """Create the Figure 1 flowchart using the analytic pools behind Table 2 IPSW."""
    output_dir = "march/publication/outputs/descriptive"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flowchart_v4_ipw.svg")
    boxes = [
        (600, 45, 400, 65, "1,041,437 total ED visits", "#D6EAF8"),
        (575, 155, 450, 80, "415,975 pediatric ED visits|Excluded: 625,462 adults", "#EAF2F8"),
        (475, 300, 650, 100, "29,575 pediatric trauma visits with an ED physician note|Excluded: 65,957 without ED notes; 320,443 non-trauma chief complaints", "#D6EAF8"),
        (30, 525, 330, 115, "TBI younger analytic pool|5,198 visits|LLM-low: 2,489; LLM-high: 2,709", "#EAF2F8"),
        (410, 515, 330, 135, "TBI older analytic pool|15,772 visits|LLM-low: 11,633; LLM-high: 4,139|5 invalid extractions removed", "#EAF2F8"),
        (790, 525, 330, 115, "CSI analytic pool|5,351 visits|LLM-low: 3,152; moderate/high: 2,199", "#EAF2F8"),
        (1170, 525, 330, 115, "IAI analytic pool|12,292 visits|LLM-low: 9,995; LLM-high: 2,297", "#EAF2F8"),
        (30, 735, 330, 100, "Random sample|50 LLM-low + 50 LLM-high|n=100", "#FCF3CF"),
        (410, 735, 330, 100, "Random sample|50 LLM-low + 50 LLM-high|n=100", "#FCF3CF"),
        (790, 725, 330, 120, "Random sample|50 low + 50 moderate + 50 high|n=150", "#FCF3CF"),
        (1170, 735, 330, 100, "Random sample|50 LLM-low + 50 LLM-high|n=100", "#FCF3CF"),
        (600, 940, 400, 70, "Validation cohort: 450 distinct ED visits", "#D6EAF8"),
    ]
    def center(box):
        return box[0] + box[2] / 2, box[1] + box[3] / 2
    def line(a, b):
        ax, ay = center(a); bx, by = center(b)
        return f'<line x1="{ax}" y1="{ay + a[3]/2}" x2="{bx}" y2="{by - b[3]/2}" stroke="#2C3E50" stroke-width="2" marker-end="url(#arrow)"/>'
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1530" height="1100" viewBox="0 0 1530 1100">',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#2C3E50"/></marker></defs>',
           '<rect width="100%" height="100%" fill="white"/>']
    svg += [line(boxes[0], boxes[1]), line(boxes[1], boxes[2])]
    svg += [line(boxes[2], b) for b in boxes[3:7]]
    svg += [line(a, b) for a, b in zip(boxes[3:7], boxes[7:11])]
    svg += [line(b, boxes[11]) for b in boxes[7:11]]
    for x, y, w, h, text, color in boxes:
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#2C3E50" stroke-width="2"/>')
        lines = text.split("|")
        start = y + h / 2 - (len(lines) - 1) * 12
        for i, value in enumerate(lines):
            svg.append(f'<text x="{x + w/2}" y="{start + i*24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#17202A">{escape(value)}</text>')
    svg.append('<text x="765" y="1060" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#17202A">Analytic pools are the denominators used for inverse-probability sampling weights. Eligibility was identified by prespecified LLM screening prompts; risk strata were generated before manual validation.</text>')
    svg.append('</svg>')
    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(svg))
    print(f"Flowchart successfully saved to {output_path}.")


from march.rule_calc import TBI, run_cspine_rule, run_iai_rule, run_tbi_older_rule, run_tbi_younger_rule

TOTAL_VISITS_IN_CDW = 1_041_437 # length EDVISITS
TOTAL_PEDIATRIC_VISITS_IN_CDW = 415_975 # length PEDIATRIC_EDVISITS
TOTAL_PEDIATRIC_VISITS_WITH_NOTES = 350_018 # length PEDIATRIC_EDVISITS_LONGEST_NOTE

VISITS_WITH_TRAUMA_CC = 29_575

def create_formatted_dfs():
    CSPINE = pd.read_csv(inclusion_extraction_path("extraction_v2_CSPINE_apr_23.csv"))
    TBI_YOUNGER = pd.read_csv(inclusion_extraction_path("extraction_TBI_YOUNGER_apr_23.csv"))
    TBI_OLDER = load_tbi_older_combined()
    IAI = pd.read_csv(inclusion_extraction_path("extraction_IAI_apr_23.csv"))

    PENETRATING_TRAUMA = pd.read_csv(exclusion_path("apr_23_all_penetrating_traumas.csv"))
    PENETRATING_TRAUMA = set(PENETRATING_TRAUMA.EncounterKey.tolist())

    CARDIAC_ARREST = pd.read_csv(exclusion_path("apr_23_all_cardiac_arrests.csv"))
    CARDIAC_ARREST = set(CARDIAC_ARREST.EncounterKey.tolist())

    CSPINE["penetrating"] = CSPINE.EncounterKey.isin(PENETRATING_TRAUMA)
    TBI_YOUNGER["penetrating"] = TBI_YOUNGER.EncounterKey.isin(PENETRATING_TRAUMA)
    TBI_OLDER["penetrating"] = TBI_OLDER.EncounterKey.isin(PENETRATING_TRAUMA)
    IAI["penetrating"] = IAI.EncounterKey.isin(PENETRATING_TRAUMA)

    CSPINE["cardiac_arrest"] = CSPINE.EncounterKey.isin(CARDIAC_ARREST)
    TBI_YOUNGER["cardiac_arrest"] = TBI_YOUNGER.EncounterKey.isin(CARDIAC_ARREST)
    TBI_OLDER["cardiac_arrest"] = TBI_OLDER.EncounterKey.isin(CARDIAC_ARREST)
    IAI["cardiac_arrest"] = IAI.EncounterKey.isin(CARDIAC_ARREST)

    BLEEDING_DISORDER = pd.read_csv(exclusion_path("apr_23_all_bleeding_disorders.csv"))
    BLEEDING_DISORDER = set(BLEEDING_DISORDER.EncounterKey.tolist())

    TBI_YOUNGER["bleeding_disorder"] = TBI_YOUNGER.EncounterKey.isin(BLEEDING_DISORDER)
    TBI_OLDER["bleeding_disorder"] = TBI_OLDER.EncounterKey.isin(BLEEDING_DISORDER)

    INTELLECTUAL_DISABILITY = pd.read_csv(exclusion_path("apr_23_all_intellectual_disability.csv"))
    INTELLECTUAL_DISABILITY = set(INTELLECTUAL_DISABILITY.EncounterKey.tolist())

    TBI_YOUNGER["intellectual_disability"] = TBI_YOUNGER.EncounterKey.isin(INTELLECTUAL_DISABILITY)
    TBI_OLDER["intellectual_disability"] = TBI_OLDER.EncounterKey.isin(INTELLECTUAL_DISABILITY)
    IAI["intellectual_disability"] = IAI.EncounterKey.isin(INTELLECTUAL_DISABILITY)

    VP_SHUNTS = set(MANUAL_LISTS["VP_SHUNTS"]) # After manual review on the shunt patients, four were kept in for new shunt

    TBI_YOUNGER["shunt"] = TBI_YOUNGER.EncounterKey.isin(VP_SHUNTS)
    TBI_OLDER["shunt"] = TBI_OLDER.EncounterKey.isin(VP_SHUNTS)

    BRAIN_TUMOR = pd.read_csv(exclusion_path("apr_23_brain_tumor.csv"))
    BRAIN_TUMOR = set(BRAIN_TUMOR.EncounterKey.tolist())
    TBI_YOUNGER["brain_tumor"] = TBI_YOUNGER.EncounterKey.isin(BRAIN_TUMOR)
    TBI_OLDER["brain_tumor"] = TBI_OLDER.EncounterKey.isin(BRAIN_TUMOR)

    TRANSFERS = pd.read_csv(cohort_path("apr_23_all_traumas_sources_and_transfers.csv"))
    assert len(TRANSFERS) == 29575

    SOURCES_TO_EXCLUDE = ["Transfer from a Hospital (Different Facility)", "Transfer from Another Health Care Facility", "Emergency Room"]
    transfer_encounters = TRANSFERS[TRANSFERS.AdmissionSource.isin(SOURCES_TO_EXCLUDE)].EncounterKey.tolist()

    CSPINE["transfer"] = CSPINE.EncounterKey.isin(transfer_encounters)
    TBI_YOUNGER["transfer"] = TBI_YOUNGER.EncounterKey.isin(transfer_encounters)
    TBI_OLDER["transfer"] = TBI_OLDER.EncounterKey.isin(transfer_encounters)
    IAI["transfer"] = IAI.EncounterKey.isin(transfer_encounters)

    # Additional exclusions
    tbi_older_transfer = MANUAL_LISTS["tbi_older_transfer"]
    tbi_older_non_trauma = MANUAL_LISTS["tbi_older_non_trauma"]
    tbi_older_penetrating = MANUAL_LISTS["tbi_older_penetrating"] # the second is an open skull fracture
    TBI_OLDER.loc[TBI_OLDER.EncounterKey.isin(tbi_older_transfer), "transfer"] = True
    TBI_OLDER.loc[TBI_OLDER.EncounterKey.isin(tbi_older_penetrating), "penetrating"] = True
    TBI_OLDER["non_injury"] = False
    TBI_OLDER.loc[TBI_OLDER.EncounterKey.isin(tbi_older_non_trauma), "non_injury"] = True

    tbi_younger_penetrating = MANUAL_LISTS["tbi_younger_penetrating"]
    TBI_YOUNGER.loc[TBI_YOUNGER.EncounterKey.isin(tbi_younger_penetrating), "penetrating"] = True

    cspine_transfers = MANUAL_LISTS["cspine_transfers"]
    CSPINE.loc[CSPINE.EncounterKey.isin(cspine_transfers), "transfer"] = True

    return CSPINE, TBI_YOUNGER, TBI_OLDER, IAI

def auto_exclusion(condition_df, name, *queries, return_ops: bool = False):
    print(f"STARTING {name} EXCLUSION")
    starting_len = VISITS_WITH_TRAUMA_CC
    print(f"{starting_len} patients starting")
    operations = [{ "col": "eligibility", "length_after": len(condition_df), "delta": VISITS_WITH_TRAUMA_CC-len(condition_df) }]
    for query in queries:
            before_len = len(condition_df)
            if query != "invalid":
                exclusion_column = query.split(" ")[0]
                condition_df = condition_df.query(query)
            else:
                # Special validation logic for TBI_OLDER
                exclusion_column = "Invalid cases (missing extraction or GCS >15)"
                condition_df = condition_df[condition_df.risk_gcs <= 15]
                condition_df = condition_df[condition_df.risk_altered.notna()]
            after_len = len(condition_df)
            operations.append({ "col": exclusion_column, "length_after": after_len, "delta": before_len - after_len })
    
    for op in operations:
        col = op["col"]
        len_after = op["length_after"]
        delta = op["delta"]

        print(f"Excluding on {col} removed {delta} patients, leaving {len_after} ({len_after/starting_len:.2%} of original)")
    
    print(f"{len(condition_df)} patients remaining")
    print(f"COMPLETED {name} EXCLUSION")
    print()

    if return_ops:
        return condition_df, operations

    return condition_df

def create_final_dfs():
    CSPINE, TBI_YOUNGER, TBI_OLDER, IAI = create_formatted_dfs()

    CSPINE = auto_exclusion(CSPINE, "CSPINE", "penetrating == False", "cardiac_arrest == False", "transfer == False")
    TBI_OLDER = auto_exclusion(TBI_OLDER, "TBI_OLDER", "penetrating == False", "cardiac_arrest == False", "transfer == False", "bleeding_disorder == False", "intellectual_disability == False", "brain_tumor == False", "shunt == False", "non_injury == False", "invalid")
    TBI_YOUNGER = auto_exclusion(TBI_YOUNGER, "TBI_YOUNGER", "penetrating == False", "cardiac_arrest == False", "transfer == False", "bleeding_disorder == False", "intellectual_disability == False", "brain_tumor == False", "shunt == False")
    IAI = auto_exclusion(IAI, "IAI", "penetrating == False", "cardiac_arrest == False", "transfer == False", "intellectual_disability == False")

    assert type(CSPINE) == pd.DataFrame and type(TBI_OLDER) == pd.DataFrame and type(TBI_YOUNGER) == pd.DataFrame and type(IAI) == pd.DataFrame

    CSPINE["rule"] = CSPINE.apply(run_cspine_rule, axis=1)
    TBI_YOUNGER["rule"] = TBI_YOUNGER.apply(run_tbi_younger_rule, axis=1)
    TBI_OLDER["rule"] = TBI_OLDER.apply(run_tbi_older_rule, axis=1)
    IAI["rule"] = IAI.apply(run_iai_rule, axis=1)

    tbi_older_force_positive = MANUAL_LISTS["tbi_older_force_positive"]
    TBI_OLDER.loc[TBI_OLDER.EncounterKey.isin(tbi_older_force_positive), "rule"] = TBI.HIGH_RISK

    tbi_younger_force_positive = MANUAL_LISTS["tbi_younger_force_positive"]
    TBI_YOUNGER.loc[TBI_YOUNGER.EncounterKey.isin(tbi_younger_force_positive), "rule"] = TBI.HIGH_RISK

    return CSPINE, TBI_YOUNGER, TBI_OLDER, IAI

def main():
    create_flowchart()

if __name__ == "__main__":
    main()
