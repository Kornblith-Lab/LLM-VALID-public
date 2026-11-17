import pandas as pd
import numpy as np

def trim_note_before_initial_assessment(df):
    ias_index = df["note_text"].str.replace(r"Initial\s{0,4}\"?\s{0,4}Assessment", "Initial Assessment", regex=True).str.find("Initial Assessment").replace(-1, np.nan)
    int_index = df["note_text"].str.find("Interpretations:").replace(-1, np.nan)
    pex_index = df["note_text"].str.find("Physical Exam").replace(-1, np.nan)
    edc_index = df["note_text"].str.find("ED Course").replace(-1, np.nan)
    edc_index = edc_index.where(edc_index >= pex_index, np.nan)
    edc_index1 = df["note_text"].str.find("ED COURSE").replace(-1, np.nan)
    edc_index1 = edc_index1.where(edc_index1 >= pex_index, np.nan)
    mdm_index = df["note_text"].str.find("Medical Decision Making").replace(-1, np.nan)
    mdm_index1 = df["note_text"].str.find("MEDICAL DECISION MAKING").replace(-1, np.nan)
    # asp_index = df["note_text"].str.find("A/P").replace(-1, np.nan)
    rfo_index = df["note_text"].str.find("Results for orders").replace(-1, np.nan) # see example_note above

    df["cutoff_index"] = pd.concat([ias_index, int_index, edc_index, edc_index1, mdm_index, mdm_index1, rfo_index], axis=1).min(axis=1)

    df_next = df[df.cutoff_index.notna()]
    df_next["trimmed_note_text"] = df_next.apply(lambda row: str(row["note_text"])[0:int(row["cutoff_index"])], axis=1)
    return df_next

PECARN_IAI_INCLUSION = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "blunt_injury" ],
    "properties": {
        "blunt_injury": {
            "type": "string",
            "description": "Did the patient experience a recent blunt torso injury consistent with the requirements described in the prompt?",
            "enum": ["No", "Yes"]
        }
    }
}

PECARN_CSPINE_INCLUSION = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "blunt_injury", "collar", "evaluation_by_trauma_team" ],
    "properties": {
        "blunt_injury": {
            "type": "string",
            "description": "Did the patient experience a recent blunt trauma consistent with the requirements described in the prompt?",
            "enum": ["No", "Yes"]
        },
        "collar": {
            "type": "string",
            "description": "Was the patient at any point placed in a cervical collar (C-collar) or C-spine precautions or on a backboard for the injury?",
            "enum": ["No", "Yes"]
        },
        "evaluation_by_trauma_team": {
            "type": "string",
            "description": "Did the trauma team or neurosurgeon evaluate the patient?",
            "enum": ["No", "Yes"]
        }
    }
}

PECARN_TBI_INCLUSION = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "recent_injury", "risk_gcs" ],
    "properties": {
        "recent_injury": {
            "type": "string",
            "description": "Did the patient experience a recent head injury consistent with the requirements described in the prompt?",
            "enum": ["No", "Yes"]
        },
        "risk_gcs": {
            "type": "number",
            "description": "The patient's Glasgow Coma Score (GCS). Document 15 if it's not present in the note. Minimum is 3. Maximum is 15."
        },
    }
}

PECARN_IAI = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "risk_gcs", "risk_walltrauma", "risk_chestwall", "risk_tenderness", "risk_abdpain", "risk_vomit", "risk_breathing" ],
    "properties": {
        "risk_gcs": {
            "type": "number",
            "description": "The patient's Glasgow Coma Score (GCS). Document 15 if it's not present in the note. Minimum is 3. Maximum is 15."
        },
        "risk_walltrauma": {
            "type": "string",
            "description": "Does the patient have abdominal wall trauma (seatbelt sign, subcutaneous air, or laceration)?",
            "enum": ["No", "Yes"]
        },
        "risk_chestwall": {
            "type": "string",
            "description": "Does the patient have chest wall trauma (seatbelt sign, subcutaneous air, or laceration)?",
            "enum": ["No", "Yes"]
        },
        "risk_tenderness": {
            "type": "string",
            "description": "Is the word \"tenderness\" or related synonym used to describe the patient's abdomen?",
            "enum": ["No", "Yes"]
        },
        "risk_abdpain": {
            "type": "string",
            "description": "Does the patient report abdominal pain or a related synonym?",
            "enum": ["No", "Yes"]
        },
        "risk_vomit": {
            "type": "string",
            "description": "Did the patient vomit?",
            "enum": ["No", "Yes"]
        },
        "risk_breathing": {
            "type": "string",
            "description": "Is the patient presenting with decreased/diminished breath sounds or related?",
            "enum": ["No", "Yes"]
        }
    }
}

PECARN_CSPINE = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "risk_gcs", "risk_mentalstatus", "risk_abc", "risk_focal", "risk_pain", "risk_rom", "risk_tenderness", "risk_torsoinj", "risk_headinj" ],
    "properties": {
        "risk_gcs": {
            "type": "number",
            "description": "The patient's Glasgow Coma Score (GCS). Document 15 if it's not present in the note. Minimum is 3. Maximum is 15."
        },
        "risk_mentalstatus": {
            "type": "string",
            "description": "What is the patient's mental status on the AVPU scale at the time of presenting at the Emergency Department?",
            "enum": ["Alert", "Verbal", "Pain", "Unresponsive"]
        },
        "risk_abc": {
            "type": "string",
            "description": "Does the patient have abnormal airway, breathing, or circulation findings?",
            "enum": ["No", "Yes"]
        },
        "risk_focal": {
            "type": "string",
            "description": "Does the patient have focal neurological deficits?",
            "enum": ["No", "Yes"]
        },
        "risk_pain": {
            "type": "string",
            "description": "Does the patient endorse posterior midline neck pain?",
            "enum": ["No", "Yes"]
        },
        "risk_rom": {
            "type": "string",
            "description": "Does the patient endorse decreased range of motion of the neck?",
            "enum": ["No", "Yes"]
        },
        "risk_tenderness": {
            "type": "string",
            "description": "Does the patient have neck tenderness on physican exam?",
            "enum": ["No", "Yes"]
        },
        "risk_torsoinj": {
            "type": "string",
            "description": "Does the patient present with a torso injury that would require surgery or hospital admission?",
            "enum": ["No", "Yes"]
        },
        "risk_headinj": {
            "type": "string",
            "description": "Does the patient present with a head injury that would require surgery or hospital admission?",
            "enum": ["No", "Yes"]
        }
    }
}

PECARN_TBI_OLDER = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "risk_altered", "risk_loc", "risk_vomit", "risk_sf", "risk_mechanism", "risk_headache" ],
    "properties": {
        "risk_altered": {
            "type": "string",
            "description": "Does this patient have altered mental status?",
            "enum": ["Yes", "No"]
        },
        "risk_loc": {
            "type": "string",
            "description": "Did the patient lose consciousness during the injury?",
            "enum": ["Yes", "Suspected", "No"]
        },
        "risk_vomit": {
            "type": "string",
            "description": "Did the patient vomit after the injury?",
            "enum": ["No", "Yes"]
        },
        "risk_sf": {
            "type": "string",
            "description": "Does the patient have clinical signs of basilar skull fracture?",
            "enum": ["No", "Yes"]
        },
        "risk_mechanism": {
            "type": "string",
            "description": "Does the patient have a severe mechanism of injury?",
            "enum": ["No", "Yes"]
        },
        "risk_headache": {
            "type": "string",
            "description": "Does the patient endorse a severe headache?",
            "enum": ["No", "Yes", "Yes (non-severe)"]
        }
    }
}

PECARN_TBI_YOUNGER = {
    "type": "object",
    "additionalProperties": False,
    "required": [ "risk_altered", "risk_hematoma", "risk_loc", "risk_mechanism", "risk_skullfx", "risk_baseline" ],
    "properties": {
        "risk_altered": {
            "type": "string",
            "description": "Does this patient have altered mental status or behaviors?",
            "enum": ["Yes", "No"]
        },
        "risk_hematoma": {
            "type": "string",
            "description": "Does this patient have a temporal, parietal, or occipital hematoma?",
            "enum": ["Yes", "No"]
        },
        "risk_loc": {
            "type": "string",
            "description": "Did the patient lose consciousness for 5 seconds or more?",
            "enum": ["Yes", "No"]
        },
        "risk_mechanism": {
            "type": "string",
            "description": "Does the patient have a severe mechanism of injury?",
            "enum": ["Yes", "No"]
        },
        "risk_skullfx": {
            "type": "string",
            "description": "Did this child have a palpable skull fracture or an unclear skull fracture exam?",
            "enum": ["Yes", "No"]
        },
        "risk_baseline": {
            "type": "string",
            "description": "Is this patient acting at baseline per their parents/caregiver?",
            "enum": ["Yes", "No"]
        }
    }
}
