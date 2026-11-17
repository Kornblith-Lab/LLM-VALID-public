from dataclasses import dataclass

import pandas as pd
from tap import tapify

from march.helpers import PECARN_CSPINE_INCLUSION, PECARN_TBI_INCLUSION, PECARN_IAI_INCLUSION
import valid.versa as versa
from sqlitedict import SqliteDict
from collections import Counter

# ALL TRAUMA CHIEF COMPLAINTS
# ('TRAUMA', 'MOTOR VEHICLE CRASH', 'ASSAULT VICTIM', 'INJURY', 'MOTOR VEHICLE CRASH', 'BICYCLE CRASH', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN', 'FALL', 'HEAD INJURY', 'FALL', 'TRAUMA', 'HEAD LACERATION', 'FACIAL INJURY', 'ASSAULT VICTIM', 'MOTOR VEHICLE CRASH', 'FACIAL LACERATION', 'CONCUSSION', 'BICYCLE CRASH', 'INJURY', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN', 'ORBITAL TRAUMA', 'FACIAL TRAUMA', 'NECK PAIN', 'TRAUMA', 'MOTOR VEHICLE CRASH', 'NECK INJURY', 'HEAD INJURY', 'ASSAULT VICTIM', 'HEADACHE', 'BACK PAIN', 'TORTICOLLIS', 'SHOULDER PAIN', 'HEAD LACERATION', 'SHOULDER INJURY', 'INJURY', 'MOTOR VEHICLE CRASH', 'BICYCLE CRASH', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN')

@dataclass
class Args:
    source: str
    mode: str

args = tapify(Args)
mode = args.mode

df = pd.read_csv(args.source)

# print(df.PrimaryChiefComplaintName.value_counts())
# exit()

if "CSPINE" in mode:
    prompt = "prompts/inclusion/cspine-inclusion-exclusion.txt"
    schema = PECARN_CSPINE_INCLUSION
elif "IAI" in mode:
    prompt = "prompts/inclusion/iai-inclusion-exclusion.txt"
    schema = PECARN_IAI_INCLUSION
    db = SqliteDict("outputs/IAI_apr_23_first_and_second_pass.sqlite", tablename="inference")
    all_encounter_keys = [k.split("_")[0] for k in db.keys()]
    encounter_key_counts = Counter(all_encounter_keys)
    encounter_keys_complete = [k for k in encounter_key_counts if encounter_key_counts[k] == 3]
    print(len(df), "rows before")
    df = df[~df.EncounterKey.isin(encounter_keys_complete)]
    print(len(df), "rows after")
    # exit()
elif "TBI" in mode:
    prompt = "prompts/inclusion/tbi-inclusion-exclusion.txt"
    schema = PECARN_TBI_INCLUSION
else:
    raise Exception("Unknown inclusion prompt type")

config = versa.ExtractionConfig(df=df,
                        prompt_file=prompt,
                        tool_calling=True,
                        llm="gpt-5-nano-2025-08-07",
                        response_schema=schema, # type: ignore
                        n_replicates=3,
                        n_workers=200)

modu = versa.Extraction(config)
df_inference = modu.extract(f"outputs/{mode}_apr_23")

# python march/05_full_inclusion.py --source cohort/apr_23_all_traumas.csv --mode TBI
# python march/05_full_inclusion.py --source cohort/apr_23_all_traumas.csv --mode IAI
# python march/05_full_inclusion.py --source cohort/apr_23_all_traumas.csv --mode CSPINE
