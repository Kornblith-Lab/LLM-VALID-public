from dataclasses import dataclass
from sqlitedict import SqliteDict

import pandas as pd
from tap import tapify

from march.helpers import PECARN_TBI_OLDER, PECARN_TBI_YOUNGER, PECARN_IAI, PECARN_CSPINE
import valid.versa as versa

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
    prompt = "prompts/extraction/cspine.txt"
    schema = PECARN_CSPINE
elif "IAI" in mode:
    prompt = "prompts/extraction/iai.txt"
    schema = PECARN_IAI
elif "TBI_YOUNGER" in mode:
    prompt = "prompts/extraction/tbi_younger.txt"
    schema = PECARN_TBI_YOUNGER
elif "TBI_OLDER" in mode:
    prompt = "prompts/extraction/tbi_older.txt"
    schema = PECARN_TBI_OLDER
else:
    raise Exception("Unknown inclusion prompt type")

if "TBI" in mode:
    print(f"Beginning with {len(df)} rows")
    df = df[df.risk_gcs >= 14]
    df = df[df.recent_injury == "Yes"]
    if "YOUNGER" in mode:
        df = df[df.Age < 2]
        print("Filtering to keep only YOUNGER patients (<2)")
    elif "OLDER" in mode:
        df = df[df.Age >= 2]
        print("Filtering to keep only OLDER patients (>=2)")
    else:
        raise Exception("Unknown mode, unable to filter by age for TBI")
    print(f"After filtering, ending with {len(df)} rows")

    print(f"For today, we are further filtering by a subset to manage costs")
    df = df[6000:]
    # print(df)
    print(f"After additional filtering, ending with the final {len(df)} rows")
    print(f"DOUBLE CHECK THE CODE to make sure we are processing the right half/subset of notes")

    y = input("Type yes (y) to continue: ")
    if y.lower() not in ["yes", "y"]: exit()
elif "CSPINE" in mode:
    ems_modes = { "Ground Ambulance", "Ground Ambulance Scene Transfer", "Ambulance", "SF Fire Department", "Rotor Wing", "Ground Ambulance Interfacility Transfer", "Transfer - System Arrival", "Fixed Wing", "American Medical Response", "King American", "Pro Transport-1", "St Joseph's Ambulance" }

    print(f"Beginning with {len(df)} rows")
    df["ems"] = df.ArrivalMethod.isin(ems_modes).map(lambda ems: "Yes" if ems else "No")
    df["has_secondary_inclusion"] = (df[["evaluation_by_trauma_team", "collar", "ems"]] == "Yes").any(axis=1)

    print("Counts of blunt_injury in the full cohort:")
    print(df.blunt_injury.value_counts())
    print("Counts of secondary inclusion among those with blunt_injury:")
    print(df[df.blunt_injury == "Yes"].has_secondary_inclusion.value_counts())

    df = df[(df.blunt_injury == "Yes") & (df.has_secondary_inclusion == True)]
    print(f"After filtering, ending with {len(df)} rows")

    y = input("Type yes (y) to continue: ")

    if y.lower() not in ["yes", "y"]: exit()
elif "IAI" in mode:
    print(f"Beginning with {len(df)} rows")
    print(df.blunt_injury.value_counts())
    df = df[df.blunt_injury == "Yes"]
    print(f"After filtering, ending with {len(df)} rows")
    y = input("Type yes (y) to continue: ")
    if y.lower() not in ["yes", "y"]: exit()
else:
    raise Exception("Unknown mode, exiting")

output_file = SqliteDict(f"outputs/extraction_{mode}_apr_23.sqlite", tablename="inference")
completed_items = pd.Series(list(output_file.keys()))
completed_items = completed_items.apply(lambda s: s.split("_")[0]).value_counts()
completed_items = completed_items[completed_items >= 3].reset_index()["index"].values.tolist()

df = df[~df.EncounterKey.isin(set(completed_items))]
print(f"{len(df)} encounters remaining")
exit()

config = versa.ExtractionConfig(df=df,
                        prompt_file=prompt,
                        tool_calling=True,
                        llm="gpt-5-mini-2025-08-07",
                        response_schema=schema, # type: ignore
                        n_replicates=3,
                        n_workers=25)

modu = versa.Extraction(config)
df_inference = modu.extract(f"outputs/extraction_{mode}_apr_23")

# python march/06_full_extraction.py --source outputs/TBI_apr_23.csv --mode TBI_YOUNGER
# python march/06_full_extraction.py --source outputs/TBI_apr_23.csv --mode TBI_OLDER
# python march/06_full_extraction.py --source outputs/IAI_apr_23.csv --mode IAI
# python march/06_full_extraction.py --source outputs/CSPINE_apr_23.csv --mode CSPINE


# TBI OLDERS
# Miss on:
# row 4752, replicate 0
# row 9614, replicate 2