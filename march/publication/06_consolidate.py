from dataclasses import dataclass
import json
from sqlitedict import SqliteDict

import pandas as pd
from tap import tapify

from march.helpers import PECARN_TBI_OLDER, PECARN_TBI_YOUNGER, PECARN_IAI, PECARN_CSPINE
import valid.versa as versa

# ALL TRAUMA CHIEF COMPLAINTS
# ('TRAUMA', 'MOTOR VEHICLE CRASH', 'ASSAULT VICTIM', 'INJURY', 'MOTOR VEHICLE CRASH', 'BICYCLE CRASH', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN', 'FALL', 'HEAD INJURY', 'FALL', 'TRAUMA', 'HEAD LACERATION', 'FACIAL INJURY', 'ASSAULT VICTIM', 'MOTOR VEHICLE CRASH', 'FACIAL LACERATION', 'CONCUSSION', 'BICYCLE CRASH', 'INJURY', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN', 'ORBITAL TRAUMA', 'FACIAL TRAUMA', 'NECK PAIN', 'TRAUMA', 'MOTOR VEHICLE CRASH', 'NECK INJURY', 'HEAD INJURY', 'ASSAULT VICTIM', 'HEADACHE', 'BACK PAIN', 'TORTICOLLIS', 'SHOULDER PAIN', 'HEAD LACERATION', 'SHOULDER INJURY', 'INJURY', 'MOTOR VEHICLE CRASH', 'BICYCLE CRASH', 'MOTORCYCLE CRASH', 'MOTOR VEHICLE VS PEDESTRIAN')

@dataclass
class Args:
    source: str = "outputs/IAI_apr_23.csv"

args = tapify(Args)

df = pd.read_csv(args.source)
prompt = "prompts/extraction/iai.txt"
schema = PECARN_IAI

config = versa.ExtractionConfig(df=df,
                        prompt_file=prompt,
                        tool_calling=True,
                        llm="gpt-5-mini-2025-08-07",
                        response_schema=schema, # type: ignore
                        n_replicates=3,
                        n_workers=25)

modu = versa.Extraction(config)

print(f"Beginning with {len(df)} rows")
print(df.blunt_injury.value_counts())
df = df[df.blunt_injury == "Yes"]
print(f"After filtering, ending with {len(df)} rows")
y = input("Type yes (y) to continue: ")
if y.lower() not in ["yes", "y"]: exit()

output_file = SqliteDict(f"outputs/extraction_IAI_apr_23.sqlite", tablename="inference")
completed_items = pd.Series(list(output_file.keys()))
completed_items = completed_items.apply(lambda s: s.split("_")[0]).value_counts()
completed_items = completed_items[completed_items >= 3].reset_index()["index"].values.tolist()

def get_agg(encounter_key: str):
    reps = []
    for r in range(0,3):
        reps.append(json.loads(output_file[f"{encounter_key}_rep{r}"]))
    return modu._aggregate_replicates(reps)
    # exit()

# df = df[~df.EncounterKey.isin(set(completed_items))]
# print(f"{len(df)} encounters remaining")

df = df.loc[df.blunt_injury == "Yes"].copy().reset_index(drop=True)
initial_len = len(df)

df["inference"] = df["EncounterKey"].apply(get_agg)
inference_df = pd.json_normalize(df["inference"]).reset_index(drop=True)

df = pd.concat([df.drop(columns=["inference"]), inference_df], axis=1)

assert len(df) == initial_len, f"{len(df)} vs {initial_len}"

print(df)

df.to_csv("outputs/extraction_IAI_apr_23.csv")