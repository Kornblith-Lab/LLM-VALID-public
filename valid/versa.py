from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Type, Union
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from pydantic import BaseModel
from openai import AzureOpenAI
import os

from sqlitedict import SqliteDict
from tqdm.auto import tqdm

load_dotenv(find_dotenv())
API_KEY: str = os.environ.get("API_KEY", "")
API_VERSION = "2025-04-01-preview"
RESOURCE_ENDPOINT = os.environ.get("RESOURCE_ENDPOINT", "")


assert len(API_KEY) == 88, f"API_KEY of length {len(API_KEY)} is not correct"
assert RESOURCE_ENDPOINT != "" 
assert API_VERSION != ""

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

@dataclass
class ToolMetadata:
    tool_name: str
    tool_description: str

# pecarn_tool = 

@dataclass
class ExtractionConfig:
    prompt_file: Path|str
    response_schema: Union[Type[BaseModel], dict]
    csv_file: Optional[Path|str] = None
    df: Optional[pd.DataFrame] = None
    note_col: str = "note_text"
    llm: str = "gpt-5-nano-2025-08-07"
    debug: Optional[int] = None
    tool_calling: bool = False
    n_replicates: int = 1
    n_workers: int = 100
    tool_metadata: ToolMetadata = field(default_factory=lambda: ToolMetadata("pecarn_extraction", "Extract PECARN predictors from clinical notes of children in the emergency department")) # type: ignore

    def __post_init__(self):
        if self.tool_calling:
            assert type(self.response_schema) == dict

        assert self.csv_file is not None or self.df is not None, "You must provide either a CSV or a dataframe as input"
        assert self.n_replicates > 0, "n_replicates must be greater than 0"

class Extraction:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.prompt_text = Path(self.config.prompt_file).read_text()
        # print(self.prompt_text)
    def extract(self, output_path: Path|str):
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        df = self.config.df.copy().reset_index() if self.config.df is not None else pd.read_csv(self.config.csv_file) # type: ignore
        db = SqliteDict(output_path.with_suffix(".sqlite"), tablename="inference", autocommit=True)

        input_token_count = 0
        output_token_count = 0

        if self.config.debug:
            df = df.head(self.config.debug)

        # Submit futures for all replicates
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as ex:
            futures = []
            for idx, row in tqdm(df.iterrows()):
                for replicate_num in range(self.config.n_replicates):
                    futures.append(ex.submit(self._extract_single, idx, row, replicate_num))

            # Dictionary to collect results by row index
            results_by_row = {}
            pbar = tqdm(as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    completion, idx, row, replicate_num = future.result()
                    if not self.config.tool_calling:
                        response = completion.choices[0].message

                        if response.refusal: raise Exception(f"Model refused: {response.refusal}")
                        if not response.parsed: raise Exception("No parsed output")

                        json_text = response.parsed.model_dump_json()
                    else:
                        response = completion.choices[0].message.tool_calls # type: ignore
                        if response is None: raise Exception(f"Model error")
                        json_text = response[0].function.arguments

                    # Parse JSON and collect results
                    parsed_result = json.loads(json_text)

                    # Store in results_by_row
                    if idx not in results_by_row:
                        results_by_row[idx] = []
                    results_by_row[idx].append(parsed_result)

                    # Store raw in database
                    db[f"{str(row.EncounterKey)}_rep{replicate_num}"] = json_text

                    input_token_count += completion.usage.prompt_tokens # type: ignore
                    output_token_count += completion.usage.completion_tokens # type: ignore
                    # pbar.update(1)
                    pbar.set_description(f"Total input {input_token_count:,} tokens and {output_token_count:,} output tokens used")
                except Exception as e:
                    print(f"Error on row {idx}, replicate {replicate_num}: {e}")
                    # pbar.update(1)

            # pbar.close()

            # Aggregate results and populate inference column
            for idx in results_by_row:
                replicates = results_by_row[idx]
                # Skip aggregation if n_replicates=1
                if self.config.n_replicates == 1:
                    result = replicates[0]
                else:
                    result = self._aggregate_replicates(replicates)
                result_json = json.dumps(result)
                df.loc[df.index == idx, "inference"] = result_json

            df.to_csv(output_path.with_suffix(".csv"), index=False)

            df["inference"] = df["inference"].apply(lambda text: json.loads(text) if not pd.isna(text) else "")
            inference_df = pd.json_normalize(df["inference"]) # type: ignore

            df = pd.concat([df.drop(columns=["inference"]), inference_df], axis=1)
            df.to_csv(output_path.with_suffix(".csv"), index=False)
            return df
    
    def _extract_single(self, idx, row: pd.Series, replicate_num: int):
        messages = [
            { "role": "system", "content": self.prompt_text },
            { "role": "user", "content": row[self.config.note_col] }
        ]

        if not self.config.tool_calling:
            completion = client.chat.completions.parse(
                model=self.config.llm,
                reasoning_effort="medium",
                messages=messages,
                response_format=self.config.response_schema, # type: ignore
            )
        else:
            completion = client.chat.completions.create(
                model=self.config.llm,
                messages=messages,
                reasoning_effort="medium",
                n=1,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "pecarn_extraction",
                            "description": "Extract PECARN predictors from clinical notes of children in the emergency department",
                            "strict": True,
                            "parameters": self.config.response_schema
                        }
                    } # type: ignore
                ], # type: ignore
                tool_choice="required"
            )

        if self.config.debug:
            print("PROMPT " * 10)
            print(self.prompt_text)
            print("=" * 20)
            print("NOTE " * 10)
            print(row[self.config.note_col])
            print("=" * 20)
            print("RESPONSE " * 10)
            print(completion.choices[0].message)
            print("=" * 20)

        return completion, idx, row, replicate_num

    def _aggregate_replicates(self, replicates: list[dict], remove_intermediates=True) -> dict:
        """
        Aggregate multiple replicate results by field type:
        - Binary Yes/No fields: average and round
        - Numeric fields: use mode
        - Categorical fields: use mode
        """
        if len(replicates) == 0:
            return {}

        if len(replicates) == 1:
            return replicates[0]

        aggregated = {}

        # Get all keys across all replicates
        all_keys = set()
        for rep in replicates:
            all_keys.update(rep.keys())

        for key in all_keys:
            values = [rep.get(key) for rep in replicates if key in rep]

            if not values:
                continue

            # Skip if any value is None or missing
            if any(v is None for v in values):
                aggregated[key] = values[0] if values else None
                continue

            # Determine field type and aggregate accordingly
            # Check if it's a numeric type
            try:
                numeric_values = [float(v) for v in values if v is not None]
                # Use mode for numeric fields
                from statistics import mode, StatisticsError
                try:
                    aggregated[key] = mode(numeric_values)
                except StatisticsError:
                    # If mode fails (all unique), use the median or first value
                    aggregated[key] = int(sorted(numeric_values)[len(numeric_values)//2])
                continue
            except (ValueError, TypeError):
                pass

            # Check if it's a binary Yes/No field
            if all(v in ["Yes", "No"] for v in values):
                # Convert to numeric, average, and round
                numeric = [1 if v == "Yes" else 0 for v in values]
                avg = sum(numeric) / len(numeric)
                aggregated[key] = "Yes" if avg >= 0.5 else "No"

            # Check for intermediate values that need mapping before Yes/No aggregation
            elif all(v in ["Yes", "No", "Yes (non-severe)", "Suspected"] for v in values) and remove_intermediates:
                # Map intermediate values
                mapped = []
                for v in values:
                    if v == "Yes (non-severe)":
                        mapped.append("No")
                    elif v == "Suspected":
                        mapped.append("Yes")
                    else:
                        mapped.append(v)

                # Now average as binary Yes/No
                numeric = [1 if v == "Yes" else 0 for v in mapped]
                avg = sum(numeric) / len(numeric)
                aggregated[key] = "Yes" if avg >= 0.5 else "No"

            # Otherwise use mode for categorical fields
            else:
                from statistics import mode, StatisticsError
                try:
                    aggregated[key] = mode(values)
                except StatisticsError:
                    print(key, values)
                    # If mode fails (all unique), use the first value
                    aggregated[key] = values[0]

        return aggregated

class InclusionDataModel(BaseModel):
    penetrating_trauma: str
    outside_imaging: str
    cpr: str
    recent_injury: str
    trivial_injury: str
    brain_tumor: str
    neurological_disorder: str

def main():
    config = ExtractionConfig(csv_file="derived/tbi.csv", 
                              prompt_file="prompts/inclusion-exclusion.txt",
                              response_schema=InclusionDataModel)
    
    modu = Extraction(config)

    modu.extract("outputs/tbi_inclusion_feb_19")

if __name__ == "__main__":
    main()
