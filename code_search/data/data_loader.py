import json
import pandas as pd
from typing import List

def load_data_from_jsonl(file_paths: List[str]) -> pd.DataFrame:
    """Loads data from one or more JSONL files into a pandas DataFrame.

    Args:
        file_paths: A list of paths to JSONL files.

    Returns:
        A pandas DataFrame containing the loaded data, with columns 'query' and 'code'.
    """
    all_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    json_line = json.loads(line)
                    all_data.append(json_line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}: {line}")
                    continue
    df = pd.DataFrame(all_data)
    if 'query' not in df.columns or 'code' not in df.columns:
        raise ValueError("JSONL files must contain 'query' and 'code' fields.")
    return df
