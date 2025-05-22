# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Dataset Combination and Splitting Script
#
# This script combines informal/formal sentence pairs from two sources:
# 1. Pre-split text files hosted at URLs.
# 2. An Excel file (`.xlsx`) located in Google Drive containing multiple sheets.
#
# It then shuffles the combined data and splits it into train, development, and test sets based on specified ratios.
#
# **Outputs:**
# *   Separate `.inf` (informal) and `.for` (formal) files for train, dev, and test splits in `OUTPUT_DIR/dataset/`.
# *   A consolidated Excel file `OUTPUT_DIR/dataset/dataset.xlsx` with columns `informal`, `formal`, and `split`.

# %% [markdown]
# ## 1. Setup and Imports

# %%
# !pip install -q pandas openpyxl requests tqdm ipywidgets python-dotenv

# %%
SCRIPT_NAME = "lutfi_20250422_combine_datasets"

import os
import pandas as pd
from pathlib import Path
import requests # Changed from 'request' to 'requests' standard library
import math
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse # To help create default filenames

# tqdm for progress bars, notebook version for Jupyter/Colab
from tqdm.notebook import tqdm

# Attempt Colab-specific imports for Drive mounting
try:
    from google.colab import drive # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Define dummy class if not in Colab to avoid NameErrors during Drive check
    class drive: # type: ignore
        @staticmethod
        def mount(path, force_remount=False): print("Google Drive mounting only available in Google Colab.")

# %% [markdown]
# ## 2. Configuration Parameters

# %%
# --- Input URLs for pre-split data ---
BASE_URL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/"
TRAIN_INF_URL = f"{BASE_URL}train.inf"
TRAIN_FOR_URL = f"{BASE_URL}train.for"
DEV_INF_URL = f"{BASE_URL}dev.inf"
DEV_FOR_URL = f"{BASE_URL}dev.for"
TEST_INF_URL = f"{BASE_URL}test.inf"
TEST_FOR_URL = f"{BASE_URL}test.for"

# --- Input Excel File from Google Drive ---
# NOTE: Requires Google Drive to be mounted in Colab.
# Example path, adjust to your actual file location within MyDrive
XLSX_DRIVE_PATH = "/content/drive/MyDrive/Uni Subjects/Bengkod U22 Bahagia/ACADEMIC CHATBOT/Lunashimu Pembakuan Teks/Datasets by Lutfi/Datasets by Lutfi 20250416.xlsx"

# --- Excel Processing Parameters ---
# Column containing the informal sentences in the Excel file
INFORMAL_COLUMN_NAME = "original_processed"
# Ordered list of columns containing potential formal sentences (first non-empty found is used)
COLUMN_NAMES_STORING_FORMAL_SENTENCES = ["formal_gemini", "formal_gtrans"]
# Column indicating whether a row should be included (must contain TRUE/True/1)
SELECTION_COLUMN_NAME = "is_selected_for_combination"

# --- Output Configuration ---
# NOTE: Adjust the base output directory as needed
OUTPUT_DIR = Path(f"/content/drive/MyDrive/Uni Subjects/Bengkod U22 Bahagia/ACADEMIC CHATBOT/Lunashimu Pembakuan Teks/Notebooks/Outputs/Lutfi/{SCRIPT_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
OUTPUT_DATASET_SUBDIR = "dataset" # Subdirectory within OUTPUT_DIR

# --- Split Ratios ---
# Must sum to 1.0
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

# --- Other Settings ---
RANDOM_SEED = 42 # For reproducible shuffling and splitting

# %% [markdown]
# ## 3. Helper Functions

# %%
def get_lines_from_url(url: str) -> List[str]:
    """Fetches text data line by line from a URL, skipping empty lines."""
    print(f"Fetching data from {url}...")
    lines = []
    try:
        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        # Decode explicitly using UTF-8, handle potential errors
        response.encoding = 'utf-8'
        # Filter out empty lines that might result from split('\n') on empty strings or trailing newlines
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        print(f"Successfully fetched {len(lines)} lines.")
    except requests.exceptions.RequestException as e:
        # Catch specific request errors (connection, timeout, HTTP errors)
        print(f"Error fetching data from {url}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"An unexpected error occurred while processing {url}: {e}")
    return lines

# %%
def validate_and_pair(informal_lines: List[str], formal_lines: List[str], source_name: str) -> List[Tuple[str, str]]:
    """Validates if informal and formal lists have the same length and pairs them."""
    if len(informal_lines) != len(formal_lines):
        print(f"Warning: Mismatch in number of lines for {source_name}. Informal: {len(informal_lines)}, Formal: {len(formal_lines)}. Skipping this source.")
        return []
    if not informal_lines: # Also handles the case where formal_lines is empty due to the length check
         print(f"Warning: No data found for {source_name}. Skipping.")
         return []
    return list(zip(informal_lines, formal_lines))

# %%
def parse_bool(value) -> bool:
    """Robustly parses various truthy values to boolean."""
    if value is None: return False
    if isinstance(value, bool): return value
    val_str = str(value).strip().lower()
    return val_str in ('true', '1', 't', 'y', 'yes')

# %% [markdown]
# ## 4. Google Drive Mount (Colab Only)

# %%
if IN_COLAB:
    print("Attempting to mount Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True) # force_remount can be helpful for debugging
        print("Google Drive mounted successfully at /content/drive")
        # Verify the Excel file path exists after mounting
        if not Path(XLSX_DRIVE_PATH).is_file():
             print(f"Warning: Excel file not found at the specified path after mounting: {XLSX_DRIVE_PATH}")
             print("Please ensure the XLSX_DRIVE_PATH variable is correct and the file exists.")
             # Depending on requirements, you might want to raise an error here or proceed without Excel data
        else:
             print(f"Confirmed Excel file exists at: {XLSX_DRIVE_PATH}")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}. Proceeding without Drive access.")
        # If Drive mount fails, Excel processing will likely fail later unless the path is accessible otherwise
else:
    print("Not running in Google Colab, skipping Google Drive mount.")
    # Check if the Excel path exists locally if not in Colab
    if not Path(XLSX_DRIVE_PATH).is_file():
         print(f"Warning: Excel file not found at the specified local path: {XLSX_DRIVE_PATH}")
         print("Ensure the file exists or adjust the XLSX_DRIVE_PATH variable.")


# %% [markdown]
# ## 5. Load Data from Sources

# %%
all_sentence_pairs: List[Tuple[str, str]] = []

# --- 5.1 Load from URLs ---
print("\n--- Loading data from URLs ---")
url_sources = {
    "Train URL": (TRAIN_INF_URL, TRAIN_FOR_URL),
    "Dev URL": (DEV_INF_URL, DEV_FOR_URL),
    "Test URL": (TEST_INF_URL, TEST_FOR_URL),
}

for name, (inf_url, for_url) in url_sources.items():
    inf_lines = get_lines_from_url(inf_url)
    for_lines = get_lines_from_url(for_url)
    url_pairs = validate_and_pair(inf_lines, for_lines, name)
    if url_pairs:
        print(f"Adding {len(url_pairs)} pairs from {name}.")
        all_sentence_pairs.extend(url_pairs)
    else:
        print(f"Skipped adding pairs from {name} due to validation issues or empty data.")

# --- 5.2 Load from Excel File ---
print("\n--- Loading data from Excel file ---")
excel_pairs: List[Tuple[str, str]] = []
xlsx_path = Path(XLSX_DRIVE_PATH)

if xlsx_path.is_file():
    try:
        print(f"Reading Excel file: {xlsx_path}")
        # Read all sheets into a dictionary of DataFrames
        all_sheets_dict = pd.read_excel(xlsx_path, sheet_name=None, engine='openpyxl')
        print(f"Found {len(all_sheets_dict)} sheets: {list(all_sheets_dict.keys())}")

        for sheet_name, df in all_sheets_dict.items():
            print(f"Processing sheet: '{sheet_name}'...")
            sheet_added_count = 0
            # Check for essential columns
            required_columns = [INFORMAL_COLUMN_NAME, SELECTION_COLUMN_NAME] + COLUMN_NAMES_STORING_FORMAL_SENTENCES
            missing_columns = [col for col in required_columns if col not in df.columns]
            essential_missing = [col for col in [INFORMAL_COLUMN_NAME, SELECTION_COLUMN_NAME] if col not in df.columns]
            available_formal_cols = [col for col in COLUMN_NAMES_STORING_FORMAL_SENTENCES if col in df.columns]

            if essential_missing:
                print(f"  Skipping sheet '{sheet_name}'. Missing essential columns: {essential_missing}")
                continue
            if not available_formal_cols:
                 print(f"  Skipping sheet '{sheet_name}'. Missing all specified formal sentence columns: {COLUMN_NAMES_STORING_FORMAL_SENTENCES}")
                 continue
            if missing_columns:
                 print(f"  Note: Sheet '{sheet_name}' is missing optional formal columns: {missing_columns}")


            # Filter rows based on the selection column
            try:
                selection_mask = df[SELECTION_COLUMN_NAME].apply(parse_bool)
                df_selected = df[selection_mask].copy() # Use copy to avoid SettingWithCopyWarning
                print(f"  Found {len(df_selected)} rows marked for combination in sheet '{sheet_name}'.")
            except Exception as e:
                print(f"  Error processing selection column '{SELECTION_COLUMN_NAME}' in sheet '{sheet_name}': {e}. Skipping sheet.")
                continue

            if df_selected.empty:
                print(f"  No rows selected for combination in sheet '{sheet_name}'.")
                continue

            # Iterate through selected rows to find informal/formal pairs
            for index, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc=f"  Extracting pairs from '{sheet_name}'"):
                informal_sentence = row[INFORMAL_COLUMN_NAME]
                formal_sentence = None

                # Find the first valid formal sentence based on priority
                for formal_col in available_formal_cols:
                    potential_formal = row[formal_col]
                    # Check if it's not null/NA and not an empty string after stripping
                    if pd.notna(potential_formal) and str(potential_formal).strip():
                        formal_sentence = str(potential_formal).strip()
                        break # Found the highest priority formal sentence

                # Add the pair if both informal and formal are valid
                if pd.notna(informal_sentence) and str(informal_sentence).strip() and formal_sentence:
                    excel_pairs.append((str(informal_sentence).strip(), formal_sentence))
                    sheet_added_count += 1
                # Optionally log skipped rows
                # else:
                #     print(f"  Skipping row {index} in sheet '{sheet_name}': Informal='{informal_sentence}', Formal found='{formal_sentence is not None}'")

            print(f"  Added {sheet_added_count} valid pairs from sheet '{sheet_name}'.")

        if excel_pairs:
            print(f"\nSuccessfully extracted {len(excel_pairs)} pairs from the Excel file.")
            all_sentence_pairs.extend(excel_pairs)
        else:
            print("\nNo valid pairs extracted from the Excel file.")

    except FileNotFoundError:
        print(f"Error: Excel file not found at {xlsx_path}. Skipping Excel source.")
    except Exception as e:
        print(f"Error reading or processing Excel file {xlsx_path}: {e}")
        # Decide whether to continue without Excel data or raise the error
        # raise

else:
    print(f"Excel file not found at {xlsx_path}. Skipping Excel source.")


# %% [markdown]
# ## 6. Shuffle and Split Data

# %%
print("\n--- Shuffling and Splitting Data ---")

if not all_sentence_pairs:
    print("Error: No sentence pairs collected from any source. Cannot proceed with splitting.")
    # Exit or raise an error if no data is available
    # sys.exit(1) # Or raise ValueError("No data to process.")
else:
    total_pairs = len(all_sentence_pairs)
    print(f"Total sentence pairs collected: {total_pairs}")

    # Validate ratios
    if not math.isclose(TRAIN_RATIO + DEV_RATIO + TEST_RATIO, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0. Current sum: {TRAIN_RATIO + DEV_RATIO + TEST_RATIO}")

    # Shuffle the data
    print(f"Shuffling data with random seed: {RANDOM_SEED}...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_sentence_pairs)

    # Calculate split indices
    train_end_idx = math.floor(total_pairs * TRAIN_RATIO)
    dev_end_idx = train_end_idx + math.floor(total_pairs * DEV_RATIO)
    # The rest goes to test to ensure all data is used, handling potential float inaccuracies

    # Perform the split
    train_pairs = all_sentence_pairs[:train_end_idx]
    dev_pairs = all_sentence_pairs[train_end_idx:dev_end_idx]
    test_pairs = all_sentence_pairs[dev_end_idx:]

    # Separate into informal and formal lists for each split
    train_inf, train_for = zip(*train_pairs) if train_pairs else ([], [])
    dev_inf, dev_for = zip(*dev_pairs) if dev_pairs else ([], [])
    test_inf, test_for = zip(*test_pairs) if test_pairs else ([], [])

    print("\nData splitting complete:")
    print(f"  Train pairs: {len(train_inf)}")
    print(f"  Dev pairs  : {len(dev_inf)}")
    print(f"  Test pairs : {len(test_inf)}")
    print(f"  Total assigned: {len(train_inf) + len(dev_inf) + len(test_inf)} (should match total collected)")

    # Convert tuples from zip back to lists
    train_inf, train_for = list(train_inf), list(train_for)
    dev_inf, dev_for = list(dev_inf), list(dev_for)
    test_inf, test_for = list(test_inf), list(test_for)


# %% [markdown]
# ## 7. Write Output Files

# %%
print("\n--- Writing Output Files ---")

# Ensure output directory exists
output_dataset_dir = OUTPUT_DIR / OUTPUT_DATASET_SUBDIR
try:
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dataset_dir}")
except Exception as e:
     print(f"Error creating output directory {output_dataset_dir}: {e}")
     # Optionally raise error or exit if directory cannot be created
     # raise

# --- 7.1 Write .inf and .for files ---
split_data = {
    "train": (train_inf, train_for),
    "dev": (dev_inf, dev_for),
    "test": (test_inf, test_for),
}

for split_name, (inf_data, for_data) in split_data.items():
    inf_path = output_dataset_dir / f"{split_name}.inf"
    for_path = output_dataset_dir / f"{split_name}.for"

    try:
        with open(inf_path, 'w', encoding='utf-8') as f_inf:
            # Join sentences with newline, add trailing newline for POSIX compatibility
            f_inf.write("\n".join(inf_data) + "\n")
        print(f"  Successfully wrote {len(inf_data)} lines to {inf_path}")

        with open(for_path, 'w', encoding='utf-8') as f_for:
            f_for.write("\n".join(for_data) + "\n")
        print(f"  Successfully wrote {len(for_data)} lines to {for_path}")

    except Exception as e:
        print(f"  Error writing {split_name} files: {e}")

# --- 7.2 Write combined Excel file ---
excel_output_path = output_dataset_dir / "dataset.xlsx"

# Check if there's data to write
if train_inf or dev_inf or test_inf: # Check if any split has data
    try:
        # Create lists for the DataFrame columns
        informal_col = train_inf + dev_inf + test_inf
        formal_col = train_for + dev_for + test_for
        split_col = (['train'] * len(train_inf) +
                     ['dev'] * len(dev_inf) +
                     ['test'] * len(test_inf))

        # Create DataFrame
        output_df = pd.DataFrame({
            'informal': informal_col,
            'formal': formal_col,
            'split': split_col
        })

        # Save to Excel
        output_df.to_excel(excel_output_path, index=False, engine='openpyxl')
        print(f"\nSuccessfully wrote combined data ({len(output_df)} rows) to {excel_output_path}")

    except Exception as e:
        print(f"Error writing combined Excel file {excel_output_path}: {e}")
        if IN_COLAB and 'Errno 30' in str(e) and 'read-only file system' in str(e) and not str(excel_output_path).startswith('/content/drive'):
             print("Hint: You might be trying to save outside of the mounted Google Drive or the allowed Colab storage. Ensure the output path starts with '/content/drive/MyDrive/...' or is within the Colab temporary storage.")

else:
     print("\nSkipping combined Excel file creation as no data was processed.")


# %% [markdown]
# ## 8. Final Summary

# %%
print("\n--- Script Finished ---")
print(f"Output generated in: {output_dataset_dir.resolve()}")
print("Summary of created files:")
if 'output_df' in locals() and not output_df.empty: # Check if df was created and has data
    print(f"  - dataset.xlsx: {len(output_df)} rows")
    split_counts = output_df['split'].value_counts()
    for split_name, count in split_counts.items():
        inf_path = output_dataset_dir / f"{split_name}.inf"
        for_path = output_dataset_dir / f"{split_name}.for"
        print(f"  - {split_name}.inf: {count} lines ({'Exists' if inf_path.exists() else 'Write Failed'})")
        print(f"  - {split_name}.for: {count} lines ({'Exists' if for_path.exists() else 'Write Failed'})")
elif not all_sentence_pairs:
     print("  - No files generated as no data was collected.")
else:
     print("  - Text files (.inf/.for) might have been generated but the combined Excel file was not (or failed). Please check logs.")
