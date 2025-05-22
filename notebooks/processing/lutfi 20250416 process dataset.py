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
# # Excel Text Preprocessing Script
#
# This script reads an Excel file (.xlsx) containing multiple sheets. For each sheet, it processes text from an 'original' column using a defined Python function and writes the output to an 'original_processed' column.
#
# **Features:**
# *   Handles multi-sheet Excel files.
# *   Applies a customizable text preprocessing function.
# *   Optionally processes only rows marked with `is_selected = TRUE`.
# *   Flexible input parameter handling (CLI > Colab Secrets > Env Vars > Colab UI > Defaults).
# *   Mounts Google Drive automatically if running in Colab.
# *   Overwrites the output file if it exists.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import argparse
import pandas as pd
from pathlib import Path
import sys
import re  # For the example processing function

# Attempt Colab-specific imports and setup
try:
    from google.colab import userdata, drive # type: ignore
    import ipywidgets as widgets
    from IPython.display import display
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Define dummy classes/functions if not in Colab to avoid NameErrors
    class userdata: # type: ignore
        @staticmethod
        def get(key): return None
    class drive: # type: ignore
        @staticmethod
        def mount(path): print("Google Drive mounting only available in Google Colab.")

# %% [markdown]
# ## 2. Google Drive Mount (Colab Only)
# If running in Google Colab, this cell attempts to mount Google Drive to `/content/drive`. You can then use paths like `drive/MyDrive/your_folder/your_file.xlsx`.

# %%
if IN_COLAB:
    print("Attempting to mount Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True) # force_remount can be helpful
        print("Google Drive mounted successfully at /content/drive")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("Proceeding without drive mount. Ensure files are accessible.")
else:
    print("Not running in Colab, skipping Google Drive mount.")

# %% [markdown]
# ## 3. Text Processing Function
# **Important:** Replace the example logic below with your actual Indonesian text preprocessing steps.

# %%
def preprocess_indonesian_text(text: str) -> str:
    """
    Preprocesses a single string of Indonesian text.
    Modify this function with your specific preprocessing requirements.

    Args:
        text: The raw Indonesian text string.

    Returns:
        The processed text string.
    """
    if not isinstance(text, str):
        return "" # Handle potential non-string data gracefully

    processed_text = text

    # Replacement map
    replacement_map = {
        "[USERNAME]": "xxxuserxxx",
        "[DATE]": "xxxdatexxx",
    }
    for key, value in replacement_map.items():
        processed_text = processed_text.replace(key, value)

    # Replace all standalone numbers to xxxnumberxxx
    processed_text = re.sub(r'\b\d+\b', 'xxxnumberxxx', processed_text)

    # Remove all hashtag words
    processed_text = re.sub(r'#(\w*[0-9a-zA-Z]+\w*[0-9a-zA-Z])', '', processed_text)

    # Example preprocessing: lowercase, remove punctuation, normalize whitespace
    processed_text = processed_text.lower()
    processed_text = re.sub(r'\s+', ' ', processed_text).strip() # Consolidate multiple spaces

    return processed_text

# %% [markdown]
# ## 4. Input Parameter Handling
# Defines and retrieves input parameters using the following priority:
# 1. Command Line Arguments (CLI)
# 2. Google Colab Secrets (`userdata.get`)
# 3. Environment Variables (`os.getenv`)
# 4. Google Colab UI Parameters (defined below)
# 5. Hardcoded defaults

# %%
# --- Google Colab UI Parameter Definitions ---
# These provide interactive input fields when run in Google Colab.
# They act as a fallback *within Colab* if CLI/Secrets/Env vars aren't set.
# For paths, consider using relative paths from your Drive root, e.g., 'drive/MyDrive/data/input.xlsx'
DATASET_PATH_PARAM = "./dataset.xlsx" # @param {type:"raw"}
DATASET_OUTPUT_PATH_PARAM = None # @param {type:"raw"}
ONLY_SELECTED_PARAM = False # @param {type:"boolean"}

print("--- Determining Script Parameters ---")

# --- Hardcoded Default Values ---
DEFAULT_DATASET_PATH = ""
DEFAULT_DATASET_OUTPUT_PATH = "" # Default is empty, logic will set it to DATASET_PATH later
DEFAULT_ONLY_SELECTED = True

# --- Helper function to robustly parse boolean inputs from various sources ---
def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 't', 'y', 'yes')
    # Consider integers 0/1? For now, default to False for others.
    return False

# --- Initialize parameter variables ---
dataset_path = None
dataset_output_path = None
only_selected = None
param_source = {} # Track the origin of each parameter for clarity

# --- 1. Command Line Arguments (Highest Priority) ---
# This block only runs if the script is executed directly (not imported or in basic notebook cells)
# Use parse_known_args to avoid conflicts with args injected by environments like Jupyter
parser = argparse.ArgumentParser(description="Preprocess text in Excel sheets.", add_help=True)
parser.add_argument("--DATASET_PATH", type=str, help="Path to the input xlsx file.")
parser.add_argument("--DATASET_OUTPUT_PATH", type=str, help="Path for the output xlsx file (optional).")
parser.add_argument("--ONLY_SELECTED", type=str, help="Process only selected rows? (True/False)")

# Check if running in a non-interactive mode where CLI args are expected
if __name__ == "__main__" and not IN_COLAB and sys.stdin.isatty(): # sys.stdin.isatty is a heuristic
    print("Parsing Command Line Arguments...")
    args, unknown = parser.parse_known_args()
    if args.DATASET_PATH:
        dataset_path = args.DATASET_PATH
        param_source['DATASET_PATH'] = 'CLI'
    if args.DATASET_OUTPUT_PATH:
        dataset_output_path = args.DATASET_OUTPUT_PATH
        param_source['DATASET_OUTPUT_PATH'] = 'CLI'
    if args.ONLY_SELECTED is not None:
        only_selected = parse_bool(args.ONLY_SELECTED)
        param_source['ONLY_SELECTED'] = 'CLI'
    # print(f"CLI Args found: {args}") # Less verbose

# --- 2. Google Colab Secrets (Requires Colab Environment) ---
if IN_COLAB:
    # print("Checking Google Colab Secrets...") # Less verbose
    secret_path = None
    secret_output_path = None
    secret_only_selected = None
    try:
        secret_path = userdata.get('DATASET_PATH')
        secret_output_path = userdata.get('DATASET_OUTPUT_PATH')
        secret_only_selected = userdata.get('ONLY_SELECTED')
    except Exception as e:
        pass

    if dataset_path is None and secret_path:
        dataset_path = secret_path
        param_source['DATASET_PATH'] = 'Colab Secret'
    if dataset_output_path is None and secret_output_path:
        dataset_output_path = secret_output_path
        param_source['DATASET_OUTPUT_PATH'] = 'Colab Secret'
    if only_selected is None and secret_only_selected is not None:
        only_selected = parse_bool(secret_only_selected)
        param_source['ONLY_SELECTED'] = 'Colab Secret'

# --- 3. Environment Variables ---
# print("Checking Environment Variables...") # Less verbose
env_path = os.getenv('DATASET_PATH')
env_output_path = os.getenv('DATASET_OUTPUT_PATH')
env_only_selected = os.getenv('ONLY_SELECTED')

if dataset_path is None and env_path:
    dataset_path = env_path
    param_source['DATASET_PATH'] = 'Environment Variable'
if dataset_output_path is None and env_output_path:
    dataset_output_path = env_output_path
    param_source['DATASET_OUTPUT_PATH'] = 'Environment Variable'
if only_selected is None and env_only_selected is not None:
    only_selected = parse_bool(env_only_selected)
    param_source['ONLY_SELECTED'] = 'Environment Variable'

# --- 4. Google Colab UI Parameters (Used if running in Colab and not set by higher priority methods) ---
if IN_COLAB:
    # print("Checking Google Colab UI Parameters...") # Less verbose
    if dataset_path is None and DATASET_PATH_PARAM: # Check if the UI param has a non-empty value
        dataset_path = DATASET_PATH_PARAM
        param_source['DATASET_PATH'] = 'Colab UI Param'
    if dataset_output_path is None and DATASET_OUTPUT_PATH_PARAM: # Check if the UI param has a non-empty value
        dataset_output_path = DATASET_OUTPUT_PATH_PARAM
        param_source['DATASET_OUTPUT_PATH'] = 'Colab UI Param'
    if only_selected is None:
         # The boolean UI param always has a value (True/False), so we take it if 'only_selected' is still None
         only_selected = ONLY_SELECTED_PARAM # Directly assign the boolean value
         param_source['ONLY_SELECTED'] = 'Colab UI Param'


# --- 5. Apply Hardcoded Defaults (Lowest Priority) ---
# print("Applying defaults if necessary...") # Less verbose
if dataset_path is None:
    dataset_path = DEFAULT_DATASET_PATH
    # Only mark as default if it truly wasn't set by any other means
    param_source.setdefault('DATASET_PATH', 'Default')
if only_selected is None:
    only_selected = DEFAULT_ONLY_SELECTED
    param_source.setdefault('ONLY_SELECTED', 'Default')

# --- Final Parameter Validation and Setup ---
if not dataset_path:
    raise ValueError("DATASET_PATH is required but was not provided via any method (CLI, Secret, Env Var, Colab UI). Please specify the input file path.")

# Default output path to input path if not specified
if not dataset_output_path:
    dataset_output_path = dataset_path
    # Mark the source only if it wasn't set previously
    param_source.setdefault('DATASET_OUTPUT_PATH', f'Default (same as input)')
else:
    # Ensure the source is marked if it came from UI/Secret/Env/CLI but wasn't set earlier
     param_source.setdefault('DATASET_OUTPUT_PATH', 'Unknown') # Fallback if logic missed it

# Convert paths to Path objects for robust handling
dataset_path = Path(dataset_path)
dataset_output_path = Path(dataset_output_path)

# Ensure 'only_selected' is definitively boolean after all inputs
only_selected = parse_bool(only_selected)

# --- Print final configuration ---
print("\n--- Final Configuration ---")
print(f"Input Path : {dataset_path} (Source: {param_source.get('DATASET_PATH', 'N/A')})")
print(f"Output Path: {dataset_output_path} (Source: {param_source.get('DATASET_OUTPUT_PATH', 'N/A')})")
print(f"Only Selected: {only_selected} (Source: {param_source.get('ONLY_SELECTED', 'N/A')})")
print("-" * 27 + "\n")

# %% [markdown]
# ## 5. Core Processing Logic
# Loads the Excel file, iterates through sheets, applies the preprocessing function based on the `ONLY_SELECTED` flag, and stores results.

# %%
print(f"--- Starting Processing ---")
print(f"Loading data from: {dataset_path}")

# Validate input file existence
if not dataset_path.is_file():
    # Provide a more helpful message if in Colab and path seems relative
    if IN_COLAB and not str(dataset_path).startswith('/') and not str(dataset_path).startswith('drive/'):
         print(f"Warning: Input file not found at '{dataset_path}'. If using Google Drive, ensure the path starts with 'drive/MyDrive/...' or provide the full '/content/drive/MyDrive/...' path.")
    raise FileNotFoundError(f"Input file not found: {dataset_path}")

# Read all sheets using sheet_name=None
try:
    all_sheets_dict = pd.read_excel(dataset_path, sheet_name=None)
    print(f"Successfully loaded {len(all_sheets_dict)} sheets: {list(all_sheets_dict.keys())}")
except Exception as e:
    print(f"Error reading Excel file '{dataset_path}': {e}")
    raise

processed_sheets_dict = {} # To hold the processed dataframes

# Process each sheet individually
for sheet_name, df in all_sheets_dict.items():
    print(f"\nProcessing sheet: '{sheet_name}'...")
    df_processed = df.copy() # Work on a copy

    # Define required columns based on the mode
    required_cols = ['original']
    if only_selected:
        required_cols.append('is_selected')
    # 'original_processed' is also needed, but we can create it if missing

    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        print(f"Warning: Sheet '{sheet_name}' is missing required column(s): {missing_cols}. Skipping processing for this sheet.")
        processed_sheets_dict[sheet_name] = df # Keep original if skipping
        continue

    # Ensure 'original_processed' column exists for storing results
    if 'original_processed' not in df_processed.columns:
        # Insert it after 'original' if possible, otherwise append
        try:
            original_col_idx = df_processed.columns.get_loc('original')
            df_processed.insert(original_col_idx + 1, 'original_processed', pd.NA)
        except KeyError: # Should not happen if check above passed, but safe practice
             df_processed['original_processed'] = pd.NA
    else:
        # Optional: Clear existing content if you always want fresh processing?
        # df_processed['original_processed'] = pd.NA
        pass # Keep existing content unless explicitly processing the row

    # --- Apply the preprocessing function based on the 'only_selected' flag ---
    rows_processed_count = 0
    if only_selected:
        # Create a boolean mask based on 'is_selected'
        try:
            # Convert various truthy values to boolean, handle NAs as False
            selection_mask = df_processed['is_selected'].apply(lambda x: parse_bool(x) if pd.notna(x) else False)
        except Exception as e:
            print(f"Warning: Could not reliably parse 'is_selected' column in sheet '{sheet_name}'. Error: {e}. Attempting direct boolean conversion.")
            # Fallback attempt assuming it might already be boolean-like
            try:
                selection_mask = df_processed['is_selected'].astype(bool)
            except:
                 print(f"Error: Failed to create selection mask for sheet '{sheet_name}'. Skipping filtering; processing ALL rows instead.")
                 selection_mask = pd.Series([True] * len(df_processed)) # Process all if conversion fails

        target_rows = df_processed.loc[selection_mask, 'original']
        if not target_rows.empty:
            print(f"Applying processing to {len(target_rows)} rows where 'is_selected' is True.")
            processed_values = target_rows.apply(preprocess_indonesian_text)
            # Update 'original_processed' ONLY for these selected rows
            df_processed.loc[selection_mask, 'original_processed'] = processed_values
            rows_processed_count = len(target_rows)
        else:
            print("No rows found with 'is_selected' as True.")

    else:
        # Process all rows if 'only_selected' is False
        print(f"Applying processing to all {len(df_processed)} rows ('only_selected' is False).")
        if 'original' in df_processed.columns:
            df_processed['original_processed'] = df_processed['original'].apply(preprocess_indonesian_text)
            rows_processed_count = len(df_processed)
        else:
             print("Warning: 'original' column not found, cannot process.") # Should be caught earlier

    processed_sheets_dict[sheet_name] = df_processed
    print(f"Sheet '{sheet_name}' processing complete. {rows_processed_count} rows updated.")

# %% [markdown]
# ## 6. Save Processed Data
# Writes the processed dataframes back to a single Excel file, overwriting the file if it already exists.

# %%
print(f"\n--- Saving Processed Data ---")
print(f"Output will be saved to: {dataset_output_path}")

try:
    # Ensure the output directory exists, create if not
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ExcelWriter to save all sheets. Default mode 'w' overwrites the file.
    with pd.ExcelWriter(dataset_output_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df_processed in processed_sheets_dict.items():
            df_processed.to_excel(writer, sheet_name=sheet_name, index=False) # index=False prevents writing row numbers

    print(f"Successfully saved processed data to {dataset_output_path}")

except Exception as e:
    print(f"Error writing output file '{dataset_output_path}': {e}")
    # Provide hints for common Colab errors
    if IN_COLAB and 'Errno 30' in str(e) and 'read-only file system' in str(e) and not str(dataset_output_path).startswith('/content/drive'):
         print("Hint: You might be trying to save outside of the mounted Google Drive or the allowed Colab storage. Ensure the output path starts with '/content/drive/MyDrive/...' or is within the Colab temporary storage.")
    raise

print("\n--- Script Finished ---")
