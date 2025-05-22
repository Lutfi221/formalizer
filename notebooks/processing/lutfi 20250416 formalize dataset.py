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
# # Excel Text Formalization Script (Gemini & Google Translate)
#
# This script reads an Excel file (.xlsx) with multiple sheets, processes text using Google Gemini and/or Google Translate API, and writes the formalized text back to specified columns.
#
# **Features:**
# *   Handles multi-sheet Excel files.
# *   Formalizes text using Google Gemini and/or Google Translate.
# *   Uses `original_processed` column if available, otherwise `original`.
# *   Processes in batches with exponential backoff for API calls.
# *   Optionally processes only rows marked `is_selected = TRUE`.
# *   Optionally skips rows where the target column is already filled.
# *   Flexible input parameter handling (CLI > Colab Secrets > Env Vars > Colab UI > Defaults).
# *   Mounts Google Drive automatically if running in Colab.
# *   Overwrites the output file if it exists.

# %% [markdown]
# ## 1. Setup and Imports

# %%
# !pip install -q pandas openpyxl python-dotenv google-generativeai googletrans==4.0.0-rc1 tqdm ipywidgets requests tenacity

# %%
SCRIPT_NAME = "lutfi_20250416_formalize_dataset"

import os
import argparse
import pandas as pd
from pathlib import Path
import sys
import time
import asyncio
import math
from tqdm.notebook import tqdm  # Use notebook-friendly tqdm
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Google Cloud / Gemini specific
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, InternalServerError, GoogleAPIError
    # Using tenacity for robust retries
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai library not found. Gemini processing will be unavailable.")
    GEMINI_AVAILABLE = False
    # Define dummy exception classes if library not installed
    class ResourceExhausted(Exception): pass
    class InternalServerError(Exception): pass
    class GoogleAPIError(Exception): pass

# Google Translate specific
try:
    # googletrans is unofficial, use with caution regarding stability/rate limits
    from googletrans import Translator # type: ignore
    GTRANS_AVAILABLE = True
except ImportError:
    print("Warning: googletrans library not found. Google Translate processing will be unavailable.")
    GTRANS_AVAILABLE = False

# Attempt Colab-specific imports and setup
try:
    from google.colab import userdata, drive # type: ignore
    import ipywidgets as widgets # type: ignore
    from IPython.display import display # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Define dummy classes/functions if not in Colab to avoid NameErrors
    class userdata: # type: ignore
        @staticmethod
        def get(key, default=None): return default # Match signature
    class drive: # type: ignore
        @staticmethod
        def mount(path, force_remount=False): print("Google Drive mounting only available in Google Colab.")

# Exponential backoff parameters
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 2
MAX_DELAY_SECONDS = 120

# %% [markdown]
# ## 2. Google Drive Mount (Colab Only)

# %%
if IN_COLAB:
    print("Attempting to mount Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True)
        print("Google Drive mounted successfully at /content/drive")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}. Proceeding without mount.")
else:
    print("Not running in Colab, skipping Google Drive mount.")

# %% [markdown]
# ## 3. API Interaction Functions (Customize Here)
# Define functions to interact with Gemini and Google Translate APIs.

# %%
# --- Gemini Formalization Function ---

# Configure retry logic for API calls using tenacity
# Retry on specific Google API errors known to be potentially transient
gemini_retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS),
    retry=retry_if_exception_type((ResourceExhausted, InternalServerError, GoogleAPIError, TimeoutError)),
    reraise=True # Reraise the exception if all retries fail
)

GEMINI_CLIENT = None # Initialize globally

def _initialize_gemini_client(api_key: str):
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        if not GEMINI_AVAILABLE:
             raise ImportError("google-generativeai library not installed.")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini processing.")
        try:
            genai.configure(api_key=api_key)
            # Test connection slightly (optional, list_models can be slow)
            # models = genai.list_models()
            # print("Gemini client configured.")
            GEMINI_CLIENT = genai # Use the configured module directly
        except Exception as e:
            print(f"Error configuring Gemini client: {e}")
            GEMINI_CLIENT = None # Ensure it's None if config fails
            raise
    return GEMINI_CLIENT

# Define the prompt structure globally or pass it if needed
GEMINI_SYSTEM_INSTRUCTION = """You are a native speaker of Indonesian. You are an expert in normalizing and formalizing Indonesian text. You MUST ONLY output formalized and normalized sentences in Indonesia without any extra text. The input and output snippets are separated by a line filled `---SNIPPET---`."""
GEMINI_USER_PROMPT_TEMPLATE = """Normalize and formalize the following Indonesian texts. The input and output snippets are separated by a line filled `---SNIPPET---`. Don't change special words surrounded by `xxx` (such as xxxnumberxxx) and leave it as it is. The word placements NEED TO STAY roughly the same whenever possible. Avoid rearranging the sentences. DON'T OUTPUT ANYTHING ELSE!

{batch_input_text}"""
GEMINI_SNIPPET_SEPARATOR = "\n---SNIPPET---\n"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or choose another appropriate model

@gemini_retry_decorator
def formalize_batch_using_gemini(texts: List[str], api_key: str) -> List[str]:
    """Formalizes a batch of Indonesian text using Google Gemini."""
    client = _initialize_gemini_client(api_key)
    if not client:
        print("Error: Gemini client not initialized.")
        return [""] * len(texts) # Return empty strings on failure

    if not texts:
        return []

    # Combine texts with the separator for a single API call
    batch_input = GEMINI_SNIPPET_SEPARATOR.join(texts)
    user_prompt = GEMINI_USER_PROMPT_TEMPLATE.format(batch_input_text=batch_input)

    model = client.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=GEMINI_SYSTEM_INSTRUCTION
        )

    try:
        # print(f"DEBUG: Sending prompt to Gemini:\n{user_prompt[:500]}...") # Optional debug
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig( # type: ignore
                # candidate_count=1, # Default is 1
                # stop_sequences=['...'], # If needed
                # max_output_tokens=..., # If needed
                temperature=0.8, # Lower temperature for more deterministic formalization
                response_mime_type="text/plain"
            )
        )

        # print(f"DEBUG: Received response from Gemini:\n{response.text[:500]}...") # Optional debug

        # Check for safety ratings or blocks if necessary
        # if response.prompt_feedback.block_reason:
        #     print(f"Warning: Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
        #     return ["<GEMINI_BLOCKED>"] * len(texts)

        # Parse the response
        processed_texts = response.text.split(GEMINI_SNIPPET_SEPARATOR)

        # Validate response length
        if len(processed_texts) != len(texts):
            print(f"Warning: Gemini response count mismatch. Expected {len(texts)}, got {len(processed_texts)}. Replacing with errors.")
            processed_texts = ["<GEMINI_RESPONSE_MISMATCH>"] * len(texts)

        return [t.strip() for t in processed_texts]

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        # Depending on the tenacity setup, this might be hit after retries fail.
        # Return error markers for the entire batch.
        return ["<GEMINI_API_ERROR>"] * len(texts)


# --- Google Translate Formalization Function ---

# Retry decorator for googletrans (less specific error types available)
gtrans_retry_decorator = retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=BASE_DELAY_SECONDS, max=MAX_DELAY_SECONDS),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)), # Retry on common network issues and general exceptions
    reraise=True
)

@gtrans_retry_decorator
async def formalize_batch_using_gtrans_async(texts: List[str]) -> List[str]:
    """Formalizes a batch of Indonesian text using Google Translate (async)."""
    if not GTRANS_AVAILABLE:
        raise ImportError("googletrans library not installed.")
    if not texts:
        return []

    try:
        # Use async context manager for the translator
        async with Translator() as translator:
             # Translate from Indonesian to Indonesian - this often triggers normalization
            translations = await translator.translate(texts, src='id', dest='id')

        # Extract the text from the translation results
        # Handle potential errors where translation might not return expected object
        processed_texts: List[str] = []
        if isinstance(translations, list):
            for t in translations:
                if hasattr(t, 'text'):
                    processed_texts.append(t.text.strip())
                else:
                     print(f"Warning: Unexpected item in gtrans result: {t}. Using error marker.")
                     processed_texts.append("<GTRANS_RESULT_ERROR>")
        elif hasattr(translations, 'text'): # Handle case where single string was passed and single result returned
            processed_texts.append(translations.text.strip())
            if len(texts) > 1: # This shouldn't happen with batch input, but safety check
                print("Warning: GTrans returned single result for multiple inputs.")
                processed_texts.extend(["<GTRANS_UNEXPECTED_SINGLE>"] * (len(texts) - 1))
        else:
            print(f"Warning: Unexpected result type from gtrans: {type(translations)}. Using error markers.")
            processed_texts = ["<GTRANS_UNEXPECTED_TYPE>"] * len(texts)


        # Basic length check (though API usually matches input count)
        if len(processed_texts) != len(texts):
            print(f"Warning: Google Translate response count mismatch. Expected {len(texts)}, got {len(processed_texts)}. Padding.")
            processed_texts.extend(["<GTRANS_RESPONSE_MISMATCH>"] * (len(texts) - len(processed_texts)))
            processed_texts = processed_texts[:len(texts)]

        return processed_texts

    except Exception as e:
        print(f"Error during Google Translate API call: {e}")
        # This might be hit after retries fail.
        # Return error markers for the entire batch.
        # Note: asyncio exceptions might need specific handling depending on context
        # but tenacity should handle the retries based on the exceptions caught.
        return ["<GTRANS_API_ERROR>"] * len(texts)

def formalize_batch_using_gtrans(texts: List[str]) -> List[str]:
    """Synchronous wrapper for the async Google Translate function."""
    # Required for running async code from a sync context like a Jupyter notebook cell
    # or a standard Python script execution.
    # In more complex async applications, manage the event loop differently.
    try:
        # Check if an event loop is already running (e.g., in Jupyter/IPython)
        loop = asyncio.get_running_loop()
        # If yes, schedule the async task and wait for it
        # This might require 'nest_asyncio' in some environments if loops are nested
        # !pip install nest_asyncio
        # import nest_asyncio
        # nest_asyncio.apply()
        future = asyncio.ensure_future(formalize_batch_using_gtrans_async(texts))
        # This is a simplified way to wait; proper async handling might be needed
        while not future.done():
            loop.run_until_complete(asyncio.sleep(0.1)) # Allow other tasks to run
        return future.result()
    except RuntimeError:
        # No event loop running, start a new one
        return asyncio.run(formalize_batch_using_gtrans_async(texts))
    except Exception as e:
         print(f"Error running async GTrans wrapper: {e}")
         return ["<GTRANS_WRAPPER_ERROR>"] * len(texts)


# %% [markdown]
# ## 4. Input Parameter Handling

# %%
# --- Default Values ---
DEFAULTS = {
    "DATASET_PATH": "",
    "DATASET_OUTPUT_PATH": "", # Default is empty, logic will set it to DATASET_PATH later
    "ONLY_SELECTED": True,
    "PROCESS_GEMINI": False,
    "GEMINI_API_KEY": "",
    "PROCESS_GTRANS": False,
    "PROCESS_GEMINI_BATCH_SIZE": 16,
    "PROCESS_GTRANS_BATCH_SIZE": 16,
    "SKIP_FILLED_CELLS": True,
    "GEMINI_INPUT_COLUMN": "original_processed", # Default input col for Gemini
    "GTRANS_INPUT_COLUMN": "original_processed", # Default input col for GTrans
}

# --- Colab UI Parameter Definitions ---
# These provide interactive input fields when run in Google Colab.
# They act as a fallback *within Colab* if CLI/Secrets/Env vars aren't set.
DATASET_PATH_PARAM = "./dataset.xlsx" # @param {type:"string"}
DATASET_OUTPUT_PATH_PARAM = "" # @param {type:"string"}
ONLY_SELECTED_PARAM = True # @param {type:"boolean"}
PROCESS_GEMINI_PARAM = True # @param {type:"boolean"}
# Note: API Key is sensitive, best handled via Secrets, Env Var, or direct assignment below
# GEMINI_API_KEY_PARAM = "" # @param {type:"string"} # Avoid direct UI param for keys
PROCESS_GTRANS_PARAM = False # @param {type:"boolean"}
PROCESS_GEMINI_BATCH_SIZE_PARAM = 18 # @param {type:"integer"}
PROCESS_GTRANS_BATCH_SIZE_PARAM = 16 # @param {type:"integer"}
SKIP_FILLED_CELLS_PARAM = True # @param {type:"boolean"}
GEMINI_INPUT_COLUMN_PARAM = "original_processed" # @param ["original", "original_processed"]
GTRANS_INPUT_COLUMN_PARAM = "original_processed" # @param ["original", "original_processed"]

# Helper to parse boolean inputs robustly
def parse_bool(value):
    if value is None: return False # Treat None as False
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.lower() in ('true', '1', 't', 'y', 'yes')
    if isinstance(value, int): return value == 1
    return False

# Helper to parse integer inputs robustly
def parse_int(value, default):
    if value is None: return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# --- Parameter Gathering Logic ---
print("--- Determining Script Parameters ---")
params = {}
param_source = {} # Track the origin

# 1. Command Line Arguments (Highest Priority)
parser = argparse.ArgumentParser(description="Formalize text in Excel sheets using APIs.", add_help=False) # Use add_help=False initially
# Add all expected arguments based on DEFAULTS keys
for key, default_value in DEFAULTS.items():
    arg_type = type(default_value)
    if arg_type == bool:
        # For bools, CLI typically uses presence/absence or string 'True'/'False'
        parser.add_argument(f"--{key}", type=str, help=f"{key} (True/False)")
    elif arg_type == int:
         parser.add_argument(f"--{key}", type=int, help=f"{key} (integer)")
    else: # Default to string
        parser.add_argument(f"--{key}", type=str, help=f"{key} (string)")

# Add --help manually if needed outside standard execution
if __name__ == "__main__" and not any(arg in sys.argv for arg in ['-h', '--help']):
     # Reparse with help enabled if needed, or just parse known args
     # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
     pass # Keep help disabled if parsing within complex envs

# Check if running in a non-interactive mode where CLI args are expected
cli_args = {}
if __name__ == "__main__" and not IN_COLAB: # Simplified check for CLI context
    print("Parsing Command Line Arguments...")
    # Use parse_known_args to ignore unknown args (like those from notebooks)
    args, unknown = parser.parse_known_args()
    cli_args = vars(args)
    # print(f"CLI Args found: {cli_args}") # Debug
    # print(f"Unknown args: {unknown}") # Debug

for key in DEFAULTS:
    cli_value = cli_args.get(key)
    if cli_value is not None:
        if isinstance(DEFAULTS[key], bool):
            params[key] = parse_bool(cli_value)
        elif isinstance(DEFAULTS[key], int):
            params[key] = parse_int(cli_value, DEFAULTS[key]) # Use default if parse fails
        else:
            params[key] = cli_value # String or other types
        param_source[key] = 'CLI'

# 2. Google Colab Secrets
if IN_COLAB:
    # print("Checking Google Colab Secrets...") # Less verbose
    for key in DEFAULTS:
        if key not in params: # Only check if not set by CLI
            try:
                secret_value = userdata.get(key)
            except Exception as e:
                print(f"Warning: Could not retrieve {key} from Colab secrets. Error: {e}. Using default value.")
                secret_value = None
            if secret_value is not None:
                if isinstance(DEFAULTS[key], bool):
                    params[key] = parse_bool(secret_value)
                elif isinstance(DEFAULTS[key], int):
                     params[key] = parse_int(secret_value, DEFAULTS[key])
                else:
                    params[key] = secret_value
                param_source[key] = 'Colab Secret'

# 3. Environment Variables
# print("Checking Environment Variables...") # Less verbose
for key in DEFAULTS:
    if key not in params: # Only check if not set by higher priority
        env_value = os.getenv(key)
        if env_value is not None:
            if isinstance(DEFAULTS[key], bool):
                params[key] = parse_bool(env_value)
            elif isinstance(DEFAULTS[key], int):
                 params[key] = parse_int(env_value, DEFAULTS[key])
            else:
                params[key] = env_value
            param_source[key] = 'Environment Variable'

# 4. Google Colab UI Parameters (if in Colab)
if IN_COLAB:
    # print("Checking Google Colab UI Parameters...") # Less verbose
    colab_ui_params = {
        "DATASET_PATH": DATASET_PATH_PARAM,
        "DATASET_OUTPUT_PATH": DATASET_OUTPUT_PATH_PARAM,
        "ONLY_SELECTED": ONLY_SELECTED_PARAM,
        "PROCESS_GEMINI": PROCESS_GEMINI_PARAM,
        # "GEMINI_API_KEY": GEMINI_API_KEY_PARAM, # Avoid UI for keys
        "PROCESS_GTRANS": PROCESS_GTRANS_PARAM,
        "PROCESS_GEMINI_BATCH_SIZE": PROCESS_GEMINI_BATCH_SIZE_PARAM,
        "PROCESS_GTRANS_BATCH_SIZE": PROCESS_GTRANS_BATCH_SIZE_PARAM,
        "SKIP_FILLED_CELLS": SKIP_FILLED_CELLS_PARAM,
        "GEMINI_INPUT_COLUMN": GEMINI_INPUT_COLUMN_PARAM,
        "GTRANS_INPUT_COLUMN": GTRANS_INPUT_COLUMN_PARAM,
    }
    for key, ui_value in colab_ui_params.items():
         if key not in params: # Only check if not set by higher priority
             # Check if the UI param has a potentially meaningful value
             # (e.g., not None, or not empty string for paths)
             is_path = "PATH" in key
             if ui_value is not None and (not is_path or ui_value != ""):
                 # Types from Colab UI are generally correct (bool, int, str)
                 params[key] = ui_value
                 param_source[key] = 'Colab UI Param'

# 5. Apply Hardcoded Defaults (Lowest Priority)
# print("Applying defaults if necessary...") # Less verbose
for key, default_value in DEFAULTS.items():
    if key not in params:
        params[key] = default_value
        param_source[key] = 'Default'

# --- Final Parameter Validation and Setup ---
if not params["DATASET_PATH"]:
    raise ValueError("DATASET_PATH is required but was not provided via any method.")

if not params["DATASET_OUTPUT_PATH"]:
    params["DATASET_OUTPUT_PATH"] = params["DATASET_PATH"]
    param_source["DATASET_OUTPUT_PATH"] = 'Default (same as input)'

# Convert paths to Path objects
params["DATASET_PATH"] = Path(params["DATASET_PATH"])
params["DATASET_OUTPUT_PATH"] = Path(params["DATASET_OUTPUT_PATH"])

# Validate API key requirements
if params["PROCESS_GEMINI"] and not params["GEMINI_API_KEY"]:
    # Try one last time to get from Colab secrets if missed earlier
    if IN_COLAB:
        api_key_secret = userdata.get("GEMINI_API_KEY")
        if api_key_secret:
            params["GEMINI_API_KEY"] = api_key_secret
            param_source["GEMINI_API_KEY"] = 'Colab Secret (late check)'
        else:
             raise ValueError("GEMINI_API_KEY is required because PROCESS_GEMINI is True, but key was not found.")
    else:
         raise ValueError("GEMINI_API_KEY is required because PROCESS_GEMINI is True, but key was not found.")

if params["PROCESS_GEMINI"] and not GEMINI_AVAILABLE:
     print("Warning: PROCESS_GEMINI is True, but the 'google-generativeai' library is not installed. Gemini processing will be skipped.")
     params["PROCESS_GEMINI"] = False # Disable processing

if params["PROCESS_GTRANS"] and not GTRANS_AVAILABLE:
    print("Warning: PROCESS_GTRANS is True, but the 'googletrans' library is not installed. Google Translate processing will be skipped.")
    params["PROCESS_GTRANS"] = False # Disable processing


# Ensure batch sizes are positive integers
params["PROCESS_GEMINI_BATCH_SIZE"] = max(1, parse_int(params["PROCESS_GEMINI_BATCH_SIZE"], DEFAULTS["PROCESS_GEMINI_BATCH_SIZE"]))
params["PROCESS_GTRANS_BATCH_SIZE"] = max(1, parse_int(params["PROCESS_GTRANS_BATCH_SIZE"], DEFAULTS["PROCESS_GTRANS_BATCH_SIZE"]))


# --- Print final configuration ---
print("\n--- Final Configuration ---")
# Determine max key length for alignment
max_key_len = max(len(key) for key in params.keys())
for key, value in params.items():
    # Hide API key for security
    display_value = "****" if key == "GEMINI_API_KEY" and value else value
    source = param_source.get(key, 'N/A')
    print(f"{key:<{max_key_len}} : {display_value} (Source: {source})")
print("-" * (max_key_len + 25)) # Adjust separator length

# Initialize Gemini client early if needed
if params["PROCESS_GEMINI"]:
    try:
        _initialize_gemini_client(params["GEMINI_API_KEY"])
        print("Gemini client configured successfully.")
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}. Gemini processing might fail.")
        # Optionally disable Gemini processing if init fails critically
        # params["PROCESS_GEMINI"] = False

# %% [markdown]
# ## 5. Core Processing Logic

# %%
import time

print(f"--- Starting Processing ---")
print(f"Loading data from: {params['DATASET_PATH']}")

# Validate input file existence
if not params["DATASET_PATH"].is_file():
    if IN_COLAB and not str(params["DATASET_PATH"]).startswith('/') and not str(params["DATASET_PATH"]).startswith('drive/'):
         print(f"Warning: Input file not found at '{params['DATASET_PATH']}'. If using Google Drive, ensure the path starts with 'drive/MyDrive/...' or provide the full '/content/drive/MyDrive/...' path.")
    raise FileNotFoundError(f"Input file not found: {params['DATASET_PATH']}")

# Read all sheets
try:
    all_sheets_dict = pd.read_excel(params["DATASET_PATH"], sheet_name=None)
    print(f"Successfully loaded {len(all_sheets_dict)} sheets: {list(all_sheets_dict.keys())}")
except Exception as e:
    print(f"Error reading Excel file '{params['DATASET_PATH']}': {e}")
    raise

processed_sheets_dict = {} # To hold the processed dataframes
overall_stats = defaultdict(lambda: defaultdict(int)) # {sheet_name: {status: count}}

# Process each sheet
for sheet_name, df in all_sheets_dict.items():
    print(f"\nProcessing sheet: '{sheet_name}'...")
    df_processed = df.copy()
    sheet_stats = overall_stats[sheet_name]

    # --- Determine Input Column ---
    # Prefer specific input column if available, else fallback, else skip
    gemini_input_col = None
    gtrans_input_col = None

    if params["PROCESS_GEMINI"]:
        if params["GEMINI_INPUT_COLUMN"] in df_processed.columns:
            gemini_input_col = params["GEMINI_INPUT_COLUMN"]
        elif "original_processed" in df_processed.columns:
            gemini_input_col = "original_processed"
            print(f"  Info: Gemini input column '{params['GEMINI_INPUT_COLUMN']}' not found, using 'original_processed'.")
        elif "original" in df_processed.columns:
             gemini_input_col = "original"
             print(f"  Info: Gemini input columns '{params['GEMINI_INPUT_COLUMN']}' and 'original_processed' not found, using 'original'.")
        else:
            print(f"  Warning: No suitable input column found for Gemini processing. Required: '{params['GEMINI_INPUT_COLUMN']}', 'original_processed', or 'original'. Skipping Gemini for this sheet.")
            params["PROCESS_GEMINI"] = False # Disable for this sheet run if input missing

    if params["PROCESS_GTRANS"]:
        if params["GTRANS_INPUT_COLUMN"] in df_processed.columns:
            gtrans_input_col = params["GTRANS_INPUT_COLUMN"]
        elif "original_processed" in df_processed.columns:
            gtrans_input_col = "original_processed"
            print(f"  Info: GTrans input column '{params['GTRANS_INPUT_COLUMN']}' not found, using 'original_processed'.")
        elif "original" in df_processed.columns:
             gtrans_input_col = "original"
             print(f"  Info: GTrans input columns '{params['GTRANS_INPUT_COLUMN']}' and 'original_processed' not found, using 'original'.")
        else:
            print(f"  Warning: No suitable input column found for Google Translate processing. Required: '{params['GTRANS_INPUT_COLUMN']}', 'original_processed', or 'original'. Skipping GTrans for this sheet.")
            params["PROCESS_GTRANS"] = False # Disable for this sheet run

    # --- Ensure Output Columns Exist ---
    if params["PROCESS_GEMINI"] and 'formal_gemini' not in df_processed.columns:
        df_processed['formal_gemini'] = pd.NA
    if params["PROCESS_GTRANS"] and 'formal_gtrans' not in df_processed.columns:
        df_processed['formal_gtrans'] = pd.NA

    # --- Identify Rows to Process ---
    process_mask = pd.Series([True] * len(df_processed), index=df_processed.index) # Start with all rows

    if params["ONLY_SELECTED"]:
        if 'is_selected' in df_processed.columns:
            try:
                # Convert various truthy values to boolean, handle NAs as False
                selection_mask = df_processed['is_selected'].apply(lambda x: parse_bool(x) if pd.notna(x) else False)
                process_mask &= selection_mask
                print(f"  Filtering based on 'is_selected' column. {selection_mask.sum()} rows initially marked for processing.")
            except Exception as e:
                print(f"  Warning: Could not reliably parse 'is_selected' column. Error: {e}. Processing all rows for this sheet.")
        else:
            print("  Warning: 'ONLY_SELECTED' is True, but 'is_selected' column not found. Processing all rows.")

    # --- Process Gemini ---
    if params["PROCESS_GEMINI"] and gemini_input_col:
        target_col = 'formal_gemini'
        batch_size = params["PROCESS_GEMINI_BATCH_SIZE"]
        print(f"\n  Processing Gemini ('{gemini_input_col}' -> '{target_col}', batch size: {batch_size})...")

        # Create mask for rows needing Gemini processing
        gemini_process_mask = process_mask.copy()
        if params["SKIP_FILLED_CELLS"]:
            # Exclude rows where target column is already filled (not NA or empty string)
             filled_mask = df_processed[target_col].notna() & (df_processed[target_col] != "")
             gemini_process_mask &= ~filled_mask
             print(f"  Skipping {filled_mask.sum()} rows where '{target_col}' is already filled.")

        # Further exclude rows with empty/NA input
        input_empty_mask = df_processed[gemini_input_col].isna() | (df_processed[gemini_input_col].astype(str).str.strip() == "")
        gemini_process_mask &= ~input_empty_mask
        if input_empty_mask.sum() > 0:
             print(f"  Skipping {input_empty_mask.sum()} rows with empty input in '{gemini_input_col}'.")


        rows_to_process_idx = df_processed.index[gemini_process_mask]
        num_rows_to_process = len(rows_to_process_idx)
        sheet_stats['gemini_total_candidates'] = num_rows_to_process
        print(f"  Found {num_rows_to_process} rows to process for Gemini.")

        if num_rows_to_process > 0:
            # Iterate in batches
            for i in tqdm(range(0, num_rows_to_process, batch_size), desc=f"  Gemini Batches"):
                time.sleep(3)
                batch_indices = rows_to_process_idx[i:min(i + batch_size, num_rows_to_process)]
                batch_texts = df_processed.loc[batch_indices, gemini_input_col].astype(str).tolist()

                try:
                    # Call the API function (which includes retries)
                    processed_batch = formalize_batch_using_gemini(batch_texts, params["GEMINI_API_KEY"])

                    # Update DataFrame
                    df_processed.loc[batch_indices, target_col] = processed_batch

                    # Update stats based on results (simple success/error check)
                    success_count = sum(1 for res in processed_batch if not (res.startswith("<GEMINI_") and res.endswith(">")))
                    error_count = len(processed_batch) - success_count
                    sheet_stats['gemini_processed'] += success_count
                    sheet_stats['gemini_errors'] += error_count

                except Exception as e:
                    # This catches errors *after* all retries failed (if reraise=True)
                    print(f"  Error processing Gemini batch (indices {batch_indices.min()}-{batch_indices.max()}) after retries: {e}")
                    # Mark batch as failed in the DataFrame
                    df_processed.loc[batch_indices, target_col] = "<GEMINI_BATCH_FAILED>"
                    sheet_stats['gemini_errors'] += len(batch_indices)
            print(f"  Gemini processing finished for sheet '{sheet_name}'.")

    # --- Process Google Translate ---
    if params["PROCESS_GTRANS"] and gtrans_input_col:
        target_col = 'formal_gtrans'
        batch_size = params["PROCESS_GTRANS_BATCH_SIZE"]
        print(f"\n  Processing Google Translate ('{gtrans_input_col}' -> '{target_col}', batch size: {batch_size})...")

        # Create mask for rows needing GTrans processing
        gtrans_process_mask = process_mask.copy()
        if params["SKIP_FILLED_CELLS"]:
            filled_mask = df_processed[target_col].notna() & (df_processed[target_col] != "")
            gtrans_process_mask &= ~filled_mask
            print(f"  Skipping {filled_mask.sum()} rows where '{target_col}' is already filled.")

        # Further exclude rows with empty/NA input
        input_empty_mask = df_processed[gtrans_input_col].isna() | (df_processed[gtrans_input_col].astype(str).str.strip() == "")
        gtrans_process_mask &= ~input_empty_mask
        if input_empty_mask.sum() > 0:
             print(f"  Skipping {input_empty_mask.sum()} rows with empty input in '{gtrans_input_col}'.")


        rows_to_process_idx = df_processed.index[gtrans_process_mask]
        num_rows_to_process = len(rows_to_process_idx)
        sheet_stats['gtrans_total_candidates'] = num_rows_to_process
        print(f"  Found {num_rows_to_process} rows to process for Google Translate.")

        if num_rows_to_process > 0:
             # Iterate in batches
            for i in tqdm(range(0, num_rows_to_process, batch_size), desc=f"  GTrans Batches"):
                batch_indices = rows_to_process_idx[i:min(i + batch_size, num_rows_to_process)]
                batch_texts = df_processed.loc[batch_indices, gtrans_input_col].astype(str).tolist()

                try:
                    # Call the synchronous wrapper for the async function
                    processed_batch = formalize_batch_using_gtrans(batch_texts)

                    # Update DataFrame
                    df_processed.loc[batch_indices, target_col] = processed_batch

                    # Update stats
                    success_count = sum(1 for res in processed_batch if not (res.startswith("<GTRANS_") and res.endswith(">")))
                    error_count = len(processed_batch) - success_count
                    sheet_stats['gtrans_processed'] += success_count
                    sheet_stats['gtrans_errors'] += error_count

                except Exception as e:
                     # This catches errors *after* all retries failed
                    print(f"  Error processing GTrans batch (indices {batch_indices.min()}-{batch_indices.max()}) after retries: {e}")
                    df_processed.loc[batch_indices, target_col] = "<GTRANS_BATCH_FAILED>"
                    sheet_stats['gtrans_errors'] += len(batch_indices)

            print(f"  Google Translate processing finished for sheet '{sheet_name}'.")

    # Store the processed dataframe for this sheet
    processed_sheets_dict[sheet_name] = df_processed
    print(f"Sheet '{sheet_name}' processing complete.")
    print(f"  Stats: {dict(sheet_stats)}")


# %% [markdown]
# ## 6. Save Processed Data

# %%
print(f"\n--- Saving Processed Data ---")
output_path = params["DATASET_OUTPUT_PATH"]
print(f"Output will be saved to: {output_path}")

try:
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write all processed sheets back to the Excel file
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df_final in processed_sheets_dict.items():
            df_final.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Successfully saved processed data to {output_path}")

except Exception as e:
    print(f"Error writing output file '{output_path}': {e}")
    if IN_COLAB and 'Errno 30' in str(e) and 'read-only file system' in str(e) and not str(output_path).startswith('/content/drive'):
         print("Hint: You might be trying to save outside of the mounted Google Drive or the allowed Colab storage. Ensure the output path starts with '/content/drive/MyDrive/...' or is within the Colab temporary storage.")
    raise

# %% [markdown]
# ## 7. Final Summary

# %%
print("\n--- Processing Summary ---")
if not overall_stats:
    print("No sheets were processed.")
else:
    for sheet_name, stats in overall_stats.items():
        print(f"\nSheet: '{sheet_name}'")
        if params['PROCESS_GEMINI']:
             total_gemini = stats.get('gemini_total_candidates', 0)
             processed_gemini = stats.get('gemini_processed', 0)
             errors_gemini = stats.get('gemini_errors', 0)
             skipped_gemini = total_gemini - processed_gemini - errors_gemini
             print(f"  Gemini: {processed_gemini}/{total_gemini} processed successfully, {errors_gemini} errors, {skipped_gemini} skipped/not attempted.")
        if params['PROCESS_GTRANS']:
             total_gtrans = stats.get('gtrans_total_candidates', 0)
             processed_gtrans = stats.get('gtrans_processed', 0)
             errors_gtrans = stats.get('gtrans_errors', 0)
             skipped_gtrans = total_gtrans - processed_gtrans - errors_gtrans
             print(f"  GTrans: {processed_gtrans}/{total_gtrans} processed successfully, {errors_gtrans} errors, {skipped_gtrans} skipped/not attempted.")
        if not params['PROCESS_GEMINI'] and not params['PROCESS_GTRANS']:
             print("  No API processing enabled for this run.")

print("\n--- Script Finished ---")
