# ---
# jupyter:
#   jupytext:
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

# %% [markdown] id="a624cf4b"
# # Indonesian Text Normalization using Fine-tuned IndoT5-Base
#
# **Version 4:** Integrated Google Drive, Improved Folder Management (Checkpoints, Data, Final Model), Checkpointing, Resume Logic, Data Saving (Metrics, Args, Duration), and Standardized Structure.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 124580, "status": "ok", "timestamp": 1744604072959, "user": {"displayName": "SHIVA WICAQSANA", "userId": "01009635231454638522"}, "user_tz": -420} id="579a411c" outputId="e24b93a2-7015-4548-a834-af2a7aee49c7"
# !pip install transformers==4.50.3 evaluate sacrebleu==2.5.1 datasets==3.5.0 torch accelerate sentencepiece google-colab --quiet

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1562, "status": "ok", "timestamp": 1744604397503, "user": {"displayName": "SHIVA WICAQSANA", "userId": "01009635231454638522"}, "user_tz": -420} id="b9IKLr975JPV" outputId="637169ee-a729-4e4b-8037-8dc9c6539a6b"
# !pip freeze

# %% executionInfo={"elapsed": 21952, "status": "ok", "timestamp": 1744604477957, "user": {"displayName": "SHIVA WICAQSANA", "userId": "01009635231454638522"}, "user_tz": -420} id="18ac1825"
import json
import os
import re # For finding latest checkpoint
import time # For timing
import evaluate
import numpy as np
import pandas as pd
import torch
from requests import request
from datasets import Dataset, DatasetDict
from IPython.display import display # For displaying DataFrames nicely
from pathlib import Path # Import pathlib

# Google Colab specific imports
try:
    from google.colab import drive, userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    print("Warning: Not running in Google Colab. Drive mounting and secrets will be skipped.")

# Transformers imports for Seq2Seq
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig
)


# %% [markdown] id="a56cb935"
# ## 1. Setup Environment

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 85869, "status": "ok", "timestamp": 1744604579850, "user": {"displayName": "SHIVA WICAQSANA", "userId": "01009635231454638522"}, "user_tz": -420} id="3a7d0086" outputId="8730ab28-1653-42d2-f3f4-dca20937f34d"
SCRIPT_NAME = "lutfi_20250412_indot5-base" # Updated script name slightly

# --- Google Drive and Secrets ---
DRIVE_MOUNT_POINT = Path('/content/drive') # Use Path object
OUTPUT_DIR = None      # Base output directory Path object
CHECKPOINT_DIR = None  # Checkpoint directory Path object (Renamed from SAVE_DIR)
FINAL_MODEL_DIR = None # Final model directory Path object
DATA_DIR = None        # Data directory Path object
CONTINUE_FROM_LATEST_CHECKPOINT = False
SAVE_FINAL_MODEL = True

if IS_COLAB:
    from google.colab import drive, userdata
    print("Running in Google Colab environment.")
    print("Mounting Google Drive...")
    try:
        # Mount expects string
        drive.mount(str(DRIVE_MOUNT_POINT), force_remount=True)
        print("Google Drive mounted successfully.")

        # --- Get base output directory from secrets ---
        print("Attempting to access secrets...")
        try:
            # Define a default base path within the mounted Drive if secret is not found
            default_output_dir = DRIVE_MOUNT_POINT / 'MyDrive' / 'colab_models'

            # --- Use a single secret for the base output directory ---
            output_dir_secret_name = 'OUTPUT_DIR'
            output_dir_str = userdata.get(output_dir_secret_name)

            if not output_dir_str:
                print(f"Warning: '{output_dir_secret_name}' secret not found. Using default base path: {default_output_dir}")
                OUTPUT_DIR = default_output_dir
            else:
                print(f"Using OUTPUT_DIR from secrets: {output_dir_str}")
                OUTPUT_DIR = Path(output_dir_str) # Convert secret string to Path

            # --- Derive specific directories using pathlib ---
            # Check if OUTPUT_DIR was successfully determined
            if OUTPUT_DIR:
                # Construct the paths relative to the base OUTPUT_DIR
                CHECKPOINT_DIR = OUTPUT_DIR / SCRIPT_NAME / "Checkpoints"
                FINAL_MODEL_DIR = OUTPUT_DIR / SCRIPT_NAME / "Final_Models" # Note plural
                DATA_DIR = OUTPUT_DIR / SCRIPT_NAME / "Data"

                print(f"Derived CHECKPOINT_DIR: {CHECKPOINT_DIR}")
                print(f"Derived FINAL_MODEL_DIR: {FINAL_MODEL_DIR}")
                print(f"Derived DATA_DIR: {DATA_DIR}")

                # --- Create directories using Path.mkdir ---
                print("Ensuring directories exist...")
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                print("Required directories ensured.")

            else:
                 print("Error: Could not determine OUTPUT_DIR. Saving/Loading disabled.")
                 # Keep directories as None
                 CHECKPOINT_DIR = None
                 FINAL_MODEL_DIR = None
                 DATA_DIR = None

            # --- Handle Continuation Flag ---
            continue_flag_secret_name = 'CONTINUE_FROM_LATEST_CHECKPOINT'
            continue_flag = userdata.get(continue_flag_secret_name)
            if continue_flag and continue_flag.lower() == 'true':
                CONTINUE_FROM_LATEST_CHECKPOINT = True
                print(f"'{continue_flag_secret_name}' is set to True. Will attempt to resume training.")
            else:
                CONTINUE_FROM_LATEST_CHECKPOINT = False # Explicitly set to False
                print(f"'{continue_flag_secret_name}' is False or not set. Starting training from scratch or using pre-trained base.")

        except Exception as e:
            print(f"Error accessing secrets or creating directories: {e}")
            print("Proceeding without Drive saving/resuming capabilities.")
            OUTPUT_DIR = None # Reset on error
            CHECKPOINT_DIR = None
            FINAL_MODEL_DIR = None
            DATA_DIR = None
            CONTINUE_FROM_LATEST_CHECKPOINT = False

    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("Proceeding as if not in Colab (local execution).")
        IS_COLAB = False # Treat as non-Colab if mount fails
        # Fall through to the non-Colab 'else' block

# --- Fallback for Local Execution (or if Colab mount/secrets failed) ---
if not IS_COLAB or OUTPUT_DIR is None: # Handle both non-Colab and Colab failures
    if not IS_COLAB:
        print("Not running in Google Colab. Defining local directories.")
    else:
        print("Colab setup failed. Defining local directories as fallback.")

    # Define a local base output directory
    OUTPUT_DIR = Path("./colab_outputs") # Use pathlib Path

    # Derive specific directories
    CHECKPOINT_DIR = OUTPUT_DIR / SCRIPT_NAME / "Checkpoints"
    FINAL_MODEL_DIR = OUTPUT_DIR / SCRIPT_NAME / "Final_Models"
    DATA_DIR = OUTPUT_DIR / SCRIPT_NAME / "Data"

    print(f"Local Base OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Local CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"Local FINAL_MODEL_DIR: {FINAL_MODEL_DIR}")
    print(f"Local DATA_DIR: {DATA_DIR}")

    # Create local directories using Path.mkdir
    print("Ensuring local directories exist...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Local directories ensured.")
    CONTINUE_FROM_LATEST_CHECKPOINT = False # Typically don't resume automatically locally


# --- Final Check and Summary ---
print("\n--- Configuration Summary ---")
print(f"IS_COLAB: {IS_COLAB}")
print(f"Base OUTPUT_DIR: {OUTPUT_DIR}")
print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
print(f"FINAL_MODEL_DIR: {FINAL_MODEL_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"CONTINUE_FROM_LATEST_CHECKPOINT: {CONTINUE_FROM_LATEST_CHECKPOINT}")

# Now you can use the Path objects CHECKPOINT_DIR, FINAL_MODEL_DIR, and DATA_DIR later in your script


# %% [markdown] id="5d320747"
# ## 2. Configuration

# %% id="91e1646b"
# --- Model ---
MODEL_CHECKPOINT = "Wikidepia/IndoT5-base"

# --- Data URLs ---
# BASE_URL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/"
BASE_URL = "https://raw.githubusercontent.com/Lutfi-Azis/lunashimu-formalizer-dataset/refs/heads/main/dataset/"
TRAIN_INF_URL = f"{BASE_URL}train.inf"
TRAIN_FOR_URL = f"{BASE_URL}train.for"
DEV_INF_URL = f"{BASE_URL}dev.inf"
DEV_FOR_URL = f"{BASE_URL}dev.for"
TEST_INF_URL = f"{BASE_URL}test.inf" # Test set included for potential final evaluation
TEST_FOR_URL = f"{BASE_URL}test.for"

# --- Preprocessing & Generation ---
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
PREFIX = "bakukan: " # T5 task prefix

# --- Training ---
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8 # Adjust based on GPU memory
EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 5 # Increase epochs slightly
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100 # Log metrics every 100 steps
SAVE_STEPS = 500   # Save checkpoint every 500 steps
EVAL_STEPS = 500   # Evaluate every 500 steps (aligned with save steps)
SAVE_TOTAL_LIMIT = 1 # Keep only the latest 1 checkpoints
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

# --- Output Directories (Set in Section 1 based on Colab/Secrets) ---
# CHECKPOINT_DIR, FINAL_MODEL_DIR, and DATA_DIR are set in the 'Setup Environment' section


# %% [markdown] id="e3278c67"
# ## 3. Load Data

# %% id="6335cc12"
def get_lines(url: str) -> list[str]:
    """Fetches text data line by line from a URL."""
    try:
        response = request("GET", url)
        response.raise_for_status() # Raise an exception for bad status codes
        # Filter out empty lines that might result from split
        return [line.strip() for line in response.text.split("\n") if line.strip()]
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return []

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2327, "status": "ok", "timestamp": 1744602720495, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="5b140971" outputId="ba53a37d-f5b9-4921-b12b-3e9e5d64f637"
print("Fetching data...")
train_inf = get_lines(TRAIN_INF_URL)
train_for = get_lines(TRAIN_FOR_URL)
dev_inf = get_lines(DEV_INF_URL)
dev_for = get_lines(DEV_FOR_URL)
test_inf = get_lines(TEST_INF_URL)
test_for = get_lines(TEST_FOR_URL)

print(f"Train lines loaded: Informal={len(train_inf)}, Formal={len(train_for)}")
print(f"Dev lines loaded:   Informal={len(dev_inf)}, Formal={len(dev_for)}")
print(f"Test lines loaded:  Informal={len(test_inf)}, Formal={len(test_for)}")

# Basic validation
raw_datasets = None
if not (len(train_inf) == len(train_for) > 0 and \
        len(dev_inf) == len(dev_for) > 0 and \
        len(test_inf) == len(test_for) > 0):
    print("\nError: Data loading issues or length mismatch between informal/formal pairs.")
    # raise ValueError("Failed to load datasets correctly.")
else:
    # Create Hugging Face Datasets
    train_dataset = Dataset.from_dict({"informal": train_inf, "formal": train_for})
    dev_dataset = Dataset.from_dict({"informal": dev_inf, "formal": dev_for})
    test_dataset = Dataset.from_dict({"informal": test_inf, "formal": test_for})

    raw_datasets = DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset,
        "test": test_dataset
    })
    print("\nDatasetDict created successfully:")
    print(raw_datasets)
    print("\nSample training example:")
    print(raw_datasets["train"][0])


# %% [markdown] id="a011fd30"
# ## 4. Load Base Tokenizer and Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 403, "referenced_widgets": ["393403fbebfe43659233a2547d1e441c", "c798138db4dd4568b8c64ab5922a2571", "c6b1a578278c429292fb40c6b6212797", "dc1097929fbb40a38ac9888963ace93b", "cd75f7fbd94d448db831ad533f986680", "18f92fddca694645b0b799c7ee0bb3ea", "3e224a84305a48f993f8598d1d169a2f", "7e214d0b22ba4d6e8460164c3263e01c", "fd7925f9bad749eea11b12f73ba6c714", "40ae5cc087e6470eb8e83f4c97853822", "3a9f342859a7415b8af072648d57a2a4", "82b752cd1e6c460396681ea7d1cee9f3", "5883075c5669429187230686baf1bf8e", "18ff8bdb94814ac9b698c19ff2d65018", "e615b27d6b774cadb002ca569baf6001", "19b61ce27b72467aa2e0c6ddd44e420b", "cceef3eea34b4edf8827d28b3fe52bdd", "af3904bdc02349b6bc91be3f4c2e712b", "826f51f242744e4ab10bfe78ab9d99a2", "243e9dcc44af463b8727fb669eb534bc", "1ac3f657a343441fbc3bd949ad5b4432", "dcd73d6760b74423a4abefa23e7df384", "e1e119a3ee0d4a2fb44d8285539e5c28", "c70bddc354374271be37303fd5bd0634", "08d19cf924a044a7ba2f4fb2c605170d", "de8fdc7c4e7e4d3a8a2ef53d914cb161", "cac338fcb64543328ab69a49ce6d4637", "dea15d616fcb43fcabde0b621542328c", "4be4b319dda949cea2324879e683a7d6", "d62fb84d0294423cabe234f3bd4a72ee", "18394e08b7a14cca86d932d2922ba0e9", "e0bc24fe07b14d7781c49894eac11ce0", "64be98981af145b58585e52e38703032", "8a8229b2c69a48c5926915630a00b75b", "9a70f4800fb14fa1bc1b9bf07612863a", "7075d039f1c84abfba43000f4709c44d", "e192a34a0e3c424e890ffc135a457bee", "2d07c1ccfb994d79a098d2b7c23c1e8c", "183a88beefcb4e2d989e94c890e220ef", "f6dc0d7cac96442699b8eb9d795956d6", "92aa9fed0e074eab9f12200567df2499", "9701c752244146febb09130e1bbbf311", "a3a367ac7b984a788b3bae8c4d13ef43", "2e45ec27f3c847f2af316a6ac3d83d61", "44b6887baa1248abb1fdee1a2ccc39b0", "8b49291fc16741babbc2ce0f5bd0f205", "01af968bc61a409088972abe429d8c06", "bec40a79e37544ea9f0330f452330d10", "e169789cf5eb470495fedb47b40792dd", "de52d00601aa44e288b36ed50d367a26", "6a351affadc946ec9ca22fc91bd6c776", "c2c114337ebc46119755838fd6aef2f2", "c8d59b341f4e407a83e86efe527bebca", "eb3a3443c6cd462b93cac6eb15ee47ee", "33ac52aa8bcf433a88b8c66688f35573", "79e2d40ec7a2414eadea31d46465ec2b", "362da50d8b5342a8a2550caaae0ae105", "0b3929a6d27f4aed8f397ef05a08eed4", "b65d5051ab0041bc929f62e53a9d52dd", "2ad0618af3ed479e99616875a0095daa", "00d28b0465d64642b5401c4c65036dc1", "75e37c2cc2614b90b4ea4bbf0795ec56", "94b3b8532889434ca8e7b3e1f777e848", "a6ff1967bfee464e84ca28e626ab59c0", "0476407ce7424619b69e5e9b2c359933", "edfedc0a18c44899a054ae142125566d"]} executionInfo={"elapsed": 12697, "status": "ok", "timestamp": 1744602736087, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="bd610231" outputId="42f2d7e8-b2cd-4619-9af0-d0e09a59b81e"
tokenizer = None
model = None
print(f"\nLoading base tokenizer and model from checkpoint: {MODEL_CHECKPOINT}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print("Base tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading base tokenizer: {e}")
    # raise e # Or handle more gracefully

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    print("Base model loaded successfully.")
    # print("\nModel Configuration:")
    # print(model.config)
except Exception as e:
    print(f"Error loading base model: {e}")
    # raise e # Or handle more gracefully


# %% [markdown] id="837e4b44"
# ## 5. Preprocess Data

# %% id="7a85963c"
def preprocess_function(examples):
    if tokenizer is None:
        raise ValueError("Tokenizer is not loaded.")

    inputs = [PREFIX + text for text in examples["informal"]]
    targets = examples["formal"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False # Padding handled by data collator
    )

    # Tokenize targets (labels)
    # Target tokens need special handling for T5 (padding ID is ignored in loss)
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False # Padding handled by data collator
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %% colab={"base_uri": "https://localhost:8080/", "height": 235, "referenced_widgets": ["e6a3a16a5a384127916a22f3e1105daf", "783316a5cd374692a922ec8e29e6438d", "d424e00ad3734e52a872a52906a53e20", "3aac43ac16de435fa4f7c40d5a7a1302", "7d42720e28ac436d8c0d396bf1a6740a", "eff3ffe9564448c5b6b1a62f84beb3f1", "7d8f1e3a48fe4a5db64d9d3aa6f2ff0f", "f316397618ec4f158c2339011253f32f", "97bd351813e144b5af3e1dc0ce47804a", "cbf985572b1940efaa26080ae6aa6672", "aa3531f8eaa44d6e847b4289c2e67bf7", "9655b10001014b76bbc9df0bc6a97dad", "8981f04e2cfe4361b1ae44b45e6e3231", "2a845d200ee94bd1acb3b56295c6fb4b", "3c8ba5c1ff5e44258232c8749afb90e6", "dc857f3b3f6c47e38c468e393cc45129", "b2eafd558d9842b897118187d545cb39", "d79c349ee48d46d5bd373af6833715e3", "175bf6d0c58c45a38ef09229d7982d53", "2c027db52cc945d5b762764098ec3894", "f29334104ea14e82b8b2ad0df19e6907", "6610e8154f94441ea3af81134f3afc4e", "b286f218d530414394d0b5b68d96ffe8", "2fa0000bd09f4d94bf4c39de25b706d9", "065998ff1c1a4b8da92d21c9e919a371", "e4910390b5614b4388abc21631082f6e", "4f4f847934de40738d7157906df85668", "56b0a81b0b2f41b099c117743108dcbd", "92c435760d134c55ae9ef1bbca900fce", "44de4e73c6324e878a90b3ed0de0d513", "154b9679b7014fc6b07facff8414169d", "93e3d04ae5704436b408c807d0464004", "650f73767510419da48820708f11b602"]} executionInfo={"elapsed": 590, "status": "ok", "timestamp": 1744602740572, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="dde7c121" outputId="8d13e4c2-cec3-41d5-caba-4814572278bf"
tokenized_datasets = None
if raw_datasets and tokenizer and model:
    print("\nApplying preprocessing...")
    try:
        # Apply processing and keep original columns temporarily if needed for inspection
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names # Remove original text columns
        )
        # Manually set format to torch (sometimes needed after map)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        print("Preprocessing complete.")
        print("\nSample preprocessed example:")
        # Print shapes to verify
        print("Input IDs shape:", tokenized_datasets["train"][0]['input_ids'].shape)
        print("Labels shape:", tokenized_datasets["train"][0]['labels'].shape)

        # Decode example for verification (optional)
        # print("\nDecoded Input Example:", tokenizer.decode(tokenized_datasets['train'][0]['input_ids'], skip_special_tokens=True))
        # Decode labels correctly, replacing -100 with pad_token_id for decoding
        # label_ids = tokenized_datasets['train'][0]['labels']
        # label_ids[label_ids == -100] = tokenizer.pad_token_id # Important for decoding
        # print("Decoded Label Example:", tokenizer.decode(label_ids, skip_special_tokens=True))


    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # import traceback; traceback.print_exc()
else:
    print("\nSkipping preprocessing due to missing data, tokenizer, or model.")


# %% [markdown] id="5a21abd4"
# ## 6. Setup Training

# %% colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["6a66cd7c30de499f9f10b872139fe846", "2e8064688259422c8dd72459a79412cb", "d3fe55d0a03a49a5be65362c869d90e0", "2c013756f6f34dde8e29144ab6aded82", "8268519a131e4008b831a63fc2bc1082", "ebeb4d4b22bf489fb50b579e9fa7c2d9", "6562e1496415489fa715f096ecdc3406", "ba2d5ca8b6c140a5812e1cb157681659", "d16cde3cee9b4e779a14491594add908", "031162cf2fd4498bb4ca3f8e910053f2", "eb2c895bd1804862bb598b90cb96e600"]} executionInfo={"elapsed": 1577, "status": "ok", "timestamp": 1744602744483, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="90f25426" outputId="10adc795-d70a-4de9-839d-92e2f68d8ac6"
data_collator = None
metric = None

# Data Collator for Seq2Seq
if tokenizer and model:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True, # Pad to the longest sequence in the batch dynamically
        label_pad_token_id=tokenizer.pad_token_id # T5 uses pad_token_id for labels too
    )
    print("\nData collator created for Seq2Seq.")
else:
    print("\nSkipping data collator creation due to missing tokenizer or model.")

# Evaluation Metric (SacreBLEU)
try:
    metric = evaluate.load("sacrebleu")
    print("SacreBLEU metric loaded.")
except Exception as e:
    print(f"Error loading SacreBLEU metric: {e}")
    metric = None

# Compute metrics function (Only loss during training runs for consistency)
# We will compute BLEU score separately after training on the best model
def compute_metrics_loss_only(eval_preds):
     # Predictions are logits, labels are token IDs
     # Loss is calculated internally by the Trainer
     return {} # Return empty dict, we rely on eval_loss

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 71, "status": "ok", "timestamp": 1744602745577, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="337c17a8" outputId="87556a97-7d84-4294-d13e-dd2802b9ca20"
training_args = None
if tokenized_datasets and CHECKPOINT_DIR: # Need CHECKPOINT_DIR for output
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),       # Save checkpoints to Drive/local path
        evaluation_strategy="steps",        # Evaluate periodically
        eval_steps=EVAL_STEPS,              # How often to evaluate
        save_strategy="steps",              # Save periodically
        save_steps=SAVE_STEPS,              # How often to save
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        load_best_model_at_end=True,        # Load the best model based on loss
        metric_for_best_model="eval_loss",  # Use validation loss to find best checkpoint
        greater_is_better=False,            # Lower loss is better
        save_total_limit=SAVE_TOTAL_LIMIT,  # Keep only the latest N checkpoints
        predict_with_generate=True,         # Needed for generation during eval/predict steps
        generation_max_length=MAX_TARGET_LENGTH, # Set generation length for eval runs
        report_to="none",                   # Disable external reporting unless configured
    )
    print(f"\nTraining arguments configured. Checkpoints will be saved to: {CHECKPOINT_DIR}")

    # --- Save training arguments to DATA_DIR ---
    if DATA_DIR:
        args_save_path = DATA_DIR / "training_args.json"
        try:
            with open(args_save_path, 'w') as f:
                # Use `to_dict()` for easy JSON serialization
                json.dump(training_args.to_sanitized_dict(), f, indent=4)
            print(f"Training arguments saved to {args_save_path}")
        except Exception as e:
            print(f"Warning: Could not save training arguments to {args_save_path}: {e}")
    else:
        print("Warning: DATA_DIR not set. Skipping saving training arguments.")

else:
    print("\nSkipping TrainingArguments setup due to missing data or CHECKPOINT_DIR.")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 520, "status": "ok", "timestamp": 1744602748070, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="387d2676" outputId="77f1fe07-add5-411b-8563-66b49dd993d0"
trainer = None
if model and training_args and tokenized_datasets and data_collator and tokenizer:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_loss_only, # Only track loss during training steps
    )
    print("Seq2SeqTrainer initialized.")
else:
    print("\nCannot initialize Trainer due to missing components.")


# %% [markdown] id="446bac75"
# ## 7. Train Model

# %% id="cab441e3"
latest_checkpoint_path = None
if CONTINUE_FROM_LATEST_CHECKPOINT and CHECKPOINT_DIR and CHECKPOINT_DIR.is_dir():
    print(f"Attempting to find latest checkpoint in {CHECKPOINT_DIR} to resume training...")
    try:
        # Need to convert Path to string for os.listdir and os.path.isdir if used
        checkpoints = [
            d for d in CHECKPOINT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            # Extract step number and find the latest
            latest_checkpoint_path = str(max(checkpoints, key=lambda x: int(re.search(r"checkpoint-(\d+)", x.name).group(1))))
            print(f"Found latest checkpoint: {latest_checkpoint_path}")
        else:
            print("No checkpoints found in the specified directory. Starting training from scratch.")
    except Exception as e:
        print(f"Error finding latest checkpoint: {e}. Starting training from scratch.")
        latest_checkpoint_path = None # Reset on error
else:
    if CONTINUE_FROM_LATEST_CHECKPOINT:
        print(f"Cannot resume: CHECKPOINT_DIR '{CHECKPOINT_DIR}' does not exist or is not a directory.")
    # Otherwise, normal start, no message needed here.

# %% colab={"base_uri": "https://localhost:8080/", "height": 523} executionInfo={"elapsed": 481514, "status": "ok", "timestamp": 1744603234937, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="2b67adcc" outputId="0fab94a2-d29e-4ff4-ff67-e93eacc23780"
train_start_time = time.time()
train_result = None

if trainer:
    print("\nStarting model training...")
    try:
        if latest_checkpoint_path:
            print(f"Resuming training from: {latest_checkpoint_path}")
            train_result = trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        else:
            print("Starting training from the beginning.")
            train_result = trainer.train()

        print("Training finished.")
        train_end_time = time.time()
        training_duration_seconds = train_end_time - train_start_time
        print(f"Total training time: {training_duration_seconds:.2f} seconds")

        # Log final training metrics from train_result
        if train_result:
            metrics = train_result.metrics
            metrics["train_duration_seconds"] = training_duration_seconds # Add duration to metrics dict
            trainer.log_metrics("train_summary", metrics)

            print(f"Training summary metrics: {metrics}")

            # --- Save training summary metrics and log history to DATA_DIR ---
            if DATA_DIR:
                summary_metrics_path = DATA_DIR / "training_summary_metrics.json"
                log_history_path = DATA_DIR / "training_log_history.json"

                try:
                    with open(summary_metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    print(f"Training summary metrics saved to {summary_metrics_path}")
                except Exception as e:
                    print(f"Warning: Could not save summary metrics: {e}")

                try:
                    log_history = trainer.state.log_history
                    with open(log_history_path, 'w') as f:
                        json.dump(log_history, f, indent=4)
                    print(f"Training log history saved to {log_history_path}")
                except Exception as e:
                    print(f"Warning: Could not save log history: {e}")
            else:
                print("Warning: DATA_DIR not set. Skipping saving training metrics and log history.")

        # Report best model checkpoint
        if trainer.state.best_model_checkpoint:
             print(f"Best model checkpoint during training (based on eval_loss): {trainer.state.best_model_checkpoint}")
             # Note: The best model is already loaded into trainer.model if load_best_model_at_end=True
        else:
             # This case might happen if evaluation didn't run or load_best_model_at_end=False
             print(f"No best model checkpoint identified (or load_best_model_at_end=False). The final model state is saved in the last checkpoint at {CHECKPOINT_DIR}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nSkipping training because the Trainer could not be initialized.")


# %% [markdown] id="ad23838a"
# ## 8. Evaluate Model (BLEU Score)

# %% id="4130dd5c"
def generate_seq2seq_predictions(dataset, model_to_eval, tokenizer_to_eval, batch_size=EVAL_BATCH_SIZE, max_gen_length=MAX_TARGET_LENGTH, prefix=PREFIX):
    """Generates predictions using the fine-tuned Seq2Seq model."""
    if not model_to_eval or not tokenizer_to_eval:
        print("Model or tokenizer not available for generation.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_eval.eval()
    model_to_eval.to(device)

    all_preds = []
    all_refs = []

    # Generation config (can customize beam search, etc.)
    generation_config = GenerationConfig(
        max_length=max_gen_length, # Use max_length for T5 generate
        num_beams=4,              # Example beam search
        early_stopping=True,
        pad_token_id=tokenizer_to_eval.pad_token_id,
        eos_token_id=tokenizer_to_eval.eos_token_id,
    )
    print(f"Starting generation with config: {generation_config}")

    total_examples = len(dataset)
    for i in range(0, total_examples, batch_size):
        batch = dataset[i : i + batch_size]
        try:
             # Assuming dataset is indexable and returns dicts or has columns
             informal_texts = batch["informal"]
             references = batch["formal"]
        except TypeError: # Handle dataset slicing if it returns a list of dicts
             informal_texts = [item["informal"] for item in batch]
             references = [item["formal"] for item in batch]
        except KeyError:
             print("Error: Could not access 'informal' or 'formal' columns in the batch.")
             continue # Skip this batch


        inputs_with_prefix = [prefix + text for text in informal_texts]
        # Tokenize inputs for the batch
        input_dict = tokenizer_to_eval(
            inputs_with_prefix,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH # Use input length here
        )
        input_dict = {k: v.to(device) for k, v in input_dict.items()}

        with torch.no_grad():
            outputs = model_to_eval.generate(**input_dict, generation_config=generation_config)

        # Decode predictions
        batch_preds = tokenizer_to_eval.batch_decode(outputs, skip_special_tokens=True)

        all_preds.extend([pred.strip() for pred in batch_preds])
        all_refs.extend([ref.strip() for ref in references])

        if (i // batch_size + 1) % 20 == 0: # Log progress less frequently
            print(f"  Generated for {i + len(batch)} / {total_examples} examples")
            if len(batch_preds) > 0:
                 print(f"    Sample Ref : {references[0]}")
                 print(f"    Sample Pred: {batch_preds[0]}")

    print(f"Generation complete for {len(all_preds)} examples.")
    return all_preds, all_refs


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 44204, "status": "ok", "timestamp": 1744603426726, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="57e5c7df" outputId="79cbd43d-1222-4d17-b32d-da462143c812"
# Perform generation and BLEU calculation on the validation set using the *best* model loaded by the trainer
validation_eval_df = None # Initialize dataframe variable
if trainer and trainer.model and tokenized_datasets and metric is not None and raw_datasets and DATA_DIR:
    print("\nEvaluating the final best model on the validation set (generating predictions for BLEU)...")

    # The trainer should have loaded the best model if load_best_model_at_end=True
    eval_model = trainer.model
    eval_tokenizer = trainer.tokenizer

    # Use the *original* validation dataset (with text columns) for evaluation input/refs
    validation_data_for_eval = raw_datasets["validation"]

    predictions, references = generate_seq2seq_predictions(
        validation_data_for_eval,
        eval_model,
        eval_tokenizer,
        batch_size=EVAL_BATCH_SIZE,
        max_gen_length=MAX_TARGET_LENGTH,
        prefix=PREFIX
    )

    if predictions is not None and references is not None:
        print("\nCalculating BLEU score...")
        references_bleu = [[ref] for ref in references] # Format for sacrebleu
        bleu_results = {}
        try:
            # Ensure predictions and references are lists of strings and handle potential None values
            valid_preds = [str(p) if p is not None else "" for p in predictions]
            # SacreBLEU expects references as list of lists of strings
            valid_refs_bleu = [[str(r) if r is not None else ""] for r in references]

            if not valid_preds or not valid_refs_bleu or len(valid_preds) != len(valid_refs_bleu):
                 print(f"Warning: Mismatch in prediction ({len(valid_preds)}) / reference ({len(valid_refs_bleu)}) counts or empty lists. Cannot compute BLEU.")
                 bleu_results = {"score": 0.0, "error": "Mismatch or empty lists"}
            else:
                # Compute BLEU
                bleu_results = metric.compute(predictions=valid_preds, references=valid_refs_bleu)
                print("\nValidation BLEU Score (Best Model):")
                # Print score nicely, handle potential missing score key
                print(json.dumps(bleu_results, indent=2))


            # --- Save evaluation results and predictions to DATA_DIR ---
            # Ensure lengths match before creating DataFrame
            min_len = min(len(validation_data_for_eval["informal"]), len(references), len(valid_preds))
            validation_eval_df = pd.DataFrame({
                "informal": validation_data_for_eval["informal"][:min_len],
                "formal_reference": references[:min_len],
                "predicted_formal": valid_preds[:min_len]
            })
            print("\nSample Validation Predictions:")
            display(validation_eval_df.head())

            eval_output_path = DATA_DIR / "validation_predictions.csv"
            bleu_output_path = DATA_DIR / "validation_bleu_results.json"

            validation_eval_df.to_csv(eval_output_path, index=False)
            print(f"Validation predictions saved to {eval_output_path}")
            with open(bleu_output_path, "w") as f:
                 json.dump(bleu_results, f, indent=4)
            print(f"BLEU results saved to {bleu_output_path}")

            # Log BLEU score to trainer state if available (might not be useful after training ends)
            # trainer.log({"eval_bleu_final": bleu_results.get("score", 0.0)})

        except Exception as e:
            print(f"Error calculating or saving BLEU score/predictions: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Prediction generation failed, skipping BLEU calculation.")

elif not DATA_DIR:
    print("\nSkipping evaluation (BLEU): DATA_DIR not set.")
else:
    print("\nSkipping evaluation (BLEU) due to missing components (trainer, data, metric) or failed training.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 615} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1744603440776, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="1vWpsPeXm31W" outputId="f2f6b00c-ad92-43e5-8e50-dffb2676ca4f"
# Display the dataframe if it was created
if validation_eval_df is not None:
    print("\n--- Validation Predictions DataFrame ---")
    display(validation_eval_df)
else:
    print("\nValidation predictions DataFrame was not generated.")


# %% [markdown] id="b191666a"
# ## 9. Save Final Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5929, "status": "ok", "timestamp": 1744603448990, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="f4123f45" outputId="08d8ba15-adb5-4626-fc3c-05fe62745450"
if trainer and trainer.model and FINAL_MODEL_DIR and SAVE_FINAL_MODEL:
    print(f"\nSaving the final best model to: {FINAL_MODEL_DIR}")
    try:
        # The trainer already holds the best model in memory if load_best_model_at_end=True
        # Use trainer.save_model which also saves tokenizer and config
        trainer.save_model(str(FINAL_MODEL_DIR))
        print(f"Final model and tokenizer saved successfully to {FINAL_MODEL_DIR}.")

        # Training arguments were already saved to DATA_DIR in section 6
        # No need to save them here again.

    except Exception as e:
        print(f"Error saving final model: {e}")
elif not FINAL_MODEL_DIR:
     print("\nSkipping final model saving: FINAL_MODEL_DIR not set.")
else:
    print("\nSkipping final model saving: (possibly due to training error).")


# %% [markdown] id="2267d9ce"
# ## 10. Inference

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 981, "status": "ok", "timestamp": 1744603458333, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="9fa262a9" outputId="d361b817-3546-4356-b967-dec92094506a"
# Load the final model from FINAL_MODEL_DIR for inference
inference_model = None
inference_tokenizer = None

if FINAL_MODEL_DIR and FINAL_MODEL_DIR.exists():
    print(f"\nLoading final model from {FINAL_MODEL_DIR} for inference...")
    try:
        # Load model and tokenizer using AutoClasses from the final saved directory
        inference_model = AutoModelForSeq2SeqLM.from_pretrained(str(FINAL_MODEL_DIR))
        inference_tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))

        # Optional: move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model.to(device)
        inference_model.eval() # Set to evaluation mode
        print(f"Inference model and tokenizer loaded successfully from {FINAL_MODEL_DIR} to device {device}.")
    except Exception as e:
        print(f"Error loading final model/tokenizer from {FINAL_MODEL_DIR}: {e}")
        inference_model = None
        inference_tokenizer = None
else:
     print(f"\nCannot load final model for inference: Directory '{FINAL_MODEL_DIR}' not found or not specified.")


# %% id="1ba761d2"
def formalize_text_t5(sentence: str, model, tokenizer, prefix=PREFIX, max_gen_len=MAX_TARGET_LENGTH):
    """Uses the loaded fine-tuned T5 model to convert informal text to formal."""
    if not model or not tokenizer:
        return "Error: Inference model or tokenizer not available."

    # Prepare input with prefix
    input_text = prefix + sentence
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)

    device = model.device # Use the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Configure generation (consistent with evaluation)
    generation_config = GenerationConfig(
        max_length=max_gen_len, # T5 uses max_length
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Decode the generated tokens
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output.strip()

    except Exception as e:
        # import traceback; traceback.print_exc() # Uncomment for detailed error
        return f"Error during generation: {e}"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3923, "status": "ok", "timestamp": 1744603470257, "user": {"displayName": "Shiva Augusta", "userId": "17123988873835353068"}, "user_tz": -420} id="ea2e3123" outputId="4aa5705b-c7a7-47a5-f957-fb91eef9ecf5"
# Test Inference with the final loaded model
if inference_model and inference_tokenizer:
    print("\n--- Testing Inference with Final Model ---")
    test_sentences = [
        "min , mau nanya . cara mengaktifkan no indosat yang sudah terblokir karena lupa isi pulsa , gmn yah ?",
        "maling yg jadi provider y begini dech jadinya",
        "kmrn aq liat promo baru, tp lupa dmn",
        "gimana caranya biar bisa dpt diskon gede?",
        "thanks ya infonya sangat membantu",
        "ga ngerti knp paket internet gw cpt bgt abisnya",
        "klo mo upgrade kartu ke 4g dmn ya?"
    ]

    results = []
    for sentence in test_sentences:
        formalized = formalize_text_t5(sentence, inference_model, inference_tokenizer, prefix=PREFIX)
        print("-" * 30)
        print(f"Informal: {sentence}")
        print(f"Formal:   {formalized}")
        results.append({"Informal": sentence, "Formal (Predicted)": formalized})

    # --- Optional: Save inference results to DATA_DIR ---
    if DATA_DIR:
        try:
            inference_df = pd.DataFrame(results)
            inference_output_path = DATA_DIR / "inference_examples.csv"
            inference_df.to_csv(inference_output_path, index=False)
            print("-" * 30)
            print(f"Inference examples saved to {inference_output_path}")
        except Exception as e:
            print(f"Could not save inference examples: {e}")
    else:
        print("Warning: DATA_DIR not set. Skipping saving inference examples.")

else:
    print("\nSkipping inference test because the final model/tokenizer could not be loaded.")

print("\n--- Script Execution Finished ---")
