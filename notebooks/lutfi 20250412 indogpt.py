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

# %% [markdown] id="51585975"
# # Indonesian Text Normalization using Fine-tuned IndoGPT (GPT-2 based)
#
# **Version 5:** Integrated new hyperparameters (Optimizer, LR Scheduler), added `metadata.json` output, and maintained robust Google Drive integration, checkpointing, and data saving.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 162862, "status": "ok", "timestamp": 1750255388036, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="177692dd" outputId="52aa3490-6e98-49bb-b761-ff7ab5e799f6"
# !pip install transformers==4.50.3 evaluate sacrebleu==2.5.1 datasets==3.5.0 torch accelerate sentencepiece google-colab --quiet
# !pip install git+https://github.com/Lutfi221/indobenchmark-toolkit.git@e49794e34e958b24606ccb2f0ae50772a374c550

# %% executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1750256296450, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="24f4d22f"
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

# Import the specific tokenizer
from indobenchmark import IndoNLGTokenizer # <--- IMPORT THE CORRECT TOKENIZER
from transformers import (
    # AutoTokenizer, # No longer used for loading this specific tokenizer
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    GenerationConfig
)

# %% [markdown] id="1d0da3e4"
# ## 1. Setup Environment

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8542, "status": "ok", "timestamp": 1750256304993, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="d5b0fe78" outputId="d5f221eb-0e42-4c96-ce1d-ba3446c172c5"
SCRIPT_NAME = "lutfi_20250412_indogpt"

# --- Google Drive and Secrets ---
DRIVE_MOUNT_POINT = Path('/content/drive') # Use Path object
OUTPUT_DIR = None      # Base output directory Path object
CHECKPOINT_DIR = None  # Checkpoint directory Path object (Renamed from SAVE_DIR)
FINAL_MODEL_DIR = None # Final model directory Path object
DATA_DIR = None        # Data directory Path object
CONTINUE_FROM_LATEST_CHECKPOINT = False
SAVE_FINAL_MODEL = True

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

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

# %% [markdown] id="f157d33c"
# ## 2. Configuration

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1750256305001, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="ca05d305"
# --- Model ---
MODEL_CHECKPOINT = "indobenchmark/indogpt"

# --- Data URLs ---
BASE_URL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/"
# BASE_URL = "https://raw.githubusercontent.com/Lutfi-Azis/lunashimu-formalizer-dataset/refs/heads/main/dataset/"
TRAIN_INF_URL = f"{BASE_URL}train.inf"
TRAIN_FOR_URL = f"{BASE_URL}train.for"
DEV_INF_URL = f"{BASE_URL}dev.inf"
DEV_FOR_URL = f"{BASE_URL}dev.for"
TEST_INF_URL = f"{BASE_URL}test.inf" # Test set included for potential final evaluation
TEST_FOR_URL = f"{BASE_URL}test.for"

# --- Preprocessing & Generation ---
MAX_LENGTH = 128 # Max sequence length for combined informal + formal text + special tokens
INFORMAL_PREFIX = "informal: "
FORMAL_PREFIX = " formal: "
# Note: The space before "formal:" is important for tokenization
MAX_NEW_TOKENS_GEN = 64 # Max tokens to generate during inference/evaluation

# --- Training ---
LEARNING_RATE = 5e-5
OPTIMIZER = "adamw_torch" # Options: adamw_torch, adamw_hf, adafactor, etc.
LR_SCHEDULER = "linear" # Options: linear, cosine, constant, etc.
TRAIN_BATCH_SIZE = 8 # Adjust based on GPU memory
EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 5 # Keep epochs low for initial test
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100 # Log metrics every 100 steps
SAVE_STEPS = 500   # Save checkpoint every 500 steps
EVAL_STEPS = 500   # Evaluate every 500 steps (aligned with save steps)
SAVE_TOTAL_LIMIT = 1 # Keep only the latest 1 checkpoints
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

# --- Output Directories (Set in Section 1) ---
# CHECKPOINT_DIR, FINAL_MODEL_DIR, DATA_DIR are set in the 'Setup Environment' section

# %% [markdown] id="4c2e36b9"
# ## 3. Load Data

# %% executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1750256305014, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="432298a3"
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2412, "status": "ok", "timestamp": 1750256307427, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="a7398051" outputId="4a85658f-65c8-4db3-a063-63325cd00fc4"
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
    # Consider raising an error or exiting if data is crucial
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

# %% [markdown] id="4f61411b"
# ## 4. Load Base Tokenizer and Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 66, "status": "ok", "timestamp": 1750256307495, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="SUWzUsQwP6w6" outputId="cf4979de-4e4e-49cc-b053-f0b65bc695e6"
AutoModelForCausalLM.from_pretrained

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2629, "status": "ok", "timestamp": 1750256310127, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="59ee86d9" outputId="c0930413-1400-48e9-d0cb-95c03f1181a6"
tokenizer = None
model = None
print(f"\nLoading specific IndoNLGTokenizer and model from checkpoint: {MODEL_CHECKPOINT}")

try:
    # Load the specific tokenizer class
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_CHECKPOINT, padding_side='right') # <--- USE SPECIFIC CLASS

    # Set pad_token if it's not already set (common for GPT-2 based models)
    if tokenizer.pad_token is None:
        print("Warning: pad_token not set. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure ID matches
    print("IndoNLGTokenizer loaded successfully.")
    print(f"EOS token: {tokenizer.eos_token}, Pad token: {tokenizer.pad_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}, Pad token ID: {tokenizer.pad_token_id}")


    # Load model using AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)
    # Ensure model's pad_token_id is aligned with tokenizer's pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model loaded successfully.")


except Exception as e:
    print(f"Error loading base model or tokenizer: {e}")
    # Depending on severity, you might want to raise e or exit
    # raise e


# %% [markdown] id="210d6c53"
# ## 5. Preprocess Data

# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1750256310163, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="e52211de"
def preprocess_function(examples):
    if tokenizer is None:
        raise ValueError("Tokenizer is not loaded.")
    if tokenizer.eos_token is None or tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have EOS and PAD tokens set.")

    inputs_text = []
    prompts_only = [] # Store prompts separately for masking calculation
    # Construct the combined text for input_ids tokenization
    for inf, form in zip(examples["informal"], examples["formal"]):
        # Text includes prompt, formal part, and EOS token
        text = f"{INFORMAL_PREFIX}{inf}{FORMAL_PREFIX}{form}{tokenizer.eos_token}"
        inputs_text.append(text)
        # Prompt is just the informal part + prefixes
        prompt = f"{INFORMAL_PREFIX}{inf}{FORMAL_PREFIX}"
        prompts_only.append(prompt)

    # Tokenize the combined text WITH padding and truncation
    model_inputs = tokenizer(
        inputs_text,
        max_length=MAX_LENGTH,
        padding="max_length", # Explicitly pad to max_length
        truncation=True,
        return_tensors="pt" # Return PyTorch tensors
    )

    # --- Create labels: shift inputs to the right and mask prompt/padding ---
    labels = model_inputs["input_ids"].clone() # Start with a copy of input_ids

    # Tokenize just the prompts to find their length accurately
    # Use add_special_tokens=False to avoid counting BOS/EOS if tokenizer adds them implicitly
    prompt_encodings = tokenizer(prompts_only, add_special_tokens=False, padding=False, truncation=False)
    prompt_token_lengths = [len(ids) for ids in prompt_encodings.input_ids]

    # Mask labels based on prompt length and padding
    pad_token_id = tokenizer.pad_token_id
    for i in range(len(labels)):
        prompt_len = prompt_token_lengths[i]
        # Ensure prompt length doesn't exceed MAX_LENGTH (shouldn't happen with overall truncation)
        prompt_len = min(prompt_len, MAX_LENGTH)

        # Mask prompt tokens: Set label tokens corresponding to the prompt to -100
        labels[i, :prompt_len] = -100

        # Mask padding tokens: Set label tokens corresponding to padding in the input_ids to -100
        labels[i][model_inputs["input_ids"][i] == pad_token_id] = -100

    model_inputs["labels"] = labels
    return model_inputs


# %% colab={"base_uri": "https://localhost:8080/", "height": 365, "referenced_widgets": ["d9c1dca38cbc4875b63a133a0269735d", "e74b25475c054212b6e085ffa90b679e", "77d42359a7474a689f09aa4b4f114aba", "a3733b12f8f74459833381665770d5c9", "ad43d20738de45a9a6907d0c6db1d65c", "4f574e9a09ce482bb6bded0e99875697", "5763961f0fa94367801814da1e3326c3", "ba9f999e309f45d78a482911b5eeba00", "d775afca21c2494bb9558d6acb34fed2", "f3862273c12244818c50e481b25610b4", "bfbe7e196cde4e7e9b7b80765e5e0a01", "c506f3d1ed0c4fa48e38d7ed077fbff7", "4e5dea1a347f4e9c836e86188c7bda6d", "cacbf41001ff4e8085f92fa66ba92ac6", "301e528ed6c04b388d571148b0a4638c", "275a76ae94fa4544b390d65188061af3", "59479f2203c8424ebcafef30aa51dae0", "b3edf1de8a174f15b0202b00996c58c8", "3e4bf875810c4b408b0b822bde05b015", "00d41d418c29402980757ab41521c8fc", "4bcc1fae97d34b34bc70f22190b57a76", "d70011ee67014e25a56f82ea79d86d08", "f8f1ff4961044de091aa6ed4475cc583", "bbfd859672f94015b54337e2166dded8", "809a43bacc90479a8bd55db09dc7cfc8", "8bbf07d4959343d59011c993ff447747", "51acb05134af45edafd0f29a5e24eaff", "817f792957fa42cba1a67df429b2f218", "6ae2669edb3e43fb9791c05202abc7a3", "f766287fa4a541b291613f09ef35b110", "41ab1b3aad3e4d74a04bf5e4f94783b9", "ba8fc6b52e884ec9bc7ac875334186fd", "5db562c49ddc42b6bb10958cf6260a04"]} executionInfo={"elapsed": 1981, "status": "ok", "timestamp": 1750256312143, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="b0bb96c9" outputId="3e8cd8c0-1569-44f4-ddc0-1a6fe6a726fe"
tokenized_datasets = None
if raw_datasets and tokenizer and model:
    print("\nApplying preprocessing...")
    try:
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names # Remove old columns
        )
        # Manually set format to torch (sometimes needed after map)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        print("Preprocessing complete.")
        print("\nSample preprocessed example:")
        # Print shapes to verify
        print("Input IDs shape:", tokenized_datasets["train"][0]['input_ids'].shape)
        print("Labels shape:", tokenized_datasets["train"][0]['labels'].shape)

        # Decode example for verification
        print("\nDecoded Input IDs:")
        print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids'], skip_special_tokens=False)) # Show special tokens
        print("\nDecoded Labels (non-masked part for verification):")
        label_example = tokenized_datasets["train"][0]['labels'].clone()
        label_example[label_example == -100] = tokenizer.pad_token_id # Replace -100 to decode
        print(tokenizer.decode(label_example, skip_special_tokens=True)) # Skip padding tokens

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback; traceback.print_exc() # Uncomment for detailed error
else:
    print("\nSkipping preprocessing due to missing data, tokenizer, or model.")

# %% [markdown] id="2911ce20"
# ## 6. Setup Training

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1240, "status": "ok", "timestamp": 1750256736878, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="001d74b0" outputId="9fb00212-5fa7-49af-e3cf-c4100390e3c6"
data_collator = None
metric = None

# Data Collator for Causal LM
if tokenizer:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # False for Causal LM (GPT-2 style)
    )
    print("\nData collator created for Causal LM.")
else:
    print("\nSkipping data collator creation due to missing tokenizer.")

# Evaluation Metric (SacreBLEU)
try:
    metric = evaluate.load("sacrebleu")
    print("SacreBLEU metric loaded.")
except Exception as e:
    print(f"Error loading SacreBLEU metric: {e}")
    metric = None

# Compute metrics function (loss only during training for Causal LM)
def compute_metrics_loss_only(eval_preds):
    # Trainer passes logits and labels by default
    # For Causal LM, we typically only monitor loss during training epochs
     return {} # Return empty dict as BLEU requires generation


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1750256736910, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="ffd683e5" outputId="4639a15c-bacd-4335-a2ae-57c0b456256d"
training_args = None
if tokenized_datasets and CHECKPOINT_DIR: # Need CHECKPOINT_DIR for output
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),       # Save checkpoints to CHECKPOINT_DIR
        evaluation_strategy="steps",        # Evaluate periodically
        eval_steps=EVAL_STEPS,              # How often to evaluate
        save_strategy="steps",              # Save periodically
        save_steps=SAVE_STEPS,              # How often to save
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        optim=OPTIMIZER,                    # Use the specified optimizer
        lr_scheduler_type=LR_SCHEDULER,     # Use the specified learning rate scheduler
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        load_best_model_at_end=True,        # Load the best model based on loss
        metric_for_best_model="eval_loss",  # Use validation loss to find best checkpoint
        greater_is_better=False,            # Lower loss is better
        save_total_limit=SAVE_TOTAL_LIMIT,  # Keep only the latest N checkpoints
        report_to="none",                   # Disable external reporting unless configured
    )
    print(f"\nTraining arguments configured. Checkpoints will be saved to: {CHECKPOINT_DIR}")

    # --- Save training arguments to DATA_DIR ---
    if DATA_DIR:
        args_save_path = DATA_DIR / "training_args.json"
        try:
            with open(args_save_path, 'w') as f:
                # Use `to_sanitized_dict()` for easy JSON serialization
                json.dump(training_args.to_sanitized_dict(), f, indent=4)
            print(f"Training arguments saved to {args_save_path}")
        except Exception as e:
            print(f"Warning: Could not save training arguments to {args_save_path}: {e}")
    else:
        print("Warning: DATA_DIR not set. Skipping saving training arguments.")

else:
    print("\nSkipping TrainingArguments setup due to missing data or CHECKPOINT_DIR.")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 67, "status": "ok", "timestamp": 1750256736978, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="ea82bdd5" outputId="3eb0127e-8a43-4c9c-9297-53b0d5099d3a"
trainer = None
if model and training_args and tokenized_datasets and data_collator and tokenizer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, # Pass the IndoNLGTokenizer instance
        data_collator=data_collator,
        compute_metrics=compute_metrics_loss_only, # Only compute loss during training
    )
    print("Trainer initialized.")
else:
    print("\nCannot initialize Trainer due to missing components.")

# %% [markdown] id="a7864b6a"
# ## 7. Train Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1750256736993, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="0561926c" outputId="c1bbcab9-95ef-405a-8bc4-210b58602470"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"elapsed": 131957, "status": "ok", "timestamp": 1750256868951, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="a7dcb860" outputId="ad9de814-c18c-4601-efdb-6c9158c9a4aa"
train_result = None
training_duration_seconds = 0
duration_per_epoch = 0

if trainer:
    if torch.cuda.is_available():
        print("Emptying CUDA cache before training...")
        torch.cuda.empty_cache()

    print("\nStarting model training...")
    train_start_time = time.time()
    try:
        if latest_checkpoint_path:
            print(f"Resuming training from: {latest_checkpoint_path}")
            # Pass the string path
            train_result = trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        else:
            print("Starting training from the beginning.")
            train_result = trainer.train()

        print("Training finished.")
        train_end_time = time.time()
        training_duration_seconds = train_end_time - train_start_time
        print(f"Total training time: {training_duration_seconds:.2f} seconds")


        # Log final training metrics from train_result
        if train_result and NUM_TRAIN_EPOCHS > 0:
            metrics = train_result.metrics
            metrics["train_duration_seconds"] = training_duration_seconds # Add duration
            trainer.log_metrics("train_summary", metrics)
            # trainer.save_metrics("train", metrics) # Optional: Saves metrics.json in checkpoint dir
            # trainer.save_state() # Saves trainer state (including logs) in checkpoint dir

            duration_per_epoch = training_duration_seconds / NUM_TRAIN_EPOCHS
            print(f"Training summary metrics: {metrics}")
            print(f"Duration per epoch: {duration_per_epoch:.2f} seconds")

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
                    # Access log history from trainer state
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
             # Should not happen if load_best_model_at_end=True and evaluation ran
             print(f"No best model checkpoint identified (or load_best_model_at_end=False). The final model state is saved in the last checkpoint at {CHECKPOINT_DIR}")


    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed error
else:
    print("\nSkipping training because the Trainer could not be initialized.")


# %% [markdown] id="43c62155"
# ## 8. Evaluate Model (BLEU Score)

# %% executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1750256868978, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="a36ec275"
def generate_formal_predictions(dataset, model_to_eval, tokenizer_to_eval, batch_size=EVAL_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS_GEN, max_prompt_len=MAX_LENGTH):
    """Generates formal text predictions using the fine-tuned Causal LM."""
    if not model_to_eval or not tokenizer_to_eval:
        print("Model or tokenizer not available for generation.")
        return None, None
    if tokenizer_to_eval.pad_token_id is None or tokenizer_to_eval.eos_token_id is None:
         print("Error: Tokenizer requires pad_token_id and eos_token_id for generation.")
         return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_eval.eval()
    model_to_eval.to(device)

    all_preds = []
    all_refs = []

    # Use GenerationConfig for cleaner parameter handling
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer_to_eval.eos_token_id,
        pad_token_id=tokenizer_to_eval.pad_token_id, # Crucial: ensure pad token ID is set
        do_sample=False, # Greedy for eval consistency
        num_beams=1,     # Change to > 1 for beam search if desired
        early_stopping=False # Can set to True with beam search
    )
    print(f"Starting generation with config: {generation_config}")

    total_examples = len(dataset)
    for i in range(0, total_examples, batch_size):
        batch = dataset[i : i + batch_size]
        # Ensure batch items are accessed correctly (assuming dataset yields dicts)
        try:
             informal_texts = batch["informal"]
             references = batch["formal"]
        except TypeError: # Handle dataset slicing if it returns a list of dicts
             informal_texts = [item["informal"] for item in batch]
             references = [item["formal"] for item in batch]
        except KeyError:
             print("Error: Could not access 'informal' or 'formal' columns in the batch.")
             continue # Skip this batch


        prompts = [f"{INFORMAL_PREFIX}{inf}{FORMAL_PREFIX}" for inf in informal_texts]

        # Tokenize prompts
        inputs = tokenizer_to_eval(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # Ensure prompt doesn't exceed max_length minus generation space
            # Subtracting max_new_tokens helps prevent truncation of essential prompt parts
            max_length=max_prompt_len - max_new_tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_lengths = inputs['input_ids'].shape[1] # Get the length of the padded+truncated prompt tokens

        with torch.no_grad():
            # Generate, feeding both input_ids and attention_mask
            outputs = model_to_eval.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config
            )

        # Decode *only* the generated part
        # The generated IDs start after the input sequence length
        generated_ids = outputs[:, input_lengths:]
        batch_preds = tokenizer_to_eval.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean up potential leading/trailing spaces
        batch_preds = [pred.strip() for pred in batch_preds]

        all_preds.extend(batch_preds)
        all_refs.extend([ref.strip() for ref in references]) # Strip refs too for consistency

        if (i // batch_size + 1) % 20 == 0: # Log progress less frequently
            print(f"  Generated for {i + len(batch)} / {total_examples} examples")
            if len(batch_preds) > 0:
                 print(f"    Sample Ref : {references[0]}")
                 print(f"    Sample Pred: {batch_preds[0]}")

    print(f"Generation complete for {len(all_preds)} examples.")
    return all_preds, all_refs

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 22924, "status": "ok", "timestamp": 1750256891904, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="34b86b2c" outputId="d64b3b22-3a64-4e2d-e44d-bc60881e572c"
# Perform generation and BLEU calculation on the validation set using the *best* model loaded by the trainer
validation_eval_df = None # Initialize dataframe variable
if trainer and trainer.model and tokenized_datasets and metric is not None and raw_datasets and DATA_DIR:
    print("\nEvaluating the final best model on the validation set (generating predictions for BLEU)...")

    # The trainer should have loaded the best model if load_best_model_at_end=True
    eval_model = trainer.model
    eval_tokenizer = trainer.tokenizer # Use the tokenizer associated with the trainer/model

    # Use the *original* validation dataset (with text columns) for evaluation input/refs
    validation_data_for_eval = raw_datasets["validation"]

    predictions, references = generate_formal_predictions(
        validation_data_for_eval,
        eval_model,
        eval_tokenizer,
        batch_size=EVAL_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS_GEN,
        max_prompt_len=MAX_LENGTH # Pass the overall max length
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
                bleu_results = metric.compute(predictions=valid_preds, references=valid_refs_bleu, lowercase=True)
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

            # Use generic filenames and save to DATA_DIR
            eval_output_path = DATA_DIR / "validation_predictions.csv"
            bleu_output_path = DATA_DIR / "validation_bleu_results.json"

            validation_eval_df.to_csv(eval_output_path, index=False)
            print(f"Validation predictions saved to {eval_output_path}")
            with open(bleu_output_path, "w") as f:
                 json.dump(bleu_results, f, indent=4)
            print(f"BLEU results saved to {bleu_output_path}")

            # Log BLEU score with the trainer if possible (might require custom logging callback)
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

# Display the dataframe if it was created
if validation_eval_df is not None:
    print("\n--- Validation Predictions DataFrame ---")
    display(validation_eval_df)
else:
    print("\nValidation predictions DataFrame was not generated.")

# %% [markdown] id="4f09ef69"
# ## 9. Save Final Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3721, "status": "ok", "timestamp": 1750256895626, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="fecaaef7" outputId="f3ac9739-4173-4eb3-920e-5c8f5f880d5e"
if trainer and trainer.model and FINAL_MODEL_DIR and SAVE_FINAL_MODEL:
    print(f"\nSaving the final best model to: {FINAL_MODEL_DIR}")
    try:
        # The trainer already holds the best model in memory if load_best_model_at_end=True
        # Use trainer.save_model which also saves tokenizer and config
        trainer.save_model(str(FINAL_MODEL_DIR)) # Convert Path to string
        print(f"Final model and tokenizer saved successfully to {FINAL_MODEL_DIR}.")

        # Training arguments and metadata were already saved to DATA_DIR in section 6.
        # No need to save them again here.

    except Exception as e:
        print(f"Error saving final model: {e}")
elif not FINAL_MODEL_DIR:
     print("\nSkipping final model saving: FINAL_MODEL_DIR not set.")
else:
    print("\nSkipping final model saving: (possibly due to training error).")

# %% [markdown] id="945e6ae6"
# ## 10. Inference

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1048, "status": "ok", "timestamp": 1750256896675, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="125bd414" outputId="eff11406-390f-4be2-b831-0fc16ff0f052"
# Load the final model from FINAL_MODEL_DIR for inference
inference_model = None
inference_tokenizer = None

if FINAL_MODEL_DIR and FINAL_MODEL_DIR.exists():
    print(f"\nLoading final model from {FINAL_MODEL_DIR} for inference...")
    try:
        # Load model using AutoModelForCausalLM
        inference_model = AutoModelForCausalLM.from_pretrained(str(FINAL_MODEL_DIR)) # Convert Path to string
        # Load the corresponding tokenizer using the specific class
        inference_tokenizer = IndoNLGTokenizer.from_pretrained(str(FINAL_MODEL_DIR)) # <--- USE SPECIFIC CLASS

        # Ensure pad token consistency after loading
        if inference_tokenizer.pad_token is None:
            print("Setting pad_token = eos_token for loaded inference tokenizer.")
            inference_tokenizer.pad_token = inference_tokenizer.eos_token
            inference_tokenizer.pad_token_id = inference_tokenizer.eos_token_id
        # Align model config if needed (should match saved config)
        inference_model.config.pad_token_id = inference_tokenizer.pad_token_id

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


# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1750256896677, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="29812ba1"
def formalize_text_gpt2(sentence: str, model, tokenizer, max_new_toks=MAX_NEW_TOKENS_GEN, max_len=MAX_LENGTH):
    """Uses the loaded fine-tuned IndoGPT model to convert informal text to formal."""
    if not model or not tokenizer:
        return "Error: Inference model or tokenizer not available."
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
         return "Error: Tokenizer requires eos_token_id and pad_token_id."

    # Prepare prompt
    prompt = f"{INFORMAL_PREFIX}{sentence}{FORMAL_PREFIX}"
    # Tokenize using the loaded IndoNLGTokenizer instance
    # Truncate prompt if it's too long to allow for generation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len - max_new_toks # Reserve space for generation
    )

    device = model.device # Use the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1] # Length of actual input tokens

    # Configure generation
    generation_config = GenerationConfig(
        max_new_tokens=max_new_toks,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, # Crucial for generation
        do_sample=False, # Use greedy or beam search for inference
        num_beams=4,     # Example beam search
        early_stopping=True
    )

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'], # Pass attention mask
                generation_config=generation_config
            )

        # Decode only the newly generated tokens (after the input prompt)
        # Output might contain the prompt, so slice it off
        generated_ids = outputs[0, input_length:]

        # Use the loaded IndoNLGTokenizer for decoding
        decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return decoded_output.strip()

    except Exception as e:
        import traceback; traceback.print_exc() # Uncomment for detailed error
        return f"Error during generation: {e}"


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1254, "status": "ok", "timestamp": 1750256897933, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="57240cd0" outputId="17b8e8af-cfa1-4039-ff58-1e99345a3df2"
# Test Inference with the final loaded model
n_inferred_sentences_per_second = 0
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
    inference_start_time = time.time()
    for sentence in test_sentences:
        formalized = formalize_text_gpt2(sentence, inference_model, inference_tokenizer)
        print("-" * 30)
        print(f"Informal: {sentence}")
        print(f"Formal:   {formalized}")
        results.append({"Informal": sentence, "Formal (Predicted)": formalized})
    inference_end_time = time.time()

    inference_duration = inference_end_time - inference_start_time
    if inference_duration > 0:
        n_inferred_sentences_per_second = len(test_sentences) / inference_duration
    print(f"\nTotal inference time for {len(test_sentences)} sentences: {inference_duration:.2f} seconds")
    print(f"Inference speed: {n_inferred_sentences_per_second:.2f} sentences/second")

    # --- Optional: Save inference results to DATA_DIR ---
    if DATA_DIR:
        try:
            inference_df = pd.DataFrame(results)
            # Use generic filename and save to DATA_DIR
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

# %% [markdown] id="new-metadata-cell"
# ## 11. Finalize and Save Metadata

# %%
# --- Finalize and save all metadata to DATA_DIR ---
if DATA_DIR:
    print("\n--- Finalizing Metadata ---")
    max_gpu_memory_usage_bytes = 0
    if torch.cuda.is_available():
        max_gpu_memory_usage_bytes = torch.cuda.max_memory_allocated()
        print(f"Max GPU memory usage: {max_gpu_memory_usage_bytes / (1024**2):.2f} MB")
    else:
        print("No GPU used, max_gpu_memory_usage will be 0.")

    metadata = {
        "informal_prefix": INFORMAL_PREFIX,
        "formal_prefix": FORMAL_PREFIX,
        "training_duration": round(training_duration_seconds, 2),
        "duration_per_epoch": round(duration_per_epoch, 2),
        "n_inferred_sentences_per_second": round(n_inferred_sentences_per_second, 2),
        "max_gpu_memory_usage": max_gpu_memory_usage_bytes
    }
    metadata_path = DATA_DIR / "metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Final metadata saved to {metadata_path}")
    except Exception as e:
        print(f"Warning: Could not save final metadata to {metadata_path}: {e}")
else:
    print("\nWarning: DATA_DIR not set. Skipping saving final metadata.")


print("\n--- Script Execution Finished ---")
