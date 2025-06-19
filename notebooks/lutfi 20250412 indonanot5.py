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
# # Indonesian Text Normalization using Fine-tuned IndoNanoT5
#
# **Version 5:** Integrated new hyperparameters (Optimizer, LR Scheduler), added `metadata.json` output, and maintained robust Google Drive integration, checkpointing, and data saving.

# %% executionInfo={"elapsed": 10433, "status": "ok", "timestamp": 1750265319738, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="579a411c"
# !pip install transformers==4.50.3 evaluate sacrebleu==2.5.1 datasets==3.5.0 torch accelerate sentencepiece google-colab --quiet

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1750265322186, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="18ac1825"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8295, "status": "ok", "timestamp": 1750265330489, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="3a7d0086" outputId="22344833-b28d-41ed-8939-05ae96a977a1"
SCRIPT_NAME = "lutfi_20250412_indonanot5" # Updated script name slightly

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

# %% [markdown] id="5d320747"
# ## 2. Configuration

# %% executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1750265330516, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="91e1646b"
# --- Model ---
MODEL_CHECKPOINT = "LazarusNLP/IndoNanoT5-base"

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
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
PREFIX = "bakukan: " # T5 task prefix

# --- Training ---
LEARNING_RATE = 5e-5
OPTIMIZER = "adamw_torch" # Options: adamw_torch, adamw_hf, adafactor, etc.
LR_SCHEDULER = "linear" # Options: linear, cosine, constant, etc.
TRAIN_BATCH_SIZE = 16 # Adjust based on GPU memory
EVAL_BATCH_SIZE = 16
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

# %% executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1750265330539, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="6335cc12"
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2315, "status": "ok", "timestamp": 1750265332857, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="5b140971" outputId="8936ef9e-c3fa-47f6-8474-ef0eca4a89df"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2846, "status": "ok", "timestamp": 1750265335705, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="bd610231" outputId="066e583e-0d0f-4ec9-e942-2b28ca61c8a4"
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

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1750265335722, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="7a85963c"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 238, "referenced_widgets": ["b55605b7a50846eba365c2926dcfc1c5", "9f08c7dc2eee40b2ba6d122080043c12", "4989e2e76b6d4849a1b71bf72bfcf850", "5fe7a398ee3c4a3cabe6a0040417ad6b", "ca6955df314e42288ae05143e5c3a897", "ce1c78e9ff014152b52194e731683901", "42696eff8da24210a8c05d191bb78b87", "8f33cc1c49534bd3baadda424854af46", "af29f0fbb9114145b4a46f5d1c5e85c5", "f9a0484045c2455dbaf0c878d78547af", "6792e75bd10b4531be43212225b58170", "4bf4d89110d14d0fb7f1a797ec9e93b5", "b9a3edee7b3a4d2a8a41a1007868e561", "ef5c8fe0e35e4dc7a6ca847b7e73875a", "1e367f39a9be4ab4bdf8eebfcb1b681b", "dbff3be4de4b4eafb0d3a28523b0b39b", "0968192d55254683a56f49bd3c63df63", "d1ece12fb2aa4ad1b0ac355784ab76b6", "f530a2e4e3dd44a3bea363afdc81fb8e", "e4257ebcb42a433fae84d16518c52e33", "d539e8da46ea4434bf10f2d8a70a74bf", "0065b71b552b459abaadae1738a20221", "21a7190a324f47d7b9f60c07d02519d2", "399843c2f25549f290c4460ce19a55e1", "2386d1c5d1974324848c68235b72c658", "d92504f600de41418a3d809bb410a72a", "5f43e61888a84c1fa687cb25df495745", "35d8e7928a954895bef68b76d6f83e01", "de572f270f494d56bcda7d17f28d13c2", "28a44db2daa9472d8d9fe3fbbc31416f", "01b01e44421f4fe3870300a976aac353", "641f7c59f1384c2fb77e76f4d0034a83", "697a66edd25c473cb449922e767c5abd"]} executionInfo={"elapsed": 1919, "status": "ok", "timestamp": 1750265337710, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="dde7c121" outputId="cb5756bb-f226-4038-b49f-051ea9584212"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1364, "status": "ok", "timestamp": 1750265339077, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="90f25426" outputId="623bff39-d9be-4048-eebd-9f90067ea244"
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1750265339098, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="337c17a8" outputId="33e353ae-fcc2-4da5-8d34-8d8093a3ca81"
training_args = None
if tokenized_datasets and CHECKPOINT_DIR: # Need CHECKPOINT_DIR for output
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),      # Save checkpoints to Drive/local path
        evaluation_strategy="steps",       # Evaluate periodically
        eval_steps=EVAL_STEPS,             # How often to evaluate
        save_strategy="steps",             # Save periodically
        save_steps=SAVE_STEPS,             # How often to save
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        optim=OPTIMIZER,                   # Use the specified optimizer
        lr_scheduler_type=LR_SCHEDULER,    # Use the specified learning rate scheduler
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        load_best_model_at_end=True,       # Load the best model based on loss
        metric_for_best_model="eval_loss", # Use validation loss to find best checkpoint
        greater_is_better=False,           # Lower loss is better
        save_total_limit=SAVE_TOTAL_LIMIT, # Keep only the latest N checkpoints
        predict_with_generate=True,        # Needed for generation during eval/predict steps
        generation_max_length=MAX_TARGET_LENGTH, # Set generation length for eval runs
        report_to="none",                  # Disable external reporting unless configured
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5484, "status": "ok", "timestamp": 1750265344583, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="387d2676" outputId="41a8f87e-8718-4bd3-81f8-9d69e60cd950"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 45, "status": "ok", "timestamp": 1750265344584, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="cab441e3" outputId="431eab4c-331b-46d0-9be8-151b579a3e08"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 448} executionInfo={"elapsed": 298935, "status": "ok", "timestamp": 1750265643515, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="2b67adcc" outputId="e28a8f4b-78e0-49f5-ab81-5f790b1456da"
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

# %% executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1750265643560, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="4130dd5c"
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
        num_beams=32,              # Example beam search
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 50871, "status": "ok", "timestamp": 1750265694441, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="57e5c7df" outputId="e7b82932-33f8-4fe8-81f7-caa2c39220be"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 616} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1750265694475, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="1vWpsPeXm31W" outputId="3cef97d2-3051-4ead-a487-a08b6a1c2b04"
# Display the dataframe if it was created
if validation_eval_df is not None:
    print("\n--- Validation Predictions DataFrame ---")
    display(validation_eval_df)
else:
    print("\nValidation predictions DataFrame was not generated.")

# %% [markdown] id="b191666a"
# ## 9. Save Final Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7460, "status": "ok", "timestamp": 1750265701940, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="f4123f45" outputId="abfd005b-5b5e-48bc-f95b-ba6fe21ca080"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2312, "status": "ok", "timestamp": 1750265704253, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="9fa262a9" outputId="7fb8e1b8-5995-4fc0-99b5-916ef4059cf0"
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


# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1750265704258, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="1ba761d2"
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
        num_beams=32,
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3942, "status": "ok", "timestamp": 1750265708202, "user": {"displayName": "Lutfi H", "userId": "07615966780902302652"}, "user_tz": -420} id="ea2e3123" outputId="e83e4eab-bc17-404a-fa67-82a2e1476c60"
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
        formalized = formalize_text_t5(sentence, inference_model, inference_tokenizer, prefix=PREFIX)
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
        "prefix": PREFIX,
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
