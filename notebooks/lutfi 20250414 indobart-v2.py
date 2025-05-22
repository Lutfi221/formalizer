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

# %% [markdown] id="35f7b6d1"
# # Indonesian Text Normalization using Fine-tuned IndoBART-v2
#
# **Adapted Version:** Using `indobenchmark/indobart-v2` (Seq2Seq BART model). Integrated Google Drive, Improved Folder Management, Checkpointing, Resume Logic, Data Saving, and Standardized Structure.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31680, "status": "ok", "timestamp": 1744688862596, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ef285908" outputId="114fe036-72da-4312-e16e-949ca7fb7296"
# !pip install "numpy<2.0"
# !pip install transformers==4.41.2 evaluate sacrebleu==2.4.2 datasets==2.19.1 torch accelerate sentencepiece google-colab --quiet
# !pip install git+https://github.com/Lutfi221/indobenchmark-toolkit.git@e49794e34e958b24606ccb2f0ae50772a374c550

# https://github.com/haotian-liu/LLaVA/issues/1808
# !pip install peft==0.10.0

# %% executionInfo={"elapsed": 24395, "status": "ok", "timestamp": 1744688887007, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="9ad067cc"
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
from indobenchmark import IndoNLGTokenizer
from transformers import (
    AutoTokenizer, # <--- USE AutoTokenizer for indobart-v2
    AutoModelForSeq2SeqLM, # <--- USE AutoModelForSeq2SeqLM for BART
    DataCollatorForSeq2Seq, # <--- USE DataCollatorForSeq2Seq
    Seq2SeqTrainer,         # <--- USE Seq2SeqTrainer
    Seq2SeqTrainingArguments,# <--- USE Seq2SeqTrainingArguments
    GenerationConfig
)

# %% [markdown] id="a73b39b7"
# ## 1. Setup Environment

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 82487, "status": "ok", "timestamp": 1744688969434, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="24db3a2a" outputId="20918020-3c7d-452c-83d0-e13d474cd5fe"
SCRIPT_NAME = "lutfi_20240516_indobart_v2" # Changed script name

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

# %% [markdown] id="5a5a7144"
# ## 2. Configuration

# %% executionInfo={"elapsed": 75, "status": "ok", "timestamp": 1744688969511, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="1dd48bb2"
# --- Model ---
MODEL_CHECKPOINT = "indobenchmark/indobart-v2" # <--- CHANGED MODEL

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
# Max length for *encoder input* and *decoder input/output*
# Can be set differently if needed (e.g., MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
MAX_LENGTH = 128
# INFORMAL_PREFIX = "informal: " # Not needed for standard BART seq2seq
# FORMAL_PREFIX = " formal: "    # Not needed for standard BART seq2seq
MAX_NEW_TOKENS_GEN = 64 # Max tokens to generate during inference/evaluation

# --- Training ---
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 16 # BART might handle larger batches
EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 1 # Train a bit longer for Seq2Seq?
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100 # Log metrics every 100 steps
SAVE_STEPS = 500   # Save checkpoint every 500 steps
EVAL_STEPS = 500   # Evaluate every 500 steps (aligned with save steps)
SAVE_TOTAL_LIMIT = 1 # Keep the latest 1 checkpoints
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available
PREDICT_WITH_GENERATE = True # Generate predictions during evaluation for BLEU

# --- Output Directories (Set in Section 1) ---
# CHECKPOINT_DIR, FINAL_MODEL_DIR, DATA_DIR are set in the 'Setup Environment' section

# %% [markdown] id="54c17c67"
# ## 3. Load Data

# %% executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1744688969515, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="c85e4113"
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1951, "status": "ok", "timestamp": 1744688971466, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="9aa853be" outputId="1fb6b16d-8058-41a0-d8f8-240d71f42548"
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

# %% [markdown] id="81478f9e"
# ## 4. Load Base Tokenizer and Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 440, "referenced_widgets": ["1d0ed21bb53046f3bc12414e0aa804d8", "be913ec79bfc4d5cb44cccfc290e3ed9", "37ce7078f2684e9c80dd0fda95bc4851", "ce3308b84b6f45e1a92927f9bbda2f1a", "fb42025f75bc42c59fdac629ef22a02c", "5c7ecbfcb04f4b4d9938f27e2fbc0f17", "fb01b1bd047e4b4384987e0fc7499d25", "7986996f0c274054b98f40bb34a1602c", "99e097e660c5406d8e09c39e985b4b76", "45f7b0affebb4aeca3e6301450eac56e", "676efe0e7f8e48ebba8552ae56bc3fc0", "a5de016d4f044731a984c921c0b6ac12", "f46f2689cbbd4693bf8b426d4f31058a", "659e68b45db8490cb4ab485f0d5dc064", "e2e5aca9d17545b6af52d7308e618904", "ead1398308bc4c2d832c43c45d7eb04c", "b8a7d3153aeb463c984256cb7ca504a5", "16419ed3baa3478ba4b306656bdc18da", "d017a0725e274e2ba0a09f20f90c7aa8", "622f9c8bf97b4ddc8f309206f189fc98", "da09f7df03c44f79acff44bc0924b8de", "0485da61e2534dc8aca6991e3f1c3f9f", "5fdd8ff1cc984b92abdd9860d54da339", "eddb33f43ea54e3bb6c3e5fff1f9012c", "de84a1c5e7c14455965a392e5036834e", "fe692ae636254fe7b7610da0d7364eda", "0a6a5efd49b0461cab6abf2bbd49488d", "405090adf8b249bca586fa79be8d93ec", "368caed11dc949a9a3a1f8b4cfbf4324", "1bfb8d66e0604d379ce9eede5dc1f4a7", "c663a23f8a6e41739b9046364b1e89ab", "1a09b1cb555b4f13990ca4753b426d05", "04810076eded48c5807e6a326cedcac8", "32c970e57316444c881436ea03750897", "3bcdd547237c4908b49abb15c4f44b27", "4771da6331b4433b9abbf660c30a0611", "874b35719d574a9cb6d7080d4ffa8112", "cec7967769c44e4eaf8c0c7c78f8a3ef", "9b458b6692e74ceba503620f229386f7", "510278f6002345f3b8c6c90b4296358d", "9502c31e02ff4c2c9ab481449bc93064", "9cc38c441015446c96b185b62d4dafff", "1c5e5197524843b0bb079923d9f2dfbc", "b0daee0313b74c858afe6294be4451b8", "7f032676f1604922b4cfd9dc3ba40f91", "7cd910eece244c07a385e8bf6600128f", "ce8053ce2c4a4041af247c7373420c93", "30d3384bf8f341f5b51793e6255daac8", "8f7fb86cdbba4553b16fa4bd1add89e8", "f90bdaabcab048b59348a0e04ebc2472", "7f93cf322a2e42e2a9af2b4b4cfa78b7", "88af5b068a0a4426a0d2c1d03be0e69e", "b3c68d172ab043929c6aea0f22a57cc2", "9c9567b0ee2c44f888e0e6a4fd478dbe", "6f0638137d28407196b58dd8709dcfe6"]} executionInfo={"elapsed": 11439, "status": "ok", "timestamp": 1744688982903, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="838949ec" outputId="e5a32978-7877-4416-ff3d-c7dd7d4d2054"
tokenizer = None
model = None
print(f"\nLoading IndoNLGTokenizer and AutoModelForSeq2SeqLM from checkpoint: {MODEL_CHECKPOINT}")

try:
    # Load tokenizer using AutoTokenizer
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print("IndoNLGTokenizer loaded successfully.")
    # Check if special tokens are present (BART usually has them)
    print(f"EOS token: {tokenizer.eos_token}, Pad token: {tokenizer.pad_token}, BOS token: {tokenizer.bos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}, Pad token ID: {tokenizer.pad_token_id}, BOS token ID: {tokenizer.bos_token_id}")


    # Load model using AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    print("Base Seq2Seq model loaded successfully.")

except Exception as e:
    print(f"Error loading base model or tokenizer: {e}")
    # Depending on severity, you might want to raise e or exit
    # raise e


# %% [markdown] id="a5a5a550"
# ## 5. Preprocess Data

# %% executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1744688982905, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="5e38009e"
def preprocess_function(examples):
    """Tokenizes informal (input) and formal (target) text for Seq2Seq."""
    if tokenizer is None:
        raise ValueError("Tokenizer is not loaded.")

    # Tokenize informal text (source)
    # Padding and truncation applied here.
    model_inputs = tokenizer(
        examples["informal"],
        max_length=MAX_LENGTH,
        padding="max_length", # Pad source sequences to MAX_LENGTH
        truncation=True,
        # return_tensors="pt" # Not needed here, map handles batching
    )

    # Tokenize formal text (target) for labels
    # Padding and truncation applied here.
    # Use tokenizer's context manager for targets in recent Transformers versions
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["formal"],
            max_length=MAX_LENGTH,
            padding="max_length", # Pad target sequences to MAX_LENGTH
            truncation=True,
            # return_tensors="pt"
        )

    # The DataCollatorForSeq2Seq will handle shifting the labels automatically
    # and replace padding token IDs in labels with -100.
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %% colab={"base_uri": "https://localhost:8080/", "height": 255, "referenced_widgets": ["689eb50442a340c79afc4977c995c9f5", "487f828d4d3b41c79fe2cc13eb255853", "f6b7ded194cd410a921ec469ab8bac48", "16455e7ef70f4321a624fd05e0b01870", "30ab5a40feef4b17a7ddc82772c09edb", "ee14f3ebefc243f69f638b931082f949", "8e9f0470d41d466f8e922324f92b4449", "2be82a2fb90547cbb9ce7f876fbe4336", "e2c2b398869a49979476807e689b4fc3", "aff36beef04d4fdd9990f536f1ffb4d8", "29d1633544a14a0c85596735c9ea37de", "2ea955a4997d4eadbf3b11b405097089", "2ba3a05f83cc4d55afce6385c93dd07a", "bb1039e4684746978ad0028186e884ad", "46ba775fce0848dc9ae584dbaacc09a9", "2692f9b69c934716b8090f5ae4a71678", "20005476854e4f0aa534e11d2211c0bc", "d3fc4ce628e2445fbf1b66a3d8269391", "ef651a022ba240049241124e08bde428", "7668ca9cf0ff4741b1bac7bf45a762ad", "8b5e68aec9064f98a6eea96ef13fda4a", "0a3ef083f5544c54a09bf59694f8af11", "95508cc8e4bb419e97bb71307d78203c", "21e365d11e00490a8e288964cb794249", "b7f23a8d203d44728bd71de1ff611688", "09f4d086726340e58227c2ac4896701e", "e46adfe9f07e4ec885f054656ef36364", "b31a3bacca234aeba34d0159c18dc6e5", "f3d81bd742fb4b36a8fe269159257f5e", "dd69948caf9548acbc447f0294c471f1", "76026eca9f104392857d81bbaf1d5691", "1475b1db272740b6bc94491cef68ec52", "e4a68f99abf042708ffadf0b127264cf"]} executionInfo={"elapsed": 1186, "status": "ok", "timestamp": 1744688984092, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="93b1e1cc" outputId="6bdfb42d-7dd1-41f6-db4e-129b5e28b853"
tokenized_datasets = None
if raw_datasets and tokenizer and model:
    print("\nApplying preprocessing...")
    try:
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
        # print("Input IDs shape:", tokenized_datasets["train"][0]['input_ids'].shape)
        # print("Attention Mask shape:", tokenized_datasets["train"][0]['attention_mask'].shape)
        # print("Labels shape:", tokenized_datasets["train"][0]['labels'].shape)

        # Decode example for verification
        # print("\nDecoded Input IDs (Informal):")
        # print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids'], skip_special_tokens=True))
        # print("\nDecoded Labels (Formal):")
        # Replace -100 used for padding in labels before decoding
        # label_ids = tokenized_datasets['train'][0]['labels'].clone()
        # label_ids[label_ids == -100] = tokenizer.pad_token_id
        # print(tokenizer.decode(label_ids, skip_special_tokens=True))

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback; traceback.print_exc()
else:
    print("\nSkipping preprocessing due to missing data, tokenizer, or model.")

# %% [markdown] id="0bb3a06c"
# ## 6. Setup Training

# %% colab={"base_uri": "https://localhost:8080/", "height": 101, "referenced_widgets": ["67f1fb76c6df4e34925159d5fad299e4", "a8106ce4ae644981969b9f572fc151e5", "c2fc438b2a08419f966329bbfa7a4c08", "cf2b3032956f43d897dd08441e6ebe4b", "f4145a1485a34996b27dfc8f1d3967de", "8f97a85ecc4943a9bdd6e45b832c193f", "050f23fce8894f5b997df6a085121840", "353b30a2a3d24b8fa88078b5f64da41e", "5b5ea3111ab44885a572c4c4bdf8e6f1", "4900e390ae7c4d579cce798ee06582cd", "1672ab4684d94e718f2d65263d33f20e"]} executionInfo={"elapsed": 1768, "status": "ok", "timestamp": 1744688985876, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="2b884e4f" outputId="f0dc7812-0822-4083-e59d-fcd800b3c503"
data_collator = None
metric = None

# Data Collator for Seq2Seq
if tokenizer and model:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model # The collator needs the model for potential label shifting/masking
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

# Compute metrics function for Seq2Seq (handles generation and BLEU)
def compute_metrics(eval_preds):
    if not metric:
        print("Warning: SacreBLEU metric not loaded. Returning empty metrics.")
        return {}

    preds, labels = eval_preds
    # preds are generated token IDs (due to predict_with_generate=True)
    # labels are the ground truth token IDs

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode generated predictions
    # Replace -100 (ignore index) and pad_token_id with pad_token for decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels
    # Replace -100 (ignore index) with pad_token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Basic post-processing: remove leading/trailing spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # SacreBLEU expects references as list of lists of strings
    decoded_labels_bleu = [[label.strip()] for label in decoded_labels]

    # Compute BLEU score
    try:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
        bleu_score = result["score"]

        # Also calculate generation length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        gen_len = np.mean(prediction_lens)

        return {"bleu": bleu_score, "gen_len": gen_len}

    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return {"bleu": 0.0, "gen_len": 0.0}


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 238, "status": "ok", "timestamp": 1744688986115, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="1015731b" outputId="0c9d24b2-5df4-4b83-887e-cf9d88fb6c41"
training_args = None
if tokenized_datasets and CHECKPOINT_DIR: # Need CHECKPOINT_DIR for output
    training_args = Seq2SeqTrainingArguments( # <--- Use Seq2SeqTrainingArguments
        output_dir=str(CHECKPOINT_DIR),       # Save checkpoints to CHECKPOINT_DIR
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
        predict_with_generate=PREDICT_WITH_GENERATE, # <--- Enable generation during eval
        load_best_model_at_end=True,        # Load the best model based on metric
        metric_for_best_model="bleu",       # <--- Use BLEU score to find best checkpoint
        greater_is_better=True,             # <--- Higher BLEU is better
        save_total_limit=SAVE_TOTAL_LIMIT,  # Keep only the latest N checkpoints
        report_to="none",                   # Disable external reporting unless configured
        # Generation specific args (can be added if needed, but defaults are often ok)
        # generation_max_length=MAX_LENGTH, # Can override model config
        # generation_num_beams=1,
    )
    print(f"\nSeq2Seq Training arguments configured. Checkpoints will be saved to: {CHECKPOINT_DIR}")

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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 730, "status": "ok", "timestamp": 1744688986844, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="2b491d4a" outputId="527d7975-403e-4139-e3a0-8f02a95e1a3b"
trainer = None
if model and training_args and tokenized_datasets and data_collator and tokenizer:
    trainer = Seq2SeqTrainer( # <--- Use Seq2SeqTrainer
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, # Pass the AutoTokenizer instance
        data_collator=data_collator,
        compute_metrics=compute_metrics, # Use the BLEU calculation function
    )
    print("Seq2SeqTrainer initialized.")
else:
    print("\nCannot initialize Trainer due to missing components.")

# %% [markdown] id="15ce4f4b"
# ## 7. Train Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 58, "status": "ok", "timestamp": 1744688986903, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="3c4a6279" outputId="0092dc0e-bac5-475c-c0da-9dfb7e9a52d2"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} executionInfo={"elapsed": 30055, "status": "ok", "timestamp": 1744689016959, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="21c40d07" outputId="cde66d26-d68e-41ef-baab-cc89b1480e0a"
train_start_time = time.time()
train_result = None

if trainer:
    print("\nStarting model training...")
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
        if train_result:
            metrics = train_result.metrics
            metrics["train_duration_seconds"] = training_duration_seconds # Add duration
            trainer.log_metrics("train_summary", metrics)
            # trainer.save_metrics("train", metrics) # Optional: Saves metrics.json in checkpoint dir
            # trainer.save_state() # Saves trainer state (including logs) in checkpoint dir

            print(f"Training summary metrics: {metrics}") # Will include eval metrics like BLEU if eval ran

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
             print(f"Best model checkpoint during training (based on {training_args.metric_for_best_model}): {trainer.state.best_model_checkpoint}")
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


# %% [markdown] id="e0723da3"
# ## 8. Evaluate Model (BLEU Score - Detailed Output)
#
# Note: BLEU score is already computed during training evaluation steps if `compute_metrics` is set up correctly. This section provides a way to generate predictions explicitly on a dataset (e.g., validation or test) using the final best model and save them alongside references.

# %% executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1744689016967, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="9540aa5a"
def generate_formal_predictions_bart(dataset, model_to_eval, tokenizer_to_eval, batch_size=EVAL_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS_GEN, num_beams=4):
    """Generates formal text predictions using the fine-tuned BART model."""
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
        pad_token_id=tokenizer_to_eval.pad_token_id,
        bos_token_id=tokenizer_to_eval.bos_token_id, # BART might use BOS
        decoder_start_token_id=model.config.decoder_start_token_id, # Important for BART
        do_sample=False, # Greedy or beam search for eval
        num_beams=num_beams,
        early_stopping=True if num_beams > 1 else False
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

        # Tokenize informal inputs ONLY
        inputs = tokenizer_to_eval(
            informal_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH # Max length for encoder input
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate, feeding only input_ids and attention_mask
            outputs = model_to_eval.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config
            )

        # Decode the entire generated sequence
        batch_preds = tokenizer_to_eval.batch_decode(outputs, skip_special_tokens=True)

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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 22318, "status": "ok", "timestamp": 1744689039289, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ad1a5201" outputId="4e26e343-5f93-4e93-ddad-75d55ec068d1"
# Perform generation and BLEU calculation on the validation set using the *best* model loaded by the trainer
# This provides detailed outputs; the BLEU score itself should match the final eval step score from training.
validation_eval_df = None # Initialize dataframe variable
if trainer and trainer.model and metric is not None and raw_datasets and DATA_DIR:
    print("\nGenerating detailed predictions on the validation set using the final best model...")

    # The trainer should have loaded the best model if load_best_model_at_end=True
    eval_model = trainer.model
    eval_tokenizer = trainer.tokenizer # Use the tokenizer associated with the trainer/model

    # Use the *original* validation dataset (with text columns) for evaluation input/refs
    validation_data_for_eval = raw_datasets["validation"]

    predictions, references = generate_formal_predictions_bart(
        validation_data_for_eval,
        eval_model,
        eval_tokenizer,
        batch_size=EVAL_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS_GEN
    )

    if predictions is not None and references is not None:
        print("\nCalculating BLEU score on generated predictions (for verification)...")
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
                print("\nValidation BLEU Score (from generated predictions):")
                # Print score nicely, handle potential missing score key
                print(json.dumps(bleu_results, indent=2))
                # Compare this score to the one reported during the trainer's final evaluation step


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

        except Exception as e:
            print(f"Error calculating or saving BLEU score/predictions: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Prediction generation failed, skipping BLEU calculation and saving.")

elif not DATA_DIR:
    print("\nSkipping detailed evaluation output generation: DATA_DIR not set.")
else:
    print("\nSkipping detailed evaluation output generation due to missing components (trainer, data, metric) or failed training.")

# Display the dataframe if it was created
if validation_eval_df is not None:
    print("\n--- Validation Predictions DataFrame ---")
    display(validation_eval_df)
else:
    print("\nValidation predictions DataFrame was not generated.")


# %% [markdown] id="e490e26b"
# ## 9. Save Final Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2552, "status": "ok", "timestamp": 1744689041848, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="5d85d8f3" outputId="56978baa-9bae-408d-c421-3c83839f58d4"
if trainer and trainer.model and FINAL_MODEL_DIR and SAVE_FINAL_MODEL:
    print(f"\nSaving the final best model to: {FINAL_MODEL_DIR}")
    try:
        # The trainer already holds the best model in memory if load_best_model_at_end=True
        # Use trainer.save_model which also saves tokenizer and config
        trainer.save_model(str(FINAL_MODEL_DIR)) # Convert Path to string
        print(f"Final model and tokenizer saved successfully to {FINAL_MODEL_DIR}.")

        # Training arguments were already saved to DATA_DIR in section 6.
        # No need to save them again here.

    except Exception as e:
        print(f"Error saving final model: {e}")
elif not FINAL_MODEL_DIR:
     print("\nSkipping final model saving: FINAL_MODEL_DIR not set.")
else:
    print("\nSkipping final model saving: (possibly due to training error or SAVE_FINAL_MODEL=False).")

# %% [markdown] id="4b5fc070"
# ## 10. Inference

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2472, "status": "ok", "timestamp": 1744689561819, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="e87af65d" outputId="2f828c2c-01e8-41e1-905b-bd7247959418"
# Load the final model from FINAL_MODEL_DIR for inference
inference_model = None
inference_tokenizer = None

if FINAL_MODEL_DIR and FINAL_MODEL_DIR.exists():
    print(f"\nLoading final model from {FINAL_MODEL_DIR} for inference...")
    try:
        # Load model using AutoModelForSeq2SeqLM
        inference_model = AutoModelForSeq2SeqLM.from_pretrained(str(FINAL_MODEL_DIR)) # Convert Path to string
        # Load the corresponding tokenizer using AutoTokenizer
        inference_tokenizer = IndoNLGTokenizer.from_pretrained(str(FINAL_MODEL_DIR))

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


# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1744689566040, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="2fec3665"
def formalize_text_bart(sentence: str, model, tokenizer, max_new_toks=MAX_NEW_TOKENS_GEN, num_beams=4):
    """Uses the loaded fine-tuned IndoBART model to convert informal text to formal."""
    if not model or not tokenizer:
        return "Error: Inference model or tokenizer not available."
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
         return "Error: Tokenizer requires eos_token_id and pad_token_id."

    # Prepare input: tokenize informal sentence directly
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True, # Add padding if processing a batch
        truncation=True,
        max_length=MAX_LENGTH # Max length for encoder input
    )

    device = model.device # Use the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Configure generation (similar to evaluation generation)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_toks,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        num_beams=num_beams,
        early_stopping=True if num_beams > 1 else False,
        do_sample=False # Usually False for inference tasks like normalization
    )

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config
            )

        # Decode the generated sequence(s)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output.strip()

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error during generation: {e}"


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4037, "status": "ok", "timestamp": 1744689570088, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="75c5c96b" outputId="c9dfe173-7d1b-4376-82f9-f540b64fc503"
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
        formalized = formalize_text_bart(sentence, inference_model, inference_tokenizer)
        print("-" * 30)
        print(f"Informal: {sentence}")
        print(f"Formal:   {formalized}")
        results.append({"Informal": sentence, "Formal (Predicted)": formalized})

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

print("\n--- Script Execution Finished ---")

# %% executionInfo={"elapsed": 63, "status": "ok", "timestamp": 1744689570153, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="76alnOhjbE5D"
