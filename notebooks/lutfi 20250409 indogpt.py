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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="51585975"
# # Indonesian Text Normalization using Fine-tuned IndoGPT (GPT-2 based)
#
# **Version 2:** Uses the required `IndoNLGTokenizer` from `indobenchmark-toolkit`.

# %% [markdown] id="1d0da3e4"
# This script fine-tunes a pre-trained GPT-2 based model (`indobenchmark/indogpt`) for the task of converting informal Indonesian text to its formal equivalent. It covers data loading, preprocessing (adapted for Causal LM), model fine-tuning, evaluation (using BLEU), and inference.

# %% id="d5b0fe78"
# !pip install transformers==4.44.2 evaluate sacrebleu datasets torch accelerate sentencepiece --quiet
# !pip install indobenchmark-toolkit --quiet  # <--- INSTALL THE REQUIRED TOOLKIT

# %% colab={"base_uri": "https://localhost:8080/", "height": 86, "referenced_widgets": ["fd19c4344142463892f9c4d01d02c93b", "edc3afecb9974c5eb52ad1f1cd81eb6d", "88138f234b92480fbc14b3fe398f4207", "12fe584eb57049aeaa1b7abacd21a03c", "f9a7e8b0c1b348a7be43b27ceddc6a2e", "46d9140ca7ac40249cad77bc4b025da9", "90e7df43974e4f7db0174e851ad3f910", "48f55b0791fe4253970085dc61ff2df8", "7908fc98fca04ec19b60cbddf4c0bbc2", "db145c5111734f299dff187b5750630a", "21150e032af24ebf9c0ad4e1c61bd5f2"]} executionInfo={"elapsed": 15849, "status": "ok", "timestamp": 1744248066684, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ca05d305" outputId="a20e28be-2029-4624-a91a-a1d80bd07f19"
import json
import evaluate
import numpy as np
import pandas as pd
import torch
from requests import request
from datasets import Dataset, DatasetDict
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
from IPython.display import display # For displaying DataFrames nicely

# %% [markdown] id="4c2e36b9"
# ## 1. Configuration

# %% id="2affc3b7"
# Model checkpoint for the GPT-2 based model
MODEL_CHECKPOINT = "indobenchmark/indogpt"

# Data URLs
# BASE_URL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/"
BASE_URL = "https://raw.githubusercontent.com/Lutfi-Azis/lunashimu-formalizer-dataset/refs/heads/main/dataset/"
TRAIN_INF_URL = f"{BASE_URL}train.inf"
TRAIN_FOR_URL = f"{BASE_URL}train.for"
DEV_INF_URL = f"{BASE_URL}dev.inf"
DEV_FOR_URL = f"{BASE_URL}dev.for"
TEST_INF_URL = f"{BASE_URL}test.inf" # Test set included for potential final evaluation
TEST_FOR_URL = f"{BASE_URL}test.for"

# Preprocessing & Generation parameters
MAX_LENGTH = 128
INFORMAL_PREFIX = "informal: "
FORMAL_PREFIX = " formal: "
# Note: The space before "formal:" is important for tokenization

# Training parameters
OUTPUT_DIR = "./indogpt-formalizer-finetuned"
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8 # Adjust based on GPU memory
EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 1 # Start with 1-3 epochs for fine-tuning
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

# %% [markdown] id="3e18fdf2"
# ## 2. Load Data

# %% id="432298a3"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3936, "status": "ok", "timestamp": 1744248070636, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="a7398051" outputId="f95d111b-09ca-48fb-cbf7-e47c56e99b6c"
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
if not (len(train_inf) == len(train_for) > 0 and \
        len(dev_inf) == len(dev_for) > 0 and \
        len(test_inf) == len(test_for) > 0):
    print("\nError: Data loading issues or length mismatch between informal/formal pairs.")
    raw_datasets = None
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
# ## 3. Load Pre-trained Tokenizer and Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3147, "status": "ok", "timestamp": 1744248073794, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="59ee86d9" outputId="736830de-8993-42a3-f732-409c115e67ea"
print(f"\nLoading specific IndoNLGTokenizer and model from checkpoint: {MODEL_CHECKPOINT}")

try:
    # Load the specific tokenizer class
    # Use IndoNLGTokenizer directly instead of AutoTokenizer
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_CHECKPOINT) # <--- USE SPECIFIC CLASS

    # Set pad_token if it's not already set (common for GPT-2 based models)
    if tokenizer.pad_token is None:
        print("Warning: pad_token not set. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    print("IndoNLGTokenizer loaded successfully.")
    print(f"EOS token: {tokenizer.eos_token}, Pad token: {tokenizer.pad_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}, Pad token ID: {tokenizer.pad_token_id}")


    # Load model using AutoModelForCausalLM (this part is usually fine)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)
    # Ensure model's pad_token_id is aligned with tokenizer's pad_token_id
    # This is crucial if we just set the tokenizer's pad_token manually
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded successfully.")


except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Raise the error or handle it more robustly depending on the use case
    raise e # Stop execution if loading fails
    # tokenizer = None
    # model = None

# %% [markdown] id="210d6c53"
# ## 4. Preprocess Data for Causal LM Fine-tuning

# %% id="e52211de"
def preprocess_function(examples):
    if tokenizer is None:
        raise ValueError("Tokenizer is not loaded.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must have an EOS token for Causal LM.")
    if tokenizer.pad_token is None:
        # Ensure pad token is set if it wasn't auto-detected
        print("Warning: pad_token not set during preprocessing. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    inputs_text = []
    # Construct the combined text for input_ids tokenization
    for inf, form in zip(examples["informal"], examples["formal"]):
        # NOTE: We do NOT add eos_token here yet, tokenizer adds it if configured,
        # or we handle it via labels. Let's rely on tokenizer adding it.
        text = f"{INFORMAL_PREFIX}{inf}{FORMAL_PREFIX}{form}{tokenizer.eos_token}"
        inputs_text.append(text)

    # Tokenize the combined text WITH padding and truncation
    model_inputs = tokenizer(
        inputs_text,
        max_length=MAX_LENGTH,
        padding="max_length", # Explicitly pad to max_length
        truncation=True
    )

    # Create labels: shift inputs to the right, mask the prompt part, AND ensure padding is -100.
    # We need to recalculate labels based on the padded/truncated input_ids.
    labels = []
    for i in range(len(model_inputs["input_ids"])):
        # Get the actual input_ids used (already padded/truncated)
        input_ids = model_inputs["input_ids"][i]

        # Find where the target (formal part) begins IN THE ORIGINAL UNPADDED TEXT.
        # This determines how much of the start of the sequence to mask.
        prompt_part = f"{INFORMAL_PREFIX}{examples['informal'][i]}{FORMAL_PREFIX}"
        # Tokenize prompt *without special tokens* or padding to find its raw length
        prompt_token_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_token_ids)

        # Create label sequence, initially copying input_ids
        label_ids = list(input_ids) # Make a mutable copy

        # Mask tokens belonging to the prompt (informal sentence + prefixes)
        # Set labels to -100 for tokens we want to ignore in loss calculation
        for j in range(prompt_length):
            # Check boundary: only mask if index j is valid for this sequence
            if j < len(label_ids):
                label_ids[j] = -100

        pad_token_id = tokenizer.pad_token_id
        for j in range(len(label_ids)):
            if input_ids[j] == pad_token_id:
                 label_ids[j] = -100

        # Optional: Verify that EOS token (if present and not part of prompt/padding) is NOT masked
        # eos_token_id = tokenizer.eos_token_id
        # if eos_token_id in input_ids:
        #     eos_indices = [k for k, token_id in enumerate(input_ids) if token_id == eos_token_id]
        #     for eos_idx in eos_indices:
        #         if eos_idx >= prompt_length and input_ids[eos_idx] != pad_token_id : # Should not be pad, but double check
        #              # If it got masked by prompt logic somehow (unlikely) or padding logic, revert if needed
        #              if label_ids[eos_idx] == -100:
        #                  # This indicates an issue, but for now let's assume prompt masking is correct.
        #                  # The padding mask above is the most critical part.
        #                  pass

        labels.append(label_ids)

        assert len(label_ids) == MAX_LENGTH, f"Label length mismatch: {len(label_ids)} vs {MAX_LENGTH}"
        assert len(input_ids) == MAX_LENGTH, f"Input ID length mismatch: {len(input_ids)} vs {MAX_LENGTH}"


    model_inputs["labels"] = labels
    return model_inputs

# %% colab={"base_uri": "https://localhost:8080/", "height": 307, "referenced_widgets": ["5e2b9e656ef94106a223eddac16e309d", "6531843f879e4e23bc863ca145e8eeff", "9d7308d9c80a4a089ab61e350c17de35", "acaaf3f335a84ed3b0bc8e4687af5425", "ca1cee9559884d35b065f0b342c2d0a4", "677d8ef57b9e45099178bd5485001bdf", "21c1d950a78a4b16bc6ee58fa80c4442", "1a85a239b77a487e952980b98e840e1d", "7cdae18cb14c4efd8e5205b887b50ba7", "9ea5bab5fac843ed8c72972333fcb68d", "f88690a690a9413e8b1e5b1cac794260", "6a7491dcbfa44114acd18727cddfae79", "74f9a19cd4d24b0e8dcc3d9e82ea909c", "80b4e3547aaf4245a2a470304a5f1b3d", "e5b9a5d9e72d479cb2b69057383cf414", "50d990d06b6c45fdbee7b40619cdb07c", "6c976b29dae9485db7d5600ca1b1c0d3", "c1c51d5b2d254a3fa17770cfb3e4b217", "91cc53257b214ffd8ec6de43a2b1f60e", "07966e4055ee4802984c559665ffc591", "9dea0af9ecf24390a9adf59fc0e62e48", "e35c84e87dfa44289b39217fda6fbfe2", "bff0b84333eb42ea96bf2bb07697bcf1", "8b08a671f31e41f8935e7bd3ded53d30", "eccce465edce4dc58799282ff30d07d0", "a3f53c91eed34b68a2d547f23911556e", "db81d4c93ef74820b7d021285b180c2e", "f941a405f0164715b5ce2755c4cff35f", "379b7e5c01b547cf9cee563bc2c31160", "1636767b43bc42e9ac885f950e77dd93", "1aa5fcb7f8c141c49595e0dfa370f5ba", "264ce64678e3429e93ac5442fc7cdd4c", "5bd2373fe997405ab927f786845bc48f"]} executionInfo={"elapsed": 1708, "status": "ok", "timestamp": 1744248471328, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="b0bb96c9" outputId="26b4d0b2-c489-48d3-9a63-3782b4854649"
if raw_datasets and tokenizer:
    print("\nApplying preprocessing...")
    # Ensure to overwrite or use a new variable name if needed
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names # Remove old columns
    )
    print("Preprocessing complete.")
    print("\nSample preprocessed example:")
    print("Input IDs:", tokenized_datasets["train"][0]['input_ids'])
    print("Labels:", tokenized_datasets["train"][0]['labels'])
    # Decode to verify masking (optional)
    # print("\nDecoded Input:")
    # print(tokenizer.decode(tokenized_datasets["train"][0]['input_ids']))
    print("\nDecoded Labels (non-masked part):")
    label_example = [tok if tok != -100 else tokenizer.pad_token_id for tok in tokenized_datasets["train"][0]['labels']]
    print(tokenizer.decode(label_example, skip_special_tokens=True)) # skip_special_tokens removes padding visualization
else:
    print("\nSkipping preprocessing due to previous errors.")
    tokenized_datasets = None


# %% [markdown] id="2911ce20"
# ## 5. Setup Training

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1744248471345, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="001d74b0" outputId="84f7879f-8283-415a-88bd-fcb733cb6196"
# Data Collator for Causal LM
if tokenizer:
    # DataCollator works with the tokenizer object regardless of its specific class
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # False for Causal LM (GPT-2 style)
    )
    print("\nData collator created for Causal LM.")
else:
    data_collator = None
    print("\nSkipping data collator creation due to missing tokenizer.")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1114, "status": "ok", "timestamp": 1744248472461, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ffd683e5" outputId="eb15441c-6e23-4503-84cb-b4c4af65c90e"
# Evaluation Metric (SacreBLEU)
try:
    metric = evaluate.load("sacrebleu")
    print("\nSacreBLEU metric loaded.")
except Exception as e:
    print(f"Error loading SacreBLEU metric: {e}")
    metric = None

# Compute metrics (loss only during training)
def compute_metrics_loss_only(eval_preds):
     return {}

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1744248472476, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ea82bdd5" outputId="e99cc018-e143-4aa3-dac8-310fac93f62f"
# Training Arguments
if tokenized_datasets:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Use validation loss to find best checkpoint
        greater_is_better=False,          # Lower loss is better
        report_to="none",
        # Important: Gradient accumulation can help if batches don't fit memory
        # gradient_accumulation_steps = 2, # Example: process 2 batches before update
    )
    print(f"\nTraining arguments configured. Output directory: {OUTPUT_DIR}")
else:
    training_args = None
    print("\nSkipping TrainingArguments setup due to missing data.")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35, "status": "ok", "timestamp": 1744248472513, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="0561926c" outputId="f265fe1d-0ad0-4493-cd2b-66ffd705e229"
# Initialize Trainer
if model and training_args and tokenized_datasets and data_collator and tokenizer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, # Pass the IndoNLGTokenizer instance
        data_collator=data_collator,
        compute_metrics=compute_metrics_loss_only,
    )
    print("Trainer initialized.")
else:
    trainer = None
    print("\nCannot initialize Trainer due to missing components.")

# %% [markdown] id="43c62155"
# ## 6. Train (Fine-tune) the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 127} executionInfo={"elapsed": 3571687, "status": "ok", "timestamp": 1744255615713, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="a36ec275" outputId="36552d49-cd5d-4324-e27f-fd863e5f5cfa"
if trainer:
    print("\nStarting model fine-tuning...")
    try:
        train_result = trainer.train()
        print("Fine-tuning finished.")

        # Save final model, metrics, and state
        trainer.save_model() # Saves the tokenizer too (important!)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(f"Training metrics: {metrics}")
        if trainer.state.best_model_checkpoint:
             print(f"Best model saved to {trainer.state.best_model_checkpoint}")
        else:
             # Should not happen if load_best_model_at_end=True and evaluation ran
             print(f"Final model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Consider adding logging traceback: import traceback; traceback.print_exc()
else:
    print("\nSkipping training because the Trainer could not be initialized.")


# %% [markdown] id="4f09ef69"
# ## 7. Evaluate the Fine-tuned Model (BLEU Score)

# %% id="fecaaef7"
def generate_formal_predictions(dataset, model, tokenizer, batch_size=EVAL_BATCH_SIZE, max_gen_length=MAX_LENGTH//2):
    """Generates formal text predictions using the fine-tuned Causal LM."""
    if not model or not tokenizer:
        print("Model or tokenizer not available for generation.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_refs = []
    max_new_tokens = max_gen_length

    # Use GenerationConfig for cleaner parameter handling
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False, # Greedy for eval consistency
        num_beams=1      # Change to > 1 for beam search
    )
    print(f"Starting generation with config: {generation_config}")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [f"{INFORMAL_PREFIX}{inf}{FORMAL_PREFIX}" for inf in batch["informal"]]
        references = batch["formal"]

        # Tokenize using the IndoNLGTokenizer
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # Ensure prompt doesn't exceed max_length minus generation space
            max_length=MAX_LENGTH - max_new_tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Decode using IndoNLGTokenizer, skipping special tokens to clean output
        full_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        batch_preds = []
        for j, full_text in enumerate(full_decoded):
            prompt_text_approx = prompts[j] # Use original prompt for finding split point
            # Find the *last* occurrence of the formal prefix in the decoded output
            formal_prefix_index = full_text.rfind(FORMAL_PREFIX.strip()) # Use stripped prefix for robustness

            if formal_prefix_index != -1:
                # Extract text *after* the prefix
                generated_part = full_text[formal_prefix_index + len(FORMAL_PREFIX.strip()):].strip()
            else:
                # Fallback: Try to remove the prompt part based on approximate length (less reliable)
                prompt_decoded_approx = tokenizer.decode(inputs['input_ids'][j], skip_special_tokens=True)
                if full_text.startswith(prompt_decoded_approx):
                    generated_part = full_text[len(prompt_decoded_approx):].strip()
                     # Clean if prefix got generated unexpectedly
                    if generated_part.startswith(FORMAL_PREFIX.strip()):
                         generated_part = generated_part[len(FORMAL_PREFIX.strip()):].strip()
                else:
                    # Cannot reliably determine split point
                    print(f"Warning: Could not cleanly extract generation for example {i+j}. Output: '{full_text}'")
                    generated_part = "" # Default to empty if extraction fails

            # Final cleanup (though skip_special_tokens should handle most)
            if tokenizer.eos_token:
                generated_part = generated_part.replace(tokenizer.eos_token, "").strip()

            batch_preds.append(generated_part)


        all_preds.extend(batch_preds)
        all_refs.extend(references)

        if (i // batch_size + 1) % 10 == 0:
            print(f"Generated for {i + len(batch)} / {len(dataset)} examples")
            if len(batch_preds) > 0:
                 print(f"  Sample Ref: {references[0]}")
                 print(f"  Sample Pred: {batch_preds[0]}")

    print(f"Generation complete for {len(all_preds)} examples.")
    return all_preds, all_refs

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1744255702620, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="125bd414" outputId="60c63fd6-ab39-4db6-cf29-e474c045efa2"
# Perform generation and BLEU calculation on the validation set
if trainer and tokenized_datasets and metric and raw_datasets:
    print("\nEvaluating the fine-tuned model on the validation set (generating predictions)...")

    # Determine model and tokenizer to use for evaluation
    # If load_best_model_at_end=True, trainer.model is the best one.
    eval_model = trainer.model
    eval_tokenizer = tokenizer # Use the same tokenizer instance

    # Use the *original* validation dataset (with text) for evaluation
    validation_data_for_eval = raw_datasets["validation"]

    predictions, references = generate_formal_predictions(
        validation_data_for_eval,
        eval_model,
        eval_tokenizer, # Pass the IndoNLGTokenizer
        batch_size=EVAL_BATCH_SIZE,
        max_gen_length=MAX_LENGTH // 2
    )

    if predictions is not None:
        print("\nCalculating BLEU score...")
        references_bleu = [[ref] for ref in references] # Format for sacrebleu
        try:
            # Ensure predictions and references are lists of strings
            valid_preds = [str(p) for p in predictions]
            valid_refs = [[str(r)] for r_list in references_bleu for r in r_list] # Flatten and stringify inner lists too

            bleu_results = metric.compute(predictions=valid_preds, references=valid_refs)
            print("\nValidation BLEU Score:")
            print(bleu_results)

            eval_df = pd.DataFrame({
                "informal": validation_data_for_eval["informal"],
                "formal_reference": references,
                "predicted_formal": predictions
            })
            display(eval_df.head())
            eval_output_path = f"{OUTPUT_DIR}/validation_predictions_gpt2_v2.csv"
            eval_df.to_csv(eval_output_path, index=False)
            print(f"Validation predictions saved to {eval_output_path}")

            # Log BLEU score with the trainer if possible
            trainer.log({"eval_bleu": bleu_results["score"]})
            with open(f"{OUTPUT_DIR}/eval_bleu_results_v2.json", "w") as f:
                 json.dump(bleu_results, f, indent=4)

        except Exception as e:
            print(f"Error calculating or saving BLEU score: {e}")
    else:
        print("Prediction generation failed, skipping BLEU calculation.")

else:
    print("\nSkipping evaluation (BLEU) due to missing components or failed training.")


# %% [markdown] id="8f9c9beb"
# ## 8. Perform Simple Inference

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1744255616153, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="b2fa94cb" outputId="01907f64-c8e3-46c5-eef8-7b964c761827"
# Load the best model for inference
inference_model = None
inference_tokenizer = None

if trainer:
    try:
        # Determine path to the best model checkpoint
        if trainer.state.best_model_checkpoint:
            checkpoint_path = trainer.state.best_model_checkpoint
            print(f"\nLoading best model from checkpoint for inference: {checkpoint_path}")
        else:
            checkpoint_path = OUTPUT_DIR # Fallback to final save dir
            print(f"\nLoading model from final save directory for inference: {checkpoint_path}")

        # Load model
        inference_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        # Load the corresponding tokenizer using the specific class
        inference_tokenizer = IndoNLGTokenizer.from_pretrained(checkpoint_path) # <--- USE SPECIFIC CLASS

        # Ensure pad token consistency after loading
        if inference_tokenizer.pad_token is None:
            print("Setting pad_token = eos_token for loaded inference tokenizer.")
            inference_tokenizer.pad_token = inference_tokenizer.eos_token
        inference_model.config.pad_token_id = inference_tokenizer.pad_token_id


        # Optional: move to GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # inference_model.to(device)
        print("Inference model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model/tokenizer from checkpoint {checkpoint_path}: {e}")
        inference_model = None
        inference_tokenizer = None
else:
     print("\nTrainer object not available. Cannot load fine-tuned model automatically.")
     # You could manually try loading from OUTPUT_DIR here if needed


def formalize_text_gpt2(sentence: str, model, tokenizer, max_new_toks=64):
    """Uses the fine-tuned IndoGPT model to convert informal text to formal."""
    if not model or not tokenizer:
        return "Error: Inference model or tokenizer not available."
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
         return "Error: Tokenizer requires eos_token_id and pad_token_id."

    # Prepare prompt
    prompt = f"{INFORMAL_PREFIX}{sentence}{FORMAL_PREFIX}"
    # Tokenize using the IndoNLGTokenizer instance
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH - max_new_toks)

    device = model.device # Use the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Configure generation
    generation_config = GenerationConfig(
        max_new_tokens=max_new_toks,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, # Crucial for generation
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Decode only the newly generated tokens (after the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        # Use the IndoNLGTokenizer for decoding
        decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return decoded_output.strip()

    except Exception as e:
        # import traceback; traceback.print_exc() # Uncomment for detailed error
        return f"Error during generation: {e}"


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1744255616168, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ff16e062" outputId="3ebe8084-439f-4a4f-9839-40a94d486baf"
# Test Inference
if inference_model and inference_tokenizer:
    print("\nTesting inference...")
    test_sentences = [
        "min , mau nanya . cara mengaktifkan no indosat yang sudah terblokir karena lupa isi pulsa , gmn yah ?",
        "maling yg jadi provider y begini dech jadinya",
        "kmrn aq liat promo baru, tp lupa dmn",
        "gimana caranya biar bisa dpt diskon gede?",
        "thanks ya infonya sangat membantu",
        "ga ngerti knp paket internet gw cpt bgt abisnya",
        "klo mo upgrade kartu ke 4g dmn ya?"
    ]

    for sentence in test_sentences:
        formalized = formalize_text_gpt2(sentence, inference_model, inference_tokenizer)
        print("-" * 30)
        print(f"Informal: {sentence}")
        print(f"Formal:   {formalized}")
else:
    print("\nSkipping inference test because model/tokenizer could not be loaded.")
