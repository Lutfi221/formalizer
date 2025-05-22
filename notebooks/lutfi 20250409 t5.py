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

# %% [markdown] id="36010afc"
# # Indonesian Text Normalization using Pre-trained IndoNanoT5

# %% [markdown] id="128cacf8"
# This script fine-tunes a pre-trained T5 model (`LazarusNLP/IndoNanoT5-base`) for the task of converting informal Indonesian text to its formal equivalent. It covers data loading, preprocessing, model fine-tuning, evaluation, and inference.

# %% id="252da30c"
# !pip install transformers evaluate sacrebleu datasets torch accelerate sentencepiece --quiet

# %% id="58d4936c"
import json
import evaluate
import numpy as np
import pandas as pd
from requests import request
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from IPython.display import display # For displaying DataFrames nicely


# %% [markdown] id="40a2ff88"
# ## 1. Load Data

# %% id="5384ba58"
def get_lines(url: str) -> list[str]:
    """Fetches text data line by line from a URL."""
    try:
        response = request("GET", url)
        response.raise_for_status() # Raise an exception for bad status codes
        # Filter out empty lines that might result from split
        return [line for line in response.text.split("\n") if line]
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return []


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 297, "status": "ok", "timestamp": 1744209336047, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="ef582d15" outputId="8f547a02-8745-4789-9d83-deb0d90303c5"
print("Fetching data...")
informal_lines = []
formal_lines   = []
# BASE_URL = "https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/"
BASE_URL = "https://raw.githubusercontent.com/Lutfi-Azis/lunashimu-formalizer-dataset/refs/heads/main/dataset/"

for split_name in ("train", "dev", "test"):
    inf_url = f"{BASE_URL}{split_name}.inf"
    for_url = f"{BASE_URL}{split_name}.for"

    inf_lines_split = get_lines(inf_url)
    for_lines_split = get_lines(for_url)

    # Basic check to ensure corresponding lines were fetched
    if len(inf_lines_split) == len(for_lines_split) and len(inf_lines_split) > 0:
        informal_lines.extend(inf_lines_split)
        formal_lines.extend(for_lines_split)
        print(f"Loaded {len(inf_lines_split)} lines from {split_name} split.")
    else:
        print(f"Warning: Mismatch or empty data for {split_name} split. Inf: {len(inf_lines_split)}, For: {len(for_lines_split)}")

print(f"\nTotal lines loaded: Informal={len(informal_lines)}, Formal={len(formal_lines)}")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 64, "status": "ok", "timestamp": 1744209336113, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="16ac6a13" outputId="813e950f-6a47-4afd-fa3f-55a807515e32"
if len(informal_lines) == len(formal_lines) and len(informal_lines) > 0:
    dataset = Dataset.from_dict({"informal": informal_lines, "formal": formal_lines})
    print("\nDataset created successfully.")

    # Split into training and validation sets
    # Using a standard 80/20 split for demonstration
    split_ds = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    print("\nDataset splits:")
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")

    print("\nSample training example:")
    print(train_ds[0])
else:
    print("\nError: Cannot create dataset due to data loading issues or length mismatch.")
    # Handle the error appropriately, maybe exit or raise an exception
    # For this script, we'll print the error and potentially fail later
    train_ds, val_ds = None, None # Set to None to indicate failure

# %% [markdown] id="60a21adf"
# ## 2. Load Pre-trained Tokenizer and Model

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 43, "status": "ok", "timestamp": 1744209336157, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="01ce0bbd" outputId="2a80f705-f35f-4ba2-b74d-db99abceb37c"
model_checkpoint = "LazarusNLP/IndoNanoT5-base"
print(f"\nLoading tokenizer and model from checkpoint: {model_checkpoint}")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 265, "status": "ok", "timestamp": 1744209336423, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="f8003f40" outputId="017243bf-e8c8-40dc-882a-bd7eace47bcb"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 698, "status": "ok", "timestamp": 1744209337142, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="6ab113e4" outputId="f6bee605-09ef-4b47-e577-563e57e3c710"
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    print("Model loaded successfully.")
    # Optional: Check model config if needed
    # print("\nModel Configuration:")
    # print(model.config)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# %% [markdown] id="58f8e6e5"
# ## 3. Preprocess Data

# %% id="2e040331"
max_input_length = 128  # Adjust based on your data/model
max_target_length = 128 # Adjust based on your data/model
prefix = "bakukan: " # T5-style prefix for the task

def preprocess_function(examples):
    if tokenizer is None:
        raise ValueError("Tokenizer is not loaded.")

    inputs = [prefix + text for text in examples["informal"]]
    targets = examples["formal"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False # Padding will be handled by the data collator
    )

    # Tokenize targets (labels)
    # Use text_target for Seq2Seq models as per documentation
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=False # Padding will be handled by the data collator
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %% colab={"base_uri": "https://localhost:8080/", "height": 203, "referenced_widgets": ["2a927713079b4f13b9792569d6c8073a", "fa8a89d70d0d4b38a603b4930d45d94b", "3753114a168f4d49bfa0afef93a7a033", "78b117cd21504282a823477e1ecc07c1", "55a0d87d5fd9458ab0a3675dbfcf5550", "d251fa9e69f745de8d1187d0a87c61f2", "737d3595a38e46768521b29da158ee14", "2613f9125c48459e8557731eb8eec95e", "ecff12f82dbb4dda9a8765740b4d7b2c", "fdd8171c4b1b4854aef146c97478f59d", "a96f12badebe4f77a1f2266a8aa93d33", "c65d6f2317e349c1b8b05399e3ce2718", "1eff339ffe524587824860dc89e3c809", "c4c5c452f5c644d99b3dc3f82adcad7c", "0925bf70a6a0492eb55861a6d0e276de", "6f5e5b49f7014726b1a6df490a8405b6", "52404ef954c8426080cb8bff6de451d4", "c8e664007666445bb47d98c96627bfd3", "07ff1bf2ec9c49ba909756869da2b594", "88fcf978c42a40cfb0226668ad42bc97", "dc6285f90333426d883d87d862748230", "97f5e6cd162c41479e63112726f3bd6d"]} executionInfo={"elapsed": 2242, "status": "ok", "timestamp": 1744209339420, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="2028d6c0" outputId="4123356f-86d3-4c82-d6d1-395da46f1cd5"
if train_ds and val_ds and tokenizer:
    print("\nApplying preprocessing...")
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)
    print("Preprocessing complete.")
    print("\nSample preprocessed example (input_ids and labels):")
    print(train_ds[0]['input_ids'])
    print(train_ds[0]['labels'])
else:
    print("\nSkipping preprocessing due to previous errors.")

# %% [markdown] id="43736d13"
# ## 4. Setup Training

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1744209339440, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="bb529f7e" outputId="c0352c51-478f-46af-e2ef-00c840f25eb0"
# Handles dynamic padding during batch creation
if tokenizer and model:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True # Pad to the longest sequence in the batch
    )
    print("\nData collator created.")
else:
    data_collator = None
    print("\nSkipping data collator creation due to missing tokenizer or model.")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1744209339447, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="3343a056" outputId="fb4078d0-c65b-4ef9-c211-e1cf0b9937a0"
try:
    metric = evaluate.load("sacrebleu")
    print("\nSacreBLEU metric loaded.")
except Exception as e:
    print(f"Error loading SacreBLEU metric: {e}")
    metric = None

def compute_metrics(eval_preds):
    if not metric or not tokenizer:
        print("Warning: Metric or tokenizer not available, skipping metric computation.")
        return {}

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0] # Some models return tuples

    # Decode predictions, skipping special tokens
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels (used for padding/masked tokens) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode labels, skipping special tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-processing: remove leading/trailing whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # SacreBLEU expects references to be a list of lists
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Compute SacreBLEU score
    try:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # Return score, removing the verbose 'counts', 'totals', etc.
        return {"bleu": result["score"]}
    except Exception as e:
        print(f"Error during metric computation: {e}")
        return {"bleu": 0.0} # Return default value on error


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1744209339456, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="abbb74b6" outputId="f37d6226-eb13-48cd-c010-7875b7333e41"
# Adjust parameters based on your resources and needs
output_directory = "./indot5-formalizer-finetuned"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_directory,
    evaluation_strategy="epoch",       # Evaluate at the end of each epoch
    save_strategy="epoch",             # Save model checkpoint at the end of each epoch
    logging_strategy="steps",
    logging_steps=100,                 # Log training loss every 100 steps
    learning_rate=5e-5,                # Lower learning rate for fine-tuning
    per_device_train_batch_size=8,     # Adjust based on GPU memory
    per_device_eval_batch_size=8,      # Adjust based on GPU memory
    num_train_epochs=1,                # Start with a few epochs for fine-tuning
    weight_decay=0.01,                 # Regularization
    predict_with_generate=True,        # Needed for Seq2Seq evaluation
    fp16=False,                        # Set to True if GPU supports mixed precision
    # Important: load the best model at the end based on metric
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu", # Metric to monitor for best model
    greater_is_better=True,            # Higher BLEU is better
    report_to="none",                  # Disable reporting to WandB/TensorBoard if not needed
    # generation_max_length=max_target_length # Optional: Control max generation length during evaluation
)
print(f"\nTraining arguments configured. Output directory: {output_directory}")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1744209339461, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="0629f4c5" outputId="b2e1ae30-bc50-4742-d370-6e950a77dbb3"
if model and train_ds and val_ds and tokenizer and data_collator:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Seq2SeqTrainer initialized.")
else:
    trainer = None
    print("\nCannot initialize Trainer due to missing components (model, data, tokenizer, or collator).")

# %% [markdown] id="8fb4aad6"
# ## 5. Train (Fine-tune) the Model

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 209} id="213b08b1" outputId="186b927e-a2ac-42c0-8162-30a62ccafd1a"
if trainer:
    print("\nStarting model fine-tuning...")
    try:
        train_result = trainer.train()
        print("Fine-tuning finished.")

        # Save final metrics and state
        trainer.save_model() # Saves the tokenizer too
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(f"Best model saved to {trainer.state.best_model_checkpoint}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
else:
    print("\nSkipping training because the Trainer could not be initialized.")

# %% [markdown] id="92e595ac"
# ## 6. Evaluate the Fine-tuned Model

# %% colab={"background_save": true} id="301e68f3"
# if trainer:
#     print("\nEvaluating the fine-tuned model on the validation set...")
#     try:
#         eval_results = trainer.evaluate()
#         print("Evaluation finished.")
#         trainer.log_metrics("eval", eval_results)
#         trainer.save_metrics("eval", eval_results)

#         print("\nEvaluation Results:")
#         # Display results nicely
#         eval_results_df = pd.DataFrame([eval_results])
#         display(eval_results_df)

#     except Exception as e:
#         print(f"An error occurred during evaluation: {e}")
# else:
#     print("\nSkipping evaluation because the Trainer was not initialized or training failed.")

# %% id="inWyTfAT4wjf"
# prompt: For every sentence in the validation dataset, generate a prediction. This will be used for evaluating the bleu score.

if trainer and val_ds and tokenizer:
    print("\nGenerating predictions for the validation set...")
    predictions = trainer.predict(val_ds)
    print("Predictions generated.")

    # Process and display predictions (example)
    predicted_ids = predictions.predictions
    predicted_sentences = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    # Create a DataFrame for better visualization
    val_df = pd.DataFrame(val_ds)
    val_df["predicted"] = predicted_sentences
    display(val_df[["informal", "formal", "predicted"]])

    # Now you can use 'predicted_sentences' and the corresponding 'formal' sentences
    # from 'val_ds' for BLEU score calculation.
else:
    print("\nSkipping prediction generation due to missing components or previous errors.")


# %% id="777Q4RHI7Ia1"
val_df.to_csv("val_df-t5-pretrained.csv")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 509, "status": "ok", "timestamp": 1744213998515, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="2E8KLFjl8dTt" outputId="824dc22c-6c47-4192-9bfd-da6bf17ebccf"
# prompt: Evaluate bleu from the val_df

metric = evaluate.load("sacrebleu")

# Assuming 'val_df' is already created and contains columns 'formal' and 'predicted'
references = [[text] for text in val_df["formal"]]  # BLEU expects a list of lists
predictions = val_df["predicted"]

results = metric.compute(predictions=predictions, references=references)
results


# %% [markdown] id="9c1fd2f0"
# ## 7. Perform Simple Inference

# %% id="52a56fcc"
# Ensure we are using the fine-tuned model loaded by the trainer
# If `load_best_model_at_end=True`, `trainer.model` should be the best one.
# Alternatively, explicitly load from the saved checkpoint.

# Let's load explicitly from the best checkpoint if available for clarity
if trainer and trainer.state.best_model_checkpoint:
    best_checkpoint_path = trainer.state.best_model_checkpoint
    print(f"\nLoading best model from checkpoint for inference: {best_checkpoint_path}")
    try:
        inference_model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint_path)
        inference_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
        # If using GPU, move model to device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # inference_model.to(device)
        print("Inference model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading best model checkpoint: {e}")
        inference_model = None
        inference_tokenizer = None
elif trainer: # Fallback to the model currently in the trainer object
    print("\nUsing model currently loaded in the Trainer for inference.")
    inference_model = trainer.model
    inference_tokenizer = trainer.tokenizer
else:
    print("\nCannot perform inference, model not available.")
    inference_model = None
    inference_tokenizer = None


def formalize_text(sentence: str):
    """Uses the fine-tuned model to convert informal text to formal."""
    if not inference_model or not inference_tokenizer:
        return "Error: Inference model or tokenizer not available."

    # Prepare input
    input_text = prefix + sentence
    inputs = inference_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Move inputs to the same device as the model if using GPU
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    try:
        outputs = inference_model.generate(
            **inputs,
            max_length=max_target_length, # Set max generation length
            num_beams=4,                  # Beam search for potentially better results
            early_stopping=True
        )
        # Decode the generated tokens
        decoded_output = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output.strip()
    except Exception as e:
        return f"Error during generation: {e}"


# %% id="b5f872f7"
if inference_model and inference_tokenizer:
    print("\nTesting inference...")
    test_sentences = [
        "min , mau nanya . cara mengaktifkan no indosat yang sudah terblokir karena lupa isi pulsa , gmn yah ?",
        "maling yg jadi provider y begini dech jadinya",
        "kmrn aq liat promo baru, tp lupa dmn",
        "gimana caranya biar bisa dpt diskon gede?",
        "thanks ya infonya sangat membantu"
    ]

    for sentence in test_sentences:
        formalized = formalize_text(sentence)
        print("-" * 30)
        print(f"Informal: {sentence}")
        print(f"Formal:   {formalized}")
else:
    print("\nSkipping inference test.")
