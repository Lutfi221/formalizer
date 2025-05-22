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

# %% executionInfo={"elapsed": 3581, "status": "ok", "timestamp": 1742801011248, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="YBhHP06MNa1H"
# !pip install transformers evaluate sacrebleu datasets tokenizers --quiet

# %% executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1742801011262, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="eNV0CH6YNnnx"
import json
import evaluate
import numpy as np
import pandas as pd

from requests import request
from datasets import Dataset, DatasetDict

# Tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Transformers & Seq2Seq
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


# %% [markdown] id="7HY7Unf2OtCO"
# # Download and Load Data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1304, "status": "ok", "timestamp": 1742801012571, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="crINR2nMOTHX" outputId="f2faf2f3-dceb-419f-c036-309e6b097921"
def get_lines(url: str) -> list[str]:
    text = request("GET", url).text
    return text.split("\n")

# informal_lines = get_lines("https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/dev.inf")
# formal_lines   = get_lines("https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/dev.for")

informal_lines = []
formal_lines   = []

for name in ("dev", "test", "train"):
  for ext in ("inf", "for"):
    lines = get_lines(f"https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/{name}.{ext}")
    if ext == "inf":
      informal_lines.extend(lines)
    else:
      formal_lines.extend(lines)


# Create a Dataset from the lists
dataset = Dataset.from_dict({"informal": informal_lines, "formal": formal_lines})

# For demonstration, we’ll split into train and validation sets.
# In reality, you'd have separate data files for train/validation/test.
split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split_ds["train"], split_ds["test"]

print("Sample training example:")
print(train_ds[0])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1742801012587, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="7Cx_y4YdAU63" outputId="7a881328-9ce6-4665-f08d-56544df7f953"
print(train_ds.shape, val_ds.shape)

# %% [markdown] id="xu1JFyhBOp18"
# # Train a Custom Tokenizer

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 303, "status": "ok", "timestamp": 1742801012900, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="cu--aTLnOVyR" outputId="489b3344-28c7-4574-85ca-d49a59409945"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    special_tokens=[
        '[UNK]', 'xxxnumberxxx', 'xxxdatexxx', 'xxxtimexxx',
        'xxxphonexxx', 'xxxpercentxxx', 'xxxuserxxx'
    ],
    continuing_subword_prefix="##",
    min_frequency=20
)

tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer on both the informal and formal texts
tokenizer.train_from_iterator(train_ds["informal"], trainer)
tokenizer.train_from_iterator(train_ds["formal"], trainer)

# You now have a raw Tokenizer object. Let's check its vocab size:
print("\nTrained tokenizer vocab size:", tokenizer.get_vocab_size())

# %% [markdown] id="bJ49kwyFO5iR"
# ## Inspect some of the vocabulary

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 78, "status": "ok", "timestamp": 1742801012982, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="mai4zU0pO24V" outputId="7c3ba805-c7f3-48c3-e942-13cbef6e8d9a"
vocab = tokenizer.get_vocab()
print(json.dumps(vocab, indent=2))

# %% [markdown] id="QyV5gogDPF1l"
# # Wrap Tokenizer with PreTrainedTokenizerFast

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39, "status": "ok", "timestamp": 1742801013030, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="wMO-VnVXPHm4" outputId="187ef8a5-cd27-4ebe-b068-0f43d60503d2"
# We need a PreTrainedTokenizerFast wrapper to use in a Hugging Face model.
# We specify the same special tokens we used earlier:
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",   # or define a new [PAD] token if you prefer
    eos_token="</s>",
    bos_token="<s>",
    sep_token="</s>"
)

# Ensure that the model sees these tokens as special:
wrapped_tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "[UNK]",
    "pad_token": "[UNK]"  # or your dedicated pad token
})

# Double-check the final vocab size with the special tokens included:
print("Final tokenizer vocab size:", wrapped_tokenizer.vocab_size)

# %% [markdown] id="eFCJWLi3PMfU"
# # Create a bart Model Config and Initialize

# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1742801013045, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="_KdWbheAPNt6"
from transformers import BartConfig, BartForConditionalGeneration

# Define a Bart config from scratch
config = BartConfig(
    vocab_size=wrapped_tokenizer.vocab_size,
    d_model=128,            # dimension of embeddings
    encoder_ffn_dim=256,    # dimension of feed-forward in encoder
    decoder_ffn_dim=256,    # dimension of feed-forward in decoder
    encoder_layers=2,       # number of encoder layers
    decoder_layers=2,       # number of decoder layers
    encoder_attention_heads=4,  # number of attention heads in encoder
    decoder_attention_heads=4,  # number of attention heads in decoder
    dropout=0.1,
    pad_token_id=wrapped_tokenizer.pad_token_id,
    decoder_start_token_id=wrapped_tokenizer.pad_token_id
)

model = BartForConditionalGeneration(config)

# Ensure the pad token is set properly
model.config.pad_token_id = wrapped_tokenizer.pad_token_id


# %% [markdown] id="mXlHcKoCPYSy"
# # Preprocess Data (Tokenize / Encode)

# %% colab={"base_uri": "https://localhost:8080/", "height": 138, "referenced_widgets": ["c92ccd1ddacd4583a37d5671e1d4f4bb", "89e83df150994a9e851da015e1acf265", "d97f5f836fd649b9855897a0f7ff1d3e", "a520c1b047b34c179e21885c1aed07bd", "9076b55bcef64bc3a0c67a086119b2a2", "56637cd9068e45f2b21ee5a886a8aa80", "27e16cd6be454fc2a99eb160f3b719fc", "99cd2f2add3c47ebbf74833d763071e4", "8e3ca4b6648f4bd5b27c1e8d741c229f", "a39aab62c78348d3bd472d3cfb46aca8", "f3b93ce2396649dcb875ee7416c0bb7c", "e1a09794683c47d8b28d41b23b64400b", "30c3859bfdf441119998817498bb44c2", "29ff7947ab1f4f329713fb0e6f0a310a", "70901469283e49c18dd879510144b614", "e2144ed1aa194c1b87782356a434a7e3", "884e612c2bf548f197bb897cc43e0a83", "3c441c4e5f9c484aa81d8a05c9bee522", "cec599ef8d6541d8bf386e1b0daf0462", "8d958ca52b3c47ddb246c9dc88c0759c", "2ed98f3f05104783b477f4146264c2ea", "b170c9cf9ae54cb6bf30ef4f72466d33"]} executionInfo={"elapsed": 504, "status": "ok", "timestamp": 1742801013549, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Fk5T97DXPxWB" outputId="3d985556-c0ab-4891-8e67-d264379f7fbc"
def preprocess_function(examples):
    # We can prepend a prompt like "formalize: " for bart-like tasks
    inputs = ["formalize: " + txt for txt in examples["informal"]]
    targets = examples["formal"]

    # Model inputs
    model_inputs = wrapped_tokenizer(
        inputs,
        max_length=128,
        truncation=True
    )

    # Model targets
    with wrapped_tokenizer.as_target_tokenizer():
        labels = wrapped_tokenizer(
            targets,
            max_length=128,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing to train/validation
train_ds = train_ds.map(preprocess_function, batched=True)
val_ds   = val_ds.map(preprocess_function, batched=True)

# %% [markdown] id="GRRNOcY9P1LY"
# # Create a data collator

# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1742801013567, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="jZ9rW6XgP4Rs"
data_collator = DataCollatorForSeq2Seq(
    tokenizer=wrapped_tokenizer,
    model=model,
    padding=True
)

# %% [markdown] id="MRnwj55_P6b5"
# # Define Training Arguments and Trainer

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1742801013606, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="3XwJe3vkP78O" outputId="478830ee-79d4-42f4-91f5-540e73047d46"
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-formalizer-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=50,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    predict_with_generate=True,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,  # Use your BART model here
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

print(trainer.args.device)


# %% [markdown] id="pMQobuJiP-zZ"
# # Train the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 1487982, "status": "ok", "timestamp": 1742802501593, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="cZ3QSSS-QCBd" outputId="6f42e92e-4d35-4653-ae36-74cf48d968fb"
trainer.train()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 39157, "status": "ok", "timestamp": 1742802540753, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="M0foaJZIAuVM" outputId="5fb7ac7b-5235-4e4b-92b2-b84f8a9382ef"
# prompt: Export the trainer and download the model

# Export the trainer and download the model
trainer.save_model("./bart-formalizer-model")

from google.colab import files
# !zip -r /content/bart-formalizer-model.zip /content/bart-formalizer-model

# files.download("/content/bart-formalizer-model.zip")


# %% [markdown] id="ekaS7-WSQKR2"
# # Evaluate the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"elapsed": 2625, "status": "ok", "timestamp": 1742802543390, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="1ICE-d8aQLH_" outputId="5541f744-b747-40cd-d253-cd94149d1412"
# We can use a BLEU or SacreBLEU metric for measuring closeness
# to the reference 'formal' text.
metric = evaluate.load("sacrebleu")

# Create a small function for evaluation:
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Decode predictions
    decoded_preds = wrapped_tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Decode labels
    # Replace all -100 in the labels as the tokenizer might insert them as padding
    labels = np.where(labels != -100, labels, wrapped_tokenizer.pad_token_id)
    decoded_labels = wrapped_tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute sacrebleu
    result = metric.compute(predictions=decoded_preds, references=[[lbl] for lbl in decoded_labels])
    return {"bleu": result["score"]}

# Evaluate
eval_results = trainer.evaluate(metric_key_prefix="eval")
print(eval_results)
# {'eval_loss': 4.6298298835754395, 'eval_runtime': 0.8378, 'eval_samples_per_second': 51.323, 'eval_steps_per_second': 13.129, 'epoch': 8.0}

# Display eval_results as table with key and value as column names
eval_results_df = pd.DataFrame(eval_results.items(), columns=["key", "value"])
display(eval_results_df)

# %% [markdown] id="ZZJKyKV7Ta_Q"
# # Inference (Test the Model)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4949, "status": "ok", "timestamp": 1742803947352, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="YoNRkwGZNbZs" outputId="67672fac-6b72-46cd-b67d-4f8429830016"
# !pip install rouge-score

# %% colab={"base_uri": "https://localhost:8080/", "height": 470} executionInfo={"elapsed": 1922, "status": "error", "timestamp": 1742807063355, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="lBYX26EzTZmp" outputId="963f6c98-29fe-4032-a616-1bd62632e957"
# We can use a BLEU or SacreBLEU metric for measuring closeness
# to the reference 'formal' text.
metric = evaluate.load("sacrebleu")
rogue = evaluate.load("rogue")

# Create a small function for evaluation:
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Decode predictions
    decoded_preds = wrapped_tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Decode labels
    # Replace all -100 in the labels as the tokenizer might insert them as padding
    labels = np.where(labels != -100, labels, wrapped_tokenizer.pad_token_id)
    decoded_labels = wrapped_tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute sacrebleu
    result = metric.compute(predictions=decoded_preds, references=[[lbl] for lbl in decoded_labels])
    # Compute rogue
    result_rogue = rogue.compute(predictions=decoded_preds, references=[[lbl] for lbl in decoded_labels])
    return {"bleu": result["score"], "rogue": result["rogue"]}

# Evaluate
eval_results = trainer.evaluate(metric_key_prefix="eval")
print(eval_results)
# {'eval_loss': 4.6298298835754395, 'eval_runtime': 0.8378, 'eval_samples_per_second': 51.323, 'eval_steps_per_second': 13.129, 'epoch': 8.0}

# Display eval_results as table with key and value as column names
eval_results_df = pd.DataFrame(eval_results.items(), columns=["key", "value"])
display(eval_results_df)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 208, "status": "ok", "timestamp": 1742803240609, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="5kW9I_JaAU7j" outputId="4afc9c5a-4859-440b-d040-1742821b11c4"
# prompt: Test the inference using a sentence

# Assuming you have already trained and loaded the model and tokenizer
# as shown in the previous code

def inference(sentence):
  inputs = wrapped_tokenizer("formalize: " + sentence, return_tensors="pt")
  if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
  outputs = model.generate(**inputs)
  decoded_output = wrapped_tokenizer.decode(outputs[0], skip_special_tokens=True)
  return decoded_output

# Example usage
sentence = "min , mau nanya . cara mengaktifkan no indosat yang sudah terblokir karena lupa isi pulsa , gmn yah ?"
# sentence = "maling yg jadi provider y begini dech jadinya"
formalized_sentence = inference(sentence)
print(f"Informal: {sentence}")
print(f"Formal: {formalized_sentence}")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1742802546594, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="oIB2P0L6-1tD" outputId="174c2917-f31d-4985-b534-a73049f1ee60"
# prompt: Test the inference using a sentence

# Assuming you have already trained and loaded the model and tokenizer
# as shown in the previous code

def inference(sentence):
  inputs = wrapped_tokenizer("formalize: " + sentence, return_tensors="pt")
  if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
  outputs = model.generate(**inputs)
  decoded_output = wrapped_tokenizer.decode(outputs[0], skip_special_tokens=True)
  return decoded_output

def inference_encoded(sentence):
  inputs = wrapped_tokenizer("formalize: " + sentence, return_tensors="pt")
  if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
  outputs = model.generate(**inputs)
  return outputs

def decode_inference(model_outputs):
  decoded_output = wrapped_tokenizer.decode(model_outputs[0], skip_special_tokens=True)
  return decoded_output

# Example usage
# sentence = "min, mau nanya."

sentence = val_ds[0]["informal"]

formalized_sentence = inference(sentence)
print(f"Informal: {sentence}")
print(f"Formal: {formalized_sentence}")


# %% executionInfo={"elapsed": 57568, "status": "ok", "timestamp": 1742802604164, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="RAuZOz78WVjS"
# prompt: Evaluate the model using testing or validation data. Generate a list of predicted sentences. Use the list with sacrebleu. Use the inference function

predicted_sentences_encoded = []
for example in val_ds:
  decoded_output = inference(example["informal"])
  predicted_sentences_encoded.append(inference_encoded(example["informal"]))


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 188, "status": "ok", "timestamp": 1742802604350, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="u0ZyiUye19is" outputId="84897f60-7fb2-4852-cc35-05fde8416d40"

references = [example["formal"] for example in val_ds]
bleu = metric.compute(predictions=[decode_inference(p) for p in predicted_sentences_encoded], references=references)
bleu


# %% colab={"base_uri": "https://localhost:8080/", "height": 167} executionInfo={"elapsed": 51, "status": "error", "timestamp": 1742802604403, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Q1FXDvBzyhLT" outputId="4dd8d70a-cfa6-4a6f-f91e-9df9237c61a9"
for s in predicted_sentences:
  print(s)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 62800, "status": "ok", "timestamp": 1742807178373, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Jbo80QyEZh7p" outputId="8d87db7c-8d1d-4472-f632-6224d0729893"
from google.colab import drive
drive.mount('/content/drive')

# %% executionInfo={"elapsed": 6889, "status": "ok", "timestamp": 1742807188367, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="vJwrqMOfZjeH"
# !mv 'bart-formalizer-model.20250324 160554.zip' '/content/drive/MyDrive/Shares/Bengkod U22 Bahagia/ACADEMIC CHATBOT/Lunashimu Pembakuan Teks/Models'
