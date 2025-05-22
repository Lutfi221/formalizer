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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25865, "status": "ok", "timestamp": 1742799718452, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="YBhHP06MNa1H" outputId="d107610e-ec24-450d-8ee1-0dce810d0996"
# !pip install transformers evaluate sacrebleu datasets tokenizers --quiet

# %% id="eNV0CH6YNnnx"
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
    T5Config,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


# %% [markdown] id="7HY7Unf2OtCO"
# # Download and Load Data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 950, "status": "ok", "timestamp": 1742799777527, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="crINR2nMOTHX" outputId="6198ac4a-8947-4e56-864f-f55c6b044dd2"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1742799777537, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="7Cx_y4YdAU63" outputId="9ea7f7f0-8481-44fb-f06d-abdffa6352d9"
print(train_ds.shape, val_ds.shape)

# %% [markdown] id="xu1JFyhBOp18"
# # Train a Custom Tokenizer

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 502, "status": "ok", "timestamp": 1742799778040, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="cu--aTLnOVyR" outputId="5e1f4fd5-ae5b-4dc7-c05a-7d5998eb204b"

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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 112, "status": "ok", "timestamp": 1742799778154, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="mai4zU0pO24V" outputId="5e01434d-a171-43c0-c3cb-a2fb3850ca56"
vocab = tokenizer.get_vocab()
print(json.dumps(vocab, indent=2))

# %% [markdown] id="QyV5gogDPF1l"
# # Wrap Tokenizer with PreTrainedTokenizerFast

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1742799778176, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="wMO-VnVXPHm4" outputId="b7b7fc90-2c26-4675-87bc-d904308e2d05"
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
# # Create a T5 Model Config and Initialize

# %% id="_KdWbheAPNt6"
# Let's define a T5 config from scratch.
# For real usage, you might use t5-small or a different pretrained checkpoint
# and then just resize token embeddings to match your custom tokenizer's vocab.
config = T5Config(
    vocab_size=wrapped_tokenizer.vocab_size,
    d_model=128,            # dimension of embeddings
    d_ff=256,               # dimension of feed-forward
    num_layers=2,           # number of encoder/decoder layers
    num_heads=4,            # number of attention heads
    dropout_rate=0.1,
    pad_token_id=wrapped_tokenizer.pad_token_id,
    decoder_start_token_id=wrapped_tokenizer.pad_token_id
)

model = T5ForConditionalGeneration(config)


# If your tokenizer does not have a dedicated pad_token, you must tell model
# what token to use as the pad token.
model.config.pad_token_id = wrapped_tokenizer.pad_token_id


# %% [markdown] id="mXlHcKoCPYSy"
# # Preprocess Data (Tokenize / Encode)

# %% colab={"base_uri": "https://localhost:8080/", "height": 138, "referenced_widgets": ["08de80b6956344a594e3ab0328203900", "2965b3ec75124ba1937a9e85fc5afaad", "e61c4aec6206453cb5a11efdad329022", "c7e3b76f85a74cd1a12acde4798f7d55", "25f728f0e2a441e280391ea43041f742", "75dd6fee3f544cde95a903da534af53f", "fa1f21a5b68e464db3ec17e6c6f45e3d", "c23491063754436c949a4ad87becb338", "53d5b54e03a4434a923e58c359901e2e", "1bbbf629a21d4ed3aefb6e713e7926ec", "857cbc5e652f457b95e36d638274c78f", "d22fd4a1cbc84ec4b91c7f700043e54e", "a74e4dd282464ce4892953d4b8c6297c", "5c6d3b6743bf4c18b16e233f4ecce21c", "dd907aaf31f14688bbc2b6ac7d9d8d34", "7b51e64a81a948f49b9924c97acedfdf", "6ef467baf4a84705b962ef67cc0df94a", "736ae255228f4b568ed9e002e01c36b4", "8190a140756b496bba4d261851fc34e4", "579e6001ac0e4459b2f6a0a98aaa1d08", "80bda51f11714054b25383e0ffe2356b", "2cf09d5436f941e78c0cc1b82f2d193f"]} executionInfo={"elapsed": 1240, "status": "ok", "timestamp": 1742799779509, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Fk5T97DXPxWB" outputId="4e4d5f59-eebb-4600-95eb-5065057cbc89"
def preprocess_function(examples):
    # We can prepend a prompt like "formalize: " for T5-like tasks
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

# %% id="jZ9rW6XgP4Rs"
data_collator = DataCollatorForSeq2Seq(
    tokenizer=wrapped_tokenizer,
    model=model,
    padding=True
)

# %% [markdown] id="MRnwj55_P6b5"
# # Define Training Arguments and Trainer

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 61, "status": "ok", "timestamp": 1742799779698, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="3XwJe3vkP78O" outputId="071d3d95-e974-4856-c472-1e182d2934d0"
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-formalizer-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=20,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    predict_with_generate=True,
    # This ensures we load the best model weights (based on eval metrics) at the end
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

# %% [markdown] id="pMQobuJiP-zZ"
# # Train the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 816} executionInfo={"elapsed": 808793, "status": "ok", "timestamp": 1742800588492, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="cZ3QSSS-QCBd" outputId="d58c4523-ad63-4a31-b1cb-04c123aeb58e"
trainer.train()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 19814, "status": "ok", "timestamp": 1742800608308, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="M0foaJZIAuVM" outputId="168aba8c-dd85-4a98-9592-5e02686729ce"
# prompt: Export the trainer and download the model

# Export the trainer and download the model
trainer.save_model("./t5-formalizer-model")

from google.colab import files
# !zip -r /content/t5-formalizer-model.zip /content/t5-formalizer-model

# files.download("/content/t5-formalizer-model.zip")


# %% [markdown] id="ekaS7-WSQKR2"
# # Evaluate the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 409, "referenced_widgets": ["a0c66736d2ef43298e43dcfa73590501", "14882137951e45fe9ce4e52d6b1a7bc5", "eb0912eaf1954619a8654c1665255835", "f9a37450c7aa4470a683beb9a529d3b1", "de34d34fedcf409f937ceb6006ca7720", "15bd975548aa44059aff31817a1b85ab", "476521edb5304078b774f056a462a7d7", "07907c3466af44edb6e585b9e6d8377a", "9eb8ccb395c7448f986e82f4b8b6e081", "ff006a251d3949c1a01e078d8beadf42", "d8b290e396174a21aae1cdf46e3d35c2"]} executionInfo={"elapsed": 4479, "status": "ok", "timestamp": 1742800612789, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="1ICE-d8aQLH_" outputId="4d77777d-9107-4781-ad38-763122118a15"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"elapsed": 3422, "status": "ok", "timestamp": 1742800616213, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="lBYX26EzTZmp" outputId="dc4f2d49-41d4-4dfe-b7e4-b650a9c9c0e1"
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 55, "status": "ok", "timestamp": 1742803017994, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="oIB2P0L6-1tD" outputId="ea5f39f8-1ed5-4d8b-b46c-0b322a2e81ed"
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
# sentence = "min , mau nanya . cara mengaktifkan no indosat yang sudah terblokir karena lupa isi pulsa , gmn yah ?"
sentence = "maling yg jadi provider y begini dech jadinya"
formalized_sentence = inference(sentence)
print(f"Informal: {sentence}")
print(f"Formal: {formalized_sentence}")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27292, "status": "ok", "timestamp": 1742800739971, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Bw0CeQpbBEVk" outputId="25d6014f-00f3-4c47-9600-cba8d3e0ab35"
from google.colab import drive
drive.mount('/content/drive')

# %% id="tSnxsHMNDczM"
# # !mv 't5-formalizer-model.20250324 141805.zip' '/content/drive/MyDrive/Shares/Bengkod U22 Bahagia/ACADEMIC CHATBOT/Lunashimu Pembakuan Teks/Models'
