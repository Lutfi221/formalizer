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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 34353, "status": "ok", "timestamp": 1741763009773, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="wJGSRcvw9w3v" outputId="afb88122-918d-41e0-a577-f6d7ae10b581"
# %pip install googletrans==4.0.0-rc1
# %pip install datasets

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8904, "status": "ok", "timestamp": 1741763018685, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="StMIFHsw-WJM" outputId="2363382b-e9dd-4978-cf04-25e03e7af424"
# !pip install transformers evaluate sacrebleu datasets tokenizers --quiet

# %% executionInfo={"elapsed": 41177, "status": "ok", "timestamp": 1741763059870, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="iC4HA8xICrs0"
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

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1741763059885, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="CON8EtTE9jGL"
import os
from googletrans import Translator
# Menggunakan library googletrans -> wrapper (pembungkus) tidak resmi untuk Google Translate.
# Library ini mengakses layanan Google Translate secara gratis melalui web, tanpa memerlukan kunci API (API key).
from datasets import Dataset, DatasetDict


# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1741763059895, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="UMWDr1sg95qa"
# Fungsi untuk membaca file dev.inf
def read_inf_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1741763059906, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="vne5Lfv297rN"
# Fungsi untuk menerjemahkan kalimat informal ke formal menggunakan Google Translate
def translate_informal_to_formal(informal_sentences):
    translator = Translator()
    formal_sentences = []
    for sentence in informal_sentences:
        try:
            translated = translator.translate(sentence, src='id', dest='en').text
            translated = translator.translate(translated, src='en', dest='id').text
            formal_sentences.append(translated)
        except Exception as e:
            print(f"Error translating sentence: {sentence}. Error: {e}")
            formal_sentences.append(sentence)  # Jika gagal, gunakan kalimat asli
    return formal_sentences



# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1741763059918, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="_VEop7nW9-gU"
# Fungsi untuk membuat dataset Huggingface
def create_huggingface_dataset(informal_sentences, formal_sentences):
    data = [{'translation': {'for': formal, 'inf': inf}} for formal, inf in zip(formal_sentences, informal_sentences)]
    dataset = Dataset.from_dict({'translation': data})
    return dataset


# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1741763059928, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="NvWx6ain-AL1"
# Fungsi untuk split dataset menjadi train, test, dan validation
def split_dataset(dataset):
    dataset_dict = dataset.train_test_split(test_size=0.2)
    test_valid = dataset_dict['test'].train_test_split(test_size=0.5)
    return DatasetDict({
        'train': dataset_dict['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })


# %% executionInfo={"elapsed": 239, "status": "ok", "timestamp": 1741763060177, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Dq18Xg_F-xX6"
def get_lines(url: str) -> list[str]:
    text = request("GET", url).text
    return text.split("\n")

informal_lines = get_lines("https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/dev.inf")
formal_lines   = get_lines("https://raw.githubusercontent.com/haryoa/stif-indonesia/refs/heads/main/data/labelled/dev.for")

# Write informal_lines to '/content/dev.inf'
with open('/content/dev.inf', 'w', encoding='utf-8') as file:
    for line in informal_lines:
        file.write(line + '\n')

# %% executionInfo={"elapsed": 12588, "status": "ok", "timestamp": 1741763072772, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="VFoKNXxo-B1L"
# Path ke file dev.inf
file_path = '/content/dev.inf'  # Sesuaikan path jika file berada di lokasi lain

# Baca kalimat informal dari file
informal_sentences = read_inf_file(file_path)
informal_sentences = informal_sentences[:10]

# display(informal_sentences)

# Terjemahkan kalimat informal ke formal
formal_sentences = translate_informal_to_formal(informal_sentences)

# display(formal_sentences)

# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1741763072801, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="g-7pK2MQ-CiG"

# Buat dataset Huggingface
dataset = create_huggingface_dataset(informal_sentences, formal_sentences)

# display(dataset)

# Split dataset menjadi train, test, dan validation
dataset_dict = split_dataset(dataset)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1741763072835, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="p8WapI4YHJz7" outputId="c4701960-4654-4969-cdd9-1aa1a34d6264"
dataset_dict

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 46, "status": "ok", "timestamp": 1741763072884, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="EYXZHUqvEe3R" outputId="5d75c6c6-819c-4167-b4f4-912a6461fd48"
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers, trainers
from datasets import load_dataset

# Load dataset dari Hugging Face
# dataset = load_dataset("haryoaw/stif-indonesia")

# Inisialisasi tokenizer dengan model BPE
tokenizer = Tokenizer(models.BPE())

# Tambahkan normalizer untuk lowercasing dan NFKC normalization
tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])

# Gunakan pretokenizer WhitespaceSplit untuk pemisahan kata yang lebih optimal
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Gunakan decoder yang sesuai dengan model BPE
tokenizer.decoder = decoders.BPEDecoder()

# Ambil data untuk training tokenizer
texts = [sentence for sentence in (formal_sentences + informal_sentences)]

# Inisialisasi trainer untuk BPE
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Training tokenizer dengan dataset
tokenizer.train_from_iterator(texts, trainer)

# Simpan tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer berhasil dilatih dan disimpan!")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1741763072927, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="wUQGA_LyFNbx" outputId="203c8ffc-bae6-4005-ae4d-8d5929991cf6"
test_sentence = formal_sentences[0]
encoded = tokenizer.encode(test_sentence)
print("Tokens:", encoded.tokens)
print("Input IDs:", encoded.ids)

# Dekode kembali
decoded_text = tokenizer.decode(encoded.ids)
print("Decoded Text:", decoded_text)

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1741763145405, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="nnEV20OoG46M"
# split_ds["train"]

# %% executionInfo={"elapsed": 59, "status": "ok", "timestamp": 1741763145495, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="5vaLlfnQJl1S"
dataset = Dataset.from_dict({"informal": informal_sentences, "formal": formal_sentences})

split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split_ds["train"], split_ds["test"]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1741763145534, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="zUgeHxGBHgwT" outputId="2477feab-ccab-4532-8548-78d79583e43e"
# Get [PAD] token id
tokenizer.token_to_id("[PAD]")

# %% executionInfo={"elapsed": 335, "status": "ok", "timestamp": 1741763145868, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="Phl6P2FqFm3U"
# Let's define a T5 config from scratch.
# For real usage, you might use t5-small or a different pretrained checkpoint
# and then just resize token embeddings to match your custom tokenizer's vocab.
config = T5Config(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=128,            # dimension of embeddings
    d_ff=256,               # dimension of feed-forward
    num_layers=2,           # number of encoder/decoder layers
    num_heads=4,            # number of attention heads
    dropout_rate=0.1,
    pad_token_id=tokenizer.token_to_id("[PAD]"),
    decoder_start_token_id=tokenizer.token_to_id("[PAD]")
)

model = T5ForConditionalGeneration(config)


# If your tokenizer does not have a dedicated pad_token, you must tell model
# what token to use as the pad token.
model.config.pad_token_id = tokenizer.token_to_id("[PAD]")


# %% [markdown] id="kWgkUAHAIVOL"
#

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1741763145875, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="A-c5TVXwGXmE"
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1741763145888, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="lDEYCoB6KMbY" outputId="e0c7e8a4-2941-468c-8066-dd148aedcd4d"
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

# %% executionInfo={"elapsed": 151, "status": "ok", "timestamp": 1741763146037, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="OZyykOg6KQ4z"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 170, "referenced_widgets": ["b5590e48946940dab69cf7449499f333", "e3976190a0e848ca8807ddd59c341aee", "9aaeb361043f46708f5665deeb2caf03", "2d56860b061348ecb8f397800e255f6e", "89adae821b8d41ef908462e16194ab65", "94d2d7f6f1ec4f128afdbc69673f07b3", "a0c710dd2e8b48989a572b956c253f05", "d9d9551a22c04657bdb0984c885db3ab", "cdaa56f524d547f09549332a57dfccdf", "642f2ae38ec94310868015b3bbf056a7", "09e98b9a8eb24dc29beee21ade16cbe6", "a1f49d7fc9364e98b374cd8ce7b8998a", "bbf8951e73664294b1380dc01a85dff3", "bf8995b079464cd1b859b3abe0956b10", "b238333fab3e41dd9fa59a37351379d3", "5265ad3020884ec8a49cf9949507da26", "b2b3e0d950434d6db4191a8ae6735da7", "d73c9b76d591497b9aed0b5ecd0d8063", "8c65c512e74545b1ad5d7b4811bfab45", "f464a9ca09d743e8a8a2e3e303a07de0", "42c8f4f3a93e425e9f9c177f2f9dcaa0", "28f1f068f8164d11a4c6a4d905e0527d"]} executionInfo={"elapsed": 63, "status": "ok", "timestamp": 1741763146097, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="EpVBu7BGKSs6" outputId="35d499f3-d347-49cf-adee-cd4f273b5c0a"
def preprocess_function(examples):
    # We can prepend a prompt like "formalize: " for T5-like tasks
    print(examples)
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

# %% executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1741763146135, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="5W8ZkWQXLytm"
data_collator = DataCollatorForSeq2Seq(
    tokenizer=wrapped_tokenizer,
    model=model,
    padding=True
)

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1741763146147, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="M68Mf2IPMLOh"
import os
os.environ["WANDB_DISABLED"] = "true"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 694, "status": "ok", "timestamp": 1741763146838, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="KBf5LbIlL81U" outputId="e8bf1492-e151-4559-e0a1-06685a5d648a"
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-formalizer-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    predict_with_generate=True,
    # This ensures we load the best model weights (based on eval metrics) at the end
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 415} executionInfo={"elapsed": 2987, "status": "ok", "timestamp": 1741763149828, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="nyxbWuvRL-h_" outputId="abe0318d-6154-4ed0-cb2a-9ac9a22c4120"
trainer.train()

# %% colab={"base_uri": "https://localhost:8080/", "height": 399, "referenced_widgets": ["2e55f880241f45f9a8422a1432168916", "622c402637774d91a009aba7463d7ab7", "c29748b2a677430cabf57789381c8c77", "5739001ace0040bf8cc6c5967d5c33cf", "a994d0031b8f420d9ff0515c4068dbf2", "3d38ffbb5ac0417ca7ed518db27d83c5", "e18050d6aa3f4de192236b9dcf63ee1c", "6e17e2bdedb14597be57bfc828b20928", "4198b3955e564bacb7a5678bcb87abcc", "ca581155e5cb48d7b2179f774ec44aee", "8daa3d51c89442b884cda28b8a6defd1"]} executionInfo={"elapsed": 1174, "status": "ok", "timestamp": 1741763151004, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="_0kyaj2LMAFh" outputId="9bae429b-12c0-48b9-d9aa-918c3396a9cc"
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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1741763151052, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="040nGQv8PF6q" outputId="0f4a335b-1a25-4a0b-c211-464bb5c52296"
wrapped_tokenizer.convert_tokens_to_ids(wrapped_tokenizer.tokenize("Hallo"))

# %% colab={"base_uri": "https://localhost:8080/", "height": 408} executionInfo={"elapsed": 52, "status": "error", "timestamp": 1741763151106, "user": {"displayName": "LUTFI HAFIIZHUDIN", "userId": "17498998759297632849"}, "user_tz": -420} id="1Lx4pKaNMWt0" outputId="d5ace4d7-6fa0-4501-e801-73559284dd04"
# Try translating a sentence

new_sentence = "Gw suka bgt belajar tt artis"


# input_ids = wrapped_tokenizer.convert_tokens_to_ids(wrapped_tokenizer.tokenize(new_sentence))
input_ids = wrapped_tokenizer(
        [new_sentence],
        max_length=128,
        truncation=True
    )
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
