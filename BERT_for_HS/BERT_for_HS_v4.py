# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
import re
from symspellpy import SymSpell
from datasets import Dataset

# Load the tokenizer and model
TOKENIZER_PATH = ''
MODEL_PATH = ''

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Initialize spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# Initialize Named Entity Recognition pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected_text = suggestions[0].term if suggestions else text
    entities = ner_pipeline(corrected_text)
    for entity in entities:
        if entity['entity'] == 'B-ORG':
            corrected_text = corrected_text.replace(entity['word'], "BRAND")
    return corrected_text

# Load and preprocess data
df = pd.read_csv("combined_data.csv")
df['HS_2digit'] = df['HS CODE'].astype(str).str[:2]
valid_hs_codes = df['HS_2digit'].value_counts()[df['HS_2digit'].value_counts() >= 250].index
df = df[df['HS_2digit'].isin(valid_hs_codes)].dropna(subset=['PRODUCT DESCRIPTION'])
df['Structured Description'] = df['PRODUCT DESCRIPTION'].astype(str).apply(normalize_text)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Structured Description'], df['HS_2digit'], test_size=0.10, random_state=42, stratify=df['HS_2digit']
)

# Encode labels
label_encoder = LabelEncoder()
train_labels = torch.tensor(label_encoder.fit_transform(train_labels))
val_labels = torch.tensor(label_encoder.transform(val_labels))

def preprocess(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=64, return_tensors="pt")

# Tokenize text
def create_dataset(texts, labels):
    encodings = preprocess(texts)
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

train_dataset = create_dataset(train_texts, train_labels)
val_dataset = create_dataset(val_texts, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(valid_hs_codes), local_files_only=True, ignore_mismatched_sizes=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=500,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
SAVE_PATH = ""
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# Load and preprocess test data
df_sample = pd.read_excel("indiaHS2.xlsx")
# df_sample = df_class.sample(n=5000, random_state=42)
df_sample['HS_2digit'] = df_sample['HS CODE'].astype(str).str[:2]
df_sample = df_sample[df_sample['HS_2digit'].isin(label_encoder.classes_)]

test_texts = df_sample['PRODUCT DESCRIPTION'].tolist()
test_labels = torch.tensor(label_encoder.transform(df_sample['HS_2digit']))
test_dataset = create_dataset(test_texts, test_labels)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

# Evaluate model
trainer = Trainer(model=model, compute_metrics=compute_metrics)
results = trainer.evaluate(test_dataset)
print("Test Set Evaluation Results:", results)

# Predict and save results
predictions = trainer.predict(test_dataset).predictions.argmax(axis=-1)
predicted_labels = label_encoder.inverse_transform(predictions)
actual_labels = label_encoder.inverse_transform(test_labels.numpy())
df_sample['Predicted Category'] = predicted_labels
df_sample['Actual Category'] = actual_labels

df_sample.to_csv("predicted_results.csv", index=False)

print(df_sample[['PRODUCT DESCRIPTION', 'Actual Category', 'Predicted Category']].head(20))
