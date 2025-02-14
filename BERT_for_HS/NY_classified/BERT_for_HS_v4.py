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

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import torch
import pandas as pd
import re

# Load the tokenizer from the uploaded directory
tokenizer = BertTokenizer.from_pretrained('/home/gridsan/kpower/BERT_for_HS/BERT')

def normalize_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def preprocess(texts):
    # Tokenize the texts
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load and preprocess the data
df = pd.read_csv(r"combined_data_2.csv")

# df2= pd.read_csv(r"combined_data.csv")

# df = pd.concat(df, " ", df2)

# Extract first 2 digits of HS codes and convert to string
df['HS_2digit'] = df['HS CODE'].astype(str).str[:2]

# Get HS codes with more than 250 occurrences
valid_hs_codes = df['HS_2digit'].value_counts()[df['HS_2digit'].value_counts() >= 250].index

# Filter DataFrame to keep only rows with valid HS codes
df = df[df['HS_2digit'].isin(valid_hs_codes)]

# Print new counts after filtering
print("\nHS Code (2-digit) counts after filtering (>= 250 occurrences):")
print(df['HS_2digit'].value_counts().sort_index())
print(f"\nNumber of HS codes remaining: {len(valid_hs_codes)}")

# Clean the data
df = df.dropna(subset=['PRODUCT DESCRIPTION'])  # Remove NaN values
df['PRODUCT DESCRIPTION'] = df['PRODUCT DESCRIPTION'].astype(str)  # Ensure all entries are strings
df = df[df['PRODUCT DESCRIPTION'].str.strip() != '']  # Remove empty strings

# Create a structured product description
df['Structured Description'] = df['PRODUCT DESCRIPTION'].apply(normalize_text)

# Split the data into train and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Structured Description'], 
    df['HS_2digit'], 
    test_size=0.10,
    random_state=42,
    stratify=df['HS_2digit']
)

# Convert both train_texts and val_texts to lists
train_texts = train_texts.tolist()
val_texts = val_texts.tolist()

print("Sample train_texts:", train_texts[:5])

# Convert labels to numeric values
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Convert labels to tensor
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# Preprocess the data
train_encodings = preprocess(train_texts)
val_encodings = preprocess(val_texts)

# Prepare data in dictionary format
train_data = {
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
}
val_data = {
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
}

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# Load the model from the local directory with specified number of labels
model = BertForSequenceClassification.from_pretrained(
    '/home/gridsan/kpower/BERT_for_HS/BERT', 
    num_labels=len(df['HS_2digit'].unique()),
    local_files_only=True,
    ignore_mismatched_sizes=True  # Add this parameter to handle the different number of classes
)

# Training arguments (removing duplicate and enhancing configuration)
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,  # Increased epochs for better learning
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Keep only the 2 best checkpoints
    logging_dir='./logs',
    logging_steps=100,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=500,
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
)

# Initialize Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
save_path = r"/home/gridsan/kpower/BERT_for_HS/trained_models_hs"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

df_class = pd.read_excel(r"indiaHS2.xlsx")

# Sample 5000 rows from df_class
# df_sample = df_class.sample(n=5000, random_state=42)
df_sample = df_class

# Extract first 2 digits of HS codes and convert to string
df_sample['HS_2digit'] = df_sample['HS CODE'].astype(str).str[:2]

# Step 1: Identify valid labels from the training set
valid_labels = set(label_encoder.classes_)

# Step 2: Filter the test set to only include valid labels
df_sample = df_sample[df_sample['HS_2digit'].isin(valid_labels)]

# Extract PRODUCT DESCRIPTION as test_texts
test_texts = df_sample['PRODUCT DESCRIPTION'].tolist()
test_labels = df_sample['HS_2digit'].tolist()

# Step 1: Check for NaN or non-numeric values
test_labels = [str(label) for label in test_labels if pd.notna(label)]

# Step 2: Convert test_labels to numeric format using the same LabelEncoder
encoded_labels = []
for label in test_labels:
    try:
        encoded_label = label_encoder.transform([label])[0]  # Transform each label individually
        encoded_labels.append(encoded_label)
    except ValueError as e:
        print(f"Skipping label {label} due to error: {e}")

test_labels = torch.tensor(encoded_labels)  # Convert encoded labels to tensor

# Step 1: Tokenize the test set
test_encodings = tokenizer(
    test_texts,  # List of test descriptions
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

# Prepare test data in dictionary format
test_data = {
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels  # Now in tensor format with numeric labels
}

# Convert to Dataset format
test_dataset = Dataset.from_dict(test_data)

# Create evaluation-specific training arguments
eval_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=32,
    eval_strategy="no",  # Changed from "steps" to "no" for evaluation
)

# Initialize evaluation trainer
trainer = Trainer(
    model=model,
    args=eval_args,  # Use the evaluation-specific arguments
    compute_metrics=compute_metrics
)

# Evaluate and generate predictions
results = trainer.evaluate(test_dataset)
predictions = trainer.predict(test_dataset)

# Process predictions and create output DataFrame
predicted_labels = predictions.predictions.argmax(axis=-1)
predicted_labels_named = label_encoder.inverse_transform(predicted_labels)
actual_labels_named = label_encoder.inverse_transform(test_labels.numpy())

df_sample['Predicted Category'] = predicted_labels_named
df_sample['Actual Category'] = actual_labels_named

# Print the evaluation results
print("Test Set Evaluation Results:", results)

# Step 3: Print the first few rows and save the DataFrame with predictions to a CSV
print(df_sample[['PRODUCT DESCRIPTION', 'Actual Category', 'Predicted Category']].head(20))

# Save the DataFrame with predictions to a CSV file for further analysis
df_sample.to_csv(r"/home/gridsan/kpower/BERT_for_HS/predicted_results_v4.csv", index=False)

df_sample.sample(20)
