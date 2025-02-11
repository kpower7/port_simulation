from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import re
from sklearn.preprocessing import LabelEncoder

# Function to normalize text (same as in the original script)
def normalize_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# Load the trained model and tokenizer
model_path = "/home/gridsan/kpower/BERT_for_HS/trained_models_hs"

tokenizer = BertTokenizer.from_pretrained('/home/gridsan/kpower/BERT_for_HS/BERT')
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the label encoder from the original training data
# We need to recreate the label encoder with the same classes as during training
df_original = pd.read_csv("/home/gridsan/kpower/BERT_for_HS/combined_data_2.csv")
df_original['HS_2digit'] = df_original['HS CODE'].astype(str).str[:2]
valid_hs_codes = df_original['HS_2digit'].value_counts()[df_original['HS_2digit'].value_counts() >= 250].index
df_original = df_original[df_original['HS_2digit'].isin(valid_hs_codes)]

label_encoder = LabelEncoder()
label_encoder.fit(df_original['HS_2digit'])

# Load the new data
new_data_path = '/home/gridsan/kpower/BERT_for_HS/december_final_merged_clean.csv'
new_data = pd.read_csv(new_data_path)

# Preprocess the 'productDescription' column
new_data['Structured Description'] = new_data['productDescription'].apply(normalize_text)

# Tokenize the new data
new_encodings = tokenizer(
    new_data['Structured Description'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Prepare new data in dictionary format
new_data_dict = {
    'input_ids': new_encodings['input_ids'],
    'attention_mask': new_encodings['attention_mask']
}

# Convert to Dataset format
new_dataset = Dataset.from_dict(new_data_dict)

# Create evaluation-specific training arguments
eval_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=32,
    eval_strategy="no",
)

# Initialize evaluation trainer
trainer = Trainer(
    model=model,
    args=eval_args,
)

# Predict HS codes for the new data
new_predictions = trainer.predict(new_dataset)
new_predicted_labels = new_predictions.predictions.argmax(axis=-1)
new_predicted_labels_named = label_encoder.inverse_transform(new_predicted_labels)

# Add predictions to the DataFrame
new_data['Predicted HS Code'] = new_predicted_labels_named

# Save the DataFrame with predictions to a CSV file
new_data.to_csv('/home/gridsan/kpower/BERT_for_HS/december_final_predictions.csv', index=False)

# Print the first few rows of the predictions
print(new_data[['productDescription', 'Predicted HS Code']].head())
print(f"Predictions completed for {len(new_data)} items and saved to december_final_predictions.csv")