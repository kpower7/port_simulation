import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# File paths
train_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_train_IND.csv.gz'
test_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_test_IND.csv.gz'
valid_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_valid_IND.csv.gz'

# Function to load datasets
def load_data():
    print("Loading datasets...")
    dtypes = {'PRODUCT DESCRIPTION': 'str', 'PRODUCT DESCRIPTION_ASCII': 'str'}
    
    try:
        train_df = pd.read_csv(train_path, dtype=dtypes)
        test_df = pd.read_csv(test_path, dtype=dtypes)
        valid_df = pd.read_csv(valid_path, dtype=dtypes)
        print("Datasets loaded successfully!")
        
        # Extract 2-digit HS codes
        train_df['HS_2digit'] = train_df['HS CODE6'].astype(str).str[:2]
        test_df['HS_2digit'] = test_df['HS CODE6'].astype(str).str[:2]
        valid_df['HS_2digit'] = valid_df['HS CODE6'].astype(str).str[:2]
        
        return train_df, test_df, valid_df
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None

# Method 1: Frequency-based keyword extraction
def extract_keywords_by_frequency(df, hs_code, top_n=20):
    """Extract keywords for a specific HS code based on frequency compared to other codes"""
    # Filter for the target HS code
    target_df = df[df['HS_2digit'] == hs_code]
    other_df = df[df['HS_2digit'] != hs_code]
    
    if len(target_df) == 0:
        print(f"No data found for HS code {hs_code}")
        return None
    
    print(f"Analyzing {len(target_df)} descriptions for HS code {hs_code}")
    
    # Preprocess text
    def preprocess(text):
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        # Remove digits
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Apply preprocessing
    target_texts = target_df['PRODUCT DESCRIPTION'].apply(preprocess)
    other_texts = other_df['PRODUCT DESCRIPTION'].apply(preprocess)
    
    # Create a list of all words in target texts
    target_words = ' '.join(target_texts).split()
    other_words = ' '.join(other_texts).split()
    
    # Count word frequencies
    target_word_counts = Counter(target_words)
    other_word_counts = Counter(other_words)
    
    # Calculate total word counts
    total_target_words = sum(target_word_counts.values())
    total_other_words = sum(other_word_counts.values())
    
    # Calculate word frequencies
    target_word_freq = {word: count/total_target_words for word, count in target_word_counts.items()}
    other_word_freq = {word: count/total_other_words for word, count in other_word_counts.items()}
    
    # Calculate TF-IDF like score (frequency in target / frequency in other)
    word_scores = {}
    for word in target_word_freq:
        if word in other_word_freq and len(word) > 2:  # Only consider words longer than 2 characters
            word_scores[word] = target_word_freq[word] / (other_word_freq[word] + 0.0001)  # Add small constant to avoid division by zero
    
    # Sort words by score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N words
    return sorted_words[:top_n]

# Method 2: BERT attention-based keyword extraction
def extract_keywords_with_bert(df, hs_code, top_n=20):
    """Extract keywords for a specific HS code using BERT attention weights"""
    try:
        # Load pre-trained BERT model and tokenizer
        model_name = "bert-base-multilingual-uncased"  # Use the same model as in the notebook
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Filter for the target HS code
        target_df = df[df['HS_2digit'] == hs_code]
        
        if len(target_df) == 0:
            print(f"No data found for HS code {hs_code}")
            return None
        
        # Sample a subset of descriptions if there are too many
        if len(target_df) > 100:
            target_df = target_df.sample(100, random_state=42)
        
        # Preprocess and tokenize text
        descriptions = target_df['PRODUCT DESCRIPTION'].tolist()
        
        # Get attention scores for each token
        all_tokens = []
        all_scores = []
        
        for desc in descriptions:
            # Tokenize input
            inputs = tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
            
            # Get model's attention
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            
            # Get attention weights (average across all heads and layers)
            attentions = outputs.attentions  # This is a tuple of tensors
            
            # Average attention across all layers and heads
            avg_attention = torch.mean(torch.cat([att.mean(dim=1) for att in attentions]), dim=0)
            
            # Get token attention scores (average attention received by each token)
            token_scores = avg_attention.mean(dim=0).cpu().numpy()
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Store tokens and scores
            for token, score in zip(tokens, token_scores):
                if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                    all_tokens.append(token)
                    all_scores.append(score)
        
        # Create a dictionary of token scores
        token_scores = {}
        for token, score in zip(all_tokens, all_scores):
            if token in token_scores:
                token_scores[token] += score
            else:
                token_scores[token] = score
        
        # Normalize by frequency
        token_counts = Counter(all_tokens)
        for token in token_scores:
            token_scores[token] /= token_counts[token]
        
        # Sort tokens by score
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out short tokens and special characters
        filtered_tokens = [(token, score) for token, score in sorted_tokens 
                           if len(token) > 2 and re.match(r'^[a-zA-Z]+$', token)]
        
        # Return top N tokens
        return filtered_tokens[:top_n]
    
    except Exception as e:
        print(f"Error in BERT keyword extraction: {e}")
        return None

# Method 3: Extract common phrases/n-grams
def extract_common_phrases(df, hs_code, top_n=20, ngram_range=(2, 3)):
    """Extract common phrases (n-grams) for a specific HS code"""
    # Filter for the target HS code
    target_df = df[df['HS_2digit'] == hs_code]
    
    if len(target_df) == 0:
        print(f"No data found for HS code {hs_code}")
        return None
    
    # Preprocess text
    def preprocess(text):
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Apply preprocessing
    target_texts = target_df['PRODUCT DESCRIPTION'].apply(preprocess)
    
    # Use CountVectorizer to extract n-grams
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(target_texts)
    
    # Get feature names (n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum up n-gram counts across all documents
    ngram_counts = X.sum(axis=0).A1
    
    # Create a dictionary of n-gram counts
    ngram_dict = {feature_names[i]: ngram_counts[i] for i in range(len(feature_names))}
    
    # Sort n-grams by count
    sorted_ngrams = sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N n-grams
    return sorted_ngrams[:top_n]

# Main function to analyze keywords for a specific HS code
def analyze_hs_code_keywords(hs_code, top_n=20):
    # Load data
    train_df, test_df, valid_df = load_data()
    if train_df is None:
        return
    
    # Combine datasets for more comprehensive analysis
    combined_df = pd.concat([train_df, test_df, valid_df])
    
    print(f"\n===== KEYWORD ANALYSIS FOR HS CODE {hs_code} =====")
    
    # Get basic statistics for this HS code
    hs_count = combined_df[combined_df['HS_2digit'] == hs_code].shape[0]
    total_count = combined_df.shape[0]
    print(f"Number of products with HS code {hs_code}: {hs_count} ({hs_count/total_count*100:.2f}% of total)")
    
    # Method 1: Frequency-based keywords
    print("\n1. Frequency-based keywords:")
    freq_keywords = extract_keywords_by_frequency(combined_df, hs_code, top_n)
    if freq_keywords:
        for word, score in freq_keywords:
            print(f"{word}: {score:.4f}")
    
    # Method 2: BERT-based keywords (optional, as it requires BERT model)
    try:
        print("\n2. BERT attention-based keywords:")
        bert_keywords = extract_keywords_with_bert(combined_df, hs_code, top_n)
        if bert_keywords:
            for word, score in bert_keywords:
                print(f"{word}: {score:.4f}")
    except Exception as e:
        print(f"BERT keyword extraction skipped: {e}")
    
    # Method 3: Common phrases
    print("\n3. Common phrases:")
    common_phrases = extract_common_phrases(combined_df, hs_code, top_n)
    if common_phrases:
        for phrase, count in common_phrases:
            print(f"{phrase}: {count}")
    
    # Save results to file
    results = {
        'hs_code': hs_code,
        'count': hs_count,
        'percentage': hs_count/total_count*100,
        'freq_keywords': freq_keywords if freq_keywords else [],
        'common_phrases': common_phrases if common_phrases else []
    }
    
    # Try to include BERT keywords if available
    try:
        results['bert_keywords'] = bert_keywords if bert_keywords else []
    except:
        pass
    
    # Save to CSV
    output_file = f"hs_code_{hs_code}_keywords.csv"
    
    # Create DataFrames for each keyword type
    dfs = []
    
    if freq_keywords:
        freq_df = pd.DataFrame(freq_keywords, columns=['Keyword', 'Score'])
        freq_df['Method'] = 'Frequency-based'
        dfs.append(freq_df)
    
    if 'bert_keywords' in results and results['bert_keywords']:
        bert_df = pd.DataFrame(results['bert_keywords'], columns=['Keyword', 'Score'])
        bert_df['Method'] = 'BERT-based'
        dfs.append(bert_df)
    
    if common_phrases:
        phrase_df = pd.DataFrame(common_phrases, columns=['Phrase', 'Count'])
        phrase_df['Method'] = 'Common Phrases'
        dfs.append(phrase_df)
    
    # Combine and save
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze keywords for specific HS codes')
    parser.add_argument('hs_code', type=str, help='2-digit HS code to analyze')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top keywords to extract')
    
    args = parser.parse_args()
    
    analyze_hs_code_keywords(args.hs_code, args.top_n)
