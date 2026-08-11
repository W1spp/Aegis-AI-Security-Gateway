import pandas as pd
import string
import math
from collections import Counter
from datasets import load_dataset

# 1. Load the raw dataset
dataset = load_dataset("deepset/prompt-injections")
df = dataset['train'].to_pandas()

# --- PREPROCESSING & FEATURE ENGINEERING ---

# Feature 1: Word Count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Feature 2: Total Prompt Length (Character count)
df['prompt_length'] = df['text'].apply(lambda x: len(str(x)))

# Feature 3: Special Character Ratio (Hackers use lots of { } < > [ ] * \)
def special_char_ratio(text):
    text = str(text)
    if len(text) == 0: return 0
    
    # Count characters that are punctuation/special symbols
    special_chars = [char for char in text if char in string.punctuation]
    return len(special_chars) / len(text)

df['special_char_ratio'] = df['text'].apply(special_char_ratio)

# Feature 4: Shannon Entropy (How mathematically "random" the text is)
# Normal sentences have low entropy; obfuscated hacker code has high entropy.
def calculate_entropy(text):
    text = str(text)
    if len(text) == 0: return 0
    
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

df['entropy'] = df['text'].apply(calculate_entropy)

# --- THE RESULT ---
# Drop the original text column so only the math remains for Scikit-Learn
X = df[['word_count', 'prompt_length', 'special_char_ratio', 'entropy']]
y = df['label']

print(X.head())