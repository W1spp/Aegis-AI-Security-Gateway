import pandas as pd
import numpy as np
import string
import math
import joblib
from collections import Counter
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. FEATURE EXTRACTION FUNCTIONS ---
def special_char_ratio(text):
    text = str(text)
    if len(text) == 0: return 0
    special_chars = [char for char in text if char in string.punctuation]
    return len(special_chars) / len(text)

def calculate_entropy(text):
    text = str(text)
    if len(text) == 0: return 0
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in probabilities)

def extract_features_from_df(df):
    features = pd.DataFrame()
    features['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    features['prompt_length'] = df['text'].apply(lambda x: len(str(x)))
    features['special_char_ratio'] = df['text'].apply(special_char_ratio)
    features['entropy'] = df['text'].apply(calculate_entropy)
    return features

# --- 2. LOAD & PREPROCESS DATA ---
print("📥 Loading dataset from Hugging Face...")
raw_dataset = load_dataset("deepset/prompt-injections")
df = raw_dataset['train'].to_pandas()

print("⚙️ Engineering mathematical features...")
X = extract_features_from_df(df)
y = df['label']

# Split into 80% Train and 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MULTI-ALGORITHM PIPELINE ---
print("🧠 Training StandardScaler & KMeans Clustering...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Algorithm 1: K-Means Clustering (Identifies structural archetypes)
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Append cluster ID as a 5th feature to the scaled data
X_train_enriched = np.column_stack((X_train_scaled, train_clusters))
X_test_enriched = np.column_stack((X_test_scaled, test_clusters))

# Algorithm 2: Random Forest Classifier (Makes final threat decision)
print("🌲 Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_enriched, y_train)

# --- 4. EVALUATE PERFORMANCE ---
y_pred = clf.predict(X_test_enriched)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Training Complete! Model Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# --- 5. SAVE MODEL FILES FOR FLASK ---
print("💾 Saving model files to disk...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans.pkl')
joblib.dump(clf, 'firewall_model.pkl')
print("🎉 Success! Saved 'scaler.pkl', 'kmeans.pkl', and 'firewall_model.pkl'.")