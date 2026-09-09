import os
import string
import math
import requests
from collections import Counter
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack

app = Flask(__name__)

# --- 1. LOAD TRAINED MODELS & CONFIG ---
print("📥 Loading trained model files...")
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
firewall_model = joblib.load('firewall_model.pkl')
tfidf = joblib.load('tfidf.pkl')

HF_TOKEN = os.environ.get("HF_TOKEN")

# --- 2. FEATURE EXTRACTION FUNCTIONS ---
def special_char_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    special_chars = [char for char in text if char in string.punctuation]
    return len(special_chars) / len(text)

def calculate_entropy(text):
    text = str(text)
    if len(text) == 0:
        return 0
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in probabilities)

# --- 3. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json or {}
    user_prompt = data.get('prompt', '')
    
    if not user_prompt.strip():
        return jsonify({"status": "error", "message": "Empty prompt provided."}), 400

    # A. Calculate structural features
    word_cnt = len(user_prompt.split())
    prompt_len = len(user_prompt)
    spec_ratio = special_char_ratio(user_prompt)
    ent_score = calculate_entropy(user_prompt)
    
    features_df = pd.DataFrame([{
        'word_count': word_cnt,
        'prompt_length': prompt_len,
        'special_char_ratio': spec_ratio,
        'entropy': ent_score
    }])
    
    # B. Run through Scikit-Learn Pipeline
    scaled_features = scaler.transform(features_df)
    cluster_id = int(kmeans.predict(scaled_features)[0])
    text_features = tfidf.transform([user_prompt])
    final_features = hstack([scaled_features, [[cluster_id]], text_features])
    
    # C. Predict Threat Probability (Class 1 = Malicious)
    threat_prob = float(firewall_model.predict_proba(final_features)[0][1])
    
    # Deterministic Whitelist for common greetings & safe one-liners
    safe_greetings = ["hi", "hello", "hey", "hlo", "he", "test", "yo", "sup", "howdy", "good morning", "yes"]
    if user_prompt.strip().lower() in safe_greetings:
        threat_prob = 0.01  # Force 1.0% safe score
        
    threat_percentage = round(threat_prob * 100, 1)

    telemetry = {
        "word_count": word_cnt,
        "prompt_length": prompt_len,
        "special_char_ratio": round(spec_ratio, 3),
        "entropy": round(ent_score, 2),
        "cluster_id": f"Archetype_{cluster_id}",
        "threat_percentage": threat_percentage
    }

    # D. Decision Gate
    if threat_prob > 0.40:
        return jsonify({
            "status": "blocked",
            "telemetry": telemetry,
            "message": "⚠️ SECURITY ALERT: Malicious Prompt Injection Pattern Detected!"
        })
    else:
        # SAFE PROMPT: Route to native free HF inference endpoint
        try:
            api_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful and concise AI assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                ai_answer = result["choices"][0]["message"]["content"].strip()
            elif "error" in result:
                ai_answer = f"Model notice: {result['error']}"
            else:
                ai_answer = str(result)

        except Exception as e:
            ai_answer = f"Gateway passed prompt, but upstream connection failed: {str(e)}"

        return jsonify({
            "status": "allowed",
            "telemetry": telemetry,
            "answer": ai_answer
        })

if __name__ == '__main__':
    print("🚀 Starting Firewall Web Server at http://127.0.0.1:5000")
    app.run(debug=True)