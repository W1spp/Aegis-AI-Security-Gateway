import os
import string
import math
from collections import Counter
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from scipy.sparse import hstack

app = Flask(__name__)

# --- 1. LOAD TRAINED MODELS & HF CLIENT ---
print("📥 Loading trained model files...")
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
firewall_model = joblib.load('firewall_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Your specific Hugging Face Token (Loaded securely from the server)
HF_TOKEN = os.environ.get("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

# --- 2. FEATURE EXTRACTION FUNCTIONS ---
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

# --- 3. ROUTES ---
@app.route('/')
def home():
    # Serves the new landing page
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Serves the main firewall application
    return render_template('dashboard.html')

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json or {}
    user_prompt = data.get('prompt', '')
    
    if not user_prompt.strip():
        return jsonify({"status": "error", "message": "Empty prompt provided."}), 400

    # A. Calculate real-time structural features from the incoming string
    word_cnt = len(user_prompt.split())
    prompt_len = len(user_prompt)
    spec_ratio = special_char_ratio(user_prompt)
    ent_score = calculate_entropy(user_prompt)
    
    # B. Format into DataFrame for Scikit-Learn
    features_df = pd.DataFrame([{
        'word_count': word_cnt,
        'prompt_length': prompt_len,
        'special_char_ratio': spec_ratio,
        'entropy': ent_score
    }])
    
    # C. Run through the Scikit-Learn Pipeline
    # 1. Scale math features and predict cluster
    scaled_features = scaler.transform(features_df)
    cluster_id = int(kmeans.predict(scaled_features)[0])
    
    # 2. Extract semantic text features using TF-IDF (Translates words to weights)
    text_features = tfidf.transform([user_prompt])
    
    # 3. Combine scaled features + cluster ID + text features using hstack
    final_features = hstack([scaled_features, [[cluster_id]], text_features])
    
    # 4. Predict threat probability (Class 1 = Malicious)
    threat_prob = float(firewall_model.predict_proba(final_features)[0][1])
        
    threat_percentage = round(threat_prob * 100, 1)

    # Telemetry metrics dictionary to return to frontend
    telemetry = {
        "word_count": word_cnt,
        "prompt_length": prompt_len,
        "special_char_ratio": round(spec_ratio, 3),
        "entropy": round(ent_score, 2),
        "cluster_id": f"Archetype_{cluster_id}",
        "threat_percentage": threat_percentage
    }

    # D. Decision Gate
    if threat_prob > 0.35:
        # THREAT DETECTED: Block and return metrics without calling HF
        return jsonify({
            "status": "blocked",
            "telemetry": telemetry,
            "message": "⚠️ SECURITY ALERT: Malicious Prompt Injection Pattern Detected!"
        })
    else:
        # SAFE PROMPT: Call Hugging Face API for real answer
        try:
            # google/gemma-2-2b-it is officially hosted directly on HF's free serverless tier
            messages = [{"role": "user", "content": user_prompt}]
            hf_response = hf_client.chat_completion(
                model="google/gemma-2-2b-it",
                messages=messages,
                max_tokens=250
            )
            ai_answer = hf_response.choices[0].message.content
        except Exception as e:
            try:
                # Fallback: simple text generation if chat router denies the pipeline
                fallback_resp = hf_client.text_generation(
                    user_prompt,
                    model="google/gemma-2-2b-it",
                    max_new_tokens=150
                )
                ai_answer = fallback_resp
            except Exception as inner_err:
                ai_answer = f"AI Service Notice: Gateway verified prompt as safe (Threat: {threat_percentage}%), but HF endpoint returned: {str(inner_err)}"
                
        return jsonify({
            "status": "allowed",
            "telemetry": telemetry,
            "answer": ai_answer
        })

if __name__ == '__main__':
    print("🚀 Starting Firewall Web Server at http://127.0.0.1:5000")
    app.run(debug=True)