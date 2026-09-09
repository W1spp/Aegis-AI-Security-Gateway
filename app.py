import os
import string
import math
import requests
import pandas as pd
import joblib

from collections import Counter
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack


app = Flask(__name__)


# ============================================================
# 1. LOAD TRAINED MODELS & CONFIG
# ============================================================

print("📥 Loading trained model files...")

scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
firewall_model = joblib.load("firewall_model.pkl")
tfidf = joblib.load("tfidf.pkl")

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN environment variable is not set.")


# ============================================================
# 2. FEATURE EXTRACTION FUNCTIONS
# ============================================================

def special_char_ratio(text):
    text = str(text)

    if len(text) == 0:
        return 0

    special_chars = [
        char for char in text
        if char in string.punctuation
    ]

    return len(special_chars) / len(text)


def calculate_entropy(text):
    text = str(text)

    if len(text) == 0:
        return 0

    probabilities = [
        count / len(text)
        for count in Counter(text).values()
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
    )


# ============================================================
# 3. ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ============================================================
# 4. AI SECURITY GATEWAY
# ============================================================

@app.route("/ask", methods=["POST"])
def ask_ai():

    data = request.json or {}

    user_prompt = data.get("prompt", "")

    if not user_prompt.strip():
        return jsonify({
            "status": "error",
            "message": "Empty prompt provided."
        }), 400


    # --------------------------------------------------------
    # A. Calculate structural features
    # --------------------------------------------------------

    word_cnt = len(user_prompt.split())
    prompt_len = len(user_prompt)
    spec_ratio = special_char_ratio(user_prompt)
    ent_score = calculate_entropy(user_prompt)

    features_df = pd.DataFrame([{
        "word_count": word_cnt,
        "prompt_length": prompt_len,
        "special_char_ratio": spec_ratio,
        "entropy": ent_score
    }])


    # --------------------------------------------------------
    # B. Run through Scikit-Learn pipeline
    # --------------------------------------------------------

    scaled_features = scaler.transform(features_df)

    cluster_id = int(
        kmeans.predict(scaled_features)[0]
    )

    text_features = tfidf.transform([user_prompt])

    final_features = hstack([
        scaled_features,
        [[cluster_id]],
        text_features
    ])


    # --------------------------------------------------------
    # C. Predict threat probability
    # Class 1 = malicious
    # --------------------------------------------------------

    threat_prob = float(
        firewall_model.predict_proba(final_features)[0][1]
    )


    # --------------------------------------------------------
    # Deterministic whitelist for common safe prompts
    # --------------------------------------------------------

    safe_greetings = [
        "hi",
        "hello",
        "hey",
        "hlo",
        "he",
        "test",
        "yo",
        "sup",
        "howdy",
        "good morning",
        "yes"
    ]

    if user_prompt.strip().lower() in safe_greetings:
        threat_prob = 0.01


    threat_percentage = round(
        threat_prob * 100,
        1
    )


    # --------------------------------------------------------
    # Telemetry
    # --------------------------------------------------------

    telemetry = {
        "word_count": word_cnt,
        "prompt_length": prompt_len,
        "special_char_ratio": round(spec_ratio, 3),
        "entropy": round(ent_score, 2),
        "cluster_id": f"Archetype_{cluster_id}",
        "threat_percentage": threat_percentage
    }


    # --------------------------------------------------------
    # D. SECURITY DECISION GATE
    # --------------------------------------------------------

    if threat_prob > 0.40:

        return jsonify({
            "status": "blocked",
            "telemetry": telemetry,
            "message": (
                "⚠️ SECURITY ALERT: "
                "Malicious Prompt Injection Pattern Detected!"
            )
        })


    # ========================================================
    # E. SAFE PROMPT → HUGGING FACE
    # ========================================================

    try:

        if not HF_TOKEN:

            return jsonify({
                "status": "error",
                "telemetry": telemetry,
                "message": (
                    "HF_TOKEN is not configured on the server."
                )
            }), 500


        # Current Hugging Face OpenAI-compatible router
        api_url = (
            "https://router.huggingface.co/"
            "v1/chat/completions"
        )


        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }


        payload = {

            # Explicitly use Featherless AI because this
            # model is currently served there through HF.
            "model": (
                "meta-llama/"
                "Llama-3.2-1B-Instruct:"
                "featherless-ai"
            ),

            "messages": [

                {
                    "role": "system",
                    "content": (
                        "You are a helpful and concise "
                        "AI assistant."
                    )
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            "max_tokens": 250,
            "temperature": 0.7,

            "stream": False
        }


        # ----------------------------------------------------
        # Send request to Hugging Face
        # ----------------------------------------------------

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )


        # Try to decode JSON
        try:
            result = response.json()
        except ValueError:

            return jsonify({
                "status": "error",
                "telemetry": telemetry,
                "message": (
                    "Hugging Face returned a non-JSON response."
                ),
                "http_status": response.status_code
            }), 502


        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        if (
            response.ok
            and "choices" in result
            and len(result["choices"]) > 0
        ):

            message = result["choices"][0].get(
                "message",
                {}
            )

            ai_answer = message.get(
                "content",
                ""
            ).strip()


            if not ai_answer:
                ai_answer = (
                    "The model returned an empty response."
                )


        # ----------------------------------------------------
        # Hugging Face API error
        # ----------------------------------------------------

        elif "error" in result:

            error_info = result["error"]

            if isinstance(error_info, dict):
                error_message = error_info.get(
                    "message",
                    str(error_info)
                )
            else:
                error_message = str(error_info)


            ai_answer = (
                f"Model notice: {error_message}"
            )


        # ----------------------------------------------------
        # Unknown response
        # ----------------------------------------------------

        else:

            ai_answer = str(result)


        # ----------------------------------------------------
        # Return gateway response
        # ----------------------------------------------------

        return jsonify({

            "status": "allowed",

            "telemetry": telemetry,

            "answer": ai_answer

        })


    # ========================================================
    # F. UPSTREAM CONNECTION ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        return jsonify({

            "status": "error",

            "telemetry": telemetry,

            "message": (
                "Gateway passed prompt, but the "
                "Hugging Face request timed out."
            )

        }), 504


    except requests.exceptions.RequestException as e:

        return jsonify({

            "status": "error",

            "telemetry": telemetry,

            "message": (
                "Gateway passed prompt, but the "
                f"upstream connection failed: {str(e)}"
            )

        }), 502


    except Exception as e:

        return jsonify({

            "status": "error",

            "telemetry": telemetry,

            "message": (
                f"Unexpected gateway error: {str(e)}"
            )

        }), 500


# ============================================================
# 5. RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "🚀 Starting Aegis AI Security Gateway..."
    )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )