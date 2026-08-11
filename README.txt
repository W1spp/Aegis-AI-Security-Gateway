===============================================================================
               AI SECURITY GATEWAY & JAILBREAK FIREWALL MONITOR
===============================================================================

[ OVERVIEW ]
-------------------------------------------------------------------------------
The AI Security Gateway & Jailbreak Firewall Monitor is a real-time cybersecurity
middleware designed to detect and block malicious prompt injection and jailbreak
attacks targeting Large Language Models (LLMs). 

Instead of relying on heavy, slow AI models to check incoming prompts, this system
uses a lightweight, high-speed Scikit-Learn Machine Learning pipeline running 
locally. It extracts structural metadata and character entropy from incoming text 
to classify threats before deciding whether to forward requests to the LLM.


[ KEY FEATURES ]
-------------------------------------------------------------------------------
* Real-Time Telemetry:
  Calculates Word Count, Total Length, Special Character Ratio, and Shannon 
  Entropy for every incoming prompt in real time.

* Multi-Algorithm Machine Learning Pipeline:
  - StandardScaler: Normalizes input features into uniform statistical z-scores.
  - KMeans Clustering: Groups prompts into structural "Archetypes".
  - RandomForestClassifier: Computes exact threat probability percentage.

* Cost & Latency Optimization:
  Blocks malicious prompts at the middleware layer, saving cloud API tokens and
  preventing malicious payload execution.

* Cloud LLM Integration:
  Safely routes clean prompts to open-source LLMs (Qwen 2.5 7B) hosted via 
  Hugging Face Serverless Inference API.

* Interactive Split-Screen Dashboard:
  Provides a modern Web UI displaying live chat responses on the left and 
  real-time security telemetry & threat gauges on the right.


[ TECH STACK ]
-------------------------------------------------------------------------------
* Language:       Python 3.10+
* Web Middleware: Flask
* Machine Learn:  Scikit-Learn, Pandas, NumPy, Joblib
* Data Source:    Hugging Face Datasets (`deepset/prompt-injections`)
* AI API:         Hugging Face Serverless Inference API (Qwen 2.5 7B)
* Frontend:       HTML5, CSS3, JavaScript (Fetch API)


[ PROJECT STRUCTURE ]
-------------------------------------------------------------------------------
Jailbreak-Firewall/
├── preprocess.py        # Dataset downloader & feature extraction test script
├── train_model.py       # ML training script (generates .pkl files)
├── app.py               # Flask REST API server & Security Gateway
├── scaler.pkl           # Saved StandardScaler model
├── kmeans.pkl           # Saved KMeans Clustering model
├── firewall_model.pkl   # Saved RandomForestClassifier model
└── templates/
    └── index.html       # Web Dashboard interface


[ INSTALLATION & SETUP ]
-------------------------------------------------------------------------------
1. Clone / Create Project Folder:
   $ mkdir Jailbreak-Firewall && cd Jailbreak-Firewall

2. Install Python Dependencies:
   $ pip install flask pandas numpy scikit-learn joblib datasets huggingface_hub

3. Train the Firewall Model:
   Download the dataset and build the model pipeline by running:
   $ python train_model.py

   (This will generate scaler.pkl, kmeans.pkl, and firewall_model.pkl)

4. Configure API Token:
   Open `app.py` and replace HF_TOKEN with your Hugging Face Access Token:
   HF_TOKEN = "hf_YOUR_TOKEN_HERE"


[ RUNNING THE APPLICATION ]
-------------------------------------------------------------------------------
1. Launch the Flask Server:
   $ python app.py

2. Access the Dashboard:
   Open your web browser and navigate to:
   http://127.0.0.1:5000


[ DETAILED WORKFLOW & ARCHITECTURE ]
-------------------------------------------------------------------------------
1. INPUT: User enters text into the browser dashboard.
2. FEATURE EXTRACTION: `app.py` computes:
   - Special Character Ratio = (Punctuation Count) / (Total Characters)
   - Shannon Entropy = -SUM( P(x) * log2(P(x)) )
3. ML PIPELINE:
   - Features are normalized via `scaler.pkl`.
   - Archetype category assigned via `kmeans.pkl`.
   - Threat Probability Score calculated via `firewall_model.pkl`.
4. DECISION GATE:
   - IF Threat Score > 50%: Request is BLOCKED immediately (HTTP response sent).
   - IF Threat Score <= 50%: Prompt is forwarded to Hugging Face API for completion.


[ DISCLAIMER & SECURITY NOTE ]
-------------------------------------------------------------------------------
Never publish API tokens to public version control repositories (e.g., GitHub). 
Store sensitive credentials in environment variables (`.env`) for production 
deployments.

===============================================================================