# 🎣 Phishing URL Detector

## 📌 Model Description  

This project is a machine learning web app that detects whether a given URL is potentially phishing or reliably legitimate. It combines **character-level TF-IDF n-grams** with **engineered numerical features** to make predictions.  

Three models were tested: 

- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**  

The **best model was selected based on recall**, since reducing false negatives (phishing URLs classified as safe) is the highest priority.  

---

## Current Best Model

  -- XGBoost with 

**Recall** = 0.9910
**Precision** = 0.9968 
**Accuracy** = 0.9961 

## ✅ Intended Uses & ⚠️ Limitations  

### Intended Uses 

- Detect phishing URLs in security applications (browsers, firewalls, email filters).  
- Demonstration of phishing detection for educational or research purposes.  

### Limitations  

- Model selection focused on **recall** → may lead to higher **false positives**.  
- Limited to **top 1000 n-grams** → some rare phishing tricks may not be captured.  
- No **domain-level metadata** (e.g., WHOIS, SSL, DNS info).  

---

## 🛠️ Training Procedure  

- **Data:**  
  - `phishing_dataset.csv` → Phishing URLs.  
  - `url_dataset.csv` → Legitimate URLs.  

- **Split:**  
  Train-test split (80:20) with stratification.  

- **Preprocessing:**  
  - `TfidfVectorizer(max_features=1000, analyzer="char", ngram_range=(2,5))`  
  - `StandardScaler()` for numerical features  

- **Validation:**  
  Stratified 5-fold cross-validation  

- **Metric:**  
  Recall (primary), with precision and accuracy for reference  

---

# 🧩 Tech Stack

**Language:** Python

**Libraries:** XGBoost, Scikit-learn, Pandas, Joblib, Gradio

**Deployment:** Hugging Face Spaces

**Version Control:** Git & GitHub

---


## 🚀 Demo & Usage

Try the application directly in the Space interface!

- **Enter URL:** Paste the complete URL you wish to check.

- **Analyze:** Click the Analyze URL button.

- **Review:** The app displays the final classification (Phishing/Legitimate) and the associated probability score.

- **Clear:** Use the Clear button to reset the fields for a new test.


- **Prediction:** Binary classification (1=Phishing, 0=Legitimate).

- **Best Model:** Selected from Logistic Regression, Random Forest, and XGBoost.

- **Feature Engineering:** Features include URL length, presence of IP addresses, dash count, and other structural and linguistic signals.

---

# 📦 Requirements for Deployment

For this application to build and run successfully on Hugging Face Spaces, the following files and folders must be committed to your repository:

- phishing_detector_app.py: Main application code.


- artifacts/best_model.pkl: The saved trained model pipeline.

- src/components/feature_extraction.py: The custom Python module required for feature extraction.

---

# 🗂️ Project Structure

phishing-url-detector/
│
├── artifacts/                  # Saved model (best_model.pkl) and related artifacts.
│
├── src/
│   ├── components/
│   │   ├── feature_extraction.py   # Extracts lexical & structural URL features
│   │   └── preprocess.py           # Preprocessing pipeline for dataset
│   │
│   └── pipeline/
│       ├── prediction_pipeline.py  # Loads model and predicts new URLs
│       └── training_pipeline.py    # Handles model training & evaluation
│
├── phishing_detector_app.py    # Main Gradio app for deployment (HF Space).
├── requirements.txt            # Python dependencies.
├── README.md                   # Hugging Face Space configuration (required YAML header).
├── README-GitHub.md            # Detailed project documentation (this file).
└── LICENSE                     # Project license (e.g., MIT).                           


---

# 👉 Live Demo on Hugging Face:
🔗 https://huggingface.co/spaces/srinija1176/phishing-url-detector

