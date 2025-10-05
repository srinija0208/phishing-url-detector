import joblib
import pandas as pd
import sys

from src.components.feature_extraction import extract_features

def load_model(model_path):

    try:
        model = joblib.load(model_path)
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def predict_urls(model, urls):

    
    if isinstance(urls,str):
        urls = [urls]  # Convert single URL to list

    # Create DataFrame
    df = pd.DataFrame({'URL': urls})

    # Extract features
    df = extract_features(df)

    # drop label column if exists
    if 'label' in df.columns:
        df = df.drop('label',axis=1)

    # Make predictions
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]  # Probability of being phishing

    results = pd.DataFrame({'URL':urls, 'Prediction':pred, 'Phishing_Probability':prob})

    return results

if __name__ == "__main__":

    model_path = "models/best_model.pkl"
    model = load_model(model_path)

    test_urls = [
        "http://secure-login-paypal.com/account/verify",
        "https://www.google.com",
    ]

    results = predict_urls(test_urls, model)
    print(results)