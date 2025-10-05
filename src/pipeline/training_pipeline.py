import os
import joblib
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


from src.components.feature_extraction import extract_features
from src.components.preprocess import combine_prepare_data   



def train_and_save_pipeline(phishing_url_path, general_url_path, model_path): 

    # Step 1: Data Preprocessing 
    data = combine_prepare_data(phishing_url_path, general_url_path) 

    if data is None or data.empty: 
        print("Data loading failed.") 
        return
    
    # Step 2: Feature Extraction 
    data = extract_features(data) 
    
    # Step 3: Train-Test Split 
    X = data.drop('label',axis=1)
    y = data['label'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) 

    
    # Step 4: preprocessing - vectorize urls and scale numerical features 
    preprocessor = ColumnTransformer(transformers=[ 
                    ('tfidf', TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(2,5)), 'URL'), 
                    ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns.tolist()) 
                    ], 
                    remainder='drop') 
    

    # Step 5: Define models 
    models = { 

        'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', solver='saga', n_jobs=-1, tol=1e-3,random_state=42), 
        'Random Forest': RandomForestClassifier(class_weight='balanced',n_estimators=50,random_state=42, n_jobs=-1), 
        'XGBoost': xgb.XGBClassifier(n_estimators=50, n_jobs=-1, random_state=42) 

        } 
    

    best_model = None 
    best_model_name = None 
    best_recall = 0 
    best_cm = None

    # step 6: Train and evaluate each model 
    for model_name, model in models.items():

        print(f"\n --- {model_name} ---") 
        
        # create pipeline - preprocessing + model 
        pipeline = Pipeline(steps=[ ('preprocessor', preprocessor), ('classifier', model) ]) 

        
        #Cross Validation (StratifiedKFold) 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='recall')

        print(f"Cross-validation recall scores: {cv_scores}") 
        print(f"Mean Recall: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}") 
        

        # Final Training and Test Evaluation

        pipeline.fit(X_train, y_train) 
        y_pred = pipeline.predict(X_test) 


        acc = accuracy_score(y_test, y_pred) 
        prec = precision_score(y_test, y_pred) 
        rec = recall_score(y_test, y_pred) 
        cm = confusion_matrix(y_test, y_pred) 

        print("Accuracy:", acc) 
        print("Precision:", prec) 
        print("Recall:", rec) 
        print("Confusion Matrix:\n", cm) 
        
        # check if this model has the best recall so far 

        if rec > best_recall: 
            
            best_recall = rec 
            best_model = pipeline 
            best_model_name = model_name 
            best_cm = cm 

    print(f"\n Best Model based on Recall: {best_model_name}") 
    print(f"Recall: {best_recall:.4f}") 
    print("Confusion Matrix of Best Model:\n", best_cm) 
            
    # Step 6: Save the best model 
    if best_model is not None: 
        joblib.dump(best_model, model_path) 
        print(f"Best model saved to {model_path}") 
        
            
if __name__ == "__main__": 

    # project root
     
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
    data_dir = os.path.join(project_root, "data", "Phishing URL dataset") 


    phishing_path = os.path.join(data_dir, "phishing_dataset.csv") 
    general_path = os.path.join(data_dir, "url_dataset.csv") 
    model_path = os.path.join(project_root, "artifacts", "best_model.pkl") 
    
    # create artifacts folder if not exists 
    os.makedirs(os.path.dirname(model_path), exist_ok=True) 


    if not os.path.exists(phishing_path) or not os.path.exists(general_path): 

        print("could not locate one or both data files") 
        print(f"Expected: {phishing_path}, {general_path}")

    else: 

        train_and_save_pipeline(phishing_path, general_path, model_path)
 