import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """
    Trains and evaluates multiple classifiers including a Majority Class Baseline,
    Logistic Regression, Random Forest, Gradient Boosting, and XGBoost/HistGradientBoosting
    on a 20% holdout test set with leak-free features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Majority Class Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state)
    else:
        models['Hist Gradient Boosting'] = HistGradientBoostingClassifier(random_state=random_state)
        
    results = {}
    best_model_name = None
    best_f1 = -1.0
    trained_models = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'Majority Class Baseline']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(preds, dtype=float)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(preds, dtype=float)
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.5
            
        cm = confusion_matrix(y_test, preds)
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm.tolist()
        }
        trained_models[name] = model
        
        # Select best ML model (excluding baseline)
        if name != 'Majority Class Baseline' and f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    if best_model_name is None:
        best_model_name = 'Random Forest'
        
    best_model = trained_models[best_model_name]
    
    return {
        'results_summary': pd.DataFrame(results).T,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'trained_models': trained_models
    }

def get_feature_importances(model, feature_names):
    """
    Extracts feature importances or logistic coefficients.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
        
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    return df_imp

def explain_single_prediction(model, scaler, input_vector_df, feature_names):
    """
    Provides explainable AI insight for a single instance input,
    deconstructing positive and negative contributing factors.
    """
    if hasattr(model, 'coef_'):
        input_scaled = scaler.transform(input_vector_df)
        coefs = model.coef_[0]
        prob = model.predict_proba(input_scaled)[0, 1]
        
        contributions = input_scaled[0] * coefs
        df_contrib = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions
        }).sort_values(by='contribution', key=abs, ascending=False)
        
        increasing_factors = df_contrib[df_contrib['contribution'] > 0].head(3)['feature'].tolist()
        decreasing_factors = df_contrib[df_contrib['contribution'] < 0].head(3)['feature'].tolist()
    else:
        prob = model.predict_proba(input_vector_df)[0, 1]
        df_imp = get_feature_importances(model, feature_names)
        increasing_factors = df_imp.head(3)['feature'].tolist()
        decreasing_factors = df_imp.tail(3)['feature'].tolist()
        
    return {
        'completion_probability': round(float(prob), 4),
        'completion_pct': round(float(prob) * 100.0, 1),
        'increasing_factors': increasing_factors,
        'decreasing_factors': decreasing_factors
    }

def save_trained_pipeline(best_model, scaler, feature_names, model_dir="models"):
    """
    Saves trained ML pipeline objects to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "model_classifier.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    print(f"[ML Engine] Model artifacts saved to '{model_dir}'.")

if __name__ == "__main__":
    from src.data_processing import process_and_clean_data
    from src.feature_engineering import calculate_behavioral_features, build_ml_feature_set
    
    _, _, _, df_master, _ = process_and_clean_data()
    df_eng, _ = calculate_behavioral_features(df_master)
    X, y, feature_cols = build_ml_feature_set(df_eng)
    
    eval_res = train_and_evaluate_models(X, y)
    print("[ML Modeling] Training complete. Results summary:")
    print(eval_res['results_summary'])
    save_trained_pipeline(eval_res['best_model'], eval_res['scaler'], feature_cols)
