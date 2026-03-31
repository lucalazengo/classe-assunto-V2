---
name: skill_train_evaluate_lgbm
description: Automated pipeline for TF-IDF + LightGBM text classification. Training, evaluation (F1-Score, Precision, Recall), hyperparameter tuning, and model persistence for legal text classification.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Skill: Train & Evaluate LightGBM Classifier

> Automated pipeline for training a TF-IDF + LightGBM text classifier.
> Designed for multi-class legal text classification (Nível 2 of the cascade).

---

## ⚠️ How to Use This Skill

This skill defines the **training and evaluation pipeline** for the statistical classifier (Nível 2). It receives cleaned text data and produces a trained model artifact with evaluation metrics. Follow the steps sequentially.

---

## 1. Pipeline Overview

```
Input CSV (clean) → Train/Test Split (stratified)
                  → TF-IDF Vectorization (fit on train)
                  → LightGBM Training (with class weights)
                  → Evaluation (F1-Macro, per-class report)
                  → Artifact Saving (model + vectorizer + metadata)
```

---

## 2. Data Preparation

### Stratified Split

```python
from sklearn.model_selection import train_test_split

def prepare_data(df, text_col='inteiro_teor', target_col='classe', test_size=0.2):
    """
    Split data with stratification to preserve class distribution.
    
    Returns: X_train, X_test, y_train, y_test
    """
    # Remove rows with null text or target
    df_clean = df.dropna(subset=[text_col, target_col])
    
    # Remove classes with < min_samples (can't stratify with 1 sample)
    min_samples = 2
    class_counts = df_clean[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_clean = df_clean[df_clean[target_col].isin(valid_classes)]
    
    X = df_clean[text_col]
    y = df_clean[target_col]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
```

---

## 3. TF-IDF Vectorization

### Configuration Principles

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer(max_features=50000, ngram_range=(1, 2)):
    """
    Create TF-IDF vectorizer optimized for legal text.
    
    Key decisions:
    - max_features: 50k balances vocabulary coverage vs memory
    - ngram_range: (1,2) captures bigrams like "ação cobrança"
    - sublinear_tf: True applies log scaling (better for long documents)
    - min_df: 2 removes terms appearing in only 1 document
    - max_df: 0.95 removes terms in 95%+ documents (near-stopwords)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        analyzer='word',
    )
```

> [!IMPORTANT]
> **Always fit the vectorizer on training data ONLY, then transform test data.** Fitting on full data causes data leakage.

```python
# CORRECT:
vectorizer = create_vectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # transform only!

# WRONG:
# X_all_tfidf = vectorizer.fit_transform(X_all)  # LEAKAGE!
```

---

## 4. LightGBM Training

### Configuration

```python
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

def train_lgbm(X_train, y_train, num_classes):
    """
    Train LightGBM classifier with class-weight balancing.
    
    Key decisions:
    - objective: multi_softmax for probability outputs
    - class_weight: 'balanced' handles imbalance automatically
    - n_estimators: 500 with early stopping
    - learning_rate: 0.05 (moderate, avoids overfitting)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(X_train, y_encoded)
    
    return model, le
```

### Hyperparameter Tuning (Optional, Phase 2)

```python
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM hyperparameter search."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = lgb.LGBMClassifier(**params, objective='multiclass', 
                                class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return f1_score(y_val, y_pred, average='macro')
```

---

## 5. Evaluation

### Metrics Suite

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

def evaluate_model(model, le, X_test, y_test, vectorizer=None):
    """
    Full evaluation suite for the trained model.
    
    Returns dict with all metrics and reports.
    """
    y_encoded = le.transform(y_test)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Decode predictions back to original labels
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Overall metrics
    metrics = {
        'f1_macro': f1_score(y_encoded, y_pred, average='macro'),
        'f1_weighted': f1_score(y_encoded, y_pred, average='weighted'),
        'precision_macro': precision_score(y_encoded, y_pred, average='macro'),
        'recall_macro': recall_score(y_encoded, y_pred, average='macro'),
        'accuracy': accuracy_score(y_encoded, y_pred),
    }
    
    # Per-class report
    report = classification_report(
        y_test, y_pred_labels,
        output_dict=True, zero_division=0
    )
    
    # Confidence analysis
    max_proba = y_pred_proba.max(axis=1)
    metrics['mean_confidence'] = float(max_proba.mean())
    metrics['pct_above_85'] = float((max_proba > 0.85).mean())
    metrics['pct_above_90'] = float((max_proba > 0.90).mean())
    
    return {
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_encoded, y_pred).tolist(),
        'confidence_distribution': max_proba.tolist(),
    }
```

### Cascade Threshold Analysis

```python
def analyze_thresholds(y_pred_proba, y_true, thresholds=[0.80, 0.85, 0.90, 0.95]):
    """
    Analyze what happens at different confidence thresholds.
    Shows tradeoff: higher threshold → fewer predictions but higher precision.
    """
    results = []
    max_proba = y_pred_proba.max(axis=1)
    y_pred = y_pred_proba.argmax(axis=1)
    
    for t in thresholds:
        mask = max_proba >= t
        coverage = mask.mean()
        
        if mask.sum() > 0:
            f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        else:
            f1 = 0.0
        
        results.append({
            'threshold': t,
            'coverage': round(coverage, 4),
            'f1_macro': round(f1, 4),
            'n_predictions': int(mask.sum()),
            'n_fallback': int((~mask).sum()),
        })
    
    return results
```

---

## 6. Artifact Persistence

```python
import joblib
import json
from datetime import datetime

def save_model_artifacts(model, vectorizer, le, metrics, output_dir, version='v1'):
    """
    Save all model artifacts with metadata.
    
    Saves:
    - model.joblib (trained LightGBM model)
    - vectorizer.joblib (fitted TF-IDF vectorizer)
    - label_encoder.joblib (fitted LabelEncoder)
    - metadata.json (params, metrics, timestamp)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    joblib.dump(model, f"{output_dir}/model_{version}.joblib")
    joblib.dump(vectorizer, f"{output_dir}/vectorizer_{version}.joblib")
    joblib.dump(le, f"{output_dir}/label_encoder_{version}.joblib")
    
    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LightGBM',
        'vectorizer_type': 'TF-IDF',
        'n_features': vectorizer.max_features,
        'n_classes': len(le.classes_),
        'classes': le.classes_.tolist(),
        'metrics': metrics,
        'params': model.get_params(),
    }
    
    with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Model artifacts saved to {output_dir}")
```

---

## 7. Full Training Pipeline

```python
def run_training_pipeline(csv_path, text_col, target_col, output_dir):
    """
    End-to-end training pipeline.
    
    1. Load data
    2. Clean text (using skill_legal_text_cleaner)
    3. Split (stratified)
    4. Vectorize (TF-IDF)
    5. Train (LightGBM)
    6. Evaluate
    7. Save artifacts
    """
    # 1. Load
    df = pd.read_csv(csv_path, sep='#', encoding='utf-8')
    
    # 2. Clean (apply legal text cleaner)
    df[text_col] = df[text_col].apply(clean_legal_text)
    
    # 3. Split
    X_train, X_test, y_train, y_test = prepare_data(df, text_col, target_col)
    
    # 4. Vectorize
    vectorizer = create_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Train
    num_classes = y_train.nunique()
    model, le = train_lgbm(X_train_tfidf, y_train, num_classes)
    
    # 6. Evaluate
    results = evaluate_model(model, le, X_test_tfidf, y_test)
    
    # 7. Save
    save_model_artifacts(model, vectorizer, le, results['metrics'], output_dir)
    
    return results
```

---

## 8. Dependencies

```
Required packages:
├── lightgbm >= 4.0
├── scikit-learn >= 1.3
├── pandas >= 2.0
├── joblib (included with sklearn)
├── optuna >= 3.0 (optional, for hyperparameter tuning)
└── matplotlib / seaborn (for visualization)
```

---

> **Remember**: The statistical model (Nível 2) is the workhorse. Optimize for F1-Macro on imbalanced multi-class. Define confidence threshold to decide when to fallback to Nível 3.
