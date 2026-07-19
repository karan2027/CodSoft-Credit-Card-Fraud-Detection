import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def generate_synthetic_data(n_samples=20000, fraud_rate=0.017, random_seed=42):
    """
    Generates a realistic synthetic credit card dataset matching the schema and 
    approximate statistical properties of the Kaggle European dataset.
    """
    np.random.seed(random_seed)
    
    n_fraud = max(5, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud
    
    # 1. Time feature: 0 to 172800 seconds (2 days)
    time_legit = np.random.uniform(0, 172800, n_legit)
    time_fraud = np.random.uniform(0, 172800, n_fraud)
    
    # 2. Amount feature: log-normally distributed
    # Legit transactions average around $88, Fraud around $122
    amount_legit = np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit)
    amount_fraud = np.random.lognormal(mean=3.8, sigma=1.4, size=n_fraud)
    
    # Cap amounts to be within realistic bounds of original dataset
    amount_legit = np.clip(amount_legit, 0, 25691.16)
    amount_fraud = np.clip(amount_fraud, 0, 2125.87)
    
    # 3. PCA Features V1 to V28
    # For Legit (Class 0), features are standard normals
    V_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    
    # For Fraud (Class 1), features have shifted means based on the original dataset
    # We define the average shifts for each feature based on Kaggle dataset analysis
    fraud_means = {
        0: -4.77,  1: 3.62,   2: -7.03,  3: 4.54,   4: -3.15,
        5: -1.39,  6: -5.56,  7: 0.57,   8: -2.58,  9: -5.67,
        10: 3.80,  11: -6.25, 12: -0.11, 13: -6.90, 14: -0.09,
        15: -4.13, 16: -8.96, 17: -3.11, 18: 0.68,  19: 0.37,
        20: 0.71,  21: 0.01,  22: -0.04, 23: -0.11, 24: 0.04,
        25: 0.05,  26: 0.17,  27: 0.08
    }
    
    # Some features have higher variance in fraud transactions
    V_fraud = np.zeros((n_fraud, 28))
    for i in range(28):
        mean_shift = fraud_means.get(i, 0.0)
        # Use slightly higher standard deviation for fraud transactions to model wider distribution
        std_dev = 2.0 if abs(mean_shift) > 2.0 else 1.2
        V_fraud[:, i] = np.random.normal(loc=mean_shift, scale=std_dev, size=n_fraud)
        
    # Combine everything
    legit_df = pd.DataFrame(V_legit, columns=[f'V{i+1}' for i in range(28)])
    legit_df.insert(0, 'Time', time_legit)
    legit_df['Amount'] = amount_legit
    legit_df['Class'] = 0
    
    fraud_df = pd.DataFrame(V_fraud, columns=[f'V{i+1}' for i in range(28)])
    fraud_df.insert(0, 'Time', time_fraud)
    fraud_df['Amount'] = amount_fraud
    fraud_df['Class'] = 1
    
    df = pd.concat([legit_df, fraud_df], axis=0).reset_index(drop=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def try_download_subset(filepath='creditcard.csv'):
    """
    Attempts to download a subset of the credit card fraud dataset 
    from a public GitHub mirror. Returns True if successful.
    """
    # Using a reliable Github repository hosting a subset or compressed version
    url = "https://raw.githubusercontent.com/nsethi31/Credit-Card-Fraud-Detection/master/creditcard.csv"
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def balance_dataset(X, y, strategy='undersample', random_seed=42):
    """
    Balances the dataset using either undersampling or oversampling (SMOTE-like or simple replication).
    """
    df = pd.concat([X, y], axis=1)
    legit = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]
    
    n_fraud = len(fraud)
    n_legit = len(legit)
    
    if strategy == 'undersample':
        # Sample legit transactions to match number of fraud transactions (as in original project)
        legit_sample = legit.sample(n=min(n_legit, n_fraud), random_state=random_seed)
        balanced_df = pd.concat([legit_sample, fraud], axis=0)
    elif strategy == 'oversample':
        # Simple random oversampling of the fraud class to match legit transactions
        fraud_oversample = fraud.sample(n=n_legit, replace=True, random_state=random_seed)
        balanced_df = pd.concat([legit, fraud_oversample], axis=0)
    else:
        # No balancing
        balanced_df = df
        
    # Shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    X_bal = balanced_df.drop(columns='Class')
    y_bal = balanced_df['Class']
    
    return X_bal, y_bal

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, random_seed=42):
    """
    Trains the selected model and returns training/testing performance metrics.
    """
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=random_seed)
    elif model_name == "K-Nearest Neighbors":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Probabilities for ROC curve
    if hasattr(clf, "predict_proba"):
        y_test_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = clf.decision_function(X_test)
        
    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "model": clf
    }
    
    # ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    metrics["fpr"] = fpr.tolist()
    metrics["tpr"] = tpr.tolist()
    metrics["auc"] = roc_auc
    
    return metrics
