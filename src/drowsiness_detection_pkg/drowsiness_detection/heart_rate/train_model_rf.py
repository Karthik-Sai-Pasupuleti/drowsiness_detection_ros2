"""
train_model.py
----------------
Random Forest training pipeline.
- Automatically creates a 'model_run_results' folder.
- Saves Images, Model, and CSV data reports.
"""

import os
import sys
import numpy as np
import pandas as pd

# Use a non-interactive backend for safe saving without GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# SETTINGS (Now Relative Paths)
# ---------------------------
# Assuming the csv is in the same folder as this script. 
# If not, put the full path here: r"C:\Path\To\Your\drowsiness_dataset.csv"
CSV_FILE_NAME = r"src/drowsiness_detection_ros2/src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Datasets/engineered_features_with_hrv_hr_time.csv"
CSV_FILE_PATH = os.path.join(os.getcwd(), CSV_FILE_NAME)

# This creates a folder named 'model_run_results' right next to this script
OUTPUT_DIR = os.path.join(os.getcwd(), 'model_run_results')

TARGET_COLUMN = 'Label'

# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str):
    """Creates the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Output directory ready: {path}")

def savefig_safe(figpath: str):
    """Safely save current Matplotlib figure."""
    try:
        plt.savefig(figpath, bbox_inches='tight')
        print(f"[INFO] Saved figure to '{os.path.basename(figpath)}'")
    except Exception as e:
        print(f"[ERROR] Could not save figure: {e}")
    finally:
        plt.close()

# ---------------------------
# Data Loading & Processing
# ---------------------------

def load_data(csv_path: str):
    try:
        print(f"[INFO] Loading data from {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found at: {csv_path}")
            print("Please ensure 'drowsiness_dataset.csv' is in the script folder or update CSV_FILE_PATH.")
            return None
            
        df = pd.read_csv(csv_path)
        print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def engineer_features(df: pd.DataFrame):
    # 1. Define columns to remove (Target + Non-numeric Timestamps)
    cols_to_remove = [TARGET_COLUMN, 'window_start_time', 'window_end_time']
    
    # 2. Separate Target (y)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in dataset.")
    y = df[TARGET_COLUMN]
    
    # 3. Create Features (X) by dropping specific columns
    drop_actual = [c for c in cols_to_remove if c in df.columns]
    X = df.drop(columns=drop_actual)
    
    # 4. SAFETY STEP: Keep only numbers
    X = X.select_dtypes(include=[np.number])

    # 5. CRITICAL FIX: Replace Infinity with NaN, then fill all NaNs with 0
    # This handles both "empty" cells and "infinite" math errors
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
        
    print(f"[INFO] Features prepared. X: {X.shape}, y: {y.shape}")
    
    # Double check for safety
    if not np.isfinite(X).all().all():
        print("[WARN] Infinite values might still persist. Forcing fit...")
        
    return X, y

# ---------------------------
# Visualization
# ---------------------------

def plot_label_distribution(y: pd.Series, output_dir: str):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Drowsiness Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    savefig_safe(os.path.join(output_dir, '01_label_distribution.png'))

def plot_key_feature_distributions(X: pd.DataFrame, output_dir: str):
    # Attempt to plot specific heart rate features if they exist
    key_features = ['HRV_RMSSD', 'HRV_SDNN', 'HRV_HF', 'HRV_LFHF', 'HRV_SampEn']
    features_to_plot = [f for f in key_features if f in X.columns]
    
    # If specific keys aren't found, just plot the first 5 columns
    if not features_to_plot:
        features_to_plot = X.columns[:5]

    plt.figure(figsize=(15, 4))
    for i, feature in enumerate(features_to_plot[:3]): # Plot up to 3 for brevity
        plt.subplot(1, 3, i + 1)
        sns.histplot(X[feature], kde=True, bins=20)
        plt.title(f'Distribution: {feature}')
    plt.tight_layout()
    savefig_safe(os.path.join(output_dir, '02_feature_distributions.png'))

# ---------------------------
# Train & Evaluate
# ---------------------------

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, output_dir: str):
    unique_classes = np.unique(y)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train Model
    print('[INFO] Training Random Forest...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # --- SAVE CSV RESULTS ---
    print('\n[INFO] Saving CSV results...')
    
    # 1. Save Predictions to CSV
    results_df = X_test.copy()
    results_df['Actual_Label'] = y_test
    results_df['Predicted_Label'] = y_pred
    results_csv_path = os.path.join(output_dir, 'test_set_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"   -> Saved predictions to: {results_csv_path}")

    # 2. Save Feature Importance to CSV
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    feat_csv_path = os.path.join(output_dir, 'feature_importances.csv')
    feature_importance_df.to_csv(feat_csv_path, index=False)
    print(f"   -> Saved feature importances to: {feat_csv_path}")

    # --- VISUALS & REPORTS ---
    
    # Classification Report
    report_text = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(output_dir, '00_classification_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"   -> Saved classification report txt")

    # Confusion Matrix Image
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    savefig_safe(os.path.join(output_dir, '03_confusion_matrix.png'))

    # Feature Importance Image
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    savefig_safe(os.path.join(output_dir, '04_feature_importances.png'))

    # Save Model
    try:
        import joblib
        model_path = os.path.join(output_dir, 'rf_model.joblib')
        joblib.dump(model, model_path)
        print(f"   -> Saved model object to: {model_path}")
    except Exception as e:
        print(f"[WARN] Could not save model object: {e}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    print('=' * 30)
    print('   AUTO-SAVE TRAINING PIPELINE')
    print('=' * 30)

    # 1. Create Output Folder
    _ensure_dir(OUTPUT_DIR)

    # 2. Load
    df = load_data(CSV_FILE_PATH)
    
    if df is not None:
        # 3. Process
        try:
            X, y = engineer_features(df)
            
            # 4. Visualize Data
            plot_label_distribution(y, OUTPUT_DIR)
            plot_key_feature_distributions(X, OUTPUT_DIR)
            
            # 5. Train & Save All Results
            train_and_evaluate(X, y, OUTPUT_DIR)
            
            print('\n' + '=' * 30)
            print(f"DONE! All files saved in:\n{OUTPUT_DIR}")
            print('=' * 30)
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")
    else:
        print("\n[STOP] Could not load data. Check file path.")