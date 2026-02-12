"""
explore_and_save_plots.py
---------------------------------
Loads a CSV and saves all exploratory diagrams as .PNG files to a **hardcoded** output folder.
This script adapts the plotting ideas from your `load_data.py` (correlation heatmap,
numeric distributions, boxplots, categorical counts, pairplot), but **saves to disk**
instead of calling `plt.show()`.

Edit these two constants as needed:
- CSV_FILE_PATH: location of your CSV
- OUTPUT_DIR: where .png files are written

No CLI args used.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# HARD-CODED PATHS
# ---------------------------
CSV_FILE_PATH = r'src/drowsiness_detection_ros2/src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Datasets/physiotagger_full_2025-12-07T12-51-14.csv'
OUTPUT_DIR    = r'src/drowsiness_detection_ros2/src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Original_Data_Result'  # <-- change to your preferred folder

# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_current_fig(name: str):
    _ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, name)
    try:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"[INFO] Saved: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {name}: {e}")
    finally:
        plt.close()

# ---------------------------
# Loading
# ---------------------------

def load_data(file_path: str):
    print(f"[INFO] Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("[INFO] Data loaded successfully.")
        print("\n--- Data Info ---")
        print(df.info())
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Basic Statistics (numeric) ---")
        print(df.describe())
        return df
    except FileNotFoundError:
        print("[ERROR] File not found. Check CSV_FILE_PATH.")
        return None
    except pd.errors.EmptyDataError:
        print("[ERROR] CSV appears to be empty.")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error while loading data: {e}")
        return None

# ---------------------------
# Plots -> PNGs
# ---------------------------

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: pd.Index):
    if len(numeric_cols) == 0:
        print("[INFO] No numerical columns for correlation heatmap.")
        return
    print("[INFO] Generating correlation heatmap...")
    plt.figure(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    _save_current_fig('correlation_heatmap.png')


def plot_numerical_distributions(df: pd.DataFrame, numeric_cols: pd.Index):
    if len(numeric_cols) == 0:
        print("[INFO] No numerical columns for distributions.")
        return
    print(f"[INFO] Generating histograms for {len(numeric_cols)} numerical features...")
    num_plots = len(numeric_cols)
    num_cols_grid = 3
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 5 * num_rows_grid))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    _save_current_fig('numerical_distributions.png')


def plot_numerical_boxplots(df: pd.DataFrame, numeric_cols: pd.Index):
    if len(numeric_cols) == 0:
        print("[INFO] No numerical columns for boxplots.")
        return
    print(f"[INFO] Generating box plots for {len(numeric_cols)} numerical features...")
    num_plots = len(numeric_cols)
    num_cols_grid = 3
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 5 * num_rows_grid))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    _save_current_fig('numerical_boxplots.png')


def plot_categorical_counts(df: pd.DataFrame, categorical_cols: pd.Index):
    if len(categorical_cols) == 0:
        print("[INFO] No categorical columns for count plots.")
        return
    print(f"[INFO] Generating count plots for {len(categorical_cols)} categorical features...")
    num_plots = len(categorical_cols)
    num_cols_grid = 2
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(18, 7 * num_rows_grid))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        if df[col].nunique() > 50:
            top_20 = df[col].value_counts().nlargest(20).index
            sns.countplot(y=col, data=df[df[col].isin(top_20)], order=top_20, ax=ax)
            ax.set_title(f'Top 20 Counts for {col} (High Cardinality)')
        else:
            sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
            ax.set_title(f'Counts for {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    _save_current_fig('categorical_counts.png')


def plot_pairplot(df: pd.DataFrame, numeric_cols: pd.Index):
    if len(numeric_cols) == 0:
        print("[INFO] No numerical columns for pairplot.")
        return
    print("[INFO] Generating pairplot (may take time)...")
    cols_to_plot = numeric_cols
    if len(numeric_cols) > 5:
        cols_to_plot = numeric_cols.to_series().sample(5, random_state=42).sort_index()
        print(f"[INFO] Too many numeric columns ({len(numeric_cols)}). Sampling 5 for pairplot.")
    g = sns.pairplot(df[cols_to_plot], corner=True)
    g.fig.suptitle('Pairplot of Numerical Features', y=1.02)
    # Save the figure from PairGrid
    _ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, 'pairplot.png')
    try:
        g.savefig(out_path, bbox_inches='tight')
        print(f"[INFO] Saved: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save pairplot.png: {e}")
    plt.close('all')

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    _ensure_dir(OUTPUT_DIR)

    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")

    df = load_data(CSV_FILE_PATH)
    if df is None:
        print("[FAILURE] Data loading failed.")
        sys.exit(1)

    print(f"\n[SUCCESS] Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns.")
    print("\n--- Starting EDA (PNG outputs) ---")

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    plot_correlation_heatmap(df, numeric_cols)
    plot_numerical_distributions(df, numeric_cols)
    plot_numerical_boxplots(df, numeric_cols)
    plot_categorical_counts(df, categorical_cols)
    plot_pairplot(df, numeric_cols)

    print("\n[INFO] PNGs written to:", OUTPUT_DIR)
