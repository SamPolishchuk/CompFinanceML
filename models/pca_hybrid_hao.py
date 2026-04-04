"""
Hao's hybrid model:
PCA + Logistic Regression
PCA + SVM

This script reads the cleaned startup dataset from:
data/startup_data_cleaned.csv

It builds a hybrid pipeline in which:
1. The raw features are preprocessed.
2. PCA is applied as an unsupervised dimensionality reduction step.
3. Logistic Regression and SVM are trained on the PCA representation.

For comparison, the script also fits raw-feature baselines:
- Raw Logistic Regression
- Raw SVM

All comments and printed explanations are intentionally written in English.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
TEST_SIZE = 0.20

# PCA can take either:
# - a float in (0,1], meaning the fraction of variance to retain
# - an integer, meaning the exact number of principal components
PCA_COMPONENTS = 0.95

# Change this only if your cleaned dataset uses a different target column name.
TARGET_COL = None

# Candidate target column names for automatic detection.
TARGET_CANDIDATES = ["status", "outcome", "target", "label", "labels", "result"]

# These label sets are used to map the target into a binary task.
# Positive class = success
# Negative class = failure
SUCCESS_LABELS = {
    "ipo", "acquisition", "acquired", "success", "successful", "1", "true", "yes"
}
FAILURE_LABELS = {
    "failure", "failed", "closed", "0", "false", "no"
}

# SVM kernel can be changed to "linear" if needed.
SVM_KERNEL = "rbf"


# ============================================================
# 2. PATHS
# ============================================================

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    REPO_ROOT = Path.cwd()

DATA_PATH = REPO_ROOT / "data" / "startup_data_cleaned.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "hao_pca_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def detect_target_column(df: pd.DataFrame, manual_target: str | None = None) -> str:
    """
    Detect the target column.
    If manual_target is provided, it is used directly.
    Otherwise, the function searches common target column names.
    """
    if manual_target is not None:
        if manual_target not in df.columns:
            raise ValueError(f"Target column '{manual_target}' was not found in the dataset.")
        return manual_target

    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "No target column was detected automatically. "
        f"Please set TARGET_COL manually. Available columns: {list(df.columns)}"
    )


def build_binary_target(series: pd.Series) -> pd.Series:
    """
    Convert the original target column into a binary target:
    1 = success
    0 = failure

    The function supports:
    - already-binary numeric targets
    - common text labels such as IPO, Acquisition, acquired, closed, failed
    """
    # Case 1: already numeric binary
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(series.dropna().unique().tolist())
        if set(unique_values).issubset({0, 1}):
            return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index, dtype=float)

    for label in normalized.unique():
        if label in FAILURE_LABELS or "fail" in label or "close" in label:
            mapped.loc[normalized == label] = 0
        elif label in SUCCESS_LABELS or "ipo" in label or "acqui" in label or "success" in label:
            mapped.loc[normalized == label] = 1

    unresolved = normalized[mapped.isna()].unique().tolist()
    if len(unresolved) > 0:
        raise ValueError(
            "Some target labels could not be mapped into binary classes. "
            f"Unresolved labels: {unresolved}. "
            "Please update SUCCESS_LABELS / FAILURE_LABELS in the configuration section."
        )

    return mapped.astype(int)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing transformer:
    - numeric features: median imputation + standard scaling
    - categorical features: most-frequent imputation + one-hot encoding
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_dir):
    """
    Fit a model, predict on the test set, compute metrics,
    and save the confusion matrix and classification report.
    """
    print("=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
    }

    print("Classification report:")
    report = classification_report(y_test, y_pred, target_names=["Failure", "Success"], zero_division=0)
    print(report)

    print("Metric summary:")
    for k, v in results.items():
        if k != "model":
            print(f"{k}: {v:.4f}")
    print()

    report_path = output_dir / f"{model_name.lower().replace(' ', '_')}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failure", "Success"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()

    return results


def save_pca_artifacts(fitted_pipeline: Pipeline, X_train: pd.DataFrame, output_dir: Path):
    """
    Save PCA explained variance, loadings, and a 2D PCA scatter plot.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    pca = fitted_pipeline.named_steps["pca"]

    X_train_preprocessed = preprocessor.transform(X_train)
    X_train_pca = pca.transform(X_train_preprocessed)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    variance_df = pd.DataFrame({
        "principal_component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance": cumulative_variance
    })
    variance_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_cumulative_explained_variance.png", dpi=200)
    plt.close()

    feature_names = preprocessor.get_feature_names_out()
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings.to_csv(output_dir / "pca_loadings_all.csv")

    with open(output_dir / "pca_top_loadings.txt", "w", encoding="utf-8") as f:
        max_pc = min(3, pca.n_components_)
        for i in range(max_pc):
            pc = f"PC{i+1}"
            f.write("=" * 60 + "\n")
            f.write(f"Top positive loadings for {pc}\n")
            f.write("=" * 60 + "\n")
            f.write(loadings[pc].sort_values(ascending=False).head(10).to_string())
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write(f"Top negative loadings for {pc}\n")
            f.write("=" * 60 + "\n")
            f.write(loadings[pc].sort_values(ascending=True).head(10).to_string())
            f.write("\n\n\n")

    if X_train_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.65, s=30)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Training Data in PCA Space")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pca_2d_projection.png", dpi=200)
        plt.close()

    print("=" * 70)
    print("PCA summary")
    print("=" * 70)
    print(f"Number of retained components: {pca.n_components_}")
    print(f"Total explained variance retained: {cumulative_variance[-1]:.4f}")
    print()


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("Step 1: Loading cleaned dataset")
print("=" * 70)
print(f"Reading file from: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Make sure the file exists inside the repo under data/startup_data_cleaned.csv"
    )

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

print(f"Dataset shape: {df.shape}")
print("Columns:")
print(df.columns.tolist())
print()


# ============================================================
# 5. DETECT TARGET AND BUILD BINARY LABEL
# ============================================================

print("=" * 70)
print("Step 2: Detecting and preparing the target column")
print("=" * 70)

target_col = detect_target_column(df, TARGET_COL)
print(f"Detected target column: {target_col}")
print("Original target value counts:")
print(df[target_col].value_counts(dropna=False))
print()

df = df.dropna(subset=[target_col]).copy()
df["binary_target"] = build_binary_target(df[target_col])

print("Binary target distribution:")
print(df["binary_target"].value_counts())
print()


# ============================================================
# 6. BUILD FEATURE MATRIX
# ============================================================

print("=" * 70)
print("Step 3: Building feature matrix")
print("=" * 70)

X = df.drop(columns=[target_col, "binary_target"])
y = df["binary_target"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print()

print("Numeric columns:")
print(X.select_dtypes(include=[np.number]).columns.tolist())
print()

print("Categorical columns:")
print(X.select_dtypes(exclude=[np.number]).columns.tolist())
print()


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("=" * 70)
print("Step 4: Creating train-test split")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print()


# ============================================================
# 8. BUILD PREPROCESSOR
# ============================================================

preprocessor = make_preprocessor(X_train)


# ============================================================
# 9. BUILD MODELS
# ============================================================

print("=" * 70)
print("Step 5: Building model pipelines")
print("=" * 70)

# Raw baseline: Logistic Regression
raw_logreg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))
])

# Raw baseline: SVM
raw_svm = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel=SVM_KERNEL, probability=True, random_state=SEED))
])

# Hao's hybrid model: PCA + Logistic Regression
pca_logreg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
    ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))
])

# Hao's hybrid model: PCA + SVM
pca_svm = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
    ("classifier", SVC(kernel=SVM_KERNEL, probability=True, random_state=SEED))
])

print("Model pipelines are ready.")
print()


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        model=raw_logreg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="Raw Logistic Regression",
        output_dir=OUTPUT_DIR
    )
)

results.append(
    evaluate_model(
        model=raw_svm,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="Raw SVM",
        output_dir=OUTPUT_DIR
    )
)

results.append(
    evaluate_model(
        model=pca_logreg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="PCA Logistic Regression",
        output_dir=OUTPUT_DIR
    )
)

results.append(
    evaluate_model(
        model=pca_svm,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="PCA SVM",
        output_dir=OUTPUT_DIR
    )
)


# ============================================================
# 11. SAVE RESULTS TABLE
# ============================================================

print("=" * 70)
print("Step 6: Saving model comparison results")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
print(results_df)
print()

results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df[metric])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(f"Model Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"compare_{metric}.png", dpi=200)
    plt.close()


# ============================================================
# 12. SAVE PCA ARTIFACTS FROM HAO'S HYBRID MODEL
# ============================================================

# We use the fitted PCA + Logistic Regression pipeline to inspect the PCA step.
# The PCA step would be the same in the PCA + SVM pipeline because the training data
# and preprocessing steps are identical.

save_pca_artifacts(pca_logreg, X_train, OUTPUT_DIR)

print("=" * 70)
print("Finished successfully.")
print("=" * 70)
print(f"All outputs were saved to: {OUTPUT_DIR}")
