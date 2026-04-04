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

SEED = 42
TEST_SIZE = 0.20
PCA_COMPONENTS = 0.95
SVM_KERNEL = "rbf"

TARGET_COL = None
TARGET_CANDIDATES = ["status", "outcome", "target", "label", "result"]

SUCCESS_LABELS = {"ipo", "acquisition", "acquired", "success", "successful", "1", "true", "yes"}
FAILURE_LABELS = {"failure", "failed", "closed", "0", "false", "no"}

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    REPO_ROOT = Path.cwd()

DATA_PATH = REPO_ROOT / "data" / "startup_data_cleaned.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "hao_pca_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_target_column(df, manual_target=None):
    if manual_target is not None:
        if manual_target not in df.columns:
            raise ValueError(f"Target column '{manual_target}' not found.")
        return manual_target

    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(f"No target column found. Available columns: {list(df.columns)}")


def build_binary_target(series):
    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().unique().tolist())
        if unique_values.issubset({0, 1}):
            return series.astype(int)

    s = series.astype(str).str.strip().str.lower()
    y = pd.Series(np.nan, index=series.index)

    for val in s.unique():
        if val in FAILURE_LABELS or "fail" in val or "close" in val:
            y.loc[s == val] = 0
        elif val in SUCCESS_LABELS or "ipo" in val or "acqui" in val or "success" in val:
            y.loc[s == val] = 1

    unresolved = s[y.isna()].unique().tolist()
    if unresolved:
        raise ValueError(f"Unresolved target labels: {unresolved}")

    return y.astype(int)


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_dir):
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

    report = classification_report(y_test, y_pred, target_names=["Failure", "Success"], zero_division=0)
    with open(output_dir / f"{model_name.lower().replace(' ', '_')}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Failure", "Success"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(model_name)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_cm.png", dpi=200)
    plt.close()

    return results


def save_pca_outputs(fitted_pipeline, X_train, output_dir):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    pca = fitted_pipeline.named_steps["pca"]

    X_train_processed = preprocessor.transform(X_train)
    X_train_pca = pca.transform(X_train_processed)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    variance_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained))],
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative
    })
    variance_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
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
    loadings.to_csv(output_dir / "pca_loadings.csv")

    with open(output_dir / "pca_top_loadings.txt", "w", encoding="utf-8") as f:
        for i in range(min(3, pca.n_components_)):
            pc = f"PC{i+1}"
            f.write(f"{pc} top positive loadings\n")
            f.write(loadings[pc].sort_values(ascending=False).head(10).to_string())
            f.write("\n\n")
            f.write(f"{pc} top negative loadings\n")
            f.write(loadings[pc].sort_values(ascending=True).head(10).to_string())
            f.write("\n\n\n")

    if X_train_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.6, s=20)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Training Data in PCA Space")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pca_2d_projection.png", dpi=200)
        plt.close()


print(f"Reading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

target_col = detect_target_column(df, TARGET_COL)
df = df.dropna(subset=[target_col]).copy()
df["binary_target"] = build_binary_target(df[target_col])

X = df.drop(columns=[target_col, "binary_target"])
y = df["binary_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

preprocessor = make_preprocessor(X_train)

pca_logreg = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
    ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))
])

pca_svm = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
    ("classifier", SVC(kernel=SVM_KERNEL, probability=True, random_state=SEED))
])

results = []
results.append(evaluate_model(pca_logreg, X_train, X_test, y_train, y_test, "PCA Logistic Regression", OUTPUT_DIR))
results.append(evaluate_model(pca_svm, X_train, X_test, y_train, y_test, "PCA SVM", OUTPUT_DIR))

results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

save_pca_outputs(pca_logreg, X_train, OUTPUT_DIR)

print(results_df)
print(f"Saved to: {OUTPUT_DIR}")
