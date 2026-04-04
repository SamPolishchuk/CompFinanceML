# =============================================================================
# CELL 1 — IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, classification_report


# =============================================================================
# CELL 2 — LOAD & PREPROCESS  (same as your existing notebook)
# =============================================================================
df = pd.read_csv('/Users/Matthew/Desktop/repositories/ml/CompFinanceML/data/startup_data.csv')

drop_cols = [
    'Unnamed: 0', 'Unnamed: 6', 'id', 'object_id', 'name',
    'zip_code', 'city', 'state_code', 'state_code.1',
    'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at',
    'category_code', 'labels'
]
df = df.drop(columns=drop_cols)
df = df.rename(columns={'status': 'outcome'})

# Milestone imputation
df['has_milestone'] = df['age_first_milestone_year'].notna().astype(int)
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(0)
df['age_last_milestone_year']  = df['age_last_milestone_year'].fillna(0)

# Encode target
df['outcome'] = df['outcome'].map({'acquired': 1, 'closed': 0})

X = df.drop(columns=['outcome'])
y = df['outcome'].values


# =============================================================================
# CELL 3 — HOLD OUT A FINAL TEST SET (vault — touched once at the very end)
# =============================================================================
# We keep 20% as a completely untouched final test set.
# Cross-validation runs on the remaining 80% only.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Development set : {X_dev.shape[0]} samples")
print(f"Final test set  : {X_test.shape[0]} samples")
print(f"Test class balance — Closed: {(y_test==0).sum()}, Acquired: {(y_test==1).sum()}")


# =============================================================================
# CELL 4 — CROSS-VALIDATION SETUP
# =============================================================================
# StratifiedKFold preserves class proportions in every fold.
# 5 folds on ~738 samples = ~590 train / ~148 val per fold.
N_FOLDS  = 5
N_CLUSTERS = 6
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Helper: one-hot encode cluster labels and append to feature matrix
def add_cluster_ohe(X_scaled, clusters, n_clusters=N_CLUSTERS):
    ohe = np.zeros((len(clusters), n_clusters))
    for i, c in enumerate(clusters):
        ohe[i, c] = 1
    return np.hstack([X_scaled, ohe])

# Models — defined once, cloned fresh each fold via sklearn clone
from sklearn.base import clone

baseline_models = {
    'LogReg' : LogisticRegression(max_iter=1000, random_state=42),
    'SVM'    : SVC(kernel='rbf', probability=True, random_state=42),
    'MLP'    : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100,
                              early_stopping=True, random_state=42),
}

hybrid_models = {
    'LR + KMeans'  : LogisticRegression(max_iter=1000, random_state=42),
    'SVM + KMeans' : SVC(kernel='rbf', probability=True, random_state=42),
    'MLP + KMeans' : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100,
                                    early_stopping=True, random_state=42),
}


# =============================================================================
# CELL 5 — RUN CROSS-VALIDATION
# =============================================================================
# Results stored as {model_name: {'f1': [...], 'auc': [...]}}
results = {name: {'f1': [], 'auc': []} for name in list(baseline_models) + list(hybrid_models)}

X_dev_arr = X_dev.values  # numpy array for indexing with fold indices

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_arr, y_dev)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    X_tr, X_val = X_dev_arr[train_idx], X_dev_arr[val_idx]
    y_tr, y_val = y_dev[train_idx],     y_dev[val_idx]

    # --- Scale: fit on train fold only ---
    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)      # transform only — no leakage

    # --- Baseline models ---
    for name, model_template in baseline_models.items():
        model = clone(model_template)
        model.fit(X_tr_scaled, y_tr)

        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        f1  = f1_score(y_val, y_pred, average='macro')
        auc = roc_auc_score(y_val, y_prob)

        results[name]['f1'].append(f1)
        results[name]['auc'].append(auc)
        print(f"  {name:<15} F1={f1:.3f}  AUC={auc:.3f}")

    # --- K-Means: fit on train fold only ---
    # This is the critical step — K-Means must NOT see the validation fold
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(X_tr_scaled)

    tr_clusters  = km.predict(X_tr_scaled)
    val_clusters = km.predict(X_val_scaled)

    X_tr_hybrid  = add_cluster_ohe(X_tr_scaled,  tr_clusters)
    X_val_hybrid = add_cluster_ohe(X_val_scaled, val_clusters)

    # --- Hybrid models ---
    for name, model_template in hybrid_models.items():
        model = clone(model_template)
        model.fit(X_tr_hybrid, y_tr)

        y_pred = model.predict(X_val_hybrid)
        y_prob = model.predict_proba(X_val_hybrid)[:, 1]

        f1  = f1_score(y_val, y_pred, average='macro')
        auc = roc_auc_score(y_val, y_prob)

        results[name]['f1'].append(f1)
        results[name]['auc'].append(auc)
        print(f"  {name:<15} F1={f1:.3f}  AUC={auc:.3f}")


# =============================================================================
# CELL 6 — SUMMARISE CV RESULTS
# =============================================================================
print("\n" + "="*65)
print(f"{'Model':<20} {'F1 Mean':>10} {'F1 Std':>10} {'AUC Mean':>10} {'AUC Std':>10}")
print("="*65)

summary = {}
for name, scores in results.items():
    f1_mean  = np.mean(scores['f1'])
    f1_std   = np.std(scores['f1'])
    auc_mean = np.mean(scores['auc'])
    auc_std  = np.std(scores['auc'])
    summary[name] = {'f1_mean': f1_mean, 'f1_std': f1_std,
                     'auc_mean': auc_mean, 'auc_std': auc_std}
    print(f"{name:<20} {f1_mean:>10.3f} {f1_std:>10.3f} {auc_mean:>10.3f} {auc_std:>10.3f}")


# =============================================================================
# CELL 7 — VISUALISE CV RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(results.keys())
f1_means    = [summary[n]['f1_mean']  for n in model_names]
f1_stds     = [summary[n]['f1_std']   for n in model_names]
auc_means   = [summary[n]['auc_mean'] for n in model_names]
auc_stds    = [summary[n]['auc_std']  for n in model_names]

colors = ['#4C72B0','#4C72B0','#4C72B0',   # baselines — blue
          '#DD8452','#DD8452','#DD8452']     # hybrids   — orange

for ax, means, stds, title in [
    (axes[0], f1_means,  f1_stds,  'F1-Macro (mean ± std across 5 folds)'),
    (axes[1], auc_means, auc_stds, 'ROC-AUC  (mean ± std across 5 folds)'),
]:
    bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black')
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0.5, 0.9)
    ax.set_xticklabels(model_names, rotation=25, ha='right')
    ax.set_ylabel('Score')
    # Annotate bar tops with mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4C72B0', label='Baseline'),
                   Patch(facecolor='#DD8452', label='KMeans Hybrid')]
axes[0].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# CELL 8 — FINAL TEST SET EVALUATION
# (Only run this once you are happy with your CV results — this is the vault)
# =============================================================================
# Scale using the full development set
final_scaler = StandardScaler()
X_dev_scaled  = final_scaler.fit_transform(X_dev_arr)
X_test_scaled = final_scaler.transform(X_test.values)

# K-Means on full development set
final_km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
final_km.fit(X_dev_scaled)

dev_clusters  = final_km.predict(X_dev_scaled)
test_clusters = final_km.predict(X_test_scaled)

X_dev_hybrid  = add_cluster_ohe(X_dev_scaled,  dev_clusters)
X_test_hybrid = add_cluster_ohe(X_test_scaled, test_clusters)

print("\n" + "="*55)
print("FINAL TEST SET RESULTS")
print("="*55)

all_final_models = {
    'LogReg'       : (LogisticRegression(max_iter=1000, random_state=42),        X_dev_scaled,  X_test_scaled),
    'SVM'          : (SVC(kernel='rbf', probability=True, random_state=42),       X_dev_scaled,  X_test_scaled),
    'MLP'          : (MLPClassifier(hidden_layer_sizes=(128,64), max_iter=100,
                                    early_stopping=True, random_state=42),        X_dev_scaled,  X_test_scaled),
    'LR + KMeans'  : (LogisticRegression(max_iter=1000, random_state=42),        X_dev_hybrid,  X_test_hybrid),
    'SVM + KMeans' : (SVC(kernel='rbf', probability=True, random_state=42),       X_dev_hybrid,  X_test_hybrid),
    'MLP + KMeans' : (MLPClassifier(hidden_layer_sizes=(128,64), max_iter=100,
                                    early_stopping=True, random_state=42),        X_dev_hybrid,  X_test_hybrid),
}

for name, (model, X_tr_, X_te_) in all_final_models.items():
    model.fit(X_tr_, y_dev)
    y_pred = model.predict(X_te_)
    y_prob = model.predict_proba(X_te_)[:, 1]

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred, target_names=['Closed', 'Acquired']))
    print(f"F1 Macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix — {name}')
    plt.show()

