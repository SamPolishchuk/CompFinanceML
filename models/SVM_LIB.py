from pathlib import Path
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
from datetime import datetime
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

base_dir  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))
data_dir = base_dir / "data"
logging_dir = base_dir / "logging"
logging_dir.mkdir(exist_ok=True)
working_file_name = 'startup_success_dataset.csv'

startUp_pd = pd.read_csv(data_dir / working_file_name, index_col=0).copy(deep=True)

# print(type(startUp_pd))
dim = startUp_pd.shape
print(f'\n(rows, cols) = ({dim[0]} {dim[1]})\n')
print(startUp_pd.columns)

# for row in range(20):
#         for col in startUp_pd.columns:
#                 print(f'{startUp_pd.iloc[row][col]}')
#         print("\n=========================\n")
#
#
# startUp_pd['outcome'] = startUp_pd['outcome'].map({'Failure': -1,
#                                                                                                    'Acquisition': 1,
#                                                                                                    'IPO': 1})
#
#
#
# for row in range(20):
#         print(startUp_pd.iloc[row]['outcome'])
categorical_cols = [
                    'investor_type',
                    'sector',
                    'founder_background',
                    'outcome'
]

for col in categorical_cols:
    print(f"\n===== {col} =====")
    
    dist = startUp_pd[col].value_counts().to_frame(name='count')
    dist['probability'] = dist['count'] / dist['count'].sum()
    
    print(dist)

# ---- label ----
y = startUp_pd['outcome'].map({'Failure'    : -1,
                               'Acquisition': 1,
                               'IPO'        : 1})

# encode and drop label
X = pd.get_dummies(startUp_pd.drop(columns=['outcome']), columns=['investor_type', 'sector', 'founder_background'])



print(f'X: (row, col) = {X.shape}')
print(f'y: (row, col) = {y.shape}')

# ---- convert ----
X_np = X.values
y_np = y.values


# split
X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=1)


# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# tune
file_name = 'svm_log.csv'
log_file = logging_dir / file_name
run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

log_records = []

best_C, best_score = None, -np.inf
for C in [2, 3, 4, 5]:
    print(f'Training with C = {C}')
    model = SVC(C=C, class_weight='balanced')
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"C={C}, val_score={val_score:.4f}")
    
    log_records.append(
            {
                "run_id"    : run_id,
                "timestamp" : datetime.now(),
                "C"         : C,
                "val_score" : val_score,
                "train_size": len(X_train),
                "val_size"  : len(X_val),
                "test_size" : len(X_test)
            }
    )

    if val_score > best_score:
        best_C, best_score = C, val_score
        print(f'save best_C: {best_C}')
        
    print('enter next round')

# final
model = SVC(C=best_C, class_weight='balanced')
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)

print("\n==============================")
print(f'Best C     : {best_C}')
print(f'Val score  : {best_score:.4f}')
print(f'Test score : {test_score:.4f}')
print("==============================")

log_records.append({
    "run_id"    : run_id,
    "timestamp" : datetime.now(),
    "C"         : best_C,
    "val_score" : best_score,
    "test_score": test_score,
    "train_size": len(X_train),
    "val_size"  : len(X_val),
    "test_size" : len(X_test),
    "note"      : "final_model"
})


log_df = pd.DataFrame(log_records)

if log_file.exists():
    log_df.to_csv(log_file, mode='a', header=False, index=False)
else:
    log_df.to_csv(log_file, index=False)

print(f"\nLog saved to: {log_file}")