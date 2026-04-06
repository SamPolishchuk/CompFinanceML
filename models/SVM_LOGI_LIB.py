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

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

base_dir  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))



class SVM_LOGIC:
    def __init__(self,
                 data_dir: Path,
                 working_file_name: Path,
                 log_dir: Path,
                 log_file_name: Path,
                 visual_dir: Path = Path("outputs") / "SVM_LOGI",
                 model_type: str = 'SVM',
                 svm_kernel_type: str = 'linear'
                 ):
        self.__data_dir = data_dir
        self.__log_dir = log_dir
        self.__working_file = data_dir / working_file_name
        self.__log_file = log_dir / log_file_name
        self.__data_pd = pd.read_csv(self.__working_file, index_col=0).copy(deep=True)
        self.__log_dir.mkdir(exist_ok=True)
        self.__model_type = model_type.upper()
        # self.__cat_list = ['investor_type', 'sector', 'founder_background']
        self.__feature_names = None
        self.__scaler = StandardScaler()
        self.__output_dir = base_dir / visual_dir
        self.__output_dir.mkdir(parents=True, exist_ok=True)
        self.__svm_kernel_type = svm_kernel_type
    
    # def set_cat_list(self, cat_list):
    #     self.__cat_list = cat_list
    
    def __get_model(self, C: float):
        
        if self.__model_type == 'SVM':
            
            if self.__svm_kernel_type == 'linear':
                return SVC(
                        C=C,
                        kernel='linear',
                        class_weight='balanced',
                        probability=True,
                        random_state=1
                )
            
            elif self.__svm_kernel_type == 'rbf':
                return SVC(
                        C=C,
                        kernel='rbf',
                        gamma='scale',  # important
                        class_weight='balanced',
                        probability=True,
                        random_state=1
                )
            
            elif self.__svm_kernel_type == 'poly':
                return SVC(
                        C=C,
                        kernel='poly',
                        degree=3,
                        coef0=1,
                        class_weight='balanced',
                        probability=True,
                        random_state=1
                )
            
            else:
                raise ValueError(f"Unknown kernel: {self.__svm_kernel_type}")
        
        elif self.__model_type == 'LOGISTIC':
            return LogisticRegression(
                    C=C,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=1
            )
        return None


    def __get_column_stats(self):
        for col in self.__data_pd.columns:
            print(f"\n===== {col} =====")
            
            dist = self.__data_pd[col].value_counts().to_frame(name='count')
            dist['probability'] = dist['count'] / dist['count'].sum()
            print(dist)

    def print_column_name(self):
        print(self.__data_pd.columns.to_list())
        
    def print_outcome(self):
        print(self.__data_pd['outcome'].unique())
        
    def __get_data_shape(self):
        print(f'(row, col) = {self.__data_pd.shape}')
    
    def __get_label(self):
        
        y = self.__data_pd['outcome']
        
        # map explicitly (safe + clear)
        y = y.map(
                 {
                     'closed': 0,
                     'acquired': 1
                 }
        )
        
        assert not pd.isna(y).any(), "Label mapping failed (unknown labels found)"
        
        return y
    
    def __get_clean_data(self):
        X = self.__data_pd.drop(columns=['outcome']).copy()
        
        # ensure numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        self.__feature_names = X.columns
        return X

    def __get_X_Y_for_training(self):
        X, y = self.__get_clean_data(), self.__get_label()
        assert not pd.isna(y).any(), "NaN labels detected!"
        X_np = X.values
        y_np = y.values

        X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np,
                                                            test_size=0.3, random_state=1,
                                                            stratify = y_np
                                                            )
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=1 / 3, random_state=1,
                                                        stratify=y_temp
                                                        )

        X_train = self.__scaler.fit_transform(X_train)
        X_val = self.__scaler.transform(X_val)
        X_test = self.__scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, C_list=(2, 3, 4, 5)):
        log_records = []
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.__get_X_Y_for_training()
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        kernel_tag = f"_{self.__svm_kernel_type}" if self.__model_type == "SVM" else ""
        model_tag = f"{self.__model_type}_{kernel_tag}"
        
        best_C, best_score = None, -np.inf
        
        best_model = None
        for C in C_list:
            print(f"[{self.__model_type}] Training with C = {C}")
            
            model = self.__get_model(C)
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
                        "test_size" : len(X_test),
                        "model"     : self.__model_type,
                    }
            )
            
            if val_score > best_score:
                best_C = C
                best_score = val_score
                best_model = model
                
                print(f'save best_C: {best_C}')
            
            if C != C_list[-1]:
                print('enter next round')
            else:
                print('Exist training loop.')
        
        # ---- FINAL MODEL ----
        model = best_model
        model.fit(X_train, y_train)
        
        test_score = model.score(X_test, y_test)
        
        print("\n==============================")
        print(f'Best C     : {best_C}')
        print(f'Val score  : {best_score:.4f}')
        print(f'Test score : {test_score:.4f}')
        print("\n==============================\n")
        
        # ---- ROC: TRAIN vs TEST ----
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, y_train_prob)
            test_auc = roc_auc_score(y_test, y_test_prob)
            
            print(f"Train AUC: {train_auc:.4f}")
            print(f"Test  AUC: {test_auc:.4f}")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # ---- TRAIN ----
            RocCurveDisplay.from_predictions(
                    y_train, y_train_prob,
                    ax=ax,
                    name=f"Train",
                    color="blue",
                    linestyle="-",
                    linewidth=2
            )
            
            # ---- TEST ----
            RocCurveDisplay.from_predictions(
                    y_test, y_test_prob,
                    ax=ax,
                    name=f"Test",
                    color="red",
                    linestyle="--",
                    linewidth=2
            )
            
            # ---- RANDOM BASELINE ----
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random")
            
            ax.set_title(f"{self.__model_type} ROC Curve (Train vs Test)", fontsize=16)
            ax.set_xlabel("False Positive Rate", fontsize=16)
            ax.set_ylabel("True Positive Rate", fontsize=16)
            
            ax.legend(loc="lower right")
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.__output_dir / f"{model_tag}_roc_train_test.pdf", dpi=300)
            plt.close()
        
        # ---- VISUAL EVALUATION ----
        self.__evaluate_and_plot(model, X_test, y_test)
        
        results = {
                       "model"     : self.__model_type,
                       "best_C"    : best_C,
                       "val_score" : best_score,
                       "test_score": test_score,
                       "f1_macro"  : f1_score(y_test, model.predict(X_test), average='macro')
                  }
        
        pd.DataFrame([results]).to_csv(self.__output_dir / f"{self.__model_type}_results.csv", index=False)
        
        if self.__model_type == 'SVM' and self.__svm_kernel_type == 'linear':
            coef = model.coef_[0]
            print(f'Top SVM Features (Linear):\n{coef}')
            importance = pd.Series(coef, index=self.__feature_names)
            importance = importance.sort_values(key=abs, ascending=False)
            
            print("\n===== Top SVM Features =====")
            print(importance.head(15))
            
            plt.figure(figsize=(6, 4))
            importance.head(15).plot(kind='barh')
            plt.gca().invert_yaxis()
            plt.title("Top SVM Features (Linear)")
            plt.tight_layout()
            plt.savefig(self.__output_dir / "svm_feature_importance.pdf", dpi=300)
            plt.close()
        
        if self.__model_type == 'LOGISTIC':
            coef = model.coef_[0]
            print(f'Top LOGISTIC Features (Linear):\n{coef}')
            importance = pd.Series(coef, index=self.__feature_names)
            importance = importance.sort_values(key=abs, ascending=False)
            
            print("\n===== Top LOGISTIC Features =====")
            print(importance.head(15))
            
            plt.figure(figsize=(6, 4))
            importance.head(15).plot(kind='barh')
            plt.gca().invert_yaxis()
            plt.title("Top Logistic Features")
            plt.tight_layout()
            plt.savefig(self.__output_dir / "logistic_feature_importance.pdf", dpi=300)
            plt.close()
            
        # ---- LOG FINAL ----
        log_records.append(
                {
                    "run_id"    : run_id,
                    "timestamp" : datetime.now(),
                    "C"         : best_C,
                    "val_score" : best_score,
                    "test_score": test_score,
                    "train_size": len(X_train),
                    "val_size"  : len(X_val),
                    "test_size" : len(X_test),
                    "model"     : self.__model_type,
                    "note"      : "final_model"
                }
        )

        # ---- SAVE ONCE ----
        log_df = pd.DataFrame(log_records)
        
        if self.__log_file.exists():
            log_df.to_csv(self.__log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(self.__log_file, index=False)
        
        print(f"\nLog saved to: {self.__log_file}")
    
    def __evaluate_and_plot(self, model, X_test, y_test):
        kernel_tag = f"_{self.__svm_kernel_type}" if self.__model_type == "SVM" else ""
        model_tag = f"{self.__model_type}{kernel_tag}"
        
        y_pred = model.predict(X_test)
        
        # Some models (SVM) need probability=True
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        # ------------------------------
        # Classification report
        # ------------------------------
        print("\n===== Classification Report =====")
        print(classification_report(y_test,
                                    y_pred,
                                    target_names=['closed', 'acquired']))
        
        print("F1 Macro:", f1_score(y_test, y_pred, average='macro'))
        
        if y_prob is not None:
            print("ROC-AUC:", roc_auc_score(y_test, y_prob))
            plt.figure(figsize=(5, 4))
            PrecisionRecallDisplay.from_predictions(y_test, y_prob)
            plt.title(f"{self.__model_type} Precision-Recall Curve")
            plt.grid()
            plt.savefig(self.__output_dir / f"{model_tag}_pr_curve.pdf", dpi=300)
            plt.close()
            
        
        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(16, 9))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['closed', 'acquired'],
                    yticklabels=['closed', 'acquired']
                    )
        
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{model_tag} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.__output_dir / f"{model_tag}_cm.pdf", dpi=300)
        plt.close()
        
        
        # ------------------------------
        # ROC Curve
        # ------------------------------
        if y_prob is not None:
            plt.figure(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.title(f"{model_tag} ROC Curve")
            plt.grid()
            plt.savefig(self.__output_dir / f"{model_tag}_roc.pdf", dpi=300)
            plt.close()
        
        report = classification_report(
                y_test, y_pred,
                target_names=['closed', 'acquired']
        )
        
        with open(self.__output_dir / f"{model_tag}_report.txt", "w") as f:
            f.write(report)


def main():
    data_dir = base_dir / Path("data")
    log_dir = base_dir / Path("logging")
    log_dir.mkdir(exist_ok=True)
    working_file_name = Path('startup_data_cleaned.csv')

    model_SVM = SVM_LOGIC(data_dir=data_dir,
                          working_file_name=working_file_name,
                          log_dir=log_dir,
                          log_file_name = Path('SVM_log.csv'),
                          model_type='SVM')
    model_SVM.print_column_name()
    model_SVM.print_outcome()

    model_LOGI = SVM_LOGIC(data_dir=data_dir,
                           working_file_name=working_file_name,
                           log_dir=log_dir,
                           log_file_name = Path('LOGISTIC_log.csv'),
                           model_type='LOGISTIC')

    
    model_SVM.train(C_list=np.arange(1, 10))
    model_LOGI.train(C_list=(2, 3))

if __name__ == '__main__':
    main()
