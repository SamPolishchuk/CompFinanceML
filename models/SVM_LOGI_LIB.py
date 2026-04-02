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



class SVM_LOGIC:
    def __init__(self,
                 data_dir: Path,
                 working_file_name: Path,
                 log_dir: Path,
                 log_file_name: Path,
                 model_type: str = 'SVM'):
        self.__data_dir = data_dir
        self.__log_dir = log_dir
        self.__working_file = data_dir / working_file_name
        self.__log_file = log_dir / log_file_name
        self.__data_pd = pd.read_csv(self.__working_file, index_col=0).copy(deep=True)
        self.__log_dir.mkdir(exist_ok=True)
        self.__model_type = model_type.upper()
        self.__cat_list = ['investor_type', 'sector', 'founder_background']
    
    def set_cat_list(self, cat_list):
        self.__cat_list = cat_list

    def __get_model(self, C: float):

        if self.__model_type == 'SVM':
            return SVC(C=C, class_weight='balanced')
        
        elif self.__model_type == 'LOGISTIC':
            return LogisticRegression(C=C,
                                      class_weight='balanced',
                                      max_iter=1000)
        return None
        
    def __get_column_stats(self):
        for col in self.__data_pd.columns:
            print(f"\n===== {col} =====")
            
            dist = self.__data_pd[col].value_counts().to_frame(name='count')
            dist['probability'] = dist['count'] / dist['count'].sum()
            print(dist)

    def __print_column_name(self):
        print(self.__data_pd.columns.to_list())
        
    def __get_data_shape(self):
        print(f'(row, col) = {self.__data_pd.shape}')
        
    def __get_label(self):
        y = self.__data_pd['outcome'].map({'Failure'    : -1,
                                           'Acquisition': 1,
                                           'IPO'        : 1})
        return y

    def __get_clean_data(self):
        X = pd.get_dummies(self.__data_pd.drop(columns=['outcome']), columns= self.__cat_list)
        self.__feature_names = X.columns
        return X

    def __get_X_Y_for_training(self):
        X, y = self.__get_clean_data(), self.__get_label()
        assert not pd.isna(y).any(), "NaN labels detected!"
        X_np = X.values
        y_np = y.values

        X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, C_list=(2, 3, 4, 5)):
        log_records = []
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.__get_X_Y_for_training()
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        best_C, best_score = None, -np.inf
        
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
                best_C, best_score = C, val_score
                print(f'save best_C: {best_C}')
            
            if C != C_list[-1]:
                print('enter next round')
            else:
                print('Exist training loop.')
        
        # ---- FINAL MODEL ----
        model = self.__get_model(best_C)
        model.fit(X_train, y_train)
        
        test_score = model.score(X_test, y_test)
        
        print("\n==============================")
        print(f'Best C     : {best_C}')
        print(f'Val score  : {best_score:.4f}')
        print(f'Test score : {test_score:.4f}')
        print("==============================")
        
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
            

def main():
    data_dir = base_dir / Path("data")
    log_dir = base_dir / Path("logging")
    log_dir.mkdir(exist_ok=True)
    working_file_name = Path('startup_success_dataset.csv')

    model_SVM = SVM_LOGIC(data_dir=data_dir,
                          working_file_name=working_file_name,
                          log_dir=log_dir,
                          log_file_name = Path('SVM_log.csv'),
                          model_type='SVM')

    model_LOGI = SVM_LOGIC(data_dir=data_dir,
                           working_file_name=working_file_name,
                           log_dir=log_dir,
                           log_file_name = Path('LOGISTIC_log.csv'),
                           model_type='LOGISTIC')

    model_SVM.train(C_list=(2, 3))
    model_LOGI.train(C_list=(2, 3))

if __name__ == '__main__':
    main()
