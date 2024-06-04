import numpy as np
import os
import pandas as pd
import dill
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV



from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            
            #model.fit(X_train, y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]]=test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def preprocess_data(X_train, X_test):
        try:
            le = LabelEncoder()
            # Identify categorical columns
            cat_columns = [col for col in range(X_train.shape[1]) if isinstance(X_train[0, col], str)]
            
            for col in cat_columns:
                all_values = list(X_train[:, col]) + list(X_test[:, col])
                le.fit(all_values)
                X_train[:, col] = le.transform(X_train[:, col])
                X_test[:, col] = le.transform(X_test[:, col])

            # Convert to float
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

            return X_train, X_test
        
        except Exception as e:
            raise CustomException(e, sys)