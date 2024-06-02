import sys
import os
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def  get_data_transformer_object(self):
        
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_features=['writing_score','reading_score']
            categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    
                    ("scaler",StandardScaler())
                ]
                
            )   
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                    
                    ("scaler",StandardScaler(with_mean=False))
                    
                ]
            )   
            
            logging.info(f"Categorical columns:{categorical_features}")
            logging.info(f"Numerical columns:{numerical_features}")
            
            
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_features),
                    ("cat_pipeline",cat_pipeline, categorical_features)
                    
                ]
            )
            
            return preprocessor
        
        
        except Exception as e:
            
            raise CustomException(e,sys)
    
    def initiate_data_Transformation(self, train_path, test_path):
        
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            
            numerical_features=['writing_score','reading_score']
            
            input_feature_train=train_df.drop(columns=[target_column_name],axis=1)
            output_feature_train=train_df[target_column_name]
            
            input_feature_test=test_df.drop(columns=[target_column_name],axis=1)
            output_feature_test=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            
            input_train=preprocessing_obj.fit_transform(input_feature_train)
            input_test=preprocessing_obj.transform(input_feature_test)
            
            train_array=np.c_[input_feature_train, np.array(output_feature_train)]
            test_array=np.c_[input_feature_test,np.array(output_feature_test)]
            
            logging.info(f"saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)