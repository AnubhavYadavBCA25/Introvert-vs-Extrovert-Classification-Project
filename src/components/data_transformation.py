import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger.logging import logging
from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    """
    Data Transformation Configuration Class
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    """
    Data Transformation Class
    Handles the transformation of data, including encoding categorical variables,
    scaling numerical features, and saving the preprocessor object.
    """
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer for preprocessing the data.
        Categorical features are encoded, and numerical features are scaled.
        """
        try:
            logging.info("Creating data transformer object")
            categorical_features = ['Stage_fear', 'Drained_after_socializing']
            numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
            logging.info(f"Total Categorical features: {len(categorical_features)} and features: {categorical_features}")
            logging.info(f"Total Numerical features: {len(numerical_features)} and features: {numerical_features}")

            # Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('label_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
                ('scaler', StandardScaler())
            ])

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Column transformer
            preprocessor = ColumnTransformer([
                ('cat_pipeline', categorical_pipeline, categorical_features),
                ('num_pipeline', numerical_pipeline, numerical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = "Personality"

            input_features_train = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            target_feature_train_df = target_feature_train_df.map({
                'Introvert': 1,
                'Extrovert': 0
            })
            
            input_features_test = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            target_feature_test_df = target_feature_test_df.map({
                'Introvert': 1,
                'Extrovert': 0
            })

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed")

            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)