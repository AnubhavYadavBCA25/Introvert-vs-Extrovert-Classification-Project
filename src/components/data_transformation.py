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
        self.transformation_config = DataTransformationConfig()

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
        pass