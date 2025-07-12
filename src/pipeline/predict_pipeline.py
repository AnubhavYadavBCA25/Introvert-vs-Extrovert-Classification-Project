import pandas as pd
import os, sys
from src.exception import CustomException
from src.utils import load_objects

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_objects(file_path=model_path)
            preprocessor = load_objects(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)