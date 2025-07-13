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

class CustomData:
    def __init__(
        self,
        time_spent_alone: float,
        stage_fear: str,
        social_event_attendance: int,
        going_outside: int,
        drained_after_socializing: str,
        friends_circle_size: int,
        post_frequency: int
        ):

        self.time_spent_alone = time_spent_alone
        self.stage_fear = stage_fear
        self.social_event_attendance = social_event_attendance
        self.going_outside = going_outside
        self.drained_after_socializing = drained_after_socializing
        self.friends_circle_size = friends_circle_size
        self.post_frequency = post_frequency

    def get_data_as_df(self):
        try:
            data_dict = {
                "Time_spent_Alone": [self.time_spent_alone],
                "Stage_fear": [self.stage_fear],
                "Social_event_attendance": [self.social_event_attendance],
                "Going_outside": [self.going_outside],
                "Drained_after_socializing": [self.drained_after_socializing],
                "Friends_circle_size": [self.friends_circle_size],
                "Post_frequency": [self.post_frequency]
            }

            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)