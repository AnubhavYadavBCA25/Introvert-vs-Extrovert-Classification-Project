import os, sys
from  dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger.logging import logging
from src.utils import save_objects, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Ada Boost": AdaBoostClassifier(),
                "Extra Tree": ExtraTreesClassifier(),
                "KNN": KNeighborsClassifier(),
                "Gaussian Naive Bayes": GaussianNB(),
                "XG Boost": XGBClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC()
            }

            params ={
                "Logistic Regression": {
                    "C": [1.0, 0.1, 0.01],
                    "solver": ["lbfgs", "liblinear", "sag", "saga"],
                    "max_iter": [100, 200, 300],
                },

                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },

                "Ada Boost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },

                "Extra Tree": {
                    "n_estimators": [100, 200, 300],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                },

                "Gaussian Naive Bayes": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                },

                "XG Boost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                },

                "SVM": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset.")

            save_objects(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            logging.info(f"Best model found: {best_model_name} with accuracy score: {acc_score}")

            return acc_score

        except Exception as e:
            raise CustomException(e, sys)