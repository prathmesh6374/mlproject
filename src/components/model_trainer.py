import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    # Path where the trained model will be saved
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        # Create configuration object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        
    ):
        try:
            logging.info(
                "Splitting training and test input data"
            )

            # Separate input features (X) and target variable (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create a dictionary containing different models
            models = {
                "Random Forest":     RandomForestRegressor(),
                "Decision Tree":     DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors":       KNeighborsRegressor(),
                "XGBoost":           XGBRegressor(),
                "CatBoost":          CatBoostRegressor(verbose=False),
                "AdaBoost":          AdaBoostRegressor(),
            }

            # Evaluate all models and get their R² scores
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Get the highest model score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Get the name of the model with the highest score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get the actual best model from the models dictionary
            best_model = models[best_model_name]

            # Check whether the best model has an acceptable score
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found",
                    sys
                )

            logging.info(
                "Best model found on both training and testing dataset"
            )

            # Save the best trained model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model
            predicted = best_model.predict(X_test)

            # Calculate the R² score of the best model
            test_r2_score = r2_score(
                y_test,
                predicted
            )

            return test_r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pass