# The main purpose of Data Transformation is to perform
# feature engineering and feature extraction on the dataset.

import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:

    # Path where preprocessing object will be saved
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        # Create configuration object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Create preprocessing pipelines for numerical
        and categorical columns."""

        try:
            # Numerical columns
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            # Categorical columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical preprocessing
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical preprocessing
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder()
                    ),
                    (
                        "standard",
                        StandardScaler(with_mean=False)
                    )
                ]
            )

            logging.info(
                "Categorical columns encoding completed"
            )

            logging.info(
                "Numerical columns normalization completed"
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Reading train and test data completed"
            )

            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            # Target column
            target_column_name = "math_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            # Fit and transform training data
            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Transform testing data
            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            # Combine transformed features with target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(
                "Data transformation completed"
            )

            # Save preprocessing object
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            # Return transformed arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)