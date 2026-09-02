import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    # Define where raw, training, and testing data will be stored
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )
    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )
    raw_data_path: str = os.path.join(
        "artifacts", "data.csv"
    )


class DataIngestion:

    def __init__(self):
        # Create configuration object
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started data ingestion")

        try:
            # Read the original dataset
            df = pd.read_csv("notebook/data/stud.csv")

            # Create artifacts directory if it does not exist
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save the original/raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            # Split data into 80% training and 20% testing
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save training and testing datasets
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Data ingestion completed")

            # Return paths for the next stage
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    # ---------------- DATA INGESTION ----------------
    obj = DataIngestion()

    train_data, test_data = (
        obj.initiate_data_ingestion()
    )


    # ---------------- DATA TRANSFORMATION ----------------
    data_transformation = DataTransformation()

    train_arr, test_arr, processor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )


    # ---------------- MODEL TRAINING ----------------
    model_trainer = ModelTrainer()

    print(
        model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )