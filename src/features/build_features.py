import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.geo_features import GeoDataTransformer
from src.features.time_features import DateTimeTransformer

logging.basicConfig(level=logging.INFO)


class FeaturesBuilder:
    def __init__(
        self,
        pickup_cols: list,
        dropoff_cols: list,
        input_train_data: str,
        input_test_data: str,
        save_data: bool,
        output_file_names: list,
    ):
        self.feature_pipeline = Pipeline(
            [
                (
                    "geodata",
                    GeoDataTransformer(
                        pickup_cols=pickup_cols,
                        dropoff_cols=dropoff_cols,
                    ),
                ),
                (
                    "datetime",
                    DateTimeTransformer(),
                ),
            ]
        )
        self.input_train_data = input_train_data
        self.input_test_data = input_test_data
        self.save_data = save_data
        self.output_file_names = output_file_names

    def load_data(self):
        logging.info("Loading input data from files")
        path = Path(__file__).parent.parent.parent / "data"
        train_data = pd.read_parquet(path.joinpath(self.input_train_data + ".parquet"))
        test_data = pd.read_parquet(path.joinpath(self.input_test_data + ".parquet"))
        return train_data, test_data

    def save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        logging.info("Saving processed data to files")
        path = Path(__file__).parent.parent.parent / "data" / "processed"
        train_data.to_parquet(path.joinpath(self.output_file_names[0] + ".parquet"))
        logging.info(f"Saved processed train data to {str(path.joinpath(self.output_file_names[0] + '.parquet'))}")
        test_data.to_parquet(path.joinpath(self.output_file_names[1] + ".parquet"))
        logging.info(f"Saved processed test data to {str(path.joinpath(self.output_file_names[1] + '.parquet'))}")

    def run_train_pipeline(self, train_data):
        train_data_prep = self.feature_pipeline.fit_transform(train_data)
        return train_data_prep

    def run_test_pipeline(self, test_data):
        logging.info("Running test pipeline")
        test_data_prep = self.feature_pipeline.transform(test_data)
        return test_data_prep

    def build_features(self, train_data, test_data):
        logging.info("Start - Building Features")
        logging.info("Step 1. Running train pipeline")
        train_data_prep = self.run_train_pipeline(train_data)
        logging.info("Step 2. Running test pipeline")
        test_data_prep = self.run_test_pipeline(test_data)
        if self.save_data:
            self.save_processed_data(train_data_prep, test_data_prep)
        logging.info("End - building Features")
        return train_data_prep, test_data_prep
