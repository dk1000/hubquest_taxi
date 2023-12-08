import logging

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
import os

from src.features.cluster_features import ClusterLocationTransformer, ClusterTripTransformer
from src.features.geo_features import GeoDataTransformer
from src.features.time_features import DateTimeTransformer
from utils import get_data_path

logging.basicConfig(level=logging.INFO)


class FeaturesBuilder:
    def __init__(
        self,
        geodata: dict,
        clusters_location: dict,
        clusters_trip: dict,
        input_train_data: str,
        input_test_data: str,
        save_data: bool,
        output_file_names: list,
        pipeline_dir: str,
        cols_to_drop: list = None,
    ):
        self.feature_pipeline = Pipeline(
            [
                (
                    "geodata",
                    GeoDataTransformer(**geodata),
                ),
                (
                    "datetime",
                    DateTimeTransformer(),
                ),
                (
                    "clusters_location",
                    ClusterLocationTransformer(**clusters_location),
                ),
                (
                    "clusters_trip_info",
                    ClusterTripTransformer(**clusters_trip),
                ),
            ]
        )
        self.input_train_data = input_train_data
        self.input_test_data = input_test_data
        self.save_data = save_data
        self.output_file_names = output_file_names
        self.cols_to_drop = cols_to_drop
        self.pipeline_dir = pipeline_dir
        self.pipeline_path = get_data_path() / pipeline_dir
        self.pipeline_name = f"feature_pipeline_{clusters_location['n_clusters']}_{clusters_trip['n_clusters']}.pkl"

    def load_data(self):
        logging.info("Loading input data from files")
        path = get_data_path()
        train_data = (
            pd.read_parquet(path.joinpath(self.input_train_data + ".parquet")).drop(columns=self.cols_to_drop).fillna(0)
        )
        test_data = (
            pd.read_parquet(path.joinpath(self.input_test_data + ".parquet")).drop(columns=self.cols_to_drop).fillna(0)
        )
        return train_data, test_data

    def save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        logging.info("Saving processed data to files")
        path = get_data_path() / "processed"
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

    def save_pipeline(self):
        logging.info("Saving pipeline to the file")
        if not os.path.exists(self.pipeline_path):
            os.makedirs(self.pipeline_path)
        joblib.dump(self.feature_pipeline, self.pipeline_path / self.pipeline_name)

    def build_features(self, train_data, test_data):
        logging.info("Start - Building Features")
        logging.info("Step 1. Running train pipeline")
        train_data_prep = self.run_train_pipeline(train_data)
        logging.info("Step 2. Running test pipeline")
        test_data_prep = self.run_test_pipeline(test_data)
        if self.save_data:
            self.save_processed_data(train_data_prep, test_data_prep)
            self.save_pipeline()
        logging.info("End - building Features")
        return train_data_prep, test_data_prep
