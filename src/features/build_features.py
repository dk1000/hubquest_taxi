import logging
from pathlib import Path

import pandas as pd
from geo_features import GeoDataTransformer
from sklearn.pipeline import Pipeline
from time_features import DateTimeTransformer

logging.basicConfig(level=logging.INFO)

feature_pipeline = Pipeline(
    [
        (
            "geodata",
            GeoDataTransformer(
                # TODO: arguments to be passed from config.yml file
                pickup_cols=["pickup_latitude", "pickup_longitude"],
                dropoff_cols=["dropoff_latitude", "dropoff_longitude"],
            ),
        ),
        (
            "datetime",
            DateTimeTransformer(),
        ),
    ]
)


def load_data(data_path: str):
    path = (Path(__file__).parent.parent.parent / "data").joinpath(data_path)
    return pd.read_parquet(path)


def save_data(file: pd.DataFrame, file_name: str):
    path = Path(__file__).parent.parent.parent / "data" / "processed" / (file_name + ".parquet")
    file.to_parquet(path)


def run_train_pipeline(data_path: str, out_file_name: str):
    train_data = load_data(data_path)
    train_data_prep = feature_pipeline.fit_transform(train_data)
    save_data(train_data_prep, out_file_name)
    return train_data_prep


def run_test_pipeline(data_path: str, out_file_name: str):
    test_data = load_data(data_path)
    test_data_prep = feature_pipeline.transform(test_data)
    save_data(test_data_prep, out_file_name)
    return test_data_prep


if __name__ == "__main__":
    logging.info("Building features")
    logging.info("Running train pipeline")
    train_data = run_train_pipeline("raw/train_data.parquet", "train_data_proc")
    logging.info("Running test pipeline")
    test_data = run_test_pipeline("raw/test_data.parquet", "test_data_proc")
