import logging
from pathlib import Path

import pandas as pd
from data_clean import DataCleaner
from data_split import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)

# TODO: cleaner_cols_to_check_na should be read from config.yml - those should be columns used by preprocessing/model
preparation_pipeline = Pipeline([("cleaner", DataCleaner())]).set_params(cleaner__cols_to_check_na=None)


def save_processed_training_data(data: pd.DataFrame, out_file_name: str):
    data_path = Path(__file__).parent.parent.parent / "data" / "interim" / (out_file_name + ".parquet")
    data.to_parquet(data_path)


if __name__ == "__main__":
    print("Making dataset")
    logging.info("Making dataset")
    logging.info("Splitting data into train and test sets")
    # TODO: arguments reading from config file
    train_data, test_data = train_test_split(
        input_file="final_taxi_data.parquet",
        test_size=0.3,
        save_files=True,
        output_files=("train_data", "test_data"),
    )
    train_data_prepared = preparation_pipeline.fit_transform(train_data)
    save_processed_training_data(train_data_prepared, "train_data_clean")
