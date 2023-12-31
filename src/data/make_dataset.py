import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.data_clean import DataCleaner
from src.data.data_split import train_test_split
from utils import get_data_path

logging.basicConfig(level=logging.INFO)


class DatasetMaker:
    """
    Author: Igor Adamiec
    Date: 7.11.2023
    Description: Class to split data into train and test sets, and to clean the train set from the bad data and outliers
    """

    def __init__(
        self,
        input_file_name: str,
        test_size: float,
        save_files: bool,
        iqr_par: float,
        output_train_test_file_names: tuple = None,
        output_cleaned_file_name: str = None,
        cleaner_cols_to_check_na: list = None,
    ):
        self.input_file_name = input_file_name + ".parquet"
        self.test_size = test_size
        self.save_files = save_files
        self.output_train_test_file_names = output_train_test_file_names
        self.output_cleaned_file_name = output_cleaned_file_name
        self.preparation_pipeline = Pipeline(
            [("cleaner", DataCleaner(iqr_par=iqr_par, cols_to_check_na=cleaner_cols_to_check_na))]
        )

    def make_dataset(self):
        def save_processed_training_data(data: pd.DataFrame, out_file_name: str):
            data_path = get_data_path() / "interim" / (out_file_name + ".parquet")
            logging.info(f"Cleaned train data saved into directory {str(data_path)}")
            data.to_parquet(data_path)

        logging.info("Start - Making Dataset")
        input_path = get_data_path() / "raw" / self.input_file_name
        data = pd.read_parquet(input_path)
        logging.info("Step 1. Running cleaning pipeline on data")
        data = self.preparation_pipeline.fit_transform(data)

        if self.save_files:
            save_processed_training_data(data, self.output_cleaned_file_name)

        logging.info("Step 2. Splitting data into train and test sets")
        train_data, test_data = train_test_split(
            input_df=data,
            test_size=self.test_size,
            save_files=self.save_files,
            output_files=self.output_train_test_file_names,
        )
        logging.info("Step 2. Running cleaning pipeline on training data")
        train_data_prepared = self.preparation_pipeline.fit_transform(train_data)

        if self.save_files:
            save_processed_training_data(train_data_prepared, self.output_cleaned_file_name)
        logging.info("End - Making Dataset")
        return train_data_prepared, test_data, train_data
