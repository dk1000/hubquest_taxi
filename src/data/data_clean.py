import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Author: Igor Adamiec
    Date:1.11.2023
    Description: Class that removes bad/missing/outlier data from the training set
    """

    def __init__(self, iqr_par: float, cols_to_check_na: list = None):
        if cols_to_check_na == "None":
            cols_to_check_na = None

        self.cols_to_check_na = cols_to_check_na
        self.iqr_par = iqr_par
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        logging.info("Cleaning the data")
        logging.info(f"Input data: {len(X)} observations")
        cols_to_iqr = ["fare_amount", "trip_distance"]
        iqr_dict = {col: np.nanquantile(X[col], q=[0.25, 0.75]) for col in cols_to_iqr}

        X = X.loc[
            (X["passenger_count"] > 0)
            & (X["pickup_longitude"].between(-76, -71, inclusive="both"))
            & (X["pickup_latitude"].between(39, 42, inclusive="both"))
            & (X["dropoff_longitude"].between(-76, -71, inclusive="both"))
            & (X["dropoff_latitude"].between(39, 42, inclusive="both"))
            & (X["ratecodeid"].isin([1, 2, 3, 4, 6]))
            & (
                X["fare_amount"].between(
                    0,
                    iqr_dict.get("fare_amount")[1]
                    + self.iqr_par * (iqr_dict.get("fare_amount")[1] - iqr_dict.get("fare_amount")[0]),
                    inclusive="right",
                )
            )
            & (
                X["trip_distance"].between(
                    0,
                    iqr_dict.get("trip_distance")[1]
                    + self.iqr_par * (iqr_dict.get("trip_distance")[1] - iqr_dict.get("trip_distance")[0]),
                    inclusive="right",
                )
            )
        ].dropna(subset=self.cols_to_check_na)
        logging.info(f"Output data: {len(X)} observations")
        return X
