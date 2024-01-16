import logging

import holidays
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


class DtypeConverter(BaseEstimator, TransformerMixin):
    """
    Author: Igor Adamiec
    Date: 9.01.2024
    Description: DTransformer that converts not numerical dtypes to numerical ones.
    """

    def __init__(self):
        logging.info("Conversion to data types readable by model API")
        self.dt_day_part_encoder = OrdinalEncoder(
            categories=[
                [
                    "dawn",
                    "early morning",
                    "late morning",
                    "noon",
                    "afternoon",
                    "evening",
                    "night",
                    "midnight",
                ]
            ],
            handle_unknown="error",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.dt_day_part_encoder.fit(X[["dt_day_part"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["pickup_datetime"] = pd.to_numeric(X["pickup_datetime"])
        X["dt_day_part"] = self.dt_day_part_encoder.transform(X[["dt_day_part"]])

        return X
