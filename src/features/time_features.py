import logging

import holidays
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def day_part(hour):
    if 4 <= hour < 6:
        return "dawn"
    elif 6 <= hour < 8:
        return "early morning"
    elif 8 <= hour < 11:
        return "late morning"
    elif 11 <= hour < 14:
        return "noon"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    elif 20 <= hour < 23:
        return "night"
    else:
        return "midnight"


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Author: Dawid Kowalczyk
    Date: 31 October 2023
    Description: Date & Time feature engineering
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        us_holidays = holidays.US()

        X["dt_month"] = X["pickup_datetime"].dt.month
        X["dt_weekday"] = X["pickup_datetime"].dt.weekday + 1
        X["dt_isweekend"] = X["dt_weekday"] >= 5
        X["dt_hour"] = X["pickup_datetime"].dt.hour
        X["dt_quarter"] = X["pickup_datetime"].dt.quarter
        X["dt_day_part"] = X["dt_hour"].apply(day_part)
        X["dt_isusholiday"] = X["pickup_datetime"].isin(us_holidays)
        X["dt_paydays"] = (X["pickup_datetime"].dt.day <= 3) | (X["pickup_datetime"].dt.day >= 28)

        logging.info("Datetime feature calculated.")

        return X
