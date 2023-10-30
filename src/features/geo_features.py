import logging
import pandas as pd
from haversine import haversine
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class GeoDataTransformer(BaseEstimator, TransformerMixin):
    """
    Author: Dawid Kowalczyk
    Date: 30 October 2023
    Description: Geographical long and lat data used to
    calculate distance metrics between observations in 2D space
    """

    # those should be passed in Pipeline
    # pickup_cols = ["pickup_latitude", "pickup_longitude"]
    # dropoff_cols = ["dropoff_latitude", "dropoff_longitude"]

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        return self

    def transform(self, X: pd.DataFrame, pickup_cols: list, dropoff_cols: list) -> pd.DataFrame:
        metrics = [
            distance.braycurtis,
            distance.chebyshev,
            distance.cityblock,
            distance.cosine,
            distance.euclidean,
            distance.sqeuclidean,
            haversine
        ]

        for function in metrics:
            X[f'distance_{function.__name__}'] = X.apply(
                lambda row: function(
                    [row[pickup_cols[0]], row[pickup_cols[1]]],
                    [row[dropoff_cols[0]], row[dropoff_cols[1]]],
                ),
                axis=1,
            )

        logging.info("Geographic metrics calculated.")

        return X
