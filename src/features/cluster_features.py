import logging
import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from utils import get_model_path

logger = logging.getLogger(__name__)


class ClusterLocationTransformer(BaseEstimator, TransformerMixin):
    """
    Author: Dawid Kowalczyk
    Date: 14 November 2023
    Description: Clustering module based on location coordinates
    """

    def __init__(self, n_clusters: int, force_retrain: bool, model_dir: str, geodata_columns: dict):
        self.n_clusters = n_clusters
        self.force_retrain = force_retrain
        self.model_dir = model_dir
        self.model_path = get_model_path() / self.model_dir
        self.model_name = f"coordinates_cluster_model_{self.n_clusters}.pkl"
        self.geodata_columns = geodata_columns
        self.model = None
        self.pickup_cols = geodata_columns["pickup_cols"]
        self.dropoff_cols = geodata_columns["dropoff_cols"]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.force_retrain or not (self.model_path / self.model_name).exists():
            logging.info("Retraining forced or model does not exists.")

            # prepare data for fit
            data = pd.concat(
                [
                    X[[self.pickup_cols[0], self.pickup_cols[1]]].rename(
                        columns={self.pickup_cols[0]: "longitude", self.pickup_cols[1]: "latitude"}
                    ),
                    X[[self.dropoff_cols[0], self.dropoff_cols[1]]].rename(
                        columns={self.dropoff_cols[0]: "longitude", self.dropoff_cols[1]: "latitude"}
                    ),
                ],
                axis=0,
                ignore_index=True,
            )

            # fit model
            model = KMeans(n_clusters=self.n_clusters, init="k-means++", random_state=2137)
            model.fit(data)

            # save
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            joblib.dump(model, self.model_path / self.model_name)

            # visualise model
            self.visualise_clusters_on_map(data, model)

            # assign model
            self.model = model

        # load
        if self.model is None:
            self.model = joblib.load(self.model_path / self.model_name)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["dropoff_cluster"] = self.model.predict(
            X[[self.dropoff_cols[0], self.dropoff_cols[1]]].rename(
                columns={self.dropoff_cols[0]: "longitude", self.dropoff_cols[1]: "latitude"}
            )
        )

        X["pickup_cluster"] = self.model.predict(
            X[[self.pickup_cols[0], self.pickup_cols[1]]].rename(
                columns={self.pickup_cols[0]: "longitude", self.pickup_cols[1]: "latitude"}
            )
        )

        logging.info("Geographic coordinates clusters calculated.")

        return X

    def visualise_clusters_on_map(self, data: pd.DataFrame, model) -> None:
        # plotly maps
        import plotly.express as px

        px.set_mapbox_access_token(
            "pk.eyJ1IjoiZHhrMDExMSIsImEiOiJjbG5rMnZiMWcwajR4MmpzMzBtbWh4MzUxIn0.Awp5L80wu7nmpdGvOoCunw"
        )

        # predict for visualisation
        data["cluster_label"] = model.predict(data)
        fig = px.scatter_mapbox(
            data,
            lat="latitude",
            lon="longitude",
            color="cluster_label",
            size_max=4,
        )
        fig.write_html(str(self.model_path / self.model_name) + "_clustered_map.html")

        pass
