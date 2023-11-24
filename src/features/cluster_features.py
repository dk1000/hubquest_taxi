import logging
import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from utils import get_model_path

logger = logging.getLogger(__name__)


class ClusterBase(BaseEstimator, TransformerMixin):
    """
    Author: Dawid Kowalczyk
    Date: 17 November 2023
    Description: This is a base class to build specific cluster methods upon it
    """

    def __init__(self, n_clusters: int, force_retrain: bool, model_dir: str, token: str):
        self.n_clusters = n_clusters
        self.force_retrain = force_retrain
        self.model_dir = model_dir
        self.token = token
        self.model_path = get_model_path() / self.model_dir
        self.model_name = None
        self.model = None

    def prepare_data(self, input_data) -> pd.DataFrame:
        """An empty shell for data preparation"""
        output_data = input_data
        return output_data

    def visualise_clusters_on_map(self, data: pd.DataFrame, model) -> None:
        """An empty shell for clusters visualisation"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.force_retrain or not (self.model_path / self.model_name).exists():
            logging.info("Retraining forced or model does not exists.")

            data = self.prepare_data(X)

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
        """An empty shell for data transformation"""
        return X


class ClusterLocationTransformer(ClusterBase):
    """
    Author: Dawid Kowalczyk
    Date: 14 November 2023
    Description: Clustering module based on location coordinates
    """

    def __init__(self, n_clusters: int, force_retrain: bool, model_dir: str, geodata_columns: dict, token: str):
        super().__init__(n_clusters, force_retrain, model_dir, token)

        self.model_name = f"coordinates_cluster_model_{self.n_clusters}.pkl"
        self.geodata_columns = geodata_columns
        self.pickup_cols = geodata_columns["pickup_cols"]
        self.dropoff_cols = geodata_columns["dropoff_cols"]

    def prepare_data(self, input_data) -> pd.DataFrame:
        """An empty shell for data preparation"""
        output_data = pd.concat(
            [
                input_data[[self.pickup_cols[0], self.pickup_cols[1]]].rename(
                    columns={self.pickup_cols[0]: "longitude", self.pickup_cols[1]: "latitude"}
                ),
                input_data[[self.dropoff_cols[0], self.dropoff_cols[1]]].rename(
                    columns={self.dropoff_cols[0]: "longitude", self.dropoff_cols[1]: "latitude"}
                ),
            ],
            axis=0,
            ignore_index=True,
        )
        return output_data

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
        px.set_mapbox_access_token(self.token)

        # predict for visualisation
        data.loc[:, "cluster_label"] = model.predict(data)
        fig = px.scatter_mapbox(
            data,
            lat="latitude",
            lon="longitude",
            color="cluster_label",
            size_max=4,
        )
        fig.write_html(str(self.model_path / self.model_name).replace(".pkl", "") + "_map.html")

        pass


class ClusterTripTransformer(ClusterBase):
    """
    Author: Dawid Kowalczyk
    Date: 17 November 2023
    Description: Clustering module based on trip information
    """

    def __init__(self, n_clusters: int, force_retrain: bool, model_dir: str, clustering_columns: list, token: str):
        super().__init__(n_clusters, force_retrain, model_dir, token)

        self.model_name = f"trip_cluster_model_{self.n_clusters}.pkl"
        self.clustering_columns = clustering_columns

    def prepare_data(self, input_data) -> pd.DataFrame:
        """An empty shell for data preparation"""

        output_data = input_data[self.clustering_columns]

        return output_data

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["trip_info_cluster"] = self.model.predict(X[self.clustering_columns])

        logging.info("Trip information clusters calculated.")

        return X

    def visualise_clusters_on_map(self, data: pd.DataFrame, model) -> None:
        cluster_centres_data = model.cluster_centers_

        import plotly.graph_objects as go

        fig = go.Figure()
        for row in cluster_centres_data:
            fig.add_trace(
                go.Scattermapbox(
                    mode="lines+markers",
                    lon=[row[0], row[2]],
                    lat=[row[1], row[3]],
                    marker=dict(
                        size=5,
                        symbol=[None, "marker"],
                        anglesrc="previous",
                    ),
                )
            )
        fig.update_layout(
            hovermode="closest",
            mapbox=dict(
                accesstoken=self.token,
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=cluster_centres_data.mean(axis=0)[1],
                    lon=cluster_centres_data.mean(axis=0)[0],
                ),
                pitch=0,
                zoom=10,
            ),
        )
        fig.update(layout_showlegend=False)
        fig.write_html(str(self.model_path / self.model_name).replace(".pkl", "") + "_map.html")

        pass
