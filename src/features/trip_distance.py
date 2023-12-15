import logging
import os

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from utils import get_model_path

logger = logging.getLogger(__name__)


class ModelTripDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Author: Dawid Kowalczyk
    Date: 12 December 2023
    Description: Predicting trip_distance based on other features, to have additional input for final model
    """

    def __init__(
            self,
            force_retrain: bool,
            model_dir: str,
            model_max_depth: int,
            model_columns: list,
            grid_search_parameters: dict,
    ):
        self.force_retrain = force_retrain
        self.model_dir = model_dir
        self.model_path = get_model_path() / self.model_dir
        self.model_columns = model_columns
        self.max_depth = model_max_depth
        self.grid_search_parameters = grid_search_parameters
        self.model_name = f"trip_distance_model.pkl"
        self.model = None
        logging.info("Model Trip Distance features calculation.")

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.force_retrain or not (self.model_path / self.model_name).exists():
            logging.info("Retraining forced or model does not exists.")

            y = X["trip_distance"]
            X = X[self.model_columns]

            # parameters optimisation
            model = LGBMRegressor(random_state=0, verbose=-1)
            logging.info(f"Running GridSearch with provided parameters (it will take some time...):"
                         f"\n{self.grid_search_parameters}")
            grid = GridSearchCV(estimator=model, param_grid=self.grid_search_parameters, cv=3, n_jobs=-1)
            grid.fit(X, y)

            best_params = grid.best_params_
            logging.info(f"Best parameters for trip_distance model from GridSearch:\n{best_params}")

            # fit model
            best_params.update({'random_state': 0, 'verbose': -1})
            model = LGBMRegressor(**best_params)
            model.fit(X, y)

            # evaluate model
            y_pred = model.predict(X)

            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            logging.info(f"Model performance MAE: {mae}")
            logging.info(f"Model performance MSE: {mse}")
            logging.info(f"Model performance R2: {r2}")

            # save
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            joblib.dump(model, self.model_path / self.model_name)

            # assign model
            self.model = model

        # load
        if self.model is None:
            self.model = joblib.load(self.model_path / self.model_name)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["trip_distance_prediction"] = self.model.predict(X[self.model_columns])

        logging.info("trip_distance feature calculated based on model prediction.")

        return X
