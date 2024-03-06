import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.models.model_utils import (eval_model, load_current_active_model,
                                    load_pipeline, plot_residuals)
from utils import get_model_path

logging.basicConfig(level=logging.INFO)


class ModelPredictor:
    def __init__(self, predict_params: dict, features_params: dict):
        self.data = None
        self.Y = None
        self.predictions = None
        self.predict_params = predict_params
        self.predict_year = predict_params["chosen_year"]
        self.predict_color = predict_params["chosen_color"]
        self.features_params = features_params
        pass

    def load_data(self):
        self.data = pd.read_json(self.predict_params["api_dict"][self.predict_year][self.predict_color])[
            self.features_params["cols_to_proceed"]
        ]

    def make_predictions(self):
        pipeline = load_pipeline(get_model_path() / "pipelines", self.features_params)
        model = load_current_active_model(get_model_path() / "main_models")
        data_transformed = pipeline.transform(self.data)
        X = data_transformed.drop(columns=["fare_amount"])
        self.Y = data_transformed["fare_amount"]
        self.predictions = model.predict(X)
        eval_model(model, X, self.Y, f"{self.predict_year} {self.predict_color} Taxi")

    def run_model_prediction(self):
        self.load_data()
        self.make_predictions()
        plot_residuals(self.Y, self.predictions, title=f"{self.predict_year} {self.predict_color} Taxi")
