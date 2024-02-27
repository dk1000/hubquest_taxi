import logging

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.models.model_utils import (
    compare_current_model_with_active,
    eval_model,
    hash_model_name,
    load_current_active_model,
    save_model,
    set_model_as_active,
)
from utils import get_data_path, get_model_path

logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, optim_iters: int, cv_splits: int, parameters: dict, model_columns: list):
        self.optim_iters = optim_iters
        self.cv_splits = cv_splits
        self.parameters = parameters
        self.model_cols = model_columns
        self.model = None
        self.model_best_params = None
        self.model_path = get_model_path() / "main_models"
        self.model_name = None

    def train_model(self, x_train, y_train):
        cv_splitter = TimeSeriesSplit(n_splits=self.cv_splits)

        def objective(trial):
            model_name = trial.suggest_categorical(
                "regressor",
                ["LGBM", "CatBoost"],
            )

            if model_name == "LGBM":
                num_leaves = trial.suggest_int(
                    "LGBM_num_leaves",
                    self.parameters["LGBM"]["num_leaves"]["min"],
                    self.parameters["LGBM"]["num_leaves"]["max"],
                )
                max_depth = trial.suggest_categorical("LGBM_max_depth", self.parameters["LGBM"]["max_depth"])
                learning_rate = trial.suggest_float(
                    "LGBM_learning_rate",
                    self.parameters["LGBM"]["learning_rate"]["min"],
                    self.parameters["LGBM"]["learning_rate"]["max"],
                    log=True,
                )
                n_estimators = trial.suggest_int(
                    "LGBM_n_estimators",
                    self.parameters["LGBM"]["n_estimators"]["min"],
                    self.parameters["LGBM"]["n_estimators"]["max"],
                )
                model = LGBMRegressor(
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=1,
                    verbose=-1,
                )
            elif model_name == "CatBoost":
                max_depth = trial.suggest_int(
                    "CB_max_depth", self.parameters["CB"]["max_depth"]["min"], self.parameters["CB"]["max_depth"]["max"]
                )
                learning_rate = trial.suggest_float(
                    "CB_learning_rate",
                    self.parameters["CB"]["learning_rate"]["min"],
                    self.parameters["CB"]["learning_rate"]["max"],
                    log=True,
                )
                n_estimators = trial.suggest_int(
                    "CB_n_estimators",
                    self.parameters["CB"]["n_estimators"]["min"],
                    self.parameters["CB"]["n_estimators"]["max"],
                )
                model = CatBoostRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_seed=1,
                    verbose=False,
                )

            score = cross_val_score(
                model, x_train, y_train, n_jobs=-1, cv=cv_splitter, scoring="neg_mean_absolute_error"
            )
            cv_mae = score.mean()
            return cv_mae

        def create_model_based_on_best_params(best_params: dict):
            if best_params["regressor"] == "LGBM":
                best_params = {
                    key.replace("LGBM_", ""): best_params[key] for key in best_params.keys() if key != "regressor"
                }
                best_params["random_state"] = 1
                best_params["verbose"] = -1
                model = LGBMRegressor(**best_params)
            elif best_params["regressor"] == "CatBoost":
                best_params = {
                    key.replace("CB_", ""): best_params[key] for key in best_params.keys() if key != "regressor"
                }
                best_params["random_seed"] = 1
                best_params["verbose"] = False
                model = CatBoostRegressor(**best_params)
            return model

        sampler = optuna.samplers.TPESampler(seed=1)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.optim_iters, n_jobs=-1)
        self.model_best_params = study.best_params

        model = create_model_based_on_best_params(self.model_best_params)
        model.fit(x_train, y_train)
        logging.info(f"Model trained with following params: {self.model_best_params}")
        self.model = model
        self.model_name = hash_model_name(self.model_best_params)

        save_model(self.model, self.model_path, self.model_name)

    def check_with_active_model(self, x_test, y_test):
        logging.info(f"Comparing new model with the Active one")

        active_model = load_current_active_model(self.model_path)
        if active_model is None:
            set_model_as_active(self.model, self.model_path, self.model_name)
        else:
            compare_current_model_with_active(
                x_test, y_test, self.model, active_model, self.model_path, self.model_name
            )

    @staticmethod
    def load_data(train_name, test_name):
        logging.info("Loading files with processed_data")
        train_data = pd.read_parquet(get_data_path() / "processed" / f"{train_name}.parquet")
        test_data = pd.read_parquet(get_data_path() / "processed" / f"{test_name}.parquet")
        return train_data, test_data

    def run_model_training(self, train_data, test_data):
        train_data.sort_values("pickup_datetime", inplace=True)
        train_data = train_data[self.model_cols]
        test_data = test_data[self.model_cols]
        x_train = train_data.drop(columns=["fare_amount"])
        y_train = train_data["fare_amount"]
        x_test = test_data.drop(columns=["fare_amount"])
        y_test = test_data["fare_amount"]

        self.train_model(x_train, y_train)
        eval_model(self.model, x_train, y_train, "Train")
        eval_model(self.model, x_test, y_test, "Test")
        self.check_with_active_model(x_test, y_test)
