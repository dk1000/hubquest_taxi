import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from utils import get_model_path, get_data_path
import hashlib
import joblib
from datetime import datetime
import os
import shutil


import optuna

import logging

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

        sampler = optuna.samplers.TPESampler(seed=1)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.optim_iters, n_jobs=-1)
        self.model_best_params = study.best_params

        model = create_model_based_on_best_params(self.model_best_params)
        model.fit(x_train, y_train)
        logging.info(f"Model trained with following params: {self.model_best_params}")
        self.model = model
        params = str(self.model_best_params)
        hashed_params = hashlib.md5(params.encode()).hexdigest()
        today = datetime.today().strftime("%Y%m%d%H%M%S")
        self.model_name = f"{today}_{hashed_params}.pkl"

        def save_model():
            logging.info(f"Saving model to the file")

            joblib.dump(self.model, self.model_path / self.model_name)
            logging.info(f"Saved model as {self.model_name}")

        save_model()

        pred_values = self.model.predict(x_train)
        train_mae = mean_absolute_error(y_train, pred_values)
        train_r2 = r2_score(y_train, pred_values)
        logging.info(f"Train MAE: {round(train_mae, 3)}\nTrain R2: {round(train_r2, 3)}")

    def eval_model(self, x_test, y_test):
        pred_values = self.model.predict(x_test)
        test_mae = mean_absolute_error(y_test, pred_values)
        test_r2 = r2_score(y_test, pred_values)
        logging.info(f"Test MAE: {round(test_mae, 3)}\nTest R2: {round(test_r2, 3)}")

    def compare_new_model_with_active(self, x_test, y_test):
        logging.info(f"Comparing new model with the Active one")

        def load_current_active_model():
            if not os.path.exists(self.model_path / "active"):
                os.makedirs(self.model_path / "active")
            for root, dirs, files in os.walk(self.model_path / "active"):
                if len(files) == 0:
                    logging.info(f"No active models found")
                    return None
                else:
                    return joblib.load(self.model_path / "active" / files[0])

        def set_model_as_active(model, path, name):
            # todo: add comparison with current active model
            if not os.path.exists(path / "active"):
                os.makedirs(path / "active")
            for root, dirs, files in os.walk(path / "active"):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
            logging.info(f"Making new model as active")
            for root, dirs, files in os.walk(path / "active"):
                print(files)
            joblib.dump(model, path / "active" / name)

        active_model = load_current_active_model()
        if active_model is None:
            set_model_as_active(self.model, self.model_path, self.model_name)
        else:
            current_pred = self.model.predict(x_test)
            current_mae = mean_absolute_error(y_test, current_pred)
            logging.info(f"Test MAE for current model {round(current_mae, 3)}")

            active_pred = active_model.predict(x_test)
            active_mae = mean_absolute_error(y_test, active_pred)
            logging.info(f"Test MAE for active model {round(active_mae, 3)}")

            if current_mae < active_mae:
                set_model_as_active(self.model, self.model_path, self.model_name)
            else:
                logging.info(f"Current active model is better than new one. No changes on Active Model")

    @staticmethod
    def load_data():
        logging.info("Loading files with processed_data")
        train_data = pd.read_parquet(get_data_path() / "processed" / "train_data_processed.parquet")
        test_data = pd.read_parquet(get_data_path() / "processed" / "test_data_processed.parquet")
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
        self.eval_model(x_test, y_test)
        self.compare_new_model_with_active(x_test, y_test)
