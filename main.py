import logging

import yaml

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.models.predict_model import ModelPredictor
from src.models.train_model import ModelTrainer

logging.basicConfig(level=logging.INFO)

with open("config.yaml", "r") as cfg:
    try:
        config = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        logging.error(exc)


if __name__ == "__main__":
    if config["general"]["make_dataset"]:
        dataset_maker = DatasetMaker(**config["make_dataset"])
        train_data, test_data, _ = dataset_maker.make_dataset()
    if config["general"]["build_features"]:
        features_builder = FeaturesBuilder(**config["build_features"])
        if not config["general"]["make_dataset"]:
            train_data, test_data = features_builder.load_data()
        train_data, test_data = features_builder.build_features(train_data, test_data)
    if config["general"]["train_model"]:
        model_trainer = ModelTrainer(**config["train_model"])
        if not config["general"]["build_features"]:
            train_data, test_data = model_trainer.load_data(*config["build_features"]["output_file_names"])
        model_trainer.run_model_training(train_data, test_data)
    if config["general"]["test_model"]:
        model_predictor = ModelPredictor(config["predict_model"], config["build_features"])
        model_predictor.run_model_prediction()
    elif not any(
        [
            config["general"]["make_dataset"],
            config["general"]["build_features"],
            config["general"]["train_model"],
            config["general"]["test_model"],
        ]
    ):
        logging.info("Nothing to run")
