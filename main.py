import logging
import subprocess

import yaml

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from utils import get_file_path

logging.basicConfig(level=logging.INFO)

with open("config.yaml", "r") as cfg:
    try:
        config = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        logging.error(exc)


def run_train_model():
    logging.info("Executing step: train model")
    path = get_file_path() / "models" / "train_model.py"
    subprocess.run(["python", path])
    logging.info("Finished step: train model")


def run_predict_model():
    logging.info("Executing step: predict model")
    path = get_file_path() / "models" / "predict_model.py"
    subprocess.run(["python", path])
    logging.info("Finished step: predict model")


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
        run_train_model()
    if config["general"]["test_model"]:
        run_predict_model()
    elif not any(
        [
            config["general"]["make_dataset"],
            config["general"]["build_features"],
            config["general"]["train_model"],
            config["general"]["test_model"],
        ]
    ):
        logging.info("Nothing to run")
