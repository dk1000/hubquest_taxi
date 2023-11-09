import logging
import subprocess
from pathlib import Path

import yaml

from src.data.make_dataset import DatasetMaker

logging.basicConfig(level=logging.INFO)

with open("config.yaml", "r") as cfg:
    try:
        config = yaml.safe_load(cfg)
        print(config)
    except yaml.YAMLError as exc:
        logging.error(exc)


def get_file_path():
    return Path().cwd() / "src"


def run_build_features():
    logging.info("Executing step: build features")
    path = get_file_path() / "features" / "build_features.py"
    subprocess.run(["python", path])
    logging.info("Finished step: build features")


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
    dataset_maker = DatasetMaker(**config["make_dataset"])

    if config["general"]["make_dataset"]:
        train_data, test_data, _ = dataset_maker.make_dataset()
    if config["general"]["build_features"]:
        run_build_features()
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
