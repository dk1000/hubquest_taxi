import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def get_file_path():
    return Path().cwd() / "src"


def run_make_dataset():
    logging.info("Executing step: make dataset")
    path = get_file_path() / "data" / "make_dataset.py"
    subprocess.run(["python", path])
    logging.info("Finished step: make dataset")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_dataset", type=bool)
    parser.add_argument("--build_features", type=bool)
    parser.add_argument("--train_model", type=bool)
    parser.add_argument("--predict_model", type=bool)
    args = parser.parse_args()
    if args.make_dataset is True:
        run_make_dataset()
    if args.build_features is True:
        run_build_features()
    if args.train_model is True:
        run_train_model()
    if args.predict_model is True:
        run_predict_model()
    else:
        logging.info("Nothing to run")
