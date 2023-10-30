import subprocess
from pathlib import Path
import argparse


def get_file_path():
    return Path().cwd() / "src"


def run_make_dataset():
    print("Executing step: make dataset")
    path = get_file_path() / "data" / "make_dataset.py"
    subprocess.run(["python", path])
    print("Finished step: make dataset")


def run_build_features():
    print("Executing step: build features")
    path = get_file_path() / "features" / "build_features.py"
    subprocess.run(["python", path])
    print("Finished step: build features")


def run_train_model():
    print("Executing step: train model")
    path = get_file_path() / "models" / "train_model.py"
    subprocess.run(["python", path])
    print("Finished step: train model")


def run_predict_model():
    print("Executing step: predict model")
    path = get_file_path() / "models" / "predict_model.py"
    subprocess.run(["python", path])
    print("Finished step: predict model")


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
        print("Nothing to run")
