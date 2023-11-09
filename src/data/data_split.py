from pathlib import Path

import pandas as pd


def train_test_split(
    input_file: str,
    test_size: float,
    save_files: bool,
    output_files: list,
):
    print("Splitting data into train and test sets")
    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    print(data_path)
    df = pd.read_parquet(data_path / input_file).sort_values("pickup_datetime").reset_index(drop=True)
    print(f"Input data: {len(df)} observations")
    split_idx = int(round((1 - test_size) * len(df), 0))
    train_set, test_set = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Output data:\n\ttraining set: {len(train_set)} observations\n\ttest set: {len(test_set)} observations")
    if save_files:
        train_set.to_parquet(data_path / (output_files[0] + ".parquet"))
        test_set.to_parquet(data_path / (output_files[1] + ".parquet"))
        print(f"Train and test set saved into directory: {str(data_path)}")

    return train_set, test_set
