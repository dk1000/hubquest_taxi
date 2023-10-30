import pandas as pd


def train_test_split(data_path: str="../data/raw/", test_size: float=.3,
                     save_files: bool=True):
    df = pd.read_parquet(data_path + 'final_taxi_data.parquet').sort_values('pickup_datetime').reset_index(drop=True)
    split_idx = int(round((1 - test_size) * len(df), 0))
    train_set, test_set = df.iloc[:split_idx], df.iloc[split_idx:]
    if save_files:
        train_set.to_parquet(data_path + 'train_data.parquet')
        test_set.to_parquet(data_path + 'test_data.parquet')
    return train_set, test_set
