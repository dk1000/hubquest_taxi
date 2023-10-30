from data_split import train_test_split


if __name__ == "__main__":
    print("Making dataset")
    print("Splitting data into train and test sets")
    # TODO: arguments reading from config file
    train_test_split(
        input_file="final_taxi_data.parquet", test_size=0.3, save_files=True, output_files=("train_data", "test_data")
    )
