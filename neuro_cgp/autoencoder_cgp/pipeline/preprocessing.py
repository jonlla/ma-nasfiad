import pickle
import numpy as np
import pandas as pd


def load_train_test_dict(db, machine_type, data_path, preprocessing_step):
    # 6_db_valve
    directory = f"{data_path}/{db}_db_{machine_type}/{machine_type}/preprocessed_data/"
    with open(directory + f"train_{preprocessing_step}.pickle", "rb") as file:
        df_train_dict = pickle.load(file, encoding="latin1")

    with open(directory + f"test_{preprocessing_step}.pickle", "rb") as file:
        df_test_dict = pickle.load(file, encoding="latin1")

    return df_train_dict, df_test_dict

def load_train_test_parts(data_path):
    # 6_db_valve
    df_train_dict = {}
    df_test_dict = {}
    for i in range(0,7,2):
        with open(data_path + f"/train_mel_specto32_norm_part_0{i}.pickle", "rb") as file:
            df_temp = pickle.load(file, encoding="latin1")
            df_train_dict[f"id_0{i}"] = df_temp

    for i in range(0, 7, 2):
        with open(data_path + f"/test_mel_specto32_norm_part_0{i}.pickle", "rb") as file:
            df_temp = pickle.load(file, encoding="latin1")
            df_test_dict[f"id_0{i}"] = df_temp

    return df_train_dict, df_test_dict

def df_to_neural_network_format(dataframe, dim1, dim2):
    '''
    Converts the different dataframes to the format that is fed into the neural netowrk, i.e. expands the dimension and picks only the signal from the dataframe.
    '''
    n = len(dataframe)

    X = np.zeros(shape=(n, dim1, dim2, 1))

    for i in range(n):
        X[i] = np.expand_dims(dataframe["signal_spectrum"][i], axis=-1)

    return X


def df_to_1D_neural_network_format(dataframe):
    dataframe.signal_spectrum = dataframe.signal_spectrum.apply(np.ravel)
    size = len(dataframe.signal_spectrum[0])
    X = np.zeros(shape=(len(dataframe), size))
    for i in range(len(dataframe)):
        X[i, :] = dataframe.signal_spectrum[i]
    return X
