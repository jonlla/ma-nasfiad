from autoencoder_cgp.conv_autoencoder import ConvolutionalAutoencoderEvolSearch
from autoencoder_cgp.evolutionary_components.evolution import evolutionary_search
from autoencoder_cgp.evolutionary_components.configurations import SearchConfiguration


import unittest.mock as mock
from unittest.mock import Mock
from numpy.random import randint
import pytest
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@mock.patch("autoencoder_cgp.evolutionary_components.evolution.train_individual", return_value=None)
# @mock.patch("evolutionary_components.evolution")
def test_integration_evolutionaray_search_mock_train_maximize(mock_train_ind):
    """
    Tests the combination of:
    1. method evolutionary_search
    2. whether random mutation crashes somewhere.
    @param mock_train_ind:
    @return:
    """

    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    search_config = SearchConfiguration(block_search_space=config_dict)
    X_train = Mock()
    logger_mock = Mock()
    logger_mock.log_fitness(return_value=1)
    fitness_eval_func = random_fitness
    best_individual, history = evolutionary_search(X_train, fitness_eval_func=fitness_eval_func,
                                                   fitness_func_args={}, maximize_fitness=True,
                                                   search_config=search_config, num_generations=100,
                                                   mutation_probability=0.1, num_children=5, num_rows=6, num_cols=10,
                                                   levelback=4, logger_factory=logger_mock,
                                                   input_layer_shape=(None, 32, 32, 1)
                                                   )

    assert best_individual.fitness == 100


@mock.patch("autoencoder_cgp.evolutionary_components.evolution.train_individual", return_value=None)
@mock.patch("autoencoder_cgp.evolutionary_components.individual.Individual._save_keras_model", return_value=None)
# @mock.patch("evolutionary_components.evolution")
def test_evolutionaray_search_save_best_individual(mock_train_ind, mock_save):
    """
    Tests the combination of:
    1. method evolutionary_search
    2. whether random mutation crashes somewhere.
    @param mock_train_ind:
    @return:
    """

    # clear up directory before test
    inds_path = "tests_data/saved_individuals"
    inds_dir = os.listdir(inds_path)
    for file in inds_dir:
        os.remove(inds_path + "/" + file)

    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    search_config = SearchConfiguration(block_search_space=config_dict)

    X_train = Mock()
    fitness_eval_func = random_fitness
    logger_mock = Mock()
    logger_mock.log_fitness(return_value=1)
    best_individual, history = evolutionary_search(X_train, fitness_eval_func=fitness_eval_func,
                                                   fitness_func_args={}, maximize_fitness=True,
                                                   search_config=search_config, num_generations=10,
                                                   mutation_probability=0.1, num_children=5, num_rows=6, num_cols=10,
                                                   levelback=4, path=inds_path, logger_factory=logger_mock,
                                                   input_layer_shape=(None, 32, 32, 1)
                                                   )
    inds_dir = os.listdir(inds_path)
    assert len(inds_dir) == 10 + 1  # +1 here is the evol_config object


@mock.patch("autoencoder_cgp.evolutionary_components.evolution.train_individual", return_value=None)
# @mock.patch("evolutionary_components.evolution")
def test_integration_evolutionaray_search_mock_train_maximize_plot_history(mock_train_ind):
    """
    Tests the combination of:
    1. method evolutionary_search
    2. whether random mutation crashes somewhere.
    @param mock_train_ind:
    @return:
    """

    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    search_config = SearchConfiguration(block_search_space=config_dict)
    X_train = Mock()
    logger_mock = Mock()
    logger_mock.log_fitness(return_value=1)
    fitness_eval_func = random_fitness
    best_individual, history = evolutionary_search(X_train, fitness_eval_func=fitness_eval_func,
                                                   fitness_func_args={}, maximize_fitness=True,
                                                   search_config=search_config, num_generations=100,
                                                   mutation_probability=0.1, num_children=5, num_rows=6, num_cols=10,
                                                   levelback=4, logger_factory=logger_mock,
                                                   input_layer_shape=(None, 32, 32, 1))
    try:
        plt.scatter(history["generation"], history["fitness"])
        # plt.show()
    except Exception as e:
        assert False, "Plotting did not work, due to some exception. Check if x, y sizes are the same."  # + e


@mock.patch("autoencoder_cgp.evolutionary_components.evolution.train_individual", return_value=None)
# @mock.patch("evolutionary_components.evolution")
def test_integration_evolutionaray_search_mock_train_minimize(mock_train_ind):
    """
    Tests the combination of:
    1. method evolutionary_search
    2. whether random mutation crashes somewhere.
    @param mock_train_ind:
    @return:
    """
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    search_config = SearchConfiguration(block_search_space=config_dict)
    X_train = Mock()
    logger_mock = Mock()
    logger_mock.log_fitness(return_value=1)
    fitness_eval_func = random_fitness
    best_individual, history = evolutionary_search(X_train, fitness_eval_func=fitness_eval_func,
                                                   fitness_func_args={}, maximize_fitness=False,
                                                   search_config=search_config, num_generations=100,
                                                   mutation_probability=0.1, num_children=5, num_rows=6, num_cols=10,
                                                   levelback=4, logger_factory=logger_mock,
                                                   input_layer_shape=(None, 32, 32, 1))

    assert best_individual.fitness == 0


def random_fitness(model):
    return randint(0, 100 + 1)


def test_integeration_evolutionary_search_conv_autoencoder():
    directory = "tests_data/6_dB_valve/valve/id_00/"

    with open(directory + "train_mel_specto32_unnorm.pickle", "rb") as file:
        df_train = pickle.load(file, encoding="latin1")

    with open(directory + "test_mel_specto32_unnorm.pickle", "rb") as file:
        df_test = pickle.load(file, encoding="latin1")

    # shrink the train and test for this test:
    df_train = df_train.iloc[0:50, :]
    df_test = df_test.iloc[0:50, :]

    INPUT_X = df_train["signal_spectrum"][0].shape[0]
    INPUT_Y = df_train["signal_spectrum"][0].shape[1]
    print(INPUT_X, INPUT_Y)

    X_test = df_test.drop(["segment", "anomaly"], axis=1)
    segements_test = df_test["segment"]
    anomalies_test = df_test["anomaly"]

    # Da die Daten wieder Segmentiert sind müssen die Label zusammengeführt werden:

    y_test = df_test.groupby("segment")["anomaly"].mean()

    segements_train = df_train["segment"]

    # Expading the last dimension for train and test data to (SHAPE_X, SHAPE_Y, 1)

    n = len(df_train)

    X_train = np.zeros(shape=(n, INPUT_X, INPUT_Y, 1))

    for i in range(n):
        X_train[i] = np.expand_dims(df_train["signal_spectrum"][i], axis=-1)

    m = len(df_test)

    X_test = np.zeros(shape=(m, INPUT_X, INPUT_Y, 1))

    for i in range(m):
        X_test[i] = np.expand_dims(df_test["signal_spectrum"][i], axis=-1)

    # ------------------------------Evolutionary Search-------------------------------------

    search_config = SearchConfiguration(epochs=3, max_representation_size=8192)
    model_search = ConvolutionalAutoencoderEvolSearch(X_train, search_config=search_config, num_generations=2,
                                                      num_children=2, num_cols=10)
    model_search.fit()
    best_model = model_search.best_model
    # --------------------------------------------------------------------------------------
    X_test_pred = best_model.predict(X_test)

    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_pred_flat = X_test_pred.reshape(X_test.shape[0], -1)

    reconstruction_loss_test = np.mean(np.power(X_test_flat - X_test_pred_flat, 2), axis=1)

    df_eval = pd.concat([segements_test, anomalies_test, pd.Series(reconstruction_loss_test, name="recon_loss")],
                        axis=1)
    mean_recon_loss = df_eval.groupby("segment")["recon_loss"].mean()

    assert sum(
        y_test.index != mean_recon_loss.index) == 0, "segments in y_test do not match the segments in mean_recon_loss"




@mock.patch("autoencoder_cgp.evolutionary_components.evolution.train_individual", return_value=None)
@mock.patch("autoencoder_cgp.evolutionary_components.individual.Individual._save_keras_model", return_value=None)
@mock.patch("autoencoder_cgp.evolutionary_components.individual.Individual._load_keras_model", return_value=None)
def test_warm_start_evolutionary_search(mock_train_ind, mock_save, mock_load):
    inds_path = "tests_data/saved_individuals"
    inds_dir = os.listdir(inds_path)
    for file in inds_dir:
        os.remove(inds_path + "/" + file)

    # brauche den phenotyp, reinladen
    # davon dann die generation bekommen, sollte im Individual gespeichert werden
    # was ist mit generation & fitness history

    num_gen_first_search = 5
    num_gen_second_search = 10

    X_train = Mock()
    X_train.shape = (None, 32, 32, 1)
    fitness_eval_func = random_fitness
    model_search = ConvolutionalAutoencoderEvolSearch(X_train, fitness_eval_func=fitness_eval_func,
                                                      fitness_func_args={}, num_generations=num_gen_first_search,
                                                      path=inds_path, warm_start=False)
    model_search.fit()

    model_search2 = ConvolutionalAutoencoderEvolSearch(X_train, fitness_eval_func=fitness_eval_func,
                                                       fitness_func_args={}, num_generations=num_gen_second_search,
                                                       path=inds_path, warm_start=True)
    history = model_search2.fit()

    total_generations = num_gen_first_search + num_gen_second_search

    assert len(set(history["generation"])) == total_generations
    # plt.scatter(history["generation"], history["fitness"])
    # plt.show()


warm_start_data = [(2, 2, 2, 2)]


# testcase checks whether the model from the first run is correctly loaded to make a prediction
# at the end as no other model will be trained.
# (1, 1, 1, 1)]


@pytest.mark.parametrize("num_gen1, num_gen2, num_children1, num_children2", warm_start_data)
def test_integeration_warm_startevolutionary_search(num_gen1, num_gen2, num_children1, num_children2):
    inds_path = "tests_data/saved_individuals"
    inds_dir = os.listdir(inds_path)
    for file in inds_dir:
        os.remove(inds_path + "/" + file)

    directory = "tests_data/6_dB_valve/valve/id_00/"

    with open(directory + "train_mel_specto32_unnorm.pickle", "rb") as file:
        df_train = pickle.load(file, encoding="latin1")

    with open(directory + "test_mel_specto32_unnorm.pickle", "rb") as file:
        df_test = pickle.load(file, encoding="latin1")

    # shrink the train and test for this test:
    df_train = df_train.iloc[0:50, :]
    df_test = df_test.iloc[0:50, :]

    INPUT_X = df_train["signal_spectrum"][0].shape[0]
    INPUT_Y = df_train["signal_spectrum"][0].shape[1]
    print(INPUT_X, INPUT_Y)

    X_test = df_test.drop(["segment", "anomaly"], axis=1)
    segements_test = df_test["segment"]
    anomalies_test = df_test["anomaly"]

    # Da die Daten wieder Segmentiert sind müssen die Label zusammengeführt werden:

    y_test = df_test.groupby("segment")["anomaly"].mean()

    segements_train = df_train["segment"]

    # Expading the last dimension for train and test data to (SHAPE_X, SHAPE_Y, 1)

    n = len(df_train)

    X_train = np.zeros(shape=(n, INPUT_X, INPUT_Y, 1))

    for i in range(n):
        X_train[i] = np.expand_dims(df_train["signal_spectrum"][i], axis=-1)

    m = len(df_test)

    X_test = np.zeros(shape=(m, INPUT_X, INPUT_Y, 1))

    for i in range(m):
        X_test[i] = np.expand_dims(df_test["signal_spectrum"][i], axis=-1)

    # ------------------------------Evolutionary Search-------------------------------------
    search_config = SearchConfiguration(epochs=3)
    model_search = ConvolutionalAutoencoderEvolSearch(X_train, search_config=search_config, num_generations=num_gen1,
                                                      num_children=num_children1, num_cols=10, path=inds_path)
    model_search.fit()

    model_search2 = ConvolutionalAutoencoderEvolSearch(X_train, search_config=search_config, num_generations=num_gen2,
                                                       num_children=num_children2, num_cols=10, path=inds_path,
                                                       warm_start=True)
    history = model_search2.fit()
    best_model = model_search.best_model

    assert len(set(history["generation"])) == 4

    # --------------------------------------------------------------------------------------
    X_test_pred = best_model.predict(X_test)

    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_pred_flat = X_test_pred.reshape(X_test.shape[0], -1)

    reconstruction_loss_test = np.mean(np.power(X_test_flat - X_test_pred_flat, 2), axis=1)

    df_eval = pd.concat([segements_test, anomalies_test, pd.Series(reconstruction_loss_test, name="recon_loss")],
                        axis=1)
    mean_recon_loss = df_eval.groupby("segment")["recon_loss"].mean()

    assert sum(
        y_test.index != mean_recon_loss.index) == 0, "segements in y_test do not match the segments in mean_reecon_loss"
