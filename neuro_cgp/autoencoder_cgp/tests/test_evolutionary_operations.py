import copy

import pytest
from flaky import flaky
from unittest import mock

from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
from autoencoder_cgp.evolutionary_components.networkblock import  ConvNetworkBlock
from autoencoder_cgp.evolutionary_components.operations import initialize_individual_default, get_column_phenotype, \
    get_connection_sample_range, get_layer_shapes_from_phenotype, mutate, check_autoencoder_constraints, \
    calculate_layer_size, mutate_passive_gene, mutate_active_gene_forced
from autoencoder_cgp.evolutionary_components.network_block_table import NetworkBlockTable

import pprint

from autoencoder_cgp.evolutionary_components.phenotype import Phenotype


def tuples_are_equal(t1, t2):
    return all([t1[i] == t2[i] for i in range(len(t1))])


testdata_get_column = [
    (2, 3, 2, 0),
    (2, 3, 1, 0),
    (2, 3, 3, 1),
    (2, 3, 4, 1),
    (2, 3, 5, 2),
    (2, 3, 6, 2),
    (10, 5, 15, 1),
    (10, 5, 20, 1),
    (10, 5, 50, 4),
    (10, 5, 30, 2)
]


@pytest.mark.parametrize("num_rows, num_cols, index, expected_col", testdata_get_column)
def test_get_column_phenotype(num_rows, num_cols, index, expected_col):
    col = get_column_phenotype(num_rows, index)
    assert col == expected_col


# "index, num_rows, num_cols, level_back, expected_range"
test_data_range = [
    (9, 4, 3, 1, (5, 9)),
    (12, 4, 3, 1, (5, 9)),
    (9, 4, 3, 2, (1, 9)),
    (12, 4, 3, 2, (1, 9)),
    (9, 4, 3, 3, (0, 9)),
    (12, 4, 3, 3, (0, 9)),
    (1, 4, 3, 3, (0, 1)),
    (4, 4, 3, 3, (0, 1)),
    (5, 4, 3, 3, (0, 5)),
    (5, 4, 3, 1, (1, 5)),
    (8, 4, 3, 1, (1, 5)),
    # For the output node:
    (13, 4, 3, 1, (1, 13)),
    (13, 4, 3, 2, (1, 13)),
    (13, 4, 3, 3, (1, 13))

]


@pytest.mark.parametrize("index, num_rows, num_cols, level_back, expected_range", test_data_range)
def test_get_sample_range(index, num_rows, num_cols, level_back, expected_range):
    sample_range = get_connection_sample_range(index, num_rows, num_cols, level_back)

    assert sample_range == expected_range


@mock.patch("autoencoder_cgp.evolutionary_components.operations.get_table_sample_range", return_value=3, autospec=True)
def test_initialize_individual(mock_randint):
    config_dict = {"filter_size": [1, 2], "num_filter": [8, 16], "has_pooling_layer": [False]}
    network_table = NetworkBlockTable(config_dict)

    expected_input_node = GeneNode(None, None)
    exptected_block = ConvNetworkBlock(block_type="Conv", filter_size=2, num_filter=16,
                                       has_pooling_layer=False)

    rows = 2
    cols = 3
    l = 2
    individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)
    genes = individual.genotype.gene_nodes
    num_genes = len(genes)
    assert num_genes == rows * cols + 2
    gene: GeneNode
    for i, gene in enumerate(genes):
        if i == 0:
            assert gene == expected_input_node
        elif i == num_genes - 1:
            assert gene.network_block is None
            assert gene.connection in range(0, rows * cols)
        else:
            assert gene.network_block == exptected_block


def test_mutate_active_gene_forced():
    config_dict = {"filter_size": [1, 2], "num_filter": [8, 16], "has_pooling_layer": [False]}
    network_table = NetworkBlockTable(config_dict)

    rows = 2
    cols = 3
    l = 2
    individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)

    mutated_individual = mutate_active_gene_forced(individual=individual, mutate_proba=0.1, num_rows=2, num_cols=2,
                                                   levelback=l, network_block_table=network_table,
                                                   input_layer_shape=(None, 32, 32, 1))

    assert mutated_individual.phenotype != individual.phenotype

# setting return_value=0 results in mutating every gene.
@mock.patch("autoencoder_cgp.evolutionary_components.operations.random_sample", return_value=0, autospec=True)
@flaky(max_runs=10, min_passes=1)
def test_mutate_gene(mock_random_sample):
    config_dict = {"filter_size": [1, 2, 3, 5, 6], "num_filter": [8, 16, 32, 64], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    rows = 2
    cols = 3
    l = 2

    original_individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)

    mutated_individual = mutate(individual=original_individual, mutate_proba=0.1, num_rows=rows, num_cols=cols,
                                levelback=l,
                                network_block_table=network_table,
                                input_layer_shape=(None, 32, 32, 1)
                                )

    num_diff = 0
    original_genes = original_individual.genotype.gene_nodes
    mutated_genes = mutated_individual.genotype.gene_nodes
    for i in range(1, len(original_genes)):
        if original_genes[i] != mutated_genes[i]:
            num_diff += 1

    # Check that every node has been mutated expect the input node:
    assert num_diff == len(original_genes) - 1

    # Check input and output node correctness
    assert mutated_individual.genotype.gene_nodes[0].connection == None
    assert mutated_individual.genotype.gene_nodes[0].network_block == None
    assert mutated_individual.genotype.gene_nodes[rows * cols + 1].connection != None
    assert mutated_individual.genotype.gene_nodes[rows * cols + 1].network_block == None


def test_mutate_passive_gene():
    config_dict = {"filter_size": [1, 2, 3, 4, 5], "num_filter": [8, 16], "has_pooling_layer": [False]}
    network_table = NetworkBlockTable(config_dict)

    rows = 4
    cols = 10
    l = 2
    individual_mutated = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)
    individual_original = copy.deepcopy(individual_mutated)
    mutate_passive_gene(individual_mutated, mutate_proba=0.1, num_rows=rows, num_cols=cols, levelback=l,
                        network_block_table=network_table)
    p1 = individual_mutated.phenotype
    p2 = individual_original.phenotype
    assert p1 == p2
    assert individual_original.genotype != individual_mutated.genotype


test_data_layer_shapes = [
    ((None, 32, 32, 1), [(None, 32, 32, 16),
                         (None, 16, 16, 16),
                         (None, 16, 16, 32)]),
    ((None, 32, 64, 1), [(None, 32, 64, 16),
                         (None, 16, 32, 16),
                         (None, 16, 32, 32)]),
    ((None, 256, 64, 1), [(None, 256, 64, 16),
                          (None, 128, 32, 16),
                          (None, 128, 32, 32)])
]


@pytest.mark.parametrize("input_size, expected_layer_shapes", test_data_layer_shapes)
def test_layer_shapes_from_phenotype(input_size, expected_layer_shapes):
    phenotype = Phenotype([ConvNetworkBlock("Conv", num_filter=16, filter_size=3, has_pooling_layer=True),
                           ConvNetworkBlock("Conv", num_filter=32, filter_size=5, has_pooling_layer=False)])

    layer_shapes = get_layer_shapes_from_phenotype(input_shape=input_size, phenotype=phenotype)

    # to debug this test:
    # model = generate_model_from_phenotype(phenotype, compile_model=True)
    # print(model.summary())

    for i in range(len(expected_layer_shapes)):
        assert tuples_are_equal(expected_layer_shapes[i], layer_shapes[i]), \
            f"Expected layer {i} to have shape {expected_layer_shapes[i]} but is  {layer_shapes[i]}"


@flaky(max_runs=100, min_passes=100)
def test_initialize_individual_with_representation_size_constraint():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16, 32, 64, 128, 256],
                   "has_pooling_layer": [True, False]}
    network_table = NetworkBlockTable(config_dict)

    rows = 4
    cols = 20
    l = 5
    max_representation_size = 1024
    individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l,
                                               max_representation_size=max_representation_size)

    layer_shapes = get_layer_shapes_from_phenotype(input_shape=(None, 32, 32, 1), phenotype=individual.phenotype)
    print("--------------------------------------")
    pprint.pprint(layer_shapes)
    print("--------------------------------------")

    representation_layer_size = calculate_layer_size(layer_shapes[-1])
    assert representation_layer_size <= max_representation_size


@flaky(max_runs=100, min_passes=100)
def test_initialize_individual_with_valid_representation_constraint():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16, 32, 64, 128, 256],
                   "has_pooling_layer": [True, False]}
    network_table = NetworkBlockTable(config_dict)

    rows = 4
    cols = 20
    l = 5
    individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)

    layer_shapes = get_layer_shapes_from_phenotype(input_shape=(None, 32, 32, 1), phenotype=individual.phenotype)
    print("--------------------------------------")
    pprint.pprint(layer_shapes)
    print("--------------------------------------")

    representation_layer = layer_shapes[-1]

    assert representation_layer[1] >= 1
    assert representation_layer[2] >= 1


@flaky(max_runs=100, min_passes=100)
def test_mutate_individual_with_valid_representation_size_constraint():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16, 32, 64, 128, 256],
                   "has_pooling_layer": [True, False]}

    network_table = NetworkBlockTable(config_dict)

    rows = 4
    cols = 20
    l = 5
    max_representation_size = 512

    original_individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l,
                                                        max_representation_size=max_representation_size)

    mutated_individual = mutate(individual=original_individual, mutate_proba=0.9, num_rows=rows, num_cols=cols,
                                levelback=l,
                                network_block_table=network_table,
                                input_layer_shape=(None, 32, 32, 1),
                                max_representation_size=max_representation_size)

    layer_shapes = get_layer_shapes_from_phenotype(input_shape=(None, 32, 32, 1),
                                                   phenotype=mutated_individual.phenotype)

    print("--------------------------------------")
    pprint.pprint(layer_shapes)
    print("--------------------------------------")

    representation_layer_size = calculate_layer_size(layer_shapes[-1])
    assert representation_layer_size <= max_representation_size


@flaky(max_runs=100, min_passes=100)
def test_mutate_individual_with_valid_representation_constraint():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16, 32, 64, 128, 256],
                   "has_pooling_layer": [True, False]}

    network_table = NetworkBlockTable(config_dict)

    rows = 4
    cols = 20
    l = 5
    original_individual = initialize_individual_default(network_table, num_rows=rows, num_columns=cols, levelback=l)

    mutated_individual = mutate(individual=original_individual, mutate_proba=0.9, num_rows=rows, num_cols=cols,
                                levelback=l,
                                network_block_table=network_table,
                                input_layer_shape=(None, 32, 32, 1))

    layer_shapes = get_layer_shapes_from_phenotype(input_shape=(None, 32, 32, 1),
                                                   phenotype=mutated_individual.phenotype)
    print("--------------------------------------")
    pprint.pprint(layer_shapes)
    print("--------------------------------------")

    representation_layer = layer_shapes[-1]

    assert representation_layer[1] >= 1
    assert representation_layer[2] >= 1


data_test_check_autoencoder_constraints = [
    ([(None, 32, 32, 1)], 1024, True),
    ([(None, 8, 8, 2)], 1024, True),
    ([(None, 64, 64, 2)], 1024, False),
    ([(None, 0.5, 0.5, 1)], 1024, False),

]


@pytest.mark.parametrize("layer_shapes, max_size, expected_result", data_test_check_autoencoder_constraints)
def test_check_autoencoder_constraints(layer_shapes, max_size, expected_result):
    result = check_autoencoder_constraints(layer_shapes, max_size)
    assert expected_result is result

