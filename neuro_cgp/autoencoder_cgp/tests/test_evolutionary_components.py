import copy

from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
from autoencoder_cgp.evolutionary_components.genotype import Genotype
from autoencoder_cgp.evolutionary_components.individual import Individual
from autoencoder_cgp.evolutionary_components.network_block_table import NetworkBlockTable
from autoencoder_cgp.evolutionary_components.networkblock import ConvNetworkBlock


def test_network_bock_table_generation_1():
    config_dict = {"filter_size": [1], "num_filter": [8], "has_pooling_layer": [False]}
    network_table = NetworkBlockTable(config_dict)

    created_network_combination = network_table.lookup_dict[0]

    expected_network_combination = ConvNetworkBlock(block_type="Conv", filter_size=1, num_filter=8, has_pooling_layer=False)

    assert expected_network_combination == created_network_combination


def test_network_bock_table_generation_2():
    config_dict = {"filter_size": [1], "num_filter": [8, 16], "has_pooling_layer": [False]}
    network_table = NetworkBlockTable(config_dict)

    created_network_combination_0 = network_table.lookup_dict[0]
    created_network_combination_1 = network_table.lookup_dict[1]

    expected_network_combination_0 = ConvNetworkBlock(block_type="Conv", filter_size=1, num_filter=8,
                                                      has_pooling_layer=False)
    expected_network_combination_1 = ConvNetworkBlock(block_type="Conv", filter_size=1, num_filter=16,
                                                      has_pooling_layer=False)

    assert expected_network_combination_0 == created_network_combination_0
    assert expected_network_combination_1 == created_network_combination_1


def test_deep_copy_individual():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    g1 = GeneNode(None, None)
    g2 = GeneNode(network_table.lookup_dict[6], 0)
    g3 = GeneNode(network_table.lookup_dict[3], 1)
    g4 = GeneNode(None, 2)
    i1 = Individual(Genotype([g1, g2, g3, g4]))
    i2 = copy.deepcopy(i1)
    # i2 = i1.__deepcopy__()

    assert i1 is not i2

    assert i1.genotype.gene_nodes[0].network_block is i2.genotype.gene_nodes[0].network_block
    assert i1.genotype.gene_nodes[0].connection is i2.genotype.gene_nodes[0].connection

    i1.genotype.gene_nodes[0].network_block = network_table.lookup_dict[1]
    i1.genotype.gene_nodes[0].connection = 3

    assert i1 is not i2

    assert i1.genotype.gene_nodes[0].network_block is not i2.genotype.gene_nodes[0].network_block
    assert i1.genotype.gene_nodes[0].connection is not i2.genotype.gene_nodes[0].connection


def test_save_load_individual():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    g1 = GeneNode(None, None)
    g2 = GeneNode(network_table.lookup_dict[6], 0)
    g3 = GeneNode(network_table.lookup_dict[3], 1)
    g4 = GeneNode(None, 2)
    i1 = Individual(Genotype([g1, g2, g3, g4]))

    path = "tests_data/"
    i1.save(path, "individual.pickle", None)

    i2 = Individual.load(path, "individual.pickle", None)

    assert i1 == i2



