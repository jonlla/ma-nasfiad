import tensorflow as tf

from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
from autoencoder_cgp.evolutionary_components.genotype import Genotype
from autoencoder_cgp.evolutionary_components.individual import Individual
from autoencoder_cgp.evolutionary_components.network_block_table import NetworkBlockTable
from autoencoder_cgp.evolutionary_components.networkblock import ConvNetworkBlock
from autoencoder_cgp.evolutionary_components.operations import generate_model_from_phenotype, \
    initialize_individual_default
from autoencoder_cgp.evolutionary_components.phenotype import Phenotype

models = tf.keras.models
layers = tf.keras.layers
K = tf.keras.backend


def tuples_are_equal(t1, t2):
    return all([t1[i] == t2[i] for i in range(len(t1))])


def test_generate_model_from_phenotype_1():
    only_allel = ConvNetworkBlock(block_type="Conv", filter_size=3, num_filter=8, has_pooling_layer=False)
    phenotype = Phenotype(coding_network_blocks=[only_allel])

    # for only one allel, the model is expected to have 4 layers:
    # 1. Input Layer
    # 2. Encoding Layer (Specified by allel)
    # 3. Decoding Layer (Specified by allel)
    # 4. Output Layer (which is a convolutional layer with only one filter to get the original size back)

    # input_layer = layers.Input(shape=(32, 32, 1))
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
    # output_layer = layers.Conv2D(1, (3, 3), activation=None, padding='same')(model)
    # autoencoder_model = models.Model(input_layer, output_layer)
    # autoencoder_model.compile(loss='mse', optimizer="adam")

    model = generate_model_from_phenotype(phenotype, input_layer_shape=(None, 32, 32, 1))

    config = model.get_config()

    # Check input layer
    assert config["layers"][0]["class_name"] == 'InputLayer'

    # Check encoding layer
    assert config["layers"][1]["class_name"] == 'Conv2D'
    assert config["layers"][1]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][1]["config"]["padding"] == "same"
    assert config["layers"][1]["config"]["filters"] == 8
    # Check decoding layer
    assert config["layers"][2]["class_name"] == 'Conv2D'
    assert config["layers"][2]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][2]["config"]["padding"] == "same"
    assert config["layers"][2]["config"]["filters"] == 8

    # Check output layer
    assert config["layers"][3]["class_name"] == 'Conv2D'
    assert config["layers"][3]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][3]["config"]["padding"] == "same"
    assert config["layers"][3]["config"]["filters"] == 1

    # check the output layer is actually of the same size as the input layer
    assert tuples_are_equal(model.layers[0].output_shape[0], model.layers[3].output_shape)


def test_generate_model_from_phenotype_2():
    block = ConvNetworkBlock(block_type="Conv", filter_size=3, num_filter=8, has_pooling_layer=True)
    phenotype = Phenotype(coding_network_blocks=[block])

    # for only one allel, the model is expected to have 6 layers:
    # 1. Input Layer
    # 2. Conv Layer (Specified by allel)
    # 3. Pool Layer
    # 4. Upsampling Layer
    # 5. Decoding Layer (Specified by allel)
    # 6. Output Layer (which is a convolutional layer with only one filter to get the original size back)

    # input_layer = layers.Input(shape = (32, 32, 1))
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    # model = layers.MaxPooling2D(pool_size=(2, 2))(model)
    # model = layers.UpSampling2D((2,2)) (model)
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
    # output_layer = layers.Conv2D(1, (3, 3), activation=None, padding='same')(model)
    # autoencoder_model = models.Model(input_layer, output_layer)
    # autoencoder_model.compile(loss='mse',optimizer="adam")

    model = generate_model_from_phenotype(phenotype, input_layer_shape=(None, 32, 32, 1))

    config = model.get_config()

    # Check input layer
    assert config["layers"][0]["class_name"] == 'InputLayer'

    # Check encoding layer
    assert config["layers"][1]["class_name"] == 'Conv2D'
    assert config["layers"][1]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][1]["config"]["padding"] == "same"
    assert config["layers"][1]["config"]["filters"] == 8
    assert config["layers"][2]["class_name"] == 'MaxPooling2D'
    assert config["layers"][2]["config"]["pool_size"] == (2, 2)

    # Check decoding layer
    assert config["layers"][3]["class_name"] == 'UpSampling2D'
    assert config["layers"][3]["config"]["size"] == (2, 2)
    assert config["layers"][4]["class_name"] == 'Conv2D'
    assert config["layers"][4]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][4]["config"]["padding"] == "same"
    assert config["layers"][4]["config"]["filters"] == 8

    # Check output layer
    assert config["layers"][5]["class_name"] == 'Conv2D'
    assert config["layers"][5]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][5]["config"]["padding"] == "same"
    assert config["layers"][5]["config"]["filters"] == 1

    # check the output layer is actually of the same size as the input layer
    assert tuples_are_equal(model.layers[0].output_shape[0], model.layers[5].output_shape)


def test_generate_model_from_phenotype_3():
    allel1 = ConvNetworkBlock(block_type="Conv", filter_size=3, num_filter=8, has_pooling_layer=True)
    allel2 = ConvNetworkBlock(block_type="Conv", filter_size=5, num_filter=16, has_pooling_layer=False)
    phenotype = Phenotype(coding_network_blocks=[allel1, allel2])

    # Expected model:
    # input_layer = layers.Input(shape=(32, 32, 1))
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    # model = layers.MaxPooling2D(pool_size=(2, 2))(model)
    # model = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(model)
    # model = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(model)
    # model = layers.UpSampling2D((2,2)) (model)
    # model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
    # output_layer = layers.Conv2D(1, (3, 3), activation=None, padding='same')(model)
    # autoencoder_model = models.Model(input_layer, output_layer)
    # autoencoder_model.compile(loss='mse',optimizer="adam")
    # expected_config = autoencoder_model.get_config()

    model = generate_model_from_phenotype(phenotype, input_layer_shape=(None, 32, 32, 1))

    config = model.get_config()

    # Check input layer
    assert config["layers"][0]["class_name"] == 'InputLayer'

    # Check encoding layer
    assert config["layers"][1]["class_name"] == 'Conv2D'
    assert config["layers"][1]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][1]["config"]["activation"] == "relu"
    assert config["layers"][1]["config"]["padding"] == "same"
    assert config["layers"][1]["config"]["filters"] == 8
    assert config["layers"][2]["class_name"] == 'MaxPooling2D'
    assert config["layers"][2]["config"]["pool_size"] == (2, 2)
    assert config["layers"][3]["class_name"] == 'Conv2D'
    assert config["layers"][3]["config"]["kernel_size"] == (5, 5)
    assert config["layers"][3]["config"]["padding"] == "same"
    assert config["layers"][3]["config"]["filters"] == 16

    # Check decoding layer
    assert config["layers"][4]["class_name"] == 'Conv2D'
    assert config["layers"][4]["config"]["kernel_size"] == (5, 5)
    assert config["layers"][4]["config"]["padding"] == "same"
    assert config["layers"][4]["config"]["filters"] == 16
    assert config["layers"][5]["class_name"] == 'UpSampling2D'
    assert config["layers"][5]["config"]["size"] == (2, 2)
    assert config["layers"][6]["class_name"] == 'Conv2D'
    assert config["layers"][6]["config"]["activation"] == "relu"
    assert config["layers"][6]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][6]["config"]["padding"] == "same"
    assert config["layers"][6]["config"]["filters"] == 8

    # Check output layer
    assert config["layers"][7]["class_name"] == 'Conv2D'
    assert config["layers"][7]["config"]["kernel_size"] == (3, 3)
    assert config["layers"][7]["config"]["padding"] == "same"
    assert config["layers"][7]["config"]["filters"] == 1

    # check the output layer is actually of the same size as the input layer
    assert tuples_are_equal(model.layers[0].output_shape[0], model.layers[7].output_shape)



def test_generate_model_from_phenotype_dropout():
    allel1 = ConvNetworkBlock(block_type="Conv", filter_size=3, num_filter=8, has_pooling_layer=False, dropout=0.5)
    phenotype = Phenotype(coding_network_blocks=[allel1])

    # Expected model:
    input_layer = layers.Input(shape=(32, 32, 1))
    model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    model = layers.Dropout(rate=0.1)(model)
    model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
    output_layer = layers.Conv2D(1, (3, 3), activation=None, padding='same')(model)
    autoencoder_model = models.Model(input_layer, output_layer)
    autoencoder_model.compile(loss='mse',optimizer="adam")
    expected_config = autoencoder_model.get_config()

    model = generate_model_from_phenotype(phenotype, input_layer_shape=(None, 32, 32, 1))

    config = model.get_config()
    print(model.summary())



def test_decode_phenotype_from_genotype():
    config_dict = {"filter_size": [3, 5], "num_filter": [16, 32], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    genotype_nodes = [GeneNode(None, None),
                      GeneNode(network_table.lookup_dict[1], 0),
                      GeneNode(network_table.lookup_dict[6], 1),
                      # The following two genes are intentionally non-coding:
                      GeneNode(network_table.lookup_dict[3], 1),
                      GeneNode(network_table.lookup_dict[1], 3),
                      GeneNode(None, 2)]
    test_genotype = Genotype(gene_nodes=genotype_nodes)

    expected_phenotype = Phenotype([ConvNetworkBlock("Conv", num_filter=16, filter_size=3, has_pooling_layer=True),
                                    ConvNetworkBlock("Conv", num_filter=32, filter_size=5, has_pooling_layer=False)])

    decoded_phenotype = Genotype.decode_phenotype_from_genotype(test_genotype)
    assert decoded_phenotype == expected_phenotype


def test_decode_phenotype_from_genotype2():
    config_dict = {"filter_size": [3, 5], "num_filter": [16, 32], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    genotype_nodes = [GeneNode(None, None),
                      GeneNode(network_table.lookup_dict[1], 0),
                      GeneNode(network_table.lookup_dict[6], 1),
                      GeneNode(network_table.lookup_dict[7], 2),
                      GeneNode(network_table.lookup_dict[0], 3),
                      # The following two genes are intentionally non-coding:
                      GeneNode(network_table.lookup_dict[3], 1),
                      GeneNode(network_table.lookup_dict[1], 3),
                      GeneNode(None, 4)]
    test_genotype = Genotype(gene_nodes=genotype_nodes)

    expected_phenotype = Phenotype([ConvNetworkBlock("Conv", num_filter=16, filter_size=3, has_pooling_layer=True),
                                    ConvNetworkBlock("Conv", num_filter=32, filter_size=5, has_pooling_layer=False),
                                    ConvNetworkBlock("Conv", num_filter=32, filter_size=5, has_pooling_layer=True),
                                    ConvNetworkBlock("Conv", num_filter=16, filter_size=3, has_pooling_layer=False)
                                    ])

    decoded_phenotype = Genotype.decode_phenotype_from_genotype(test_genotype)
    assert decoded_phenotype == expected_phenotype


def test_decode_phenotype_from_genotype3():
    """
    Tests the special case where input is directly connected to output.
    @return:
    """
    config_dict = {"filter_size": [3, 5], "num_filter": [16, 32], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    genotype_nodes = [GeneNode(None, None),
                      # The following two genes are intentionally non-coding:
                      GeneNode(network_table.lookup_dict[3], 1),
                      GeneNode(network_table.lookup_dict[1], 3),
                      GeneNode(None, 0)]
    test_genotype = Genotype(gene_nodes=genotype_nodes)

    expected_phenotype = Phenotype([])

    decoded_phenotype = Genotype.decode_phenotype_from_genotype(test_genotype)
    assert decoded_phenotype == expected_phenotype

def test_decode_phenotype_from_genotype_mark_as_coding():
    config_dict = {"filter_size": [3, 5], "num_filter": [16, 32], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    genotype_nodes = [GeneNode(None, None),
                      GeneNode(network_table.lookup_dict[1], 0),
                      GeneNode(network_table.lookup_dict[6], 1),
                      # The following two genes are intentionally non-coding:
                      GeneNode(network_table.lookup_dict[3], 1),
                      GeneNode(network_table.lookup_dict[1], 3),
                      # output node
                      GeneNode(None, 2)]

    test_genotype = Genotype(gene_nodes=genotype_nodes)
    ind = Individual(genotype=test_genotype)

    coding_nodes = 0
    node: GeneNode
    for node in ind.genotype.gene_nodes:
        if node.coding:
            coding_nodes += 1

    # from the list: node1, node2 are the coding nodes, input and output nodes are automatically added to a network
    # and are not marked as coding here.
    assert coding_nodes == 2



def test_integration_genereate_and_decode_individual():
    config_dict = {"filter_size": [1, 3, 5], "num_filter": [8, 16, 32, 64, 128], "has_pooling_layer": [False, True]}
    network_table = NetworkBlockTable(config_dict)

    for i in range(10):
        print(f"Initalize individual number {i}")
        individual = initialize_individual_default(network_table, num_rows=3, num_columns=20, levelback=5)
        individual: Individual
        phenotype = individual.phenotype
        model = generate_model_from_phenotype(phenotype, input_layer_shape=(None, 32, 32, 1))
        assert model is not None, "Error occurred during model creation."
        assert tuples_are_equal(model.layers[0].output_shape[0], model.layers[-1].output_shape)
        assert tuples_are_equal(model.layers[1].output_shape, model.layers[-2].output_shape)
        K.clear_session()
