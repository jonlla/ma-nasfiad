import autoencoder_cgp.evolutionary_components.individual
from autoencoder_cgp.evolutionary_components.logger import ConvModelInfoLogger
from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
from autoencoder_cgp.evolutionary_components.genotype import Genotype
from autoencoder_cgp.evolutionary_components.network_block_table import NetworkBlockTable
from autoencoder_cgp.evolutionary_components.networkblock import ConvNetworkBlock, NetworkBlock
from autoencoder_cgp.evolutionary_components.phenotype import Phenotype

import copy
import functools
import math
import numpy as np
from numpy.random import random_sample

import tensorflow as tf

models = tf.keras.models
layers = tf.keras.layers
initializers = tf.keras.initializers
callbacks = tf.keras.callbacks
optimizer = tf.keras.optimizers
activiations = tf.keras.activations


def initialize_individual_default(block_table: NetworkBlockTable, num_rows, num_columns, levelback,
                                  input_layer_shape=(None, 32, 32, 1), max_representation_size=None) -> "Individual":
    is_individual_valid = False
    while not is_individual_valid:
        len_genotype = num_rows * num_columns + 1
        genes = []

        input_node = GeneNode(None, None)
        genes.append(input_node)

        for gene_index in range(1, len_genotype):
            network_block = random_sample_block_default(block_table)
            connection = random_sample_connection(gene_index, num_rows, num_columns, levelback)
            node = GeneNode(network_block, connection)
            genes.append(node)

        # output node
        connection = random_sample_connection(gene_index, num_rows, num_columns, levelback)
        output_node = GeneNode(None, connection)
        genes.append(output_node)

        new_individual = autoencoder_cgp.evolutionary_components.individual.Individual(Genotype(genes))

        layer_shapes = get_layer_shapes_from_phenotype(input_layer_shape, new_individual.phenotype)
        is_individual_valid = check_autoencoder_constraints(layer_shapes, max_representation_size)
        # print("Generated individual does not fullfill constraints. Generating new individual..")

    return new_individual

def get_column_phenotype(num_rows, index):
    return math.ceil(index / num_rows - 1)


def get_connection_sample_range(index, num_rows, num_cols, level_back, output_from_all=True):
    col = get_column_phenotype(num_rows, index)

    #  check if it is the outputs nodes index
    if index == num_rows * num_cols + 1 and output_from_all:
        sample_range = (1, 1 + num_rows * num_cols)

    else:
        if col >= level_back:
            lower_bound = 1 + (col - level_back) * num_rows
            upper_bound = 1 + col * num_rows
        else:
            lower_bound = 0
            upper_bound = 1 + col * num_rows

        sample_range = (lower_bound, upper_bound)

    return sample_range


def random_sample_connection(index, num_rows, num_cols, levelback, output_from_all=True):
    assert index != 0, "Index=0 is the input node. No connection needs to be sampled here."
    sample_range = get_connection_sample_range(index, num_rows, num_cols, levelback, output_from_all)
    return np.random.randint(sample_range[0], sample_range[1])


def random_sample_block_default(block_table: NetworkBlockTable, prev_gene: GeneNode = None,
                                next_gene: GeneNode = None):
    block_dict = block_table.lookup_dict
    random_index = get_table_sample_range(block_dict)
    return block_dict[random_index]


def get_table_sample_range(block_dict):
    random_index = np.random.randint(0, len(block_dict))
    return random_index


def can_model_be_build(encoding_layer, previous_shape):
    if type(encoding_layer) == layers.MaxPool2D and previous_shape[1] == 1:
        return False
    else:
        return True


def generate_model_from_phenotype(phenotype: "Phenotype", input_layer_shape,
                                  model_info_logger: ConvModelInfoLogger = None, compile_model=True) -> models.Model:
    """

    Note: It is possible that a phenotype encodes a model that cannot be build. E.g. if the size after a pooling
    layer is (None, 1, 1, ?) and the next layer that should be added is again a pooling layer. In this case the last
    layer of the encoding layer will be the (None, 1, 1, ?) and then the decoded is build, ignoring the remaining
    layers in the phenotype. This ensures that a correct model can be build by this method.
    @param input_layer_shape: Expects an input shape like this: (None, 32, 32, 1)
    @param model_info_logger:
    @param phenotype:
    @param compile_model: If True model is compiled with Adam optimizer and MSE loss, if False user can
    specify these hyperparameters and then compile the model.
    @return:
    """
    if model_info_logger is None:
        model_info_logger = ConvModelInfoLogger()

    decoder_layers, encoder_layers = generate_encoder_decoder_layers(phenotype, input_layer_shape)

    # build up and compile the model
    try:
        autoencoder_model = build_autoencoder_model(decoder_layers, encoder_layers, input_layer_shape,
                                                    model_info_logger)
        if compile_model:
            opt = optimizer.Adam(learning_rate=0.001)
            autoencoder_model.compile(loss='mse', optimizer=opt)

        model_info_logger.print_model_info()

        return autoencoder_model

    except Exception as e:
        model_info_logger.print_model_info_with_error(e)
        raise e


def build_autoencoder_model(decoder_layers, encoder_layers, input_layer_shape, model_info_logger):
    input_layer = encoder_layers[0]
    current_layer = input_layer
    model_info_logger.log_custom_info(f"Input Layer: {input_layer_shape}\n")
    previous_shape = input_layer_shape

    for i, encoding_layer in enumerate(encoder_layers[1:]):
        if can_model_be_build(encoding_layer, previous_shape):
            current_layer = encoding_layer(current_layer)
            previous_shape = encoding_layer.output_shape
            model_info_logger.log_layer_info(encoding_layer, is_encoder_part=True)
        else:
            raise Exception("Model cannot be build.")

    model_info_logger.reached_end_encoder(last_layer=encoder_layers[-1])
    for decoding_layer in reversed(decoder_layers):
        current_layer = decoding_layer(current_layer)
        model_info_logger.log_layer_info(decoding_layer, is_encoder_part=False)

    output_layer = current_layer
    model_info_logger.log_custom_info("Output Layer")

    autoencoder_model = models.Model(input_layer, output_layer)
    return autoencoder_model


def generate_encoder_decoder_layers(phenotype, input_layer_shape):
    encoder_layers = []
    decoder_layers = []

    network_block: NetworkBlock
    for i, network_block in enumerate(phenotype.coding_blocks):
        if i == 0:
            input_layer = layers.Input(shape=input_layer_shape[1:])
            output_layer = layers.Conv2D(1, (3, 3), activation=None, padding='same')
            encoder_layers.append(input_layer)
            decoder_layers.append(output_layer)

        add_layer_from_conv_block(decoder_layers, encoder_layers, network_block)

    return decoder_layers, encoder_layers


def add_layer_from_conv_block(decoder_layers, encoder_layers, network_block):
    num_filter = network_block.num_filter
    filter_size = network_block.filter_size
    # add conv layer to the encoding part of the autoencoder
    conv_layer_encode = layers.Conv2D(filters=num_filter,
                                      kernel_size=filter_size,
                                      kernel_initializer=initializers.he_normal(),
                                      activation="relu",
                                      padding="same"
                                      )
    encoder_layers.append(conv_layer_encode)

    try:
        if hasattr(network_block, 'dropout'):
            if network_block.dropout != 0:
                dropout_layer = layers.Dropout(rate=network_block.dropout)
                encoder_layers.append(dropout_layer)
    except:
        pass

    # add pooling layer if defined by network_block
    if network_block.has_pooling_layer:
        pool_layer = layers.MaxPool2D(pool_size=(2, 2))
        encoder_layers.append(pool_layer)

    # add conv layer to decoding part of the autoencoder
    conv_layer_decode = layers.Conv2D(filters=num_filter,
                                      kernel_size=filter_size,
                                      activation="relu",
                                      kernel_initializer=initializers.he_normal(),
                                      padding="same"
                                      )
    decoder_layers.append(conv_layer_decode)
    if network_block.has_pooling_layer:
        upsampling_layer = layers.UpSampling2D(size=(2, 2))
        decoder_layers.append(upsampling_layer)


def mutate_active_gene_forced(individual: "Individual", mutate_proba, num_rows, num_cols, levelback,
                              network_block_table: NetworkBlockTable,
                              input_layer_shape, max_representation_size=None,
                              random_sample_block_function=random_sample_block_default
                              ):
    phenotype_old = copy.deepcopy(individual.phenotype)
    mutated_individual = mutate(individual, mutate_proba, num_rows=num_rows, num_cols=num_cols,
                                levelback=levelback, network_block_table=network_block_table,
                                input_layer_shape=input_layer_shape, max_representation_size=max_representation_size,
                                random_sample_block_function=random_sample_block_function)

    while mutated_individual.phenotype == phenotype_old:
        mutated_individual = mutate(individual, mutate_proba, num_rows=num_rows, num_cols=num_cols,
                                    levelback=levelback, network_block_table=network_block_table,
                                    input_layer_shape=input_layer_shape,
                                    max_representation_size=max_representation_size,
                                    random_sample_block_function=random_sample_block_function)
    return mutated_individual


def mutate(individual: "Individual", mutate_proba, num_rows, num_cols, levelback,
           network_block_table: NetworkBlockTable, input_layer_shape, max_representation_size=None,
           random_sample_block_function=random_sample_block_default):
    is_mutation_valid = False

    while not is_mutation_valid:
        # get the genotype from the individual
        individual_copy = copy.deepcopy(individual)
        genotype = individual_copy.genotype

        gene_nodes = genotype.gene_nodes
        for i, gene_node in enumerate(gene_nodes):
            if random_sample() <= mutate_proba:
                mutate_gene(gene_node, index=i, num_rows=num_rows, num_cols=num_cols, levelback=levelback,
                            network_block_table=network_block_table,
                            random_sample_block_function=random_sample_block_function,
                            prev_gene=gene_nodes[i - 1] if i > 0 else None,
                            next_gene=gene_nodes[i + 1] if i < len(gene_nodes) - 1 else None
                            )

        # Check Constraints
        # Individual.phenotype generates the phenotype from the genotype.
        layer_shapes = get_layer_shapes_from_phenotype(input_layer_shape, individual_copy.phenotype)
        is_mutation_valid = check_autoencoder_constraints(layer_shapes, max_representation_size)

    return individual_copy


def mutate_gene(gene_node: GeneNode, index, num_rows, num_cols, levelback, network_block_table: NetworkBlockTable,
                prev_gene=None, next_gene=None, random_sample_block_function=random_sample_block_default):
    # no mutation required for the input node
    if index == 0:
        return
    # output node mutation should only be on the connection:
    if index == num_rows * num_cols + 1:
        connection = random_sample_connection(index=index, num_rows=num_rows, num_cols=num_cols, levelback=levelback)
        gene_node.connection = connection
    # for all genes between input and output:
    else:
        block = random_sample_block_function(network_block_table, prev_gene, next_gene)
        connection = random_sample_connection(index=index, num_rows=num_rows, num_cols=num_cols, levelback=levelback)
        gene_node.network_block = block
        gene_node.connection = connection


def mutate_passive_gene(individual: "Individual", mutate_proba, num_rows, num_cols, levelback,
                        network_block_table: NetworkBlockTable,
                        random_sample_block_function=random_sample_block_default):
    passive_gene_indices = []
    gene_nodes = individual.genotype.gene_nodes
    for i in range(1, len(gene_nodes) - 1):  # exclude input and output node
        if gene_nodes[i].coding == False:
            passive_gene_indices.append(i)

    for i in passive_gene_indices:
        if random_sample() > mutate_proba:
            mutate_gene(gene_nodes[i], index=i, num_rows=num_rows, num_cols=num_cols, levelback=levelback,
                        network_block_table=network_block_table,
                        random_sample_block_function=random_sample_block_function,
                        prev_gene=gene_nodes[i - 1] if i > 0 else None,
                        next_gene=gene_nodes[i + 1] if i < len(gene_nodes) - 1 else None
                        )


def calculate_layer_size(layer_shape):
    if type(layer_shape) is not int and len(layer_shape) > 1:
        return functools.reduce(lambda x1, x2: x1 * x2, layer_shape[1:])
    else:
        return layer_shape


def get_layer_shapes_from_phenotype(input_shape, phenotype: Phenotype):
    if input_shape is None:
        raise ValueError(
            "Input shape cannot be None. This is required to check the autoencoder architecture constraints.")

    layer_shapes = []
    first_block = phenotype.coding_blocks[0]
    if len(input_shape) != 4:
        raise ValueError(f"Expected input shape of form (?,?,?,?) but instead got {input_shape}")
    x_size = input_shape[1]
    y_size = input_shape[2]

    block: ConvNetworkBlock
    for block in phenotype.coding_blocks:
        intermediate_layer = (None, x_size, y_size, block.num_filter)
        layer_shapes.append(intermediate_layer)

        if block.has_pooling_layer:
            x_size = x_size / 2
            y_size = y_size / 2
            intermediate_layer = (None, x_size, y_size, block.num_filter)
            layer_shapes.append(intermediate_layer)

    return layer_shapes


def check_autoencoder_constraints(layer_shapes, max_representation_layer_size):
    if layer_shapes is None:
        raise ValueError("No layer_shapes were passed to the constraint checker.")

    representation_layer = layer_shapes[-1]
    representation_layer_size = calculate_layer_size(representation_layer)

    if max_representation_layer_size is not None:
        valid_size = representation_layer_size <= max_representation_layer_size
    else:
        valid_size = True

    if len(representation_layer) > 2:
        return valid_size and representation_layer[1] >= 1 and representation_layer[2] >= 1
    else:
        return valid_size
