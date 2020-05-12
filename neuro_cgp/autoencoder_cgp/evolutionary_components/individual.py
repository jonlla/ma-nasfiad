# from evolutionary_components.operations import decode_phenotype_from_genotype
import tensorflow as tf

from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
from autoencoder_cgp.evolutionary_components.genotype import Genotype
from autoencoder_cgp.evolutionary_components.phenotype import Phenotype
import autoencoder_cgp.evolutionary_components.operations

models = tf.keras.models
import copy
import pickle
import os
import pprint


class Individual:

    def __init__(self, genotype: "Genotype") -> object:
        self.genotype = genotype
        self.fitness = None
        self._phenotype = self.phenotype
        self.model: models.Model = None
        self.generation = None

    @property
    def phenotype(self) -> "Phenotype":
        return Genotype.decode_phenotype_from_genotype(self.genotype)

    def __eq__(self, other):
        return self.genotype == other.genotype and self.fitness == other.fitness

    def __deepcopy__(self, memo={}):
        genotype_copy = self.genotype.__deepcopy__()
        individual_copy = Individual(genotype_copy)
        individual_copy.generation = self.generation
        individual_copy.fitness = self.fitness
        return individual_copy

    def save(self, directory, filename_individual, filename_model=None):
        # copy and save the individual, set model to None as this is not required.
        individual_copy = copy.deepcopy(self)
        if filename_model is not None:
            self._save_keras_model(directory, filename_model)

        path_individual = os.path.join(directory, filename_individual)
        with open(path_individual, 'wb') as output:
            pickle.dump(individual_copy, output, pickle.HIGHEST_PROTOCOL)



    def generate_architecture(self, input_layer_shape, compile_model=True):
        """
        :param input_layer_shape: Expects an input shape like this: (None, 32, 32, 1)
        :param compile_model: If True, Keras model is compiled with Adam optimizer. If False, model must be compiled manually.
        :return:
        """
        return autoencoder_cgp.evolutionary_components.operations.generate_model_from_phenotype(
            phenotype=self.phenotype, input_layer_shape=input_layer_shape, compile_model=compile_model)



    @staticmethod
    def load_individual(filepath) -> "Individual":
        with open(filepath, 'rb') as input:
            ind: Individual = pickle.load(input)
        return ind

    def _save_keras_model(self, directory, filename_model):
        if filename_model is not None:
            path_model = os.path.join(directory, filename_model)
            self.model.save(path_model)

    @staticmethod
    def load(directory, filename_individual, filename_model=None):
        path_individual = os.path.join(directory, filename_individual)
        with open(path_individual, 'rb') as input:
            ind: Individual = pickle.load(input)

        if filename_model is not None:
            model = Individual._load_keras_model(directory, filename_model)
            ind.model = model

        return ind

    @staticmethod
    def _load_keras_model(directory, filename_model):
        path_model = os.path.join(directory, filename_model)
        model = models.load_model(path_model)
        return model

    def print_summary(self):
        debug_print = []
        gene: "GeneNode"
        for gene in self.genotype.gene_nodes:
            debug_print.append(str(gene.network_block))
        pprint.pprint(debug_print)