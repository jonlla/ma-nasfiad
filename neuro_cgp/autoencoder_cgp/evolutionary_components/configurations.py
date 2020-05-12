import pickle
import sys


class SearchConfiguration:

    def __init__(self, network_type="Conv", block_search_space: "dict" = None, epochs=50, batch_size=16,
                 max_representation_size=None):
        """
        @param block_search_space:
        @param epochs:
        @param batch_size:
        @param max_representation_size: Default = None, i.e. the representation size is not limited.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.block_search_space = None
        assert network_type in ["Conv"]
        self.network_type = network_type
        self.initialize_block_search_space(block_search_space)
        self.max_representation_size = max_representation_size

    def initialize_block_search_space(self, block_search_space):

        # assert that the block search space is valid
        if block_search_space is not None:
            for key in block_search_space.keys():
                assert key in ["filter_size", "num_filter", "has_pooling_layer", "dropout"]

            self.block_search_space = block_search_space
        else:
            default_conv = {"filter_size": [1, 3, 5],
                            "num_filter": [8, 16, 32, 64, 128, 256],
                            "has_pooling_layer": [False, True]}

            if self.network_type == "Conv":
                self.block_search_space = default_conv
            else:
                raise ValueError("Specified network type is not allowed.")


class EvolutionConfiguration:

    def __init__(self, generation_history, fitness_history, best_fitness_history, filename_individual, filename_model):
        self.filename_model = filename_model
        self.individual_filename = filename_individual
        self.fitness_history = fitness_history
        self.best_fitness_history = best_fitness_history
        self.generation_history = generation_history

    def save(self, path):
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        try:
            with open(path, 'rb') as input:
                evol_config = pickle.load(input)
        except OSError:
            raise Exception(
                "Error when loading the configuration from a previous search. If you have not run a previous search, you need to use warm_start = False. \n")

        return evol_config
