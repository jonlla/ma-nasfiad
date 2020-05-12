from autoencoder_cgp.evolutionary_components.networkblock import ConvNetworkBlock

import itertools


class NetworkBlockTable:

    def __init__(self, config_dict, network_type="Conv"):

        assert network_type in ["Conv"]
        self.network_type = network_type
        self.config_dict = config_dict
        self.create_block_configuration()

    def create_block_configuration(self):
        """
        Creates all possible network block configurations. A block configuration is for example
        a convolutional block with filter size 5x5 and #filters 128.
        Maps all integer ids to each block configuration.
        @return:
        """
        config_values_combinations = list(itertools.product(*self.config_dict.values()))

        block_lookup_dict = {}
        for i in range(len(config_values_combinations)):
            current_config = config_values_combinations[i]
            config_params = {}
            for config_index, config_name in enumerate(self.config_dict.keys()):
                config_params[config_name] = current_config[config_index]

            if self.network_type == "Conv":
                new_network_block = ConvNetworkBlock(block_type="Conv", **config_params)
                block_lookup_dict[i] = new_network_block

        self._lookup_dict = block_lookup_dict

    @property
    def lookup_dict(self):
        if self._lookup_dict is None:
            self.create_block_configuration()
        return self._lookup_dict

    @lookup_dict.setter
    def lookup_dict(self, lookup_dict):
        self._lookup_dict = lookup_dict
