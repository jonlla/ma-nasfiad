from autoencoder_cgp.evolutionary_components.networkblock import ConvNetworkBlock


class Phenotype:
    """
    The Phenotype consists of the network blocks that make up the Autoencoder. These are called coding network blocks.
    Non-coding blocks exist in the genotype but will not influence the architecture of the autoencoder.
    When generating the phenotype from the genotype the non-coding network blocks are therefore not considered.
    """

    def __init__(self, coding_network_blocks):
        """
        @param coding_network_blocks: The network blocks that will be used to construct the final autoencoder of the individual.
        """
        assert type(coding_network_blocks) == list
        if (len(coding_network_blocks)> 0):
            assert type(coding_network_blocks[0]) == ConvNetworkBlock
        self.coding_blocks = coding_network_blocks

    def __eq__(self, other):
        if not isinstance(other, Phenotype):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if len(self.coding_blocks) != len(other.coding_blocks):
            return False
        else:
            return all([self.coding_blocks[i] == other.coding_blocks[i] for i in range(len(self.coding_blocks))])
