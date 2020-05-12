
class GeneNode:
    # has a function_gene = network_block and a connection_gene = connection

    def __init__(self, network_block, connection: int):
        """

        @type network_block: object
        @type connection: object
        """
        self.network_block = network_block
        self.connection = connection
        self.coding = False

    def __eq__(self, other):
        if not isinstance(other, GeneNode):
            return NotImplemented

        return self.network_block == other.network_block and self.connection == other.connection
    #
    # @property
    # def coding(self):
    #     if self._coding is None:
    #         raise ValueError(
    #             "Coding value is still none. That means that this gene was not used to decode a phenotype. This should be done first before checking if a GeneNode is coding.")
    #     else:
    #         return self._coding
    #
    # @coding.setter
    # def coding(self, value: bool):
    #     self._coding = value
