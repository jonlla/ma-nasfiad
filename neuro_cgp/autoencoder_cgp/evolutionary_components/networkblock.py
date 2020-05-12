class NetworkBlock:
    pass


class ConvNetworkBlock(NetworkBlock):

    def __init__(self, block_type, num_filter, filter_size, has_pooling_layer, dropout=0):
        self.block_type = block_type
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.has_pooling_layer = has_pooling_layer
        self.dropout = dropout
        # Warn: if more attributes are added also change eq and hash method!

    def __eq__(self, other):
        if not isinstance(other, ConvNetworkBlock):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.block_type == other.block_type and self.filter_size == other.filter_size and \
               self.num_filter == other.num_filter and self.has_pooling_layer == other.has_pooling_layer\
               and self.dropout == other.dropout


    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.block_type, self.filter_size, self.num_filter, self.has_pooling_layer, self.has_pooling_layer))

    def __str__(self):
        return str({"block_type": "Conv", "filter_size": self.filter_size, "num_filter": self.num_filter,
                    "has_pooling_layer": self.has_pooling_layer, "dropout": self.dropout})


