from autoencoder_cgp.evolutionary_components.gene_node import GeneNode
import copy

from autoencoder_cgp.evolutionary_components.phenotype import Phenotype


class Genotype:
    """
    The genotype is represented as a directed acyclic graph (DAG), which is represented by a two-dimensional grid of nodes.
    These nodes are called gene nodes in this class. Each gene_node has a connection gene and a function gene.
    """

    def __init__(self, gene_nodes: list):
        self.gene_nodes = gene_nodes

    def __deepcopy__(self):
        copied_gene_nodes = []
        gene: GeneNode
        for gene in self.gene_nodes:
            copied_gene_nodes.append(copy.copy(gene))
        return Genotype(copied_gene_nodes)

    def decode_phenotype_from_genotype(genotype: "Genotype")->"Phenotype":
        coding_genes = []
        genes = genotype.gene_nodes
        # select output node
        node = genes[-1]
        # skip to next node:
        node = genes[node.connection]
        node: GeneNode
        while node.connection is not None:
            coding_genes.append(node.network_block)
            node.coding = True
            node = genes[node.connection]

        # this algorithm decodes backwards.Therefore, reverse the order here for the phenotype:
        ordered_coding_genes = list(reversed(coding_genes))
        return Phenotype(ordered_coding_genes)

    def __eq__(self, other: "Genotype"):
        for i in range(len(self.gene_nodes)):
            if self.gene_nodes[i] != other.gene_nodes[i]:
                return False

        return True
