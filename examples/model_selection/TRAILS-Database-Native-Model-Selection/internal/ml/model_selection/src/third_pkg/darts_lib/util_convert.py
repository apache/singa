
from scipy.special import softmax
from .genotypes import *


def genotype(weights, steps=4, multiplier=4):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(
                W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal = _parse(softmax(weights[0], axis=-1))
    gene_reduce = _parse(softmax(weights[1], axis=-1))

    concat = range(2 + steps - multiplier, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


# from naslib
def convert_genotype_to_compact(genotype):
    """Converts Genotype to the compact representation"""
    OPS = [
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
    ]
    compact = []

    for i, cell_type in enumerate(["normal", "reduce"]):
        cell = eval("genotype." + cell_type)
        compact.append([])

        for j in range(8):
            compact[i].append((cell[j][1], OPS.index(cell[j][0])))

    compact_tuple = (tuple(compact[0]), tuple(compact[1]))
    return compact_tuple


# from naslib
def convert_compact_to_genotype(compact):
    """Converts the compact representation to a Genotype"""
    OPS = [
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
    ]
    genotype = []

    for i in range(2):
        cell = compact[i]
        genotype.append([])

        for j in range(8):
            genotype[i].append((OPS[cell[j][1]], cell[j][0]))

    return Genotype(
        normal=genotype[0],
        normal_concat=[2, 3, 4, 5],
        reduce=genotype[1],
        reduce_concat=[2, 3, 4, 5],
    )
    # TODO: need to check with Colin and/or Arber
    #  return Genotype(
    #     normal = genotype[0],
    #     normal_concat = [2, 3, 4, 5, 6],
    #     reduce = genotype[1],
    #     reduce_concat = [4, 5, 6]
    # )


# from naslib
def make_compact_mutable(compact):
    # convert tuple to list so that it is mutable
    arch_list = []
    for cell in compact:
        arch_list.append([])
        for pair in cell:
            arch_list[-1].append([])
            for num in pair:
                arch_list[-1][-1].append(num)
    return arch_list
