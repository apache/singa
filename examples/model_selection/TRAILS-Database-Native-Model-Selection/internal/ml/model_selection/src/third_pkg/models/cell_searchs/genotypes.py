##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from copy import deepcopy


def get_combination(space, num):
    combs = []
    for i in range(num):
        if i == 0:
            for func in space:
                combs.append([(func, i)])
        else:
            new_combs = []
            for string in combs:
                for func in space:
                    xstring = string + [(func, i)]
                    new_combs.append(xstring)
            combs = new_combs
    return combs


class Structure:
    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(
            genotype, tuple
        ), "invalid class of genotype : {:}".format(type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(
                node_info, tuple
            ), "invalid class of node_info : {:}".format(type(node_info))
            assert len(node_info) >= 1, "invalid length : {:}".format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(
                    node_in, tuple
                ), "invalid class of in-node : {:}".format(type(node_in))
                assert (
                    len(node_in) == 2 and node_in[1] <= idx
                ), "invalid in-node : {:}".format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
        # note that we re-order the input node in this function
        # return the-genotype-list and success [if unsuccess, it is not a connectivity]
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0:
                return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), "invalid index={:} < {:}".format(
            index, len(self)
        )
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = "|".join([x[0] + "~{:}".format(x[1]) for x in node_info])
            string = "|{:}|".format(string)
            strings.append(string)
        return "+".join(strings)

    def check_valid(self):
        nodes = {0: True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == "none" or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
        # two operations are special, i.e., none and skip_connect
        nodes = {0: "0"}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                elif consider_zero:
                    if op == "none" or nodes[xin] == "#":
                        x = "#"  # zero
                    elif op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                else:
                    if op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                cur_node.append(x)
            nodes[i_node + 1] = "+".join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                # assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
                if inode_edge[0] not in op_names:
                    return False
        return True

    def __repr__(self):
        return "{name}({node_num} nodes with {node_info})".format(
            name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__
        )

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        if isinstance(xstr, Structure):
            return xstr
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name="none"):
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(
            type(xstr)
        )
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = list((op, int(IDX)) for (op, IDX) in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes:
                    input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return Structure(genotypes)

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or isinstance(
            search_space, tuple
        ), "invalid class of search-space : {:}".format(type(search_space))
        assert (
            num >= 2
        ), "There should be at least two nodes in a neural cell instead of {:}".format(
            num
        )
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]

        for inode in range(2, num):
            cur_nodes = get_combination(search_space, inode)
            new_all_archs = []
            for previous_arch in all_archs:
                for cur_node in cur_nodes:
                    new_all_archs.append(previous_arch + [tuple(cur_node)])
            all_archs = new_all_archs
        if return_ori:
            return all_archs
        else:
            return [Structure(x) for x in all_archs]


class Struct101(Structure):
    def __init__(self, arch_id, genotype):
        super().__init__(genotype)
        self.arch_id = arch_id


ResNet_CODE = Structure(
    [
        (("nor_conv_3x3", 0),),  # node-1
        (("nor_conv_3x3", 1),),  # node-2
        (("skip_connect", 0), ("skip_connect", 2)),
    ]  # node-3
)

AllConv3x3_CODE = Structure(
    [
        (("nor_conv_3x3", 0),),  # node-1
        (("nor_conv_3x3", 0), ("nor_conv_3x3", 1)),  # node-2
        (("nor_conv_3x3", 0), ("nor_conv_3x3", 1), ("nor_conv_3x3", 2)),
    ]  # node-3
)

AllFull_CODE = Structure(
    [
        (
            ("skip_connect", 0),
            ("nor_conv_1x1", 0),
            ("nor_conv_3x3", 0),
            ("avg_pool_3x3", 0),
        ),  # node-1
        (
            ("skip_connect", 0),
            ("nor_conv_1x1", 0),
            ("nor_conv_3x3", 0),
            ("avg_pool_3x3", 0),
            ("skip_connect", 1),
            ("nor_conv_1x1", 1),
            ("nor_conv_3x3", 1),
            ("avg_pool_3x3", 1),
        ),  # node-2
        (
            ("skip_connect", 0),
            ("nor_conv_1x1", 0),
            ("nor_conv_3x3", 0),
            ("avg_pool_3x3", 0),
            ("skip_connect", 1),
            ("nor_conv_1x1", 1),
            ("nor_conv_3x3", 1),
            ("avg_pool_3x3", 1),
            ("skip_connect", 2),
            ("nor_conv_1x1", 2),
            ("nor_conv_3x3", 2),
            ("avg_pool_3x3", 2),
        ),
    ]  # node-3
)

AllConv1x1_CODE = Structure(
    [
        (("nor_conv_1x1", 0),),  # node-1
        (("nor_conv_1x1", 0), ("nor_conv_1x1", 1)),  # node-2
        (("nor_conv_1x1", 0), ("nor_conv_1x1", 1), ("nor_conv_1x1", 2)),
    ]  # node-3
)

AllIdentity_CODE = Structure(
    [
        (("skip_connect", 0),),  # node-1
        (("skip_connect", 0), ("skip_connect", 1)),  # node-2
        (("skip_connect", 0), ("skip_connect", 1), ("skip_connect", 2)),
    ]  # node-3
)

architectures = {
    "resnet": ResNet_CODE,
    "all_c3x3": AllConv3x3_CODE,
    "all_c1x1": AllConv1x1_CODE,
    "all_idnt": AllIdentity_CODE,
    "all_full": AllFull_CODE,
}
