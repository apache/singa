import copy
import hashlib
import itertools

import numpy as np

# Graphviz is optional and only required for visualization.
try:
    import graphviz  # pylint: disable=g-import-not-at-top
except ImportError:
    pass

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class NASBench101ModelSpec(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format='channels_last'):
        """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()

        self.data_format = data_format

    def _prune(self):
        """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def hash_spec(self, canonical_ops):
        """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return hash_module(self.matrix, labeling)

    def visualize(self):
        """Creates a dot graph. Can be visualized in colab directly."""
        num_vertices = np.shape(self.matrix)[0]
        g = graphviz.Digraph()
        g.node(str(0), 'input')
        for v in range(1, num_vertices - 1):
            g.node(str(v), self.ops[v])
        g.node(str(num_vertices - 1), 'output')

        for src in range(num_vertices - 1):
            for dst in range(src + 1, num_vertices):
                if self.matrix[src, dst]:
                    g.edge(str(src), str(dst))

        return g

    @classmethod
    def random_sample_one_architecture(cls, dataset_api: dict, min_size=7):
        """
    This will sample a random architecture and update the edges in the
    naslib object accordingly.
    From the NASBench repository:
    one-hot adjacency matrix
    draw [0,1] for each slot in the adjacency matrix
    """
        while True:
            matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=min_size).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
            if not dataset_api["nb101_data"].is_valid(spec):
                continue

            spec = NASBench101ModelSpec(matrix, ops)
            # only sample model with 7 nodes.
            if len(spec.matrix) == min_size:
                break

        return spec


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


def gen_is_edge_fn(bits):
    """Generate a boolean function for the edge connectivity.

  Given a bitstring FEDCBA and a 4x4 matrix, the generated matrix is
    [[0, A, B, D],
     [0, 0, C, E],
     [0, 0, 0, F],
     [0, 0, 0, 0]]

  Note that this function is agnostic to the actual matrix dimension due to
  order in which elements are filled out (column-major, starting from least
  significant bit). For example, the same FEDCBA bitstring (0-padded) on a 5x5
  matrix is
    [[0, A, B, D, 0],
     [0, 0, C, E, 0],
     [0, 0, 0, F, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

  Args:
    bits: integer which will be interpreted as a bit mask.

  Returns:
    vectorized function that returns True when an edge is present.
  """

    def is_edge(x, y):
        """Is there an edge from x to y (0-indexed)?"""
        if x >= y:
            return 0
        # Map x, y to index into bit string
        index = x + (y * (y - 1) // 2)
        return (bits >> index) % 2 == 1

    return np.vectorize(is_edge)


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).

  i.e. no disconnected or "hanging" vertices.

  It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

  Args:
    matrix: V x V upper-triangular adjacency matrix

  Returns:
    True if the there are no dangling vertices.
  """
    shape = np.shape(matrix)

    rows = matrix[:shape[0] - 1, :] == 0
    rows = np.all(rows, axis=1)  # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)  # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.

  Args:
    matrix: np.ndarray square upper-triangular adjacency matrix.
    labeling: list of int labels of length equal to both dimensions of
      matrix.

  Returns:
    MD5 hash of the matrix and labeling.
  """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.

  Args:
    graph: np.ndarray adjacency matrix.
    label: list of labels of same length as graph dimensions.
    permutation: a permutation list of ints of same length as graph dimensions.

  Returns:
    np.ndarray where vertex permutation[v] is vertex v from the original graph
  """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    edge_fn = lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1
    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def is_isomorphic(graph1, graph2):
    """Exhaustively checks if 2 graphs are isomorphic."""
    matrix1, label1 = np.array(graph1[0]), graph1[1]
    matrix2, label2 = np.array(graph2[0]), graph2[1]
    assert np.shape(matrix1) == np.shape(matrix2)
    assert len(label1) == len(label2)

    vertices = np.shape(matrix1)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(0, vertices)):
        pmatrix1, plabel1 = permute_graph(matrix1, label1, perm)
        if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
            return True

    return False
