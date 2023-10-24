


"""This is a NAS-Bench-101 version which is tensorflow independent.

Before using this API, download the data files from the links in the README.

Usage:
  # Load the data from file (this will take some time)
  nasbench = api.NASBench('/path/to/pickle/or/shelve')

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


  # Query this model from dataset
  data = nasbench.query(model_spec)

Adjacency matrices are expected to be upper-triangular 0-1 matrices within the
defined search space (7 vertices, 9 edges, 3 allowed ops). The first and last
operations must be 'input' and 'output'. The other operations should be from
config['available_ops']. Currently, the available operations are:
  CONV3X3 = "conv3x3-bn-relu"
  CONV1X1 = "conv1x1-bn-relu"
  MAXPOOL3X3 = "maxpool3x3"

When querying a spec, the spec will first be automatically pruned (removing
unused vertices and edges along with ops). If the pruned spec is still out of
the search space, an OutOfDomainError will be raised, otherwise the data is
returned.

The returned data object is a dictionary with the following keys:
  - module_adjacency: numpy array for the adjacency matrix
  - module_operations: list of operation labels
  - trainable_parameters: number of trainable parameters in the model
  - training_time: the total training time in seconds up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy

Instead of querying the dataset for a single run of a model, it is also possible
to retrieve all metrics for a given spec, using:

  fixed_stats, computed_stats = nasbench.get_metrics_from_spec(model_spec)

The fixed_stats is a dictionary with the keys:
  - module_adjacency
  - module_operations
  - trainable_parameters

The computed_stats is a dictionary from epoch count to a list of metric
dicts. For example, computed_stats[108][0] contains the metrics for the first
repeat of the provided model trained to 108 epochs. The available keys are:
  - halfway_training_time
  - halfway_train_accuracy
  - halfway_validation_accuracy
  - halfway_test_accuracy
  - final_training_time
  - final_train_accuracy
  - final_validation_accuracy
  - final_test_accuracy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import time
import shelve
import hashlib
import _pickle as pickle
import numpy as np


class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


class NASBench(object):
  """User-facing API for accessing the NASBench dataset."""

  def __init__(self, dataset_file, seed=None, data_format='pickle'):
    """Initialize dataset, this should only be done once per experiment.

    Args:
      dataset_file: path to .tfrecord file containing the dataset.
      seed: random seed used for sampling queried models. Two NASBench objects
        created with the same seed will return the same data points when queried
        with the same models in the same order. By default, the seed is randomly
        generated.
    """
    self.config = {
        'module_vertices': 7,
        'max_edges': 9,
        'num_repeats': 3,
        'available_ops': ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
    }
    random.seed(seed)

    print('Loading dataset from file... This may take a few minutes...')
    start = time.time()

    # Stores the fixed statistics that are independent of evaluation (i.e.,
    # adjacency matrix, operations, and number of parameters).
    # hash --> metric name --> scalar
    self.fixed_statistics = {}

    # Stores the statistics that are computed via training and evaluating the
    # model on CIFAR-10. Statistics are computed for multiple repeats of each
    # model at each max epoch length.
    # hash --> epochs --> repeat index --> metric name --> scalar
    self.computed_statistics = {}

    # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
    # {108} for the smaller dataset with only the 108 epochs.
    self.valid_epochs = set()

    # open the database
    if data_format == 'shelve':
        with shelve.open(dataset_file, 'r') as shelf:
          for module_hash in shelf:
            # Parse the data from the data file.
            fixed_statistics, computed_statistics = shelf[module_hash]

            self.fixed_statistics[module_hash] = fixed_statistics
            self.computed_statistics[module_hash] = computed_statistics

            self.valid_epochs.update(set(computed_statistics.keys()))
    elif data_format == 'pickle':
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
        for module_hash, stats in data.items():
            self.fixed_statistics[module_hash] = stats[0]
            self.computed_statistics[module_hash] = stats[1]

            self.valid_epochs.update(set(stats[1].keys()))
    else:
        raise Exception('Data format not supported')

    elapsed = time.time() - start
    print('Loaded dataset in %d seconds' % elapsed)

    self.history = {}
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

  def query(self, model_spec, epochs=108, stop_halfway=False):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      model_spec: ModelSpec object.
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    if epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    computed_stat = computed_stat[epochs][sampled_index]

    data = {}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']

    if stop_halfway:
      data['training_time'] = computed_stat['halfway_training_time']
      data['train_accuracy'] = computed_stat['halfway_train_accuracy']
      data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
      data['test_accuracy'] = computed_stat['halfway_test_accuracy']
    else:
      data['training_time'] = computed_stat['final_training_time']
      data['train_accuracy'] = computed_stat['final_train_accuracy']
      data['validation_accuracy'] = computed_stat['final_validation_accuracy']
      data['test_accuracy'] = computed_stat['final_test_accuracy']

    self.training_time_spent += data['training_time']
    if stop_halfway:
      self.total_epochs_spent += epochs // 2
    else:
      self.total_epochs_spent += epochs

    return data

  def is_valid(self, model_spec):
    """Checks the validity of the model_spec.

    For the purposes of benchmarking, this does not increment the budget
    counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      True if model is within space.
    """
    try:
      self._check_spec(model_spec)
    except OutOfDomainError:
      return False

    return True

  def get_budget_counters(self):
    """Returns the time and budget counters."""
    return self.training_time_spent, self.total_epochs_spent

  def reset_budget_counters(self):
    """Reset the time and epoch budget counters."""
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

  def hash_iterator(self):
    """Returns iterator over all unique model hashes."""
    return self.fixed_statistics.keys()

  def get_metrics_from_hash(self, module_hash):
    """Returns the metrics for all epochs and all repeats of a hash.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
    computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
    return fixed_stat, computed_stat

  def get_metrics_from_spec(self, model_spec):
    """Returns the metrics for all epochs and all repeats of a model.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    self._check_spec(model_spec)
    module_hash = self._hash_spec(model_spec)
    return self.get_metrics_from_hash(module_hash)

  def _check_spec(self, model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
      raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)

    if num_vertices > self.config['module_vertices']:
      raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                             % (num_vertices, config['module_vertices']))

    if num_edges > self.config['max_edges']:
      raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                             % (num_edges, self.config['max_edges']))

    if model_spec.ops[0] != 'input':
      raise OutOfDomainError('first operation should be \'input\'')
    if model_spec.ops[-1] != 'output':
      raise OutOfDomainError('last operation should be \'output\'')
    for op in model_spec.ops[1:-1]:
      if op not in self.config['available_ops']:
        raise OutOfDomainError('unsupported op %s (available ops = %s)'
                               % (op, self.config['available_ops']))

  def _hash_spec(self, model_spec):
    """Returns the MD5 hash for a provided model_spec."""
    return model_spec.hash_spec(self.config['available_ops'])


class ModelSpec(object):
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


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False

  return True


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


