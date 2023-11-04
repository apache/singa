#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
############################################################################################
# NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020 #
############################################################################################
# In this Python file, we define NASBenchMetaAPI, the abstract class for benchmark APIs.
# We also define the class ArchResults, which contains all information of a single architecture trained by one kind of hyper-parameters on three datasets.
# We also define the class ResultsCount, which contains all information of a single trial for a single architecture.
############################################################################################
# History:
# [2020.06.30] The first version.
#
import os, abc, copy, random, torch, numpy as np
from pathlib import Path
from typing import List, Text, Union, Dict, Optional
from collections import OrderedDict, defaultdict


def remap_dataset_set_names(dataset, metric_on_set, verbose=False):
  """re-map the metric_on_set to internal keys"""
  if verbose:
    print('Call internal function _remap_dataset_set_names with dataset={:} and metric_on_set={:}'.format(dataset, metric_on_set))
  if dataset == 'cifar10' and metric_on_set == 'valid':
    dataset, metric_on_set = 'cifar10-valid', 'x-valid'
  elif dataset == 'cifar10' and metric_on_set == 'test':
    dataset, metric_on_set = 'cifar10', 'ori-test'
  elif dataset == 'cifar10' and metric_on_set == 'train':
    dataset, metric_on_set = 'cifar10', 'train'
  elif (dataset == 'cifar100' or dataset == 'ImageNet16-120') and metric_on_set == 'valid':
    metric_on_set = 'x-valid'
  elif (dataset == 'cifar100' or dataset == 'ImageNet16-120') and metric_on_set == 'test':
    metric_on_set = 'x-test'
  if verbose:
    print('  return dataset={:} and metric_on_set={:}'.format(dataset, metric_on_set))
  return dataset, metric_on_set


class NASBenchMetaAPI(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __init__(self, file_path_or_dict: Optional[Union[Text, Dict]]=None, verbose: bool=True):
    """The initialization function that takes the dataset file path (or a dict loaded from that path) as input."""

  def __getitem__(self, index: int):
    return copy.deepcopy(self.meta_archs[index])

  def arch(self, index: int):
    """Return the topology structure of the `index`-th architecture."""
    if self.verbose:
      print('Call the arch function with index={:}'.format(index))
    assert 0 <= index < len(self.meta_archs), 'invalid index : {:} vs. {:}.'.format(index, len(self.meta_archs))
    return copy.deepcopy(self.meta_archs[index])

  def __len__(self):
    return len(self.meta_archs)

  def __repr__(self):
    return ('{name}({num}/{total} architectures, file={filename})'.format(name=self.__class__.__name__, num=len(self.evaluated_indexes), total=len(self.meta_archs), filename=self.filename))

  @property
  def avaliable_hps(self):
    return list(copy.deepcopy(self._avaliable_hps))

  @property
  def used_time(self):
    return self._used_time

  def reset_time(self):
    self._used_time = 0

  def simulate_train_eval(self, arch, dataset, hp='12', account_time=True):
    index = self.query_index_by_arch(arch)
    all_names = ('cifar10', 'cifar100', 'ImageNet16-120')
    assert dataset in all_names, 'Invalid dataset name : {:} vs {:}'.format(dataset, all_names)
    if dataset == 'cifar10':
      info = self.get_more_info(index, 'cifar10-valid', iepoch=None, hp=hp, is_random=True)
    else:
      info = self.get_more_info(index, dataset, iepoch=None, hp=hp, is_random=True)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    latency = self.get_latency(index, dataset)
    if account_time:
      self._used_time += time_cost
    return valid_acc, latency, time_cost, self._used_time

  def random(self):
    """Return a random index of all architectures."""
    return random.randint(0, len(self.meta_archs)-1)

  def query_index_by_arch(self, arch):
    """ This function is used to query the index of an architecture in the search space.
        In the topology search space, the input arch can be an architecture string such as '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|';
          or an instance that has the 'tostr' function that can generate the architecture string;
          or it is directly an architecture index, in this case, we will check whether it is valid or not.
        This function will return the index.
        If return -1, it means this architecture is not in the search space.
        Otherwise, it will return an int in [0, the-number-of-candidates-in-the-search-space).
    """
    if self.verbose:
      pass
      # print('Call query_index_by_arch with arch={:}'.format(arch))
    if isinstance(arch, int):
      if 0 <= arch < len(self):
        return arch
      else:
        raise ValueError('Invalid architecture index {:} vs [{:}, {:}].'.format(arch, 0, len(self)))
    elif isinstance(arch, str):
      if arch in self.archstr2index: arch_index = self.archstr2index[ arch ]
      else                         : arch_index = -1
    elif hasattr(arch, 'tostr'):
      if arch.tostr() in self.archstr2index: arch_index = self.archstr2index[ arch.tostr() ]
      else                                 : arch_index = -1
    else: arch_index = -1
    return arch_index

  def query_by_arch(self, arch, hp):
    # This is to make the current version be compatible with the old version.
    return self.query_info_str_by_arch(arch, hp)

  @abc.abstractmethod
  def reload(self, archive_root: Text = None, index: int = None):
    """Overwrite all information of the 'index'-th architecture in the search space, where the data will be loaded from 'archive_root'.
       If index is None, overwrite all ckps.
    """

  def clear_params(self, index: int, hp: Optional[Text]=None):
    """Remove the architecture's weights to save memory.
    :arg
      index: the index of the target architecture
      hp: a flag to controll how to clear the parameters.
        -- None: clear all the weights in '01'/'12'/'90', which indicates the number of training epochs.
        -- '01' or '12' or '90': clear all the weights in arch2infos_dict[index][hp].
    """
    if self.verbose:
      print('Call clear_params with index={:} and hp={:}'.format(index, hp))
    if hp is None:
      for key, result in self.arch2infos_dict[index].items():
        result.clear_params()
    else:
      if str(hp) not in self.arch2infos_dict[index]:
        raise ValueError('The {:}-th architecture only has hyper-parameters of {:} instead of {:}.'.format(index, list(self.arch2infos_dict[index].keys()), hp))
      self.arch2infos_dict[index][str(hp)].clear_params()

  @abc.abstractmethod
  def query_info_str_by_arch(self, arch, hp: Text='12'):
    """This function is used to query the information of a specific architecture."""

  def _query_info_str_by_arch(self, arch, hp: Text='12', print_information=None):
    arch_index = self.query_index_by_arch(arch)
    if arch_index in self.arch2infos_dict:
      if hp not in self.arch2infos_dict[arch_index]:
        raise ValueError('The {:}-th architecture only has hyper-parameters of {:} instead of {:}.'.format(index, list(self.arch2infos_dict[arch_index].keys()), hp))
      info = self.arch2infos_dict[arch_index][hp]
      strings = print_information(info, 'arch-index={:}'.format(arch_index))
      return '\n'.join(strings)
    else:
      print ('Find this arch-index : {:}, but this arch is not evaluated.'.format(arch_index))
      return None

  def query_meta_info_by_index(self, arch_index, hp: Text = '12'):
    """Return the ArchResults for the 'arch_index'-th architecture. This function is similar to query_by_index."""
    if self.verbose:
      print('Call query_meta_info_by_index with arch_index={:}, hp={:}'.format(arch_index, hp))
    if arch_index in self.arch2infos_dict:
      if hp not in self.arch2infos_dict[arch_index]:
        raise ValueError('The {:}-th architecture only has hyper-parameters of {:} instead of {:}.'.format(arch_index, list(self.arch2infos_dict[arch_index].keys()), hp))
      info = self.arch2infos_dict[arch_index][hp]
    else:
      raise ValueError('arch_index [{:}] does not in arch2infos'.format(arch_index))
    return copy.deepcopy(info)

  def query_by_index(self, arch_index: int, dataname: Union[None, Text] = None, hp: Text = '12'):
    """ This 'query_by_index' function is used to query information with the training of 01 epochs, 12 epochs, 90 epochs, or 200 epochs.
        ------
        If hp=01, we train the model by 01 epochs (see config in configs/nas-benchmark/hyper-opts/01E.config)
        If hp=12, we train the model by 01 epochs (see config in configs/nas-benchmark/hyper-opts/12E.config)
        If hp=90, we train the model by 01 epochs (see config in configs/nas-benchmark/hyper-opts/90E.config)
        If hp=200, we train the model by 01 epochs (see config in configs/nas-benchmark/hyper-opts/200E.config)
        ------
        If dataname is None, return the ArchResults
          else, return a dict with all trials on that dataset (the key is the seed)
        Options are 'cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'.
        -- cifar10-valid : training the model on the CIFAR-10 training set.
        -- cifar10 : training the model on the CIFAR-10 training + validation set.
        -- cifar100 : training the model on the CIFAR-100 training set.
        -- ImageNet16-120 : training the model on the ImageNet16-120 training set.
    """
    if self.verbose:
      print('Call query_by_index with arch_index={:}, dataname={:}, hp={:}'.format(arch_index, dataname, hp))
    info = self.query_meta_info_by_index(arch_index, hp)
    if dataname is None: return info
    else:
      if dataname not in info.get_dataset_names():
        raise ValueError('invalid dataset-name : {:} vs. {:}'.format(dataname, info.get_dataset_names()))
      return info.query(dataname)

  def find_best(self, dataset, metric_on_set, FLOP_max=None, Param_max=None, hp: Text = '12'):
    """Find the architecture with the highest accuracy based on some constraints."""
    if self.verbose:
      print('Call find_best with dataset={:}, metric_on_set={:}, hp={:} | with #FLOPs < {:} and #Params < {:}'.format(dataset, metric_on_set, hp, FLOP_max, Param_max))
    dataset, metric_on_set = remap_dataset_set_names(dataset, metric_on_set, self.verbose)
    best_index, highest_accuracy = -1, None
    for i, arch_index in enumerate(self.evaluated_indexes):
      arch_info = self.arch2infos_dict[arch_index][hp]
      info = arch_info.get_compute_costs(dataset)  # the information of costs
      flop, param, latency = info['flops'], info['params'], info['latency']
      if FLOP_max  is not None and flop  > FLOP_max : continue
      if Param_max is not None and param > Param_max: continue
      xinfo = arch_info.get_metrics(dataset, metric_on_set)  # the information of loss and accuracy
      loss, accuracy = xinfo['loss'], xinfo['accuracy']
      if best_index == -1:
        best_index, highest_accuracy = arch_index, accuracy
      elif highest_accuracy < accuracy:
        best_index, highest_accuracy = arch_index, accuracy
    if self.verbose:
      print('  the best architecture : [{:}] {:} with accuracy={:.3f}%'.format(best_index, self.arch(best_index), highest_accuracy))
    return best_index, highest_accuracy

  def get_net_param(self, index, dataset, seed: Optional[int], hp: Text = '12'):
    """
      This function is used to obtain the trained weights of the `index`-th architecture on `dataset` with the seed of `seed`
      Args [seed]:
        -- None : return a dict containing the trained weights of all trials, where each key is a seed and its corresponding value is the weights.
        -- a interger : return the weights of a specific trial, whose seed is this interger.
      Args [hp]:
        -- 01 : train the model by 01 epochs
        -- 12 : train the model by 12 epochs
        -- 90 : train the model by 90 epochs
        -- 200 : train the model by 200 epochs
    """
    if self.verbose:
      print('Call the get_net_param function with index={:}, dataset={:}, seed={:}, hp={:}'.format(index, dataset, seed, hp))
    info = self.query_meta_info_by_index(index, hp)
    return info.get_net_param(dataset, seed)

  def get_net_config(self, index: int, dataset: Text):
    """
      This function is used to obtain the configuration for the `index`-th architecture on `dataset`.
      Args [dataset] (4 possible options):
        -- cifar10-valid : training the model on the CIFAR-10 training set.
        -- cifar10 : training the model on the CIFAR-10 training + validation set.
        -- cifar100 : training the model on the CIFAR-100 training set.
        -- ImageNet16-120 : training the model on the ImageNet16-120 training set.
      This function will return a dict.
      ========= Some examlpes for using this function:
      config = api.get_net_config(128, 'cifar10')
    """
    if self.verbose:
      print('Call the get_net_config function with index={:}, dataset={:}.'.format(index, dataset))
    if index in self.arch2infos_dict:
      info = self.arch2infos_dict[index]
    else:
      raise ValueError('The arch_index={:} is not in arch2infos_dict.'.format(arch_index))
    info = next(iter(info.values()))
    results = info.query(dataset, None)
    results = next(iter(results.values()))
    return results.get_config(None)
  
  def get_cost_info(self, index: int, dataset: Text, hp: Text = '12') -> Dict[Text, float]:
    """To obtain the cost metric for the `index`-th architecture on a dataset."""
    if self.verbose:
      print('Call the get_cost_info function with index={:}, dataset={:}, and hp={:}.'.format(index, dataset, hp))
    info = self.query_meta_info_by_index(index, hp)
    return info.get_compute_costs(dataset)

  def get_latency(self, index: int, dataset: Text, hp: Text = '12') -> float:
    """
    To obtain the latency of the network (by default it will return the latency with the batch size of 256).
    :param index: the index of the target architecture
    :param dataset: the dataset name (cifar10-valid, cifar10, cifar100, ImageNet16-120)
    :return: return a float value in seconds
    """
    if self.verbose:
      print('Call the get_latency function with index={:}, dataset={:}, and hp={:}.'.format(index, dataset, hp))
    cost_dict = self.get_cost_info(index, dataset, hp)
    return cost_dict['latency']

  @abc.abstractmethod
  def show(self, index=-1):
    """This function will print the information of a specific (or all) architecture(s)."""

  def _show(self, index=-1, print_information=None) -> None:
    """
    This function will print the information of a specific (or all) architecture(s).

    :param index: If the index < 0: it will loop for all architectures and print their information one by one.
                  else: it will print the information of the 'index'-th architecture.
    :return: nothing
    """
    if index < 0: # show all architectures
      print(self)
      for i, idx in enumerate(self.evaluated_indexes):
        print('\n' + '-' * 10 + ' The ({:5d}/{:5d}) {:06d}-th architecture! '.format(i, len(self.evaluated_indexes), idx) + '-'*10)
        print('arch : {:}'.format(self.meta_archs[idx]))
        for key, result in self.arch2infos_dict[index].items():
          strings = print_information(result)
          print('>' * 40 + ' {:03d} epochs '.format(result.get_total_epoch()) + '>' * 40)
          print('\n'.join(strings))
        print('<' * 40 + '------------' + '<' * 40)
    else:
      if 0 <= index < len(self.meta_archs):
        if index not in self.evaluated_indexes: print('The {:}-th architecture has not been evaluated or not saved.'.format(index))
        else:
          arch_info = self.arch2infos_dict[index]
          for key, result in self.arch2infos_dict[index].items():
            strings = print_information(result)
            print('>' * 40 + ' {:03d} epochs '.format(result.get_total_epoch()) + '>' * 40)
            print('\n'.join(strings))
          print('<' * 40 + '------------' + '<' * 40)
      else:
        print('This index ({:}) is out of range (0~{:}).'.format(index, len(self.meta_archs)))

  def statistics(self, dataset: Text, hp: Union[Text, int]) -> Dict[int, int]:
    """This function will count the number of total trials."""
    if self.verbose:
      print('Call the statistics function with dataset={:} and hp={:}.'.format(dataset, hp))
    valid_datasets = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    if dataset not in valid_datasets:
      raise ValueError('{:} not in {:}'.format(dataset, valid_datasets))
    nums, hp = defaultdict(lambda: 0), str(hp)
    for index in range(len(self)):
      archInfo = self.arch2infos_dict[index][hp]
      dataset_seed = archInfo.dataset_seed
      if dataset not in dataset_seed:
        nums[0] += 1
      else:
        nums[len(dataset_seed[dataset])] += 1
    return dict(nums)


class ArchResults(object):

  def __init__(self, arch_index, arch_str):
    self.arch_index   = int(arch_index)
    self.arch_str     = copy.deepcopy(arch_str)
    self.all_results  = dict()
    self.dataset_seed = dict()
    self.clear_net_done = False

  def get_compute_costs(self, dataset):
    x_seeds = self.dataset_seed[dataset]
    results = [self.all_results[ (dataset, seed) ] for seed in x_seeds]

    flops     = [result.flop for result in results]
    params    = [result.params for result in results]
    latencies = [result.get_latency() for result in results]
    latencies = [x for x in latencies if x > 0]
    mean_latency = np.mean(latencies) if len(latencies) > 0 else None
    time_infos = defaultdict(list)
    for result in results:
      time_info = result.get_times()
      for key, value in time_info.items(): time_infos[key].append( value )
     
    info = {'flops'  : np.mean(flops),
            'params' : np.mean(params),
            'latency': mean_latency}
    for key, value in time_infos.items():
      if len(value) > 0 and value[0] is not None:
        info[key] = np.mean(value)
      else: info[key] = None
    return info

  def get_metrics(self, dataset, setname, iepoch=None, is_random=False):
    """
      This `get_metrics` function is used to obtain obtain the loss, accuracy, etc information on a specific dataset.
      If not specify, each set refer to the proposed split in NAS-Bench-201 paper.
      If some args return None or raise error, then it is not avaliable.
      ========================================
      Args [dataset] (4 possible options):
        -- cifar10-valid : training the model on the CIFAR-10 training set.
        -- cifar10 : training the model on the CIFAR-10 training + validation set.
        -- cifar100 : training the model on the CIFAR-100 training set.
        -- ImageNet16-120 : training the model on the ImageNet16-120 training set.
      Args [setname] (each dataset has different setnames):
        -- When dataset = cifar10-valid, you can use 'train', 'x-valid', 'ori-test'
        ------ 'train' : the metric on the training set.
        ------ 'x-valid' : the metric on the validation set.
        ------ 'ori-test' : the metric on the test set.
        -- When dataset = cifar10, you can use 'train', 'ori-test'.
        ------ 'train' : the metric on the training + validation set.
        ------ 'ori-test' : the metric on the test set.
        -- When dataset = cifar100 or ImageNet16-120, you can use 'train', 'ori-test', 'x-valid', 'x-test'
        ------ 'train' : the metric on the training set.
        ------ 'x-valid' : the metric on the validation set.
        ------ 'x-test' : the metric on the test set.
        ------ 'ori-test' : the metric on the validation + test set.
      Args [iepoch] (None or an integer in [0, the-number-of-total-training-epochs)
        ------ None : return the metric after the last training epoch.
        ------ an integer i : return the metric after the i-th training epoch.
      Args [is_random]:
        ------ True : return the metric of a randomly selected trial.
        ------ False : return the averaged metric of all avaliable trials.
        ------ an integer indicating the 'seed' value : return the metric of a specific trial (whose random seed is 'is_random').
    """
    x_seeds = self.dataset_seed[dataset]
    results = [self.all_results[ (dataset, seed) ] for seed in x_seeds]
    infos   = defaultdict(list)
    for result in results:
      if setname == 'train':
        info = result.get_train(iepoch)
      else:
        info = result.get_eval(setname, iepoch)
      for key, value in info.items(): infos[key].append( value )
    return_info = dict()
    if isinstance(is_random, bool) and is_random: # randomly select one
      index = random.randint(0, len(results)-1)
      for key, value in infos.items(): return_info[key] = value[index]
    elif isinstance(is_random, bool) and not is_random: # average
      for key, value in infos.items():
        if len(value) > 0 and value[0] is not None:
          return_info[key] = np.mean(value)
        else: return_info[key] = None
    elif isinstance(is_random, int): # specify the seed
      if is_random not in x_seeds: raise ValueError('can not find random seed ({:}) from {:}'.format(is_random, x_seeds))
      index = x_seeds.index(is_random)
      for key, value in infos.items(): return_info[key] = value[index]
    else:
      raise ValueError('invalid value for is_random: {:}'.format(is_random))
    return return_info

  def show(self, is_print=False):
    return print_information(self, None, is_print)

  def get_dataset_names(self):
    return list(self.dataset_seed.keys())

  def get_dataset_seeds(self, dataset):
    return copy.deepcopy( self.dataset_seed[dataset] )

  def get_net_param(self, dataset: Text, seed: Union[None, int] =None):
    """
    This function will return the trained network's weights on the 'dataset'.
    :arg
      dataset: one of 'cifar10-valid', 'cifar10', 'cifar100', and 'ImageNet16-120'.
      seed: an integer indicates the seed value or None that indicates returing all trials.
    """
    if seed is None:
      x_seeds = self.dataset_seed[dataset]
      return {seed: self.all_results[(dataset, seed)].get_net_param() for seed in x_seeds}
    else:
      xkey = (dataset, seed)
      if xkey in self.all_results:
        return self.all_results[xkey].get_net_param()
      else:
        raise ValueError('key={:} not in {:}'.format(xkey, list(self.all_results.keys())))

  def reset_latency(self, dataset: Text, seed: Union[None, Text], latency: float) -> None:
    """This function is used to reset the latency in all corresponding ResultsCount(s)."""
    if seed is None:
      for seed in self.dataset_seed[dataset]:
        self.all_results[(dataset, seed)].update_latency([latency])
    else:
      self.all_results[(dataset, seed)].update_latency([latency])

  def reset_pseudo_train_times(self, dataset: Text, seed: Union[None, Text], estimated_per_epoch_time: float) -> None:
    """This function is used to reset the train-times in all corresponding ResultsCount(s)."""
    if seed is None:
      for seed in self.dataset_seed[dataset]:
        self.all_results[(dataset, seed)].reset_pseudo_train_times(estimated_per_epoch_time)
    else:
      self.all_results[(dataset, seed)].reset_pseudo_train_times(estimated_per_epoch_time)

  def reset_pseudo_eval_times(self, dataset: Text, seed: Union[None, Text], eval_name: Text, estimated_per_epoch_time: float) -> None:
    """This function is used to reset the eval-times in all corresponding ResultsCount(s)."""
    if seed is None:
      for seed in self.dataset_seed[dataset]:
        self.all_results[(dataset, seed)].reset_pseudo_eval_times(eval_name, estimated_per_epoch_time)
    else:
      self.all_results[(dataset, seed)].reset_pseudo_eval_times(eval_name, estimated_per_epoch_time)

  def get_latency(self, dataset: Text) -> float:
    """Get the latency of a model on the target dataset. [Timestamp: 2020.03.09]"""
    latencies = []
    for seed in self.dataset_seed[dataset]:
      latency = self.all_results[(dataset, seed)].get_latency()
      if not isinstance(latency, float) or latency <= 0:
        raise ValueError('invalid latency of {:} with seed={:} : {:}'.format(dataset, seed, latency))
      latencies.append(latency)
    return sum(latencies) / len(latencies)

  def get_total_epoch(self, dataset=None):
    """Return the total number of training epochs."""
    if dataset is None:
      epochss = []
      for xdata, x_seeds in self.dataset_seed.items():
        epochss += [self.all_results[(xdata, seed)].get_total_epoch() for seed in x_seeds]
    elif isinstance(dataset, str):
      x_seeds = self.dataset_seed[dataset]
      epochss = [self.all_results[(dataset, seed)].get_total_epoch() for seed in x_seeds]
    else:
      raise ValueError('invalid dataset={:}'.format(dataset))
    if len(set(epochss)) > 1: raise ValueError('Each trial mush have the same number of training epochs : {:}'.format(epochss))
    return epochss[-1]

  def query(self, dataset, seed=None):
    """Return the ResultsCount object (containing all information of a single trial) for 'dataset' and 'seed'"""
    if seed is None:
      x_seeds = self.dataset_seed[dataset]
      return {seed: self.all_results[(dataset, seed)] for seed in x_seeds}
    else:
      return self.all_results[(dataset, seed)]

  def arch_idx_str(self):
    return '{:06d}'.format(self.arch_index)

  def update(self, dataset_name, seed, result):
    if dataset_name not in self.dataset_seed:
      self.dataset_seed[dataset_name] = []
    assert seed not in self.dataset_seed[dataset_name], '{:}-th arch alreadly has this seed ({:}) on {:}'.format(self.arch_index, seed, dataset_name)
    self.dataset_seed[ dataset_name ].append( seed )
    self.dataset_seed[ dataset_name ] = sorted( self.dataset_seed[ dataset_name ] )
    assert (dataset_name, seed) not in self.all_results
    self.all_results[ (dataset_name, seed) ] = result
    self.clear_net_done = False

  def state_dict(self):
    state_dict = dict()
    for key, value in self.__dict__.items():
      if key == 'all_results': # contain the class of ResultsCount
        xvalue = dict()
        assert isinstance(value, dict), 'invalid type of value for {:} : {:}'.format(key, type(value))
        for _k, _v in value.items():
          assert isinstance(_v, ResultsCount), 'invalid type of value for {:}/{:} : {:}'.format(key, _k, type(_v))
          xvalue[_k] = _v.state_dict()
      else:
        xvalue = value
      state_dict[key] = xvalue
    return state_dict

  def load_state_dict(self, state_dict):
    new_state_dict = dict()
    for key, value in state_dict.items():
      if key == 'all_results': # to convert to the class of ResultsCount
        xvalue = dict()
        assert isinstance(value, dict), 'invalid type of value for {:} : {:}'.format(key, type(value))
        for _k, _v in value.items():
          xvalue[_k] = ResultsCount.create_from_state_dict(_v)
      else: xvalue = value
      new_state_dict[key] = xvalue
    self.__dict__.update(new_state_dict)

  @staticmethod
  def create_from_state_dict(state_dict_or_file):
    x = ArchResults(-1, -1)
    if isinstance(state_dict_or_file, str): # a file path
      state_dict = torch.load(state_dict_or_file, map_location='cpu')
    elif isinstance(state_dict_or_file, dict):
      state_dict = state_dict_or_file
    else:
      raise ValueError('invalid type of state_dict_or_file : {:}'.format(type(state_dict_or_file)))
    x.load_state_dict(state_dict)
    return x

  # This function is used to clear the weights saved in each 'result'
  # This can help reduce the memory footprint.
  def clear_params(self):
    for key, result in self.all_results.items():
      del result.net_state_dict
      result.net_state_dict = None
    self.clear_net_done = True

  def debug_test(self):
    """This function is used for me to debug and test, which will call most methods."""
    all_dataset = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    for dataset in all_dataset:
      print('---->>>> {:}'.format(dataset))
      print('The latency on {:} is {:} s'.format(dataset, self.get_latency(dataset)))
      for seed in self.dataset_seed[dataset]:
        result = self.all_results[(dataset, seed)]
        print('  ==>> result = {:}'.format(result))
        print('  ==>> cost = {:}'.format(result.get_times()))

  def __repr__(self):
    return ('{name}(arch-index={index}, arch={arch}, {num} runs, clear={clear})'.format(name=self.__class__.__name__, index=self.arch_index, arch=self.arch_str, num=len(self.all_results), clear=self.clear_net_done))


"""
This class (ResultsCount) is used to save the information of one trial for a single architecture.
I did not write much comment for this class, because it is the lowest-level class in NAS-Bench-201 API, which will be rarely called.
If you have any question regarding this class, please open an issue or email me.
"""
class ResultsCount(object):

  def __init__(self, name, state_dict, train_accs, train_losses, params, flop, arch_config, seed, epochs, latency):
    self.name           = name
    self.net_state_dict = state_dict
    self.train_acc1es = copy.deepcopy(train_accs)
    self.train_acc5es = None
    self.train_losses = copy.deepcopy(train_losses)
    self.train_times  = None
    self.arch_config  = copy.deepcopy(arch_config)
    self.params     = params
    self.flop       = flop
    self.seed       = seed
    self.epochs     = epochs
    self.latency    = latency
    # evaluation results
    self.reset_eval()

  def update_train_info(self, train_acc1es, train_acc5es, train_losses, train_times) -> None:
    self.train_acc1es = train_acc1es
    self.train_acc5es = train_acc5es
    self.train_losses = train_losses
    self.train_times  = train_times

  def reset_pseudo_train_times(self, estimated_per_epoch_time: float) -> None:
    """Assign the training times."""
    train_times = OrderedDict()
    for i in range(self.epochs):
      train_times[i] = estimated_per_epoch_time
    self.train_times = train_times

  def reset_pseudo_eval_times(self, eval_name: Text, estimated_per_epoch_time: float) -> None:
    """Assign the evaluation times."""
    if eval_name not in self.eval_names: raise ValueError('invalid eval name : {:}'.format(eval_name))
    for i in range(self.epochs):
      self.eval_times['{:}@{:}'.format(eval_name,i)] = estimated_per_epoch_time

  def reset_eval(self):
    self.eval_names  = []
    self.eval_acc1es = {}
    self.eval_times  = {}
    self.eval_losses = {}

  def update_latency(self, latency):
    self.latency = copy.deepcopy( latency )

  def get_latency(self) -> float:
    """Return the latency value in seconds. -1 represents not avaliable ; otherwise it should be a float value"""
    if self.latency is None: return -1.0
    else: return sum(self.latency) / len(self.latency)

  def update_eval(self, accs, losses, times):  # new version
    data_names = set([x.split('@')[0] for x in accs.keys()])
    for data_name in data_names:
      assert data_name not in self.eval_names, '{:} has already been added into eval-names'.format(data_name)
      self.eval_names.append( data_name )
      for iepoch in range(self.epochs):
        xkey = '{:}@{:}'.format(data_name, iepoch)
        self.eval_acc1es[ xkey ] = accs[ xkey ]
        self.eval_losses[ xkey ] = losses[ xkey ]
        self.eval_times [ xkey ] = times[ xkey ]

  def update_OLD_eval(self, name, accs, losses): # old version
    assert name not in self.eval_names, '{:} has already added'.format(name)
    self.eval_names.append( name )
    for iepoch in range(self.epochs):
      if iepoch in accs:
        self.eval_acc1es['{:}@{:}'.format(name,iepoch)] = accs[iepoch]
        self.eval_losses['{:}@{:}'.format(name,iepoch)] = losses[iepoch]

  def __repr__(self):
    num_eval = len(self.eval_names)
    set_name = '[' + ', '.join(self.eval_names) + ']'
    return ('{name}({xname}, arch={arch}, FLOP={flop:.2f}M, Param={param:.3f}MB, seed={seed}, {num_eval} eval-sets: {set_name})'.format(name=self.__class__.__name__, xname=self.name, arch=self.arch_config['arch_str'], flop=self.flop, param=self.params, seed=self.seed, num_eval=num_eval, set_name=set_name))

  def get_total_epoch(self):
    return copy.deepcopy(self.epochs)

  def get_times(self):
    """Obtain the information regarding both training and evaluation time."""
    if self.train_times is not None and isinstance(self.train_times, dict):
      train_times = list( self.train_times.values() )
      time_info = {'T-train@epoch': np.mean(train_times), 'T-train@total': np.sum(train_times)}
    else:
      time_info = {'T-train@epoch':                 None, 'T-train@total':               None }
    for name in self.eval_names:
      try:
        xtimes = [self.eval_times['{:}@{:}'.format(name,i)] for i in range(self.epochs)]
        time_info['T-{:}@epoch'.format(name)] = np.mean(xtimes)
        time_info['T-{:}@total'.format(name)] = np.sum(xtimes)
      except:
        time_info['T-{:}@epoch'.format(name)] = None
        time_info['T-{:}@total'.format(name)] = None
    return time_info

  def get_eval_set(self):
    return self.eval_names

  # get the training information
  def get_train(self, iepoch=None):
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    if self.train_times is not None:
      xtime = self.train_times[iepoch]
      atime = sum([self.train_times[i] for i in range(iepoch+1)])
    else: xtime, atime = None, None
    return {'iepoch'  : iepoch,
            'loss'    : self.train_losses[iepoch],
            'accuracy': self.train_acc1es[iepoch],
            'cur_time': xtime,
            'all_time': atime}

  def get_eval(self, name, iepoch=None):
    """Get the evaluation information ; there could be multiple evaluation sets (identified by the 'name' argument)."""
    if iepoch is None: iepoch = self.epochs-1
    assert 0 <= iepoch < self.epochs, 'invalid iepoch={:} < {:}'.format(iepoch, self.epochs)
    def _internal_query(xname):
      if isinstance(self.eval_times,dict) and len(self.eval_times) > 0:
        xtime = self.eval_times['{:}@{:}'.format(xname, iepoch)]
        atime = sum([self.eval_times['{:}@{:}'.format(xname, i)] for i in range(iepoch+1)])
      else:
        xtime, atime = None, None
      return {'iepoch'  : iepoch,
              'loss'    : self.eval_losses['{:}@{:}'.format(xname, iepoch)],
              'accuracy': self.eval_acc1es['{:}@{:}'.format(xname, iepoch)],
              'cur_time': xtime,
              'all_time': atime}
    if name == 'valid':
      return _internal_query('x-valid')
    else:
      return _internal_query(name)

  def get_net_param(self, clone=False):
    if clone: return copy.deepcopy(self.net_state_dict)
    else: return self.net_state_dict

  def get_config(self, str2structure):
    """This function is used to obtain the config dict for this architecture."""
    if str2structure is None:
      # In this case, this is NAS-Bench-301
      if 'name' in self.arch_config and self.arch_config['name'] == 'infer.shape.tiny':
        return {'name': 'infer.shape.tiny', 'channels': self.arch_config['channels'],
                'genotype': self.arch_config['genotype'], 'num_classes': self.arch_config['class_num']}
      # In this case, this is NAS-Bench-201
      else:
        return {'name': 'infer.tiny', 'C': self.arch_config['channel'],
                'N'   : self.arch_config['num_cells'],
                'arch_str': self.arch_config['arch_str'], 'num_classes': self.arch_config['class_num']}
    else:
      # In this case, this is NAS-Bench-301
      if 'name' in self.arch_config and self.arch_config['name'] == 'infer.shape.tiny':
        return {'name': 'infer.shape.tiny', 'channels': self.arch_config['channels'],
                'genotype': str2structure(self.arch_config['genotype']), 'num_classes': self.arch_config['class_num']}
      # In this case, this is NAS-Bench-201
      else:
        return {'name': 'infer.tiny', 'C': self.arch_config['channel'],
                'N'   : self.arch_config['num_cells'],
                'genotype': str2structure(self.arch_config['arch_str']), 'num_classes': self.arch_config['class_num']}

  def state_dict(self):
    _state_dict = {key: value for key, value in self.__dict__.items()}
    return _state_dict

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  @staticmethod
  def create_from_state_dict(state_dict):
    x = ResultsCount(None, None, None, None, None, None, None, None, None, None)
    x.load_state_dict(state_dict)
    return x
