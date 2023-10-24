#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
from .api_utils import ArchResults, ResultsCount
from .api_201 import NASBench201API

# NAS_BENCH_201_API_VERSION="v1.1"  # [2020.02.25]
# NAS_BENCH_201_API_VERSION="v1.2"  # [2020.03.09]
# NAS_BENCH_201_API_VERSION="v1.3"  # [2020.03.16]
NAS_BENCH_201_API_VERSION="v2.0"    # [2020.06.30]


def test_api(path):
  """This is used to test the API of NAS-Bench-201."""
  api = NASBench201API(path)
  num = len(api)
  for i, arch_str in enumerate(api):
    print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
  indexes = [1, 2, 11, 301]
  for index in indexes:
    print('\n--- index={:} ---'.format(index))
    api.show(index)
    # show the mean loss and accuracy of an architecture
    info = api.query_meta_info_by_index(index)  # This is an instance of `ArchResults`
    res_metrics = info.get_metrics('cifar10', 'train') # This is a dict with metric names as keys
    cost_metrics = info.get_compute_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency

    # get the detailed information
    results = api.query_by_index(index, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
    print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
    for seed, result in results.items():
      print ('Latency : {:}'.format(result.get_latency()))
      print ('Train Info : {:}'.format(result.get_train()))
      print ('Valid Info : {:}'.format(result.get_eval('x-valid')))
      print ('Test  Info : {:}'.format(result.get_eval('x-test')))
      # for the metric after a specific epoch
      print ('Train Info [10-th epoch] : {:}'.format(result.get_train(10)))
    config = api.get_net_config(index, 'cifar10')
    print ('config={:}'.format(config))
  index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
  api.show(index)
  print('TEST NAS-BENCH-201 DONE.')
