#!/usr/bin/env python
from singa.model import * 

def load_data(
     workspace = None,
     backend = 'kvfile',
     nb_rbm = 0,  # the number of layers for RBM and Autoencoder 
     checkpoint_steps = 0, 
     **pvalues
   ):

  # using mnist dataset
  data_dir = 'examples/mnist'
  path_train = data_dir + '/train_data.bin'
  path_test  = data_dir + '/test_data.bin'
  if workspace == None: workspace = data_dir

  # checkpoint path to load
  checkpoint_list = None 
  if checkpoint_steps > 0:
    workerid = 0
    checkpoint_list = [] 
    for i in range(nb_rbm-1, 0, -1):
      checkpoint_list.append('examples/rbm/rbm{0}/checkpoint/step{1}-worker{2}'.format(str(i),checkpoint_steps,workerid))

  store = Store(path=path_train, backend=backend, **pvalues)
  data_train = Data(load='recordinput', phase='train', conf=store, checkpoint=checkpoint_list)

  store = Store(path=path_test, backend=backend, **pvalues)
  data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_test, workspace
