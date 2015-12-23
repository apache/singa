## SINGA-81 Add Python Helper, which enables users to construct a model (JobProto) and run Singa in Python

    SINGAROOT/tool/python
    |-- pb2 (has job_pb2.py)
    |-- singa 
        |-- model.py 
        |-- layer.py 
        |-- parameter.py 
        |-- initialization.py 
        |-- utils 
            |-- utility.py 
            |-- message.py 
        |-- datasets 
            |-- cifar10.py 
            |-- mnist.py 
            |-- rbm.py 
            |-- ae.py 
    |-- examples 
        |-- cifar10_cnn.py, mnist_mlp.py, , mnist_rbm1.py, rnnlm_usermodel.py, etc. 

### How to Run
```
bin/singa-run.sh -exe user_main.py -conf cluster.conf
```
The python code, i.e., `user_main.py`, would create the JobProto object and pass it to Driver::Train.
Currently, ./bin/singa-run.sh needs to get the cluster topology, hence we still need to pass a `cluster.conf` to it.
The cluster.conf has the configuration for a JobProto with all other fields empty except the cluster field.

Note that 'workspace' field in ClusterProto can be set in either (i) cluster.conf or (ii) python code.

#### Examples
```
cd SINGA_ROOT
bin/singa-run.sh -exe tool/python/examples/cifar10_cnn.py -conf tool/python/examples/cluster.conf
```

### Layer class (inherited)

* Data
* Dense
* Activation
* Convolution2D
* MaxPooling2D
* AvgPooling2D
* LRN2D 
* Dropout
* RBM
* Autoencoder

#### for user defined layers (IN PROGRESS) 

The following classes are designed to construct user-defined layers for RNNLM example.

* Embedding
* RNNLM
* UserLossRNNLM


### Model class

* Model class has `jobconf` (JobProto) and `layers` (layer list)

Methods in Model class

* add
	* add Layer into Model
	* 2 subclasses: Sequential model and Energy model

* compile	
	* set Updater (i.e., optimizer) and Cluster (i.e., topology) components

* fit 
	* set Training data and parameter values for the training
		* (optional) set Validatiaon data and parameter values
	* set Train_one_batch component
	* specify `with_test` field if a user wants to run singa with test data simultaneously.
	* [TODO] recieve train/validation results, e.g., accuracy, loss, ppl, etc. 

* evaluate
	* set Testing data and parameter values for the testing
	* specify `checkpoint_path` field if a user want to run singa only for testing.
	* [TODO] recieve test results, e.g., accuracy, loss, ppl, etc. 

#### Results

fit() and evaluate() return `result`, a dictionary containing

* [key]: step number
* [value]: a list of dictionay
	* 'acc' for accuracy
	* 'loss' for loss
	* 'ppl' for ppl
	* 'se' for squred error   

#### Other classes

* Store
* Algorithm
* Updater
* SGD (inherited)
* AdaGrad (inherited)
* Cluster


## MLP Example

An example (to generate job.conf for mnist)

```
X_train, X_test, workspace = mnist.load_data()

m = Sequential('mlp', sys.argv)  

m.add(Dense(2500, init='uniform', activation='tanh'))
m.add(Dense(2000, init='uniform', activation='tanh'))
m.add(Dense(1500, init='uniform', activation='tanh'))
m.add(Dense(1000, init='uniform', activation='tanh'))
m.add(Dense(500,  init='uniform', activation='tanh'))
m.add(Dense(10, init='uniform', activation='softmax')) 

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, batch_size=100, test_steps=10, test_freq=60)
```

## CNN Example

An example (to generate job.conf for cifar10)

```
X_train, X_test, workspace = cifar10.load_data()

m = Sequential('cnn', sys.argv)

m.add(Convolution2D(32, 5, 1, 2, w_std=0.0001, b_lr=2))
m.add(MaxPooling2D(pool_size=(3,3), stride=2))
m.add(Activation('relu'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution2D(32, 5, 1, 2, b_lr=2))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(3,3), stride=2))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution2D(64, 5, 1, 2))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(3,3), stride=2))

m.add(Dense(10, w_wd=250, b_lr=2, b_wd=0, activation='softmax'))

sgd = SGD(decay=0.004, lr_type='fixed', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster(workspace)
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, 1000, test_steps=30, test_freq=300)
```


## RBM Example
```
rbmid = 3                                                                                           
X_train, X_test, workspace = mnist.load_data(nb_rbm=rbmid)                                               
m = Energy('rbm'+str(rbmid), sys.argv)

out_dim = [1000, 500, 250]
m.add(RBM(out_dim, w_std=0.1, b_wd=0)) 
                                                                                         
sgd = SGD(lr=0.1, decay=0.0002, momentum=0.8)                                
topo = Cluster(workspace)                                                                    
m.compile(optimizer=sgd, cluster=topo)                                                    
m.fit(X_train, alg='cd', nb_epoch=6000)                            
```

## AutoEncoder Example
```
rbmid = 4
X_train, X_test, workspace = mnist.load_data(nb_rbm=rbmid+1)                                               
m = Sequential('autoencoder', sys.argv)

hid_dim = [1000, 500, 250, 30]
m.add(Autoencoder(hid_dim, out_dim=784, activation='sigmoid', param_share=True))

agd = AdaGrad(lr=0.01)
topo = Cluster(workspace)
m.compile(loss='mean_squared_error', optimizer=agd, cluster=topo)
m.fit(X_train, alg='bp', nb_epoch=12200)
```

### TIPS

Hidden layers for MLP can be written as
```
for n in [2500, 2000, 1500, 1000, 500]:
  m.add(Dense(n, init='uniform', activation='tanh'))
m.add(Dense(10, init='uniform', activation='softmax'))
```

Activation layer can be specified separately
```
m.add(Dense(2500, init='uniform'))
m.add(Activation('tanh'))
```

Users can explicity specify weight and bias, and their values

for example of MLP
```
par = Parameter(init='uniform', scale=0.05)
m.add(Dense(2500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(10, w_param=par, b_param=par, activation='softmax'))
```

for example of Cifar10 
```
parw = Parameter(init='gauss', std=0.0001)
parb = Parameter(init='const', value=0)
m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb, b_lr=2))
m.add(MaxPooling2D(pool_size(3,3), stride=2))
m.add(Activation('relu'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

parw.update(std=0.01)
m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size(3,3), stride=2))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution(64, 5, 1, 2, w_param=parw, b_param=parb, b_lr=1))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size(3,3), stride=2))

m.add(Dense(10, w_param=parw, w_wd=250, b_param=parb, b_lr=2, b_wd=0, activation='softmax'))
```


Alternative ways to add Data layer
```
X_train, X_test = mnist.load_data()  // parameter values are set in load_data() 
m.fit(X_train, ...)                  // Data layer for training is added
m.evaluate(X_test, ...)              // Data layer for testing is added
```
```
X_train, X_test = mnist.load_data()  // parameter values are set in load_data() 
m.add(X_train)                       // explicitly add Data layer
m.add(X_test)                        // explicitly add Data layer
```
```
store = Store(path='train.bin', batch_size=64, ...)        // parameter values are set explicitly 
m.add(Data(load='recordinput', phase='train', conf=store)) // Data layer is added
store = Store(path='test.bin', batch_size=100, ...)        // parameter values are set explicitly 
m.add(Data(load='recordinput', phase='test', conf=store))  // Data layer is added
```

### Parameter class

Users need to set parameter and initial values. For example,

* Parameter 
	* lr = (float) // learning rate
	* wd = (float) // weight decay

* Parameter initialization
	* init = (string) // one of the types, 'uniform', 'constant', 'gaussian' 
	* for uniform [default]
		* high = (float)
		* low = (float)
	* for constant
		* value = (float)
	* for gaussian
		* mean = (float)
		* std = (float)

* Weight (w_param) is gaussian with mean=0, std=0.01 at default

* Bias (b_param) is constant with value=0 at default

* How to update the parameter fields
	* for updating Weight, put 'w_' in front of field name
	* for updating Bias, put 'b_' in front of field name

Several ways to set Parameter values
```
m.add(Dense(10, w_mean=1, w_std=0.1, w_lr=2, w_wd=10, ...)
```
```
parw = Parameter(lr=2, wd=10, init='constant', value=0)
m.add(Dense(10, w_param=parw, ...)
```
```
parw = Parameter(init='constant', value=0)
m.add(Dense(10, w_param=parw, w_lr=2, w_wd=10, ...)
```

### Example to run singa

(1) Run singa for training
```
m.fit(X_train, nb_epoch=1000)
```

(2) Run singa for training and validation
```
m.fit(X_train, validate_data=X_valid, nb_epoch=1000)
```

(3) Run singa for test while training 
```
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, batch_size=100, test_steps=100)
```

(4) Run singa for test only
Assume a checkpoint exists after training
```
result = m.evaluate(X_test, batch_size=100, checkpoint_path=workspace+'/checkpoint/step100-worker0')
```
