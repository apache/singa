# Python Binding

---

Python binding provides APIs for configuring a training job following
[keras](http://keras.io/), including the configuration of neural net, training
algorithm, etc.  It replaces the configuration file (e.g., *job.conf*) in
protobuf format, which is typically long and error-prone to prepare. We will add
python functions to interact with the layer and neural net
objects (see [here](python_interactive_training.html)), which would enable users to train and debug their models
interactively.

Here is the layout of python related code,

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
    |-- examples
        |-- cifar10_cnn.py, mnist_mlp.py, , mnist_rbm1.py, mnist_ae.py, etc.
        |-- datasets
            |-- cifar10.py
            |-- mnist.py

## Compiling and running instructions

In order to use the Python APIs, users need to add the following arguments when compiling
SINGA,

    ./configure --enable-python --with-python=PYTHON_DIR
    make

where PYTHON_DIR has Python.h


The training program is launched by

    bin/singa-run.sh -exec <user_main.py>

where user_main.py creates the JobProto object and passes it to Driver::Train to
start the training.

For example,

    cd SINGAROOT
    bin/singa-run.sh -exec tool/python/examples/cifar10_cnn.py



## Examples


### MLP Example

This example uses python APIs to configure and train a MLP model over the MNIST
dataset. The configuration content is the same as that written in *SINGAROOT/examples/mnist/job.conf*.

```
X_train, X_test, workspace = mnist.load_data()

m = Sequential('mlp', sys.argv)

m.add(Dense(2500, init='uniform', activation='stanh'))
m.add(Dense(2000, init='uniform', activation='stanh'))
m.add(Dense(1500, init='uniform', activation='stanh'))
m.add(Dense(1000, init='uniform', activation='stanh'))
m.add(Dense(500,  init='uniform', activation='stanh'))
m.add(Dense(10, init='uniform', activation='softmax'))

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, batch_size=100, test_steps=10, test_freq=60)
```

### CNN Example

This example uses python APIs to configure and train a CNN model over the Cifar10
dataset. The configuration content is the same as that written in *SINGAROOT/examples/cifar10/job.conf*.


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

sgd = SGD(decay=0.004, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster(workspace)
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, 1000, test_steps=30, test_freq=300)
```


### RBM Example

This example uses python APIs to configure and train a RBM model over the MNIST
dataset. The configuration content is the same as that written in *SINGAROOT/examples/rbm*.conf*.

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

### AutoEncoder Example
This example uses python APIs to configure and train an autoencoder model over
the MNIST dataset. The configuration content is the same as that written in
*SINGAROOT/examples/autoencoder.conf*.


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

### To run SINGA on GPU

Users need to set a list of gpu ids to `device` field in fit() or evaluate().
The number of GPUs must be the same to the number of workers configured for
cluster topology.


```
gpu_id = [0]
m.fit(X_train, nb_epoch=100, with_test=True, device=gpu_id)
```

### TIPS

Hidden layers for MLP can be configured as

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

Users can explicitly specify hyper-parameters of weight and bias

```
par = Parameter(init='uniform', scale=0.05)
m.add(Dense(2500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(10, w_param=par, b_param=par, activation='softmax'))
```


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


Data can be added in this way,

```
X_train, X_test = mnist.load_data()  // parameter values are set in load_data()
m.fit(X_train, ...)                  // Data layer for training is added
m.evaluate(X_test, ...)              // Data layer for testing is added
```
or this way,

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


### Cases to run SINGA

(1) Run SINGA for training

```
m.fit(X_train, nb_epoch=1000)
```

(2) Run SINGA for training and validation

```
m.fit(X_train, validate_data=X_valid, nb_epoch=1000)
```

(3) Run SINGA for test while training

```
m.fit(X_train, nb_epoch=1000, with_test=True)
result = m.evaluate(X_test, batch_size=100, test_steps=100)
```

(4) Run SINGA for test only
Assume a checkpoint exists after training

```
result = m.evaluate(X_test, batch_size=100, checkpoint_path=workspace+'/checkpoint/step100-worker0')
```


## Implementation Details

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

### Model class

Model class has `jobconf` (JobProto) and `layers` (layer list)

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
	* specify `with_test` field if a user wants to run SINGA with test data simultaneously.
	* [TODO] recieve train/validation results, e.g., accuracy, loss, ppl, etc.

* evaluate
	* set Testing data and parameter values for the testing
	* specify `checkpoint_path` field if a user want to run SINGA only for testing.
	* [TODO] recieve test results, e.g., accuracy, loss, ppl, etc.

### Results

fit() and evaluate() return train/test results, a dictionary containing

* [key]: step number
* [value]: a list of dictionay
	* 'acc' for accuracy
	* 'loss' for loss
	* 'ppl' for ppl
	* 'se' for squred error


### Parameter class

Users need to set parameter and initial values. For example,

* Parameter (fields in Param proto)
	* lr = (float) // learning rate multiplier, used to scale the learning rate when updating parameters.
	* wd = (float) // weight decay multiplier, used to scale the weight decay when updating parameters.

* Parameter initialization (fields in ParamGen proto)
	* init = (string) // one of the types, 'uniform', 'constant', 'gaussian'
	* high = (float)  // for 'uniform'
	* low = (float)   // for 'uniform'
	* value = (float) // for 'constant'
	* mean = (float)  // for 'gaussian'
	* std = (float)   // for 'gaussian'

* Weight (`w_param`) is 'gaussian' with mean=0, std=0.01 at default

* Bias (`b_param`) is 'constant' with value=0 at default

* How to update the parameter fields
	* for updating Weight, put `w_` in front of field name
	* for updating Bias, put `b_` in front of field name

Several ways to set Parameter values

```
parw = Parameter(lr=2, wd=10, init='gaussian', std=0.1)
parb = Parameter(lr=1, wd=0, init='constant', value=0)
m.add(Convolution2D(10, w_param=parw, b_param=parb, ...)
```

```
m.add(Dense(10, w_mean=1, w_std=0.1, w_lr=2, w_wd=10, ...)
```

```
parw = Parameter(init='constant', mean=0)
m.add(Dense(10, w_param=parw, w_lr=1, w_wd=1, b_value=1, ...)
```

### Other classes

* Store
* Algorithm
* Updater
* SGD
* AdaGrad
* Cluster
