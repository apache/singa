    SINGAROOT/tool
    |-- pb2 (has job_pb2.py)
    |-- python 
        |-- model.py 
        |-- ex_cifar10_cnn.py 
        |-- ex_mnist_mlp.py 
        |-- datasets 
            |-- cifar10.py 
            |-- mnist.py 
        |-- utils 
            |-- utility.py 
            |-- message.py 

### Layer class (inherited)

* Data
* Dense
* Activation
* Convolution2D
* MaxPooling2D
* AvgPooling2D
* LRN2D 
* Dropout

### Other classes

* Store
* Parameter
* SGD
* Cluster

### Model class

* Model class has `jobconf` (JobProto) and `layers` (layer list)

Methods in Model class

* add
	* add Layer into Model

* compile	
	* set Updater (i.e., optimizer) and Cluster (i.e., topology) components

* fit 
	* set Training data and parameter values for the training
		* (optional) set Validatiaon data and parameter values
	* set Train_one_batch component

* evaluate
	* set Testing data and parameter values for the testing
	* run singa (train/test/validation) via a command script
	* recieve train/test/validation results, e.g., accuracy 
	* [IN PROGRESS] run singa via a wrapper for Driver class

* [IN PROGRESS] run singa with checkpoint
* [IN PROGRESS] run singa for particular tasks, e.g., classification/prediction


## MLP Example

An example (to generate job.conf for mnist)

```
X_train, X_test, workspace = mnist.load_data()

m = Sequential('mlp')  # inherited from Model 

par = Parameter(init='uniform', low=-0.05, high=0.05)
m.add(Dense(2500, w_param=par, b_param=par))
m.add(Activation('tanh'))
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(500, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(10, w_param=par, b_param=par, activation='softmax'))

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)
m.fit(X_train, train_steps=1000, disp_freq=10)
result = m.evaluate(X_test, batch_size=100, test_steps=10, test_freq=60)
```


## CNN Example

An example (to generate job.conf for cifar10)

```
X_train, X_test, workspace = cifar10.load_data()

m = Sequential('cnn')

parw = Parameter(init='gauss', std=0.0001)
parb = Parameter(init='const', value=0)
m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb, b_lr=2))
m.add(MaxPooling2D(pool_size(3,3), stride=2))
m.add(Activation('relu'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size(3,3), stride=2))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution(64, 5, 1, 2, w_param=parw, b_param=parb.setval(lr_scale=1)))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size(3,3), stride=2))

parw.setval(wd_scale=250)
parb.setval(lr_scale=2, wd_scale=0)
m.add(Dense(10, w_param=parw, b_param=parb, activation='softmax'))

sgd = SGD(decay=0.004, lr_type='fixed', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster(workspace)
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, 1000, disp_freq=30)
result = m.evaluate(X_test, 1000, test_steps=30, test_freq=300)
```

### TIPS

Hidden layers for MLP can be written as
```
par = Param(init='uniform', low=-0.05, high=0.05)
for n in [2500, 2000, 1500, 1000, 500]:
  m.add(Dense(n, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(10, w_param=par, b_param=par, activation='softmax'))
```

Alternative ways to write the hidden layers
```
m.add(Dense(2500, w_param=par, b_param=par))
m.add(Activation('tanh'))
```
```
m.add(Dense(2500, init='uniform', activation='softmax'))
```
```
m.add(Dense(2500, w_param=Param(init='uniform'), b_param=Param(init='gaussian')))
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

Users need to set parameter configuration and initial values. For example,

* Parameter configuration
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

Several ways to set Parameter values
```
parw = Parameter(lr=2, wd=10, init='constant', value=0)
m.add(Dense(10, w_param=parw, ...)
```
```
parw = Parameter(init='constant', value=0)
m.add(Dense(10, w_param=parw, w_lr=2, w_wd=10, ...)
```



