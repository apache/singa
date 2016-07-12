# Interactive Training using Python

---

`Layer` class ([layer.py](layer.py)) has the following methods for an interactive training.
For the basic usage of Python binding features, please refer to [python.md](python.md).

**ComputeFeature(self, \*srclys)**

* This method creates and sets up singa::Layer and maintains its source layers, then call singa::Layer::ComputeFeature(...) for data transformation.

	* `*srclys`: (an arbtrary number of) source layers

**ComputeGradient(self)**

* This method creates calls singa::Layer::ComputeGradient(...) for gradient computation.

**GetParams(self)**

* This method calls singa::Layer::GetParam() to retrieve parameter values of the layer. Currently, it returns weight and bias. Each parameter is a 2D numpy array.

**SetParams(self, \*params)**

* This method sets parameter values of the layer.
	* `*params`: (an arbitrary number of) parameters, each of which is a 2D numpy array. Typically, it sets weight and bias, 2D numpy array.

* * *

`Dummy` class is a subclass of `Layer`, which is provided to fetch input data and/or label information.
Specifically, it creates singa::DummyLayer.

**Feed(self, shape, data, aux_data)**

* This method sets input data and/or auxiary data such as labels.

	* `shape`: the shape (width and height) of dataset
	* `data`: input dataset
	* `aux_data`: auxiary dataset (e.g., labels)

In addition, `Dummy` class has two subclasses named `ImageInput` and `LabelInput`.

* `ImageInput` class will take three arguments as follows.

	**\_\_init__(self, height=None, width=None, nb_channel=1)**

* Both `ImageInput` and `LabelInput` classes have their own Feed method to call Feed of Dummy class.

	**Feed(self, data)**


<!--

Users can save or load model parameter (e.g., weight and bias) at anytime during training.
The following methods are provided in `model.py`.

**save_model_parameter(step, fout, neuralnet)**

* This method saves model parameters into the specified checkpoint (fout).

	* `step`: the step id of training
	* `fout`: the name of checkpoint (output filename)
	* `neuralnet`: neural network model, i.e., a list of layers

**load_model_parameter(fin, neuralnet, batchsize=1, data_shape=None)**

* This method loads model parameters from the specified checkpoint (fin).

	* `fin`: the name of checkpoint (input filename)
	* `neuralnet`: neural network model, i.e., a list of layers
	* `batchsize`:
	* `data_shape`:
-->

* * *

## Example scripts for the interactive training

Two example scripts are provided at [`train_mnist.py`]() and [`train_cifar10.py`](), one is training MLP model for MNIST dataset, and another is training CNN model for CIFAR10 dataset.

* Assume that `nn` is a neural network model, i.e., a list of layers. Currently, this examples considers sequential models. Example MLP and CNN are shown below.

* `load_dataset()` method loads input data and corresponding labels, each of which is a 2D numpy array.
For example, loading MNIST dataset returns x: [60000 x 784] and y: [60000 x 1]. Loading CIFAR10 dataset, x: [10000 x 3072] and y: [10000 x 1].

* `sgd` is an Updater instance. Please see [`python.md`](python.md) and [`model.py`]() for more details.

#### Basic steps for the interactive training

* Step 1: Prepare batchsized data and corresponding label information, and then input the data using `Feed()` method.

* Step 2: (a) Transform data according to neuralnet (nn) structure using `ComputeFeature()`. Note that this example considers a sequential model, so it uses a simple loop. (b) Users need to provide `label` information for loss layer to compute loss function. (c) Users can print out the training performance, e.g., loss and accuracy.

* Step 3: Compute gradient in a reverse order of neuralnet (nn) structure using `ComputeGradient()`.

* Step 4: Update parameters, e.g., weight and bias, of layers using `Update()` of the updater.

Here is an example script for the interactive training.
```
bsize = 64      # batchsize
disp_freq = 10  # step to show the training accuracy

x, y = load_dataset()

for i in range(x.shape[0] / bsize):

	# (Step1) Input data containing "bsize" samples
	xb, yb = x[i*bsize:(i+1)*bsize, :], y[i*bsize:(i+1)*bsize, :]
	nn[0].Feed(xb)
	label.Feed(yb)

	# (Step2-a) Transform data according to the neuralnet (nn) structure
	for h in range(1, len(nn)):
		nn[h].ComputeFeature(nn[h-1])

	# (Step2-b) Provide label to compute loss function
	loss.ComputeFeature(nn[-1], label)

	# (Step2-c) Print out performance, e.g., loss and accuracy
	if (i+1) % disp_freq == 0:
		print '  Step {:>3}: '.format(i+1),
		loss.display()

	# (Step3) Compute gradient in a reverse order
	loss.ComputeGradient()
	for h in range(len(nn)-1, 0, -1):
		nn[h].ComputeGradient()
		# (Step 4) Update parameter
		sgd.Update(i+1, nn[h])
```        

<a id="model"></a>
### <a href="#model">Example MLP</a>  

Here is an example MLP model with 5 fully-connected hidden layers.
Please refer to [`python.md`](python.md) and [`layer.py`]() for more details about layer definition. `SGD()` is an updater defined in [`model.py`]().

```
input = ImageInput(28, 28) # image width and height
label = LabelInput()

nn = []
nn.append(input)
nn.append(Dense(2500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(2000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(10, init='uniform'))
loss = Loss('softmaxloss')

sgd = SGD(lr=0.001, lr_type='step')

```

### <a href="#model2">Example CNN</a>  

Here is an example MLP model with 3 convolution and pooling layers.
Please refer to [`python.md`]() and [`layer.py`]() for more details about layer definition. `SGD()` is an updater defined in [`model.py`]().

```
input = ImageInput(32, 32, 3) # image width, height, channel
label = LabelInput()

nn = []
nn.append(input)
nn.append(Convolution2D(32, 5, 1, 2, w_std=0.0001, b_lr=2))
nn.append(MaxPooling2D(pool_size=(3,3), stride=2))
nn.append(Activation('relu'))
nn.append(LRN2D(3, alpha=0.00005, beta=0.75))
nn.append(Convolution2D(32, 5, 1, 2, b_lr=2))
nn.append(Activation('relu'))
nn.append(AvgPooling2D(pool_size=(3,3), stride=2))
nn.append(LRN2D(3, alpha=0.00005, beta=0.75))
nn.append(Convolution2D(64, 5, 1, 2))
nn.append(Activation('relu'))
nn.append(AvgPooling2D(pool_size=(3,3), stride=2))
nn.append(Dense(10, w_wd=250, b_lr=2, b_wd=0))
loss = Loss('softmaxloss')

sgd = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
```
