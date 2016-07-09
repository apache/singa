# Layers

---

Layer is a core abstraction in SINGA. It performs a variety of feature
transformations for extracting high-level features, e.g., loading raw features,
parsing RGB values, doing convolution transformation, etc.

The *Basic user guide* section introduces the configuration of a built-in
layer. *Advanced user guide* explains how to extend the base Layer class to
implement users' functions.

## Basic user guide

### Layer configuration

The configurations of three layers from the [MLP example](mlp.html) is shown below,

    layer {
      name: "data"
      type: kShardData
      sharddata_conf { }
      exclude: kTest
      partition_dim : 0
    }
    layer{
      name: "mnist"
      type: kMnist
      srclayers: "data"
      mnist_conf { }
    }
    layer{
      name: "fc1"
      type: kInnerProduct
      srclayers: "mnist"
      innerproduct_conf{ }
      param{ }
      param{ }
    }

There are some common fields for all kinds of layers:

  * `name`: a string used to differentiate two layers.
  * `type`: an integer used for identifying a Layer subclass. The types of built-in
  layers are listed in LayerType (defined in job.proto).
  For user-defined layer subclasses, `user_type` of string should be used instead of `type`.
  The detail is explained in the [last section](#newlayer) of this page.
  * `srclayers`: one or more layer names, for identifying the source layers.
  In SINGA, all connections are [converted](neural-net.html) to directed connections.
  * `exclude`: an enumerate value of type [Phase](), can be {kTest, kValidation,
  kTrain}. It is used to filter this layer when creating the
  [NeuralNet](neural-net.html) for the excluding phase. E.g.,
  the "data" layer would be filtered when creating the NeuralNet instance for test phase.
  * `param`: configuration for a [Param](param.html) instance.
  There can be multiple Param objects in one layer.
  * `partition_dim`: integer value indicating the partition dimension of this
  layer. -1 (the default value) for no partitioning, 0 for partitioning on batch dimension, 1 for
  partitioning on feature dimension. It is used by
  [CreateGraph](neural-net.html) for partitioning the neural net.

Different layers may have different configurations. These configurations
are defined in `<type>_conf`.  E.g., the "data" layer has `sharddata_conf` and "fc1" layer has
`innerproduct_conf`. The subsequent sections
explain the functionality of each built-in layer and how to configure it,

### Built-in Layer subclasses
SINGA has provided many built-in layers, which can be used directly to create neural nets.
These layers are categorized according to their functionalities,

  * Data layers for loading records (e.g., images) from [disk], HDFS or network into memory.
  * Parser layers for parsing features, labels, etc. from records, into [Blob](../api-v0.1.0/classsinga_1_1Blob.html).
  * Neuron layers for feature transformation, e.g., [convolution](../api-v0.1.0/classsinga_1_1ConvolutionLayer.html), [pooling](../api-v0.1.0/classsinga_1_1PoolingLayer.html), dropout, etc.
  * Loss layers for measuring the training objective loss, e.g., [cross entropy-loss] or [Euclidean loss].
  * Output layers for outputting the prediction results (e.g., probabilities of each category) onto disk or network.
  * Connection layers for connecting layers when the neural net is partitioned.

#### Input layers

Input layers load training/test data from disk or other places (e.g., HDFS or network)
into memory.

##### DataLayer

DataLayer loads training/testing data as [Record](data.html)s, which
are parsed by parser layers.

##### ShardDataLayer

[ShardDataLayer](../api-v0.1.0/classsinga_1_1ShardDataLayer.html) is a subclass of DataLayer,
which reads Records from disk file. The file should be created using
[DataShard](../api-v0.1.0/classsinga_1_1DataShard.html)
class. With the data file prepared, users configure the layer as

    type: kShardData
    sharddata_conf {
      path: "path to data shard folder"
      batchsize: int
      random_skip: int
    }

`batchsize` specifies the number of records to be trained for one mini-batch.
The first `rand() % random_skip` `Record`s will be skipped at the first
iteration. This is to enforce that different workers work on different Records.

##### LMDBDataLayer

[LMDBDataLayer] is similar to ShardDataLayer, except that the Records are
loaded from LMDB.

    type: kLMDBData
    lmdbdata_conf {
      path: "path to LMDB folder"
      batchsize: int
      random_skip: int
    }

##### ParserLayer

It get a vector of Records from DataLayer and parse features into
a Blob.

    virtual void ParseRecords(Phase phase, const vector<Record>& records, Blob<float>* blob) = 0;


##### LabelLayer

[LabelLayer](../api-v0.1.0/classsinga_1_1LabelLayer.html) is a subclass of ParserLayer.
It parses a single label from each Record. Consequently, it
will put $b$ (mini-batch size) values into the Blob. It has no specific configuration fields.


##### MnistImageLayer
[MnistImageLayer] is a subclass of ParserLayer. It parses the pixel values of
each image from the MNIST dataset. The pixel
values may be normalized as `x/norm_a - norm_b`. For example, if `norm_a` is
set to 255 and `norm_b` is set to 0, then every pixel will be normalized into
[0, 1].

    type: kMnistImage
    mnistimage_conf {
      norm_a: float
      norm_b: float
    }

##### RGBImageLayer
[RGBImageLayer](../api-v0.1.0/classsinga_1_1RGBImageLayer.html) is a subclass of ParserLayer.
It parses the RGB values of one image from each Record. It may also
apply some transformations, e.g., cropping, mirroring operations. If the
`meanfile` is specified, it should point to a path that contains one Record for
the mean of each pixel over all training images.

    type: kRGBImage
    rgbimage_conf {
      scale: float
      cropsize: int  # cropping each image to keep the central part with this size
      mirror: bool  # mirror the image by set image[i,j]=image[i,len-j]
      meanfile: "Image_Mean_File_Path"
    }

#### PrefetchLayer

[PrefetchLayer](../api-v0.1.0/classsinga_1_1PrefetchLayer.html) embeds other input layers
to do data prefeching.  It will launch a thread to call the embedded layers to load and extract features.
It ensures that the I/O task and computation task can work simultaneously.
One example PrefetchLayer configuration is,

    layer {
      name: "prefetch"
      type: kPrefetch
      sublayers {
        name: "data"
        type: kShardData
        sharddata_conf { }
      }
      sublayers {
        name: "rgb"
        type: kRGBImage
        srclayers:"data"
        rgbimage_conf { }
      }
      sublayers {
        name: "label"
        type: kLabel
        srclayers: "data"
      }
      exclude:kTest
    }

The layers on top of the PrefetchLayer should use the name of the embedded
layers as their source layers. For example, the "rgb" and "label" should be
configured to the `srclayers` of other layers.

#### Neuron Layers

Neuron layers conduct feature transformations.

##### ConvolutionLayer

[ConvolutionLayer](../api-v0.1.0/classsinga_1_1ConvolutionLayer.html) conducts convolution transformation.

    type: kConvolution
    convolution_conf {
      num_filters: int
      kernel: int
      stride: int
      pad: int
    }
    param { } # weight/filter matrix
    param { } # bias vector

The int value `num_filters` stands for the count of the applied filters; the int
value `kernel` stands for the convolution kernel size (equal width and height);
the int value `stride` stands for the distance between the successive filters;
the int value `pad` pads each with a given int number of pixels border of
zeros.

##### InnerProductLayer

[InnerProductLayer](../api-v0.1.0/classsinga_1_1InnerProductLayer.html) is fully connected with its (single) source layer.
Typically, it has two parameter fields, one for weight matrix, and the other
for bias vector. It rotates the feature of the source layer (by multiplying with weight matrix) and
shifts it (by adding the bias vector).

    type: kInnerProduct
    innerproduct_conf {
      num_output: int
    }
    param { } # weight matrix
    param { } # bias vector


##### PoolingLayer

[PoolingLayer](../api-v0.1.0/classsinga_1_1PoolingLayer.html) is used to do a normalization (or averaging or sampling) of the
feature vectors from the source layer.

    type: kPooling
    pooling_conf {
      pool: AVE|MAX // Choose whether use the Average Pooling or Max Pooling
      kernel: int   // size of the kernel filter
      pad: int      // the padding size
      stride: int   // the step length of the filter
    }

The pooling layer has two methods: Average Pooling and Max Pooling.
Use the enum AVE and MAX to choose the method.

  * Max Pooling selects the max value for each filtering area as a point of the
  result feature blob.
  * Average Pooling averages all values for each filtering area at a point of the
    result feature blob.

##### ReLULayer

[ReLuLayer](../api-v0.1.0/classsinga_1_1ReLULayer.html) has rectified linear neurons, which conducts the following
transformation, `f(x) = Max(0, x)`. It has no specific configuration fields.

##### STanhLayer

[STanhLayer](../api-v0.1.0/classsinga_1_1TanhLayer.html) uses the scaled tanh as activation function, i.e., `f(x)=1.7159047* tanh(0.6666667 * x)`.
It has no specific configuration fields.

##### SigmoidLayer

[SigmoidLayer] uses the sigmoid (or logistic) as activation function, i.e.,
`f(x)=sigmoid(x)`.  It has no specific configuration fields.


##### Dropout Layer
[DropoutLayer](../api-v0.1.0/asssinga_1_1DropoutLayer.html) is a layer that randomly dropouts some inputs.
This scheme helps deep learning model away from over-fitting.

    type: kDropout
    dropout_conf {
      dropout_ratio: float # dropout probability
    }

##### LRNLayer
[LRNLayer](../api-v0.1.0/classsinga_1_1LRNLayer.html), (Local Response Normalization), normalizes over the channels.

    type: kLRN
    lrn_conf {
      local_size: int
      alpha: float  // scaling parameter
      beta: float   // exponential number
    }

`local_size` specifies  the quantity of the adjoining channels which will be summed up.
 For `WITHIN_CHANNEL`, it means the side length of the space region which will be summed up.


#### Loss Layers

Loss layers measures the objective training loss.

##### SoftmaxLossLayer

[SoftmaxLossLayer](../api-v0.1.0/classsinga_1_1SoftmaxLossLayer.html) is a combination of the Softmax transformation and
Cross-Entropy loss. It applies Softmax firstly to get a prediction probability
for each output unit (neuron) and compute the cross-entropy against the ground truth.
It is generally used as the final layer to generate labels for classification tasks.

    type: kSoftmaxLoss
    softmaxloss_conf {
      topk: int
    }

The configuration field `topk` is for selecting the labels with `topk`
probabilities as the prediction results. It is tedious for users to view the
prediction probability of every label.

#### ConnectionLayer

Subclasses of ConnectionLayer are utility layers that connects other layers due
to neural net partitioning or other cases.

##### ConcateLayer

[ConcateLayer](../api-v0.1.0/classsinga_1_1ConcateLayer.html) connects more than one source layers to concatenate their feature
blob along given dimension.

    type: kConcate
    concate_conf {
      concate_dim: int  // define the dimension
    }

##### SliceLayer

[SliceLayer](../api-v0.1.0/classsinga_1_1SliceLayer.html) connects to more than one destination layers to slice its feature
blob along given dimension.

    type: kSlice
    slice_conf {
      slice_dim: int
    }

##### SplitLayer

[SplitLayer](../api-v0.1.0/classsinga_1_1SplitLayer.html) connects to more than one destination layers to replicate its
feature blob.

    type: kSplit
    split_conf {
      num_splits: int
    }

##### BridgeSrcLayer & BridgeDstLayer

[BridgeSrcLayer](../api-v0.1.0/classsinga_1_1BridgeSrcLayer.html) &
[BridgeDstLayer](../api-v0.1.0/classsinga_1_1BridgeDstLayer.html) are utility layers assisting data (e.g., feature or
gradient) transferring due to neural net partitioning. These two layers are
added implicitly. Users typically do not need to configure them in their neural
net configuration.

### OutputLayer

It write the prediction results or the extracted features into file, HTTP stream
or other places. Currently SINGA has not implemented any specific output layer.

## Advanced user guide

The base Layer class is introduced in this section, followed by how to
implement a new Layer subclass.

### Base Layer class

#### Members

    LayerProto layer_conf_;
    Blob<float> data_, grad_;

The base layer class keeps the user configuration in `layer_conf_`.
Almost all layers has $b$ (mini-batch size) feature vectors, which are stored
in the `data_` [Blob](../api-v0.1.0/classsinga_1_1Blob.html) (A Blob is a chunk of memory space, proposed in
[Caffe](http://caffe.berkeleyvision.org/)).
There are layers without feature vectors; instead, they use other
layers' feature vectors. In this case, the `data_` field is not used.
The `grad_` Blob is for storing the gradients of the
objective loss w.r.t. the `data_` Blob. It is necessary in [BP algorithm](../api-v0.1.0/classsinga_1_1BPWorker.html),
hence we put it as a member of the base class. For [CD algorithm](../api-v0.1.0/classsinga_1_1CDWorker.html), the `grad_`
field is not used; instead, the layer from RBM may have a Blob for the positive
phase feature and a Blob for the negative phase feature. For a recurrent layer
in RNN, the feature blob contains one vector per internal layer.

If a layer has parameters, these parameters are declared using type
[Param](param.html). Since some layers do not have
parameters, we do not declare any `Param` in the base layer class.

#### Functions

    virtual void Setup(const LayerProto& conf, const vector<Layer*>& srclayers);
    virtual void ComputeFeature(int flag, const vector<Layer*>& srclayers) = 0;
    virtual void ComputeGradient(int flag, const vector<Layer*>& srclayers) = 0;

The `Setup` function reads user configuration, i.e. `conf`, and information
from source layers, e.g., mini-batch size,  to set the
shape of the `data_` (and `grad_`) field as well
as some other layer specific fields.
<!---
If `npartitions` is larger than 1, then
users need to reduce the sizes of `data_`, `grad_` Blobs or Param objects. For
example, if the `partition_dim=0` and there is no source layer, e.g., this
layer is a (bottom) data layer, then its `data_` and `grad_` Blob should have
`b/npartitions` feature vectors; If the source layer is also partitioned on
dimension 0, then this layer should have the same number of feature vectors as
the source layer. More complex partition cases are discussed in
[Neural net partitioning](neural-net.html#neural-net-partitioning). Typically, the
Setup function just set the shapes of `data_` Blobs and Param objects.
-->
Memory will not be allocated until computation over the data structure happens.

The `ComputeFeature` function evaluates the feature blob by transforming (e.g.
convolution and pooling) features from the source layers.  `ComputeGradient`
computes the gradients of parameters associated with this layer.  These two
functions are invoked by the [TrainOneBatch](train-one-batch.html)
function during training. Hence, they should be consistent with the
`TrainOneBatch` function. Particularly, for feed-forward and RNN models, they are
trained using [BP algorithm](train-one-batch.html#back-propagation),
which requires each layer's `ComputeFeature`
function to compute `data_` based on source layers, and requires each layer's
`ComputeGradient` to compute gradients of parameters and source layers'
`grad_`. For energy models, e.g., RBM, they are trained by
[CD algorithm](train-one-batch.html#contrastive-divergence), which
requires each layer's `ComputeFeature` function to compute the feature vectors
for the positive phase or negative phase depending on the `phase` argument, and
requires the `ComputeGradient` function to only compute parameter gradients.
For some layers, e.g., loss layer or output layer, they can put the loss or
prediction result into the `metric` argument, which will be averaged and
displayed periodically.

### Implementing a new Layer subclass

Users can extend the Layer class or other subclasses to implement their own feature transformation
logics as long as the two virtual functions are overridden to be consistent with
the `TrainOneBatch` function. The `Setup` function may also be overridden to
read specific layer configuration.

The [RNNLM](rnn.html) provides a couple of user-defined layers. You can refer to them as examples.

#### Layer specific protocol message

To implement a new layer, the first step is to define the layer specific
configuration. Suppose the new layer is `FooLayer`, the layer specific
google protocol message `FooLayerProto` should be defined as

    # in user.proto
    package singa
    import "job.proto"
    message FooLayerProto {
      optional int32 a = 1;  // specific fields to the FooLayer
    }

In addition, users need to extend the original `LayerProto` (defined in job.proto of SINGA)
to include the `foo_conf` as follows.

    extend LayerProto {
      optional FooLayerProto foo_conf = 101;  // unique field id, reserved for extensions
    }

If there are multiple new layers, then each layer that has specific
configurations would have a `<type>_conf` field and takes one unique extension number.
SINGA has reserved enough extension numbers, e.g., starting from 101 to 1000.

    # job.proto of SINGA
    LayerProto {
      ...
      extensions 101 to 1000;
    }

With user.proto defined, users can use
[protoc](https://developers.google.com/protocol-buffers/) to generate the `user.pb.cc`
and `user.pb.h` files.  In users' code, the extension fields can be accessed via,

    auto conf = layer_proto_.GetExtension(foo_conf);
    int a = conf.a();

When defining configurations of the new layer (in job.conf), users should use
`user_type` for its layer type instead of `type`. In addition, `foo_conf`
should be enclosed in brackets.

    layer {
      name: "foo"
      user_type: "kFooLayer"  # Note user_type of user-defined layers is string
      [foo_conf] {      # Note there is a pair of [] for extension fields
        a: 10
      }
    }

#### New Layer subclass declaration

The new layer subclass can be implemented like the built-in layer subclasses.

    class FooLayer : public singa::Layer {
     public:
      void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
      void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
      void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

     private:
      //  members
    };

Users must override the two virtual functions to be called by the
`TrainOneBatch` for either BP or CD algorithm. Typically, the `Setup` function
will also be overridden to initialize some members. The user configured fields
can be accessed through `layer_conf_` as shown in the above paragraphs.

#### New Layer subclass registration

The newly defined layer should be registered in [main.cc](http://singa.incubator.apache.org/docs/programming-guide) by adding

    driver.RegisterLayer<FooLayer, std::string>("kFooLayer"); // "kFooLayer" should be matched to layer configurations in job.conf.

After that, the [NeuralNet](neural-net.html) can create instances of the new Layer subclass.
