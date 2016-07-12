# Parameters

---

A `Param` object in SINGA represents a set of parameters, e.g., a weight matrix
or a bias vector. *Basic user guide* describes how to configure for a `Param`
object, and *Advanced user guide* provides details on implementing users'
parameter initialization methods.

## Basic user guide

The configuration of a Param object is inside a layer configuration, as the
`Param` are associated with layers. An example configuration is like

    layer {
      ...
      param {
        name : "p1"
        init {
          type : kConstant
          value: 1
        }
      }
    }

The [SGD algorithm](overview.html) starts with initializing all
parameters according to user specified initialization method (the `init` field).
For the above example,
all parameters in `Param` "p1" will be initialized to constant value 1. The
configuration fields of a Param object is defined in [ParamProto](../api/classsinga_1_1ParamProto.html):

  * name, an identifier string. It is an optional field. If not provided, SINGA
  will generate one based on layer name and its order in the layer.
  * init, field for setting initialization methods.
  * share_from, name of another `Param` object, from which this `Param` will share
  configurations and values.
  * lr_scale, float value to be multiplied with the learning rate when
  [updating the parameters](updater.html)
  * wd_scale, float value to be multiplied with the weight decay when
  [updating the parameters](updater.html)

There are some other fields that are specific to initialization methods.

### Initialization methods

Users can set the `type` of `init` use the following built-in initialization
methods,

  * `kConst`, set all parameters of the Param object to a constant value

        type: kConst
        value: float  # default is 1

  * `kGaussian`, initialize the parameters following a Gaussian distribution.

        type: kGaussian
        mean: float # mean of the Gaussian distribution, default is 0
        std: float # standard variance, default is 1
        value: float # default 0

  * `kUniform`, initialize the parameters following an uniform distribution

        type: kUniform
        low: float # lower boundary, default is -1
        high: float # upper boundary, default is 1
        value: float # default 0

  * `kGaussianSqrtFanIn`, initialize `Param` objects with two dimensions (i.e.,
  matrix) using `kGaussian` and then
  multiple each parameter with `1/sqrt(fan_in)`, where`fan_in` is the number of
  columns of the matrix.

  * `kUniformSqrtFanIn`, the same as `kGaussianSqrtFanIn` except that the
  distribution is an uniform distribution.

  * `kUniformFanInOut`, initialize matrix `Param` objects using `kUniform` and then
  multiple each parameter with `sqrt(6/(fan_in + fan_out))`, where`fan_in +
  fan_out` sums up the number of columns and rows of the matrix.

For all above initialization methods except `kConst`, if their `value` is not
1, every parameter will be multiplied with `value`. Users can also implement
their own initialization method following the *Advanced user guide*.


## Advanced user guide

This sections describes the details on implementing new parameter
initialization methods.

### Base ParamGenerator
All initialization methods are implemented as
subclasses of the base `ParamGenerator` class.

    class ParamGenerator {
     public:
      virtual void Init(const ParamGenProto&);
      void Fill(Param*);

     protected:
      ParamGenProto proto_;
    };

Configurations of the initialization method is in `ParamGenProto`. The `Fill`
function fills the `Param` object (passed in as an argument).

### New ParamGenerator subclass

Similar to implement a new Layer subclass, users can define a configuration
protocol message,

    # in user.proto
    message FooParamProto {
      optional int32 x = 1;
    }
    extend ParamGenProto {
      optional FooParamProto fooparam_conf =101;
    }

The configuration of `Param` would be

    param {
      ...
      init {
        user_type: 'FooParam" # must use user_type for user defined methods
        [fooparam_conf] { # must use brackets for configuring user defined messages
          x: 10
        }
      }
    }

The subclass could be declared as,

    class FooParamGen : public ParamGenerator {
     public:
      void Fill(Param*) override;
    };

Users can access the configuration fields in `Fill` by

    int x = proto_.GetExtension(fooparam_conf).x();

To use the new initialization method, users need to register it in the
[main function](programming-guide.html).

    driver.RegisterParamGenerator<FooParamGen>("FooParam")  # must be consistent with the user_type in configuration

{% comment %}
### Base Param class

### Members

    int local_version_;
    int slice_start_;
    vector<int> slice_offset_, slice_size_;

    shared_ptr<Blob<float>> data_;
    Blob<float> grad_;
    ParamProto proto_;

Each Param object has a local version and a global version (inside the data
Blob). These two versions are used for synchronization. If multiple Param
objects share the same values, they would have the same `data_` field.
Consequently, their global version is the same. The global version is updated
by [the stub thread](communication.html). The local version is
updated in `Worker::Update` function which assigns the global version to the
local version. The `Worker::Collect` function is blocked until the global
version is larger than the local version, i.e., when `data_` is updated. In
this way, we synchronize workers sharing parameters.

In Deep learning models, some Param objects are 100 times larger than others.
To ensure the load-balance among servers, SINGA slices large Param objects. The
slicing information is recorded by `slice_*`. Each slice is assigned a unique
ID starting from 0. `slice_start_` is the ID of the first slice of this Param
object. `slice_offset_[i]` is the offset of the i-th slice in this Param
object. `slice_size_[i]` is the size of the i-th slice. These slice information
is used to create messages for transferring parameter values or gradients to
different servers.

Each Param object has a `grad_` field for gradients. Param objects do not share
this Blob although they may share `data_`.  Because each layer containing a
Param object would contribute gradients. E.g., in RNN, the recurrent layers
share parameters values, and the gradients used for updating are averaged from all recurrent
these recurrent layers. In SINGA, the stub thread will aggregate local
gradients for the same Param object. The server will do a global aggregation
of gradients for the same Param object.

The `proto_` field has some meta information, e.g., name and ID. It also has a
field called `owner` which is the ID of the Param object that shares parameter
values with others.

### Functions
The base Param class implements two sets of functions,

    virtual void InitValues(int version = 0);  // initialize values according to `init_method`
    void ShareFrom(const Param& other);  // share `data_` from `other` Param
    --------------
    virtual Msg* GenGetMsg(bool copy, int slice_idx);
    virtual Msg* GenPutMsg(bool copy, int slice_idx);
    ... // other message related functions.

Besides the functions for processing the parameter values, there is a set of
functions for generating and parsing messages. These messages are for
transferring parameter values or gradients between workers and servers. Each
message corresponds to one Param slice. If `copy` is false, it means the
receiver of this message is in the same process as the sender. In such case,
only pointers to the memory of parameter value (or gradient) are wrapped in
the message; otherwise, the parameter values (or gradients) should be copied
into the message.


## Implementing Param subclass
Users can extend the base Param class to implement their own parameter
initialization methods and message transferring protocols. Similar to
implementing a new Layer subclasses, users can create google protocol buffer
messages for configuring the Param subclass. The subclass, denoted as FooParam
should be registered in main.cc,

    dirver.RegisterParam<FooParam>(kFooParam);  // kFooParam should be different to 0, which is for the base Param type


  * type, an integer representing the `Param` type. Currently SINGA provides one
    `Param` implementation with type 0 (the default type). If users want
    to use their own Param implementation, they should extend the base Param
    class and configure this field with `kUserParam`

{% endcomment %}
