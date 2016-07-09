# Train-One-Batch

---

For each SGD iteration, every worker calls the `TrainOneBatch` function to
compute gradients of parameters associated with local layers (i.e., layers
dispatched to it).  SINGA has implemented two algorithms for the
`TrainOneBatch` function. Users select the corresponding algorithm for
their model in the configuration.

## Basic user guide

### Back-propagation

[BP algorithm](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) is used for
computing gradients of feed-forward models, e.g., [CNN](cnn.html)
and [MLP](mlp.html), and [RNN](rnn.html) models in SINGA.


    # in job.conf
    alg: kBP

To use the BP algorithm for the `TrainOneBatch` function, users just simply
configure the `alg` field with `kBP`. If a neural net contains user-defined
layers, these layers must be implemented properly be to consistent with the
implementation of the BP algorithm in SINGA (see below).


### Contrastive Divergence

[CD algorithm](http://www.cs.toronto.edu/~fritz/absps/nccd.pdf) is used for
computing gradients of energy models like RBM.

    # job.conf
    alg: kCD
    cd_conf {
      cd_k: 2
    }

To use the CD algorithm for the `TrainOneBatch` function, users just configure
the `alg` field to `kCD`. Uses can also configure the Gibbs sampling steps in
the CD algorthm through the `cd_k` field. By default, it is set to 1.



## Advanced user guide

### Implementation of BP

The BP algorithm is implemented in SINGA following the below pseudo code,

    BPTrainOnebatch(step, net) {
      // forward propagate
      foreach layer in net.local_layers() {
        if IsBridgeDstLayer(layer)
          recv data from the src layer (i.e., BridgeSrcLayer)
        foreach param in layer.params()
          Collect(param) // recv response from servers for last update

        layer.ComputeFeature(kForward)

        if IsBridgeSrcLayer(layer)
          send layer.data_ to dst layer
      }
      // backward propagate
      foreach layer in reverse(net.local_layers) {
        if IsBridgeSrcLayer(layer)
          recv gradient from the dst layer (i.e., BridgeDstLayer)
          recv response from servers for last update

        layer.ComputeGradient()
        foreach param in layer.params()
          Update(step, param) // send param.grad_ to servers

        if IsBridgeDstLayer(layer)
          send layer.grad_ to src layer
      }
    }


It forwards features through all local layers (can be checked by layer
partition ID and worker ID) and backwards gradients in the reverse order.
[BridgeSrcLayer](layer.html#bridgesrclayer--bridgedstlayer)
(resp. `BridgeDstLayer`) will be blocked until the feature (resp.
gradient) from the source (resp. destination) layer comes. Parameter gradients
are sent to servers via `Update` function. Updated parameters are collected via
`Collect` function, which will be blocked until the parameter is updated.
[Param](param.html) objects have versions, which can be used to
check whether the `Param` objects have been updated or not.

Since RNN models are unrolled into feed-forward models, users need to implement
the forward propagation in the recurrent layer's `ComputeFeature` function,
and implement the backward propagation in the recurrent layer's `ComputeGradient`
function. As a result, the whole `TrainOneBatch` runs
[back-propagation through time (BPTT)](https://en.wikipedia.org/wiki/Backpropagation_through_time)  algorithm.

### Implementation of CD

The CD algorithm is implemented in SINGA following the below pseudo code,

    CDTrainOneBatch(step, net) {
      # positive phase
      foreach layer in net.local_layers()
        if IsBridgeDstLayer(layer)
          recv positive phase data from the src layer (i.e., BridgeSrcLayer)
        foreach param in layer.params()
          Collect(param)  // recv response from servers for last update
        layer.ComputeFeature(kPositive)
        if IsBridgeSrcLayer(layer)
          send positive phase data to dst layer

      # negative phase
      foreach gibbs in [0...layer_proto_.cd_k]
        foreach layer in net.local_layers()
          if IsBridgeDstLayer(layer)
            recv negative phase data from the src layer (i.e., BridgeSrcLayer)
          layer.ComputeFeature(kPositive)
          if IsBridgeSrcLayer(layer)
            send negative phase data to dst layer

      foreach layer in net.local_layers()
        layer.ComputeGradient()
        foreach param in layer.params
          Update(param)
    }

Parameter gradients are computed after the positive phase and negative phase.

### Implementing a new algorithm

SINGA implements BP and CD by creating two subclasses of
the [Worker](../api/classsinga_1_1Worker.html) class:
[BPWorker](../api/classsinga_1_1BPWorker.html)'s `TrainOneBatch` function implements the BP
algorithm; [CDWorker](../api/classsinga_1_1CDWorker.html)'s `TrainOneBatch` function implements the CD
algorithm. To implement a new algorithm for the `TrainOneBatch` function, users
need to create a new subclass of the `Worker`, e.g.,

    class FooWorker : public Worker {
      void TrainOneBatch(int step, shared_ptr<NeuralNet> net, Metric* perf) override;
      void TestOneBatch(int step, Phase phase, shared_ptr<NeuralNet> net, Metric* perf) override;
    };

The `FooWorker` must implement the above two functions for training one
mini-batch and testing one mini-batch. The `perf` argument is for collecting
training or testing performance, e.g., the objective loss or accuracy. It is
passed to the `ComputeFeature` function of each layer.

Users can define some fields for users to configure

    # in user.proto
    message FooWorkerProto {
      optional int32 b = 1;
    }

    extend JobProto {
      optional FooWorkerProto foo_conf = 101;
    }

    # in job.proto
    JobProto {
      ...
      extension 101..max;
    }

It is similar as [adding configuration fields for a new layer](layer.html#implementing-a-new-layer-subclass).

To use `FooWorker`, users need to register it in the [main.cc](programming-guide.html)
and configure the `alg` and `foo_conf` fields,

    # in main.cc
    const int kFoo = 3; // worker ID, must be different to that of CDWorker and BPWorker
    driver.RegisterWorker<FooWorker>(kFoo);

    # in job.conf
    ...
    alg: 3
    [foo_conf] {
      b = 4;
    }
