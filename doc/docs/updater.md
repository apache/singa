# Updater

---

Every server in SINGA has an [Updater](../api/classsinga_1_1Updater.html)
instance that updates parameters based on gradients.
In this page, the *Basic user guide* describes the configuration of an updater.
The *Advanced user guide* present details on how to implement a new updater and a new
learning rate changing method.

## Basic user guide

There are many different parameter updating protocols (i.e., subclasses of
`Updater`). They share some configuration fields like

* `type`, an integer for identifying an updater;
* `learning_rate`, configuration for the
[LRGenerator](../api/classsinga_1_1LRGenerator.html) which controls the learning rate.
* `weight_decay`, the co-efficient for [L2 * regularization](http://deeplearning.net/tutorial/gettingstarted.html#regularization).
* [momentum](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/).

If you are not familiar with the above terms, you can get their meanings in
[this page provided by Karpathy](http://cs231n.github.io/neural-networks-3/#update).

### Configuration of built-in updater classes

#### Updater
The base `Updater` implements the [vanilla SGD algorithm](http://cs231n.github.io/neural-networks-3/#sgd).
Its configuration type is `kSGD`.
Users need to configure at least the `learning_rate` field.
`momentum` and `weight_decay` are optional fields.

    updater{
      type: kSGD
      momentum: float
      weight_decay: float
      learning_rate {
        ...
      }
    }

#### AdaGradUpdater

It inherits the base `Updater` to implement the
[AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf) algorithm.
Its type is `kAdaGrad`.
`AdaGradUpdater` is configured similar to `Updater` except
that `momentum` is not used.

#### NesterovUpdater

It inherits the base `Updater` to implements the
[Nesterov](http://arxiv.org/pdf/1212.0901v2.pdf) (section 3.5) updating protocol.
Its type is `kNesterov`.
`learning_rate` and `momentum` must be configured. `weight_decay` is an
optional configuration field.

#### RMSPropUpdater

It inherits the base `Updater` to implements the
[RMSProp algorithm](http://cs231n.github.io/neural-networks-3/#sgd) proposed by
[Hinton](http://www.cs.toronto.edu/%7Etijmen/csc321/slides/lecture_slides_lec6.pdf)(slide 29).
Its type is `kRMSProp`.

    updater {
      type: kRMSProp
      rmsprop_conf {
       rho: float # [0,1]
      }
    }

#### AdaDeltaUpdater

It inherits the base `Updater` to implements the
[AdaDelta](http://arxiv.org/abs/1212.5701) updating algorithm.
Its type is `kAdaDelta`.

    updater {
      type: kAdaDelta
      adadelta_conf {
       rho: float # [0,1]
      }
    }

#### Adam

It inherits the base `Updater` to implements the
[Adam](http://arxiv.org/pdf/1412.6980.pdf) updating algorithm.
Its type is `kAdam`.
`beta1` and `beta2` is floats, 0 < `beta` < 1, generally close to 1.

    updater {
      type: kAdam
      adam_conf {
       beta1: float # [0,1]
       beta2: float # [0,1]
      }
    }

#### AdaMax

It inherits the base `Updater` to implements the
[AdaMax](http://arxiv.org/pdf/1412.6980.pdf) updating algorithm.
Its type is `kAdamMax`.
`beta1` and `beta2` is floats, 0 < `beta` < 1, generally close to 1.

    updater {
      type: kAdamMax
      adammax_conf {
       beta1: float # [0,1]
       beta2: float # [0,1]
      }
    }

### Configuration of learning rate

The `learning_rate` field is configured as,

    learning_rate {
      type: ChangeMethod
      base_lr: float  # base/initial learning rate
      ... # fields to a specific changing method
    }

The common fields include `type` and `base_lr`. SINGA provides the following
`ChangeMethod`s.

#### kFixed

The `base_lr` is used for all steps.

#### kLinear

The updater should be configured like

    learning_rate {
      base_lr:  float
      linear_conf {
        freq: int
        final_lr: float
      }
    }

Linear interpolation is used to change the learning rate,

    lr = (1 - step / freq) * base_lr + (step / freq) * final_lr

#### kExponential

The udapter should be configured like

    learning_rate {
      base_lr: float
      exponential_conf {
        freq: int
      }
    }

The learning rate for `step` is

    lr = base_lr / 2^(step / freq)

#### kInverseT

The updater should be configured like

    learning_rate {
      base_lr: float
      inverset_conf {
        final_lr: float
      }
    }

The learning rate for `step` is

    lr = base_lr / (1 + step / final_lr)

#### kInverse

The updater should be configured like

    learning_rate {
      base_lr: float
      inverse_conf {
        gamma: float
        pow: float
      }
    }


The learning rate for `step` is

    lr = base_lr * (1 + gamma * setp)^(-pow)


#### kStep

The updater should be configured like

    learning_rate {
      base_lr : float
      step_conf {
        change_freq: int
        gamma: float
      }
    }


The learning rate for `step` is

    lr = base_lr * gamma^ (step / change_freq)

#### kFixedStep

The updater should be configured like

    learning_rate {
      fixedstep_conf {
        step: int
        step_lr: float

        step: int
        step_lr: float

        ...
      }
    }

Denote the i-th tuple as (step[i], step_lr[i]), then the learning rate for
`step` is,

    step_lr[k]

where step[k] is the smallest number that is larger than `step`.


## Advanced user guide

### Implementing a new Updater subclass

The base Updater class has one virtual function,

    class Updater{
     public:
      virtual void Update(int step, Param* param, float grad_scale = 1.0f) = 0;

     protected:
      UpdaterProto proto_;
      LRGenerator lr_gen_;
    };

It updates the values of the `param` based on its gradients. The `step`
argument is for deciding the learning rate which may change through time
(step). `grad_scale` scales the original gradient values. This function is
called by servers once it receives all gradients for the same `Param` object.

To implement a new Updater subclass, users must override the `Update` function.

    class FooUpdater : public Updater {
      void Update(int step, Param* param, float grad_scale = 1.0f) override;
    };

Configuration of this new updater can be declared similar to that of a new
layer,

    # in user.proto
    FooUpdaterProto {
      optional int32 c = 1;
    }

    extend UpdaterProto {
      optional FooUpdaterProto fooupdater_conf= 101;
    }

The new updater should be registered in the
[main function](programming-guide.html)

    driver.RegisterUpdater<FooUpdater>("FooUpdater");

Users can then configure the job as

    # in job.conf
    updater {
      user_type: "FooUpdater"  # must use user_type with the same string identifier as the one used for registration
      fooupdater_conf {
        c : 20;
      }
    }

### Implementing a new LRGenerator subclass

The base `LRGenerator` is declared as,

    virtual float Get(int step);

To implement a subclass, e.g., `FooLRGen`, users should declare it like

    class FooLRGen : public LRGenerator {
     public:
      float Get(int step) override;
    };

Configuration of `FooLRGen` can be defined using a protocol message,

    # in user.proto
    message FooLRProto {
     ...
    }

    extend LRGenProto {
      optional FooLRProto foolr_conf = 101;
    }

The configuration is then like,

    learning_rate {
      user_type : "FooLR" # must use user_type with the same string identifier as the one used for registration
      base_lr: float
      foolr_conf {
        ...
      }
    }

Users have to register this subclass in the main function,

      driver.RegisterLRGenerator<FooLRGen, std::string>("FooLR")
