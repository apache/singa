# CheckPoint

---

SINGA checkpoints model parameters onto disk periodically according to user
configured frequency. By checkpointing model parameters, we can

  1. resume the training from the last checkpointing. For example, if
    the program crashes before finishing all training steps, we can continue
    the training using checkpoint files.

  2. use them to initialize a similar model. For example, the
    parameters from training a RBM model can be used to initialize
    a [deep auto-encoder](rbm.html) model.

## Configuration

Checkpointing is controlled by two configuration fields:

* `checkpoint_after`, start checkpointing after this number of training steps,
* `checkpoint_freq`, frequency of doing checkpointing.

For example,

    # job.conf
    checkpoint_after: 100
    checkpoint_frequency: 300
    ...

Checkpointing files are located at *WORKSPACE/checkpoint/stepSTEP-workerWORKERID*.
*WORKSPACE* is configured in

    cluster {
      workspace:
    }

For the above configuration, after training for 700 steps, there would be
two checkpointing files,

    step400-worker0
    step700-worker0

## Application - resuming training

We can resume the training from the last checkpoint (i.e., step 700) by,

    ./bin/singa-run.sh -conf JOB_CONF -resume

There is no change to the job configuration.

## Application - model initialization

We can also use the checkpointing file from step 400 to initialize
a new model by configuring the new job as,

    # job.conf
    checkpoint : "WORKSPACE/checkpoint/step400-worker0"
    ...

If there are multiple checkpointing files for the same snapshot due to model
partitioning, all the checkpointing files should be added,

    # job.conf
    checkpoint : "WORKSPACE/checkpoint/step400-worker0"
    checkpoint : "WORKSPACE/checkpoint/step400-worker1"
    ...

The training command is the same as starting a new job,

    ./bin/singa-run.sh -conf JOB_CONF
