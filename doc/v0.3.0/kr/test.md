# Performance Test and Feature Extraction

----

Once SINGA finishes the training of a model, it would checkpoint the model parameters
into disk files under the [checkpoint folder](checkpoint.html). Model parameters can also be dumped
into this folder periodically during training if the
[checkpoint configuration[(checkpoint.html) fields are set. With the checkpoint
files, we can load the model parameters to conduct performance test, feature extraction and prediction
against new data.

To load the model parameters from checkpoint files, we need to add the paths of
checkpoint files in the job configuration file

    checkpoint_path: PATH_TO_CHECKPOINT_FILE1
    checkpoint_path: PATH_TO_CHECKPOINT_FILE2
    ...

The new dataset is configured by specifying the ``test_step`` and the data input
layer, e.g. the following configuration is for a dataset with 100*100 instances.

    test_steps: 100
    net {
      layer {
        name: "input"
        store_conf {
          backend: "kvfile"
          path: PATH_TO_TEST_KVFILE
          batchsize: 100
        }
      }
      ...
    }

## Performance Test

This application is to test the performance, e.g., accuracy, of the previously
trained model. Depending on the application, the test data may have ground truth
labels or not. For example, if the model is trained for image classification,
the test images must have ground truth labels to calculate the accuracy; if the
model is an auto-encoder, the performance could be measured by reconstruction error, which
does not require extra labels. For both cases, there would be a layer that calculates
the performance, e.g., the `SoftmaxLossLayer`.

The job configuration file for the cifar10 example can be used directly for testing after
adding the checkpoint path. The running command is


    $ ./bin/singa-run.sh -conf examples/cifar10/job.conf -test

The performance would be output on the screen like,


    Load from checkpoint file examples/cifar10/checkpoint/step50000-worker0
    accuracy = 0.728000, loss = 0.807645

## Feature extraction

Since deep learning models are good at learning features, feature extraction for
is a major functionality of deep learning models, e.g., we can extract features
from the fully connected layers of [AlexNet](www.cs.toronto.edu/~fritz/absps/imagenet.pdf) as image features for image retrieval.
To extract the features from one layer, we simply add an output layer after that layer.
For instance, to extract the fully connected (with name `ip1`) layer of the cifar10 example model,
we replace the `SoftmaxLossLayer` with a `CSVOutputLayer` which extracts the features into a CSV file,

    layer {
      name: "ip1"
    }
    layer {
      name: "output"
      type: kCSVOutput
      srclayers: "ip1"
      store_conf {
        backend: "textfile"
        path: OUTPUT_FILE_PATH
      }
    }

The input layer and test steps, and the running command are the same as in *Performance Test* section.

## Label Prediction

If the output layer is connected to a layer that predicts labels of images,
the output layer would then write the prediction results into files.
SINGA provides two built-in layers for generating prediction results, namely,

* SoftmaxLayer, generates probabilities of each candidate labels.
* ArgSortLayer, sorts labels according to probabilities in descending order and keep topk labels.

By connecting the two layers with the previous layer and the output layer, we can
extract the predictions of each instance. For example,

    layer {
      name: "feature"
      ...
    }
    layer {
      name: "softmax"
      type: kSoftmax
      srclayers: "feature"
    }
    layer {
      name: "prediction"
      type: kArgSort
      srclayers: "softmax"
      argsort_conf {
        topk: 5
      }
    }
    layer {
      name: "output"
      type: kCSVOutput
      srclayers: "prediction"
      store_conf {}
    }

The top-5 labels of each instance will be written as one line of the output CSV file.
Currently, above layers cannot co-exist with the loss layers used for training.
Please comment out the loss layers for extracting prediction results.
