This example trains the [RNN model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) proposed by Tomas Mikolov for [language modeling](https://en.wikipedia.org/wiki/Language_model) over text dataset contains 71350 words, provided at [RNNLM Toolkit](https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U).
The training objective (loss) is to minimize the [perplexity per word](https://en.wikipedia.org/wiki/Perplexity), which is equivalent to maximize the probability of predicting the next word given the current word in a sentence.
The purpose of this example is to show users how to implement and use their own layers for RNN in SINGA.
The example RNN model consists of six layers, namely RnnDataLayer, WordLayer, RnnLabelLayer, EmbeddingLayer, HiddenLayer, and OutputLayer. 

## File description

The files in this folder include:

* rnnlm.proto, definition of the configuration protocol of the layers.
* rnnlm.h, declaration of the layers.
* rnnlm.cc, definition of the layers.
* main.cc, main function that register the layers.
* Makefile.exmaple, Makefile for compiling all source code in this folder.
* job.conf, the job configuration for training the RNN language model.


## Data preparation

To use the RNNLM dataset, we can download it and create DataShard by typing

    # in rnnlm/ folder
    cp Makefile.example Makefile
    make download
    make create

## Compilation

The *Makefile.example* contains instructions for compiling the source code.

    # in rnnlm/ folder
    cp Makefile.example Makefile
    make rnnlm

It will generate an executable file *rnnlm.bin*.

## Running

Make sure that there is one example job configuration file, named *job.conf*.

Before running SINGA, we need to export the `LD_LIBRARY_PATH` to
include the libsinga.so by the following script.

    # at the root folder of SINGA
    export LD_LIBRARY_PATH=.libs:$LD_LIBRARY_PATH

Then, we can run SINGA as follows. 

    # at the root folder of SINGA
    ./bin/singa-run.sh -exec examples/rnnlm/rnnlm.bin -conf examples/rnnlm/job.conf

You will see the values of loss and ppl at each training step.
