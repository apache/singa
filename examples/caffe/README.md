<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->
# Use parameters pre-trained from Caffe in SINGA

In this example, we use SINGA to load the VGG parameters trained by Caffe to do image classification.

## Run this example
You can run this example by simply executing `run.sh vgg16` or `run.sh vgg19`
The script does the following work.

### Obtain the Caffe model
* Download caffe model prototxt and parameter binary file.
* Currently we only support the latest caffe format, if your model is in
    previous version of caffe, please update it to current format.(This is
    supported by caffe)
* After updating, we can obtain two files, i.e., the prototxt and parameter
    binary file.

### Prepare test images
A few sample images are downloaded into the `test` folder.

### Predict
The `predict.py` script creates the VGG model and read the parameters,

    usage: predict.py [-h] model_txt model_bin imgclass

where `imgclass` refers to the synsets of imagenet dataset for vgg models.
You can start the prediction program by executing the following command:

    python predict.py vgg16.prototxt vgg16.caffemodel synset_words.txt

Then you type in the image path, and the program would output the top-5 labels.

More Caffe models would be tested soon.
