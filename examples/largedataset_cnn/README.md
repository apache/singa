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

# Image Classification using Convolutional Neural Networks with datasets from the filesysetm

Examples inside this folder show how to train CNN models using SINGA for image classification where the dataset
is from the filesystem.

It reads the dataset from the filesystem defined by `process_data.py`. Hence, users can modify `process_data.py`
for their perference of dataset format.

In the current setting, 'classes.txt' contains the names of the classes at each line. For example, if it is a food dataset containing three classes, 'classes.txt' may read like this:

    Fish_and_chips
    bagel_and_croissant
    bak_kut_teh

Then, the directory '/Data/' contains all the folders for images of different classes, while each folder name should be the same as that appeared in 'classes.txt'. The name of an image file should not be a concern, but it should be placed inside the folder of the class it belongs to. For the same example above, the folder structure may look like this:

    Data/
        Fish_and_chips/
            fish1.jpg
            fish2.jpg
            ...
            chip1.jpg
            ...
        bagel_and_croissant/
            bagel.jpg
            ...
            croissant1.jpg
            croissant2.jpg
            ...
        bak_kut_teh/
            photo.jpg
            photo2.jpg
            ...

Before running the code, the `model` folder in `examples/cnn` should be copied to this directory.

* `train_largedata.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

* `train_mpi.py` is the script for distributed training (among multiple nodes)
  using MPI and NCCL for communication.
