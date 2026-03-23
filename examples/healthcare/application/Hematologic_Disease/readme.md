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
# Train a hematologic net model on BloodMnist dataset

This example is to train a hematologic net model over the BloodMnist dataset.

## About dataset

The BloodMNIST , as a sub set of [MedMNIST](https://medmnist.com/), is based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. 
It contains a total of 17,092 images and is organized into 8 classes. 
it is split with a ratio of 7:1:2 into training, validation and test set. 
The source images with resolution 3×360×363 pixels are center-cropped into 3×200×200, and then resized into 3×28×28.

8 classes of the dataset: 
```
"0": "basophil",
"1": "eosinophil",
"2": "erythroblast",
"3": "ig (immature granulocytes)",
"4": "lymphocyte",
"5": "monocyte",
"6": "neutrophil",
"7": "platelet"
```

## Running instructions

1. Download the pre-processed [BloodMnist dataset](https://github.com/lzjpaul/singa-healthcare/blob/main/data/bloodmnist/bloodmnist.tar.gz) to the folder (pathToDataset), which contains a few training samples and test samples. For the complete BloodMnist dataset, please download it via this [link](https://github.com/gzrp/bloodmnist/blob/master/bloodmnist.zip).

2. Start the training

```bash
python train.py hematologicnet -dir pathToDataset
```
