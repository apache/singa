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

# Singa for Malaria Detection Task

## Malaria

Malaria is caused by parasites and could be transmitted through infected mosquitoes. There are about 200 million cases worldwide, and about 400,000 deaths per year, therefore, malaria does lots of harm to global health.

Although Malaria is a curable disease, inadequate diagnostics make it harder to reduce mortality, as a result, a fast and reliable diagnostic test is a promising and effective way to fight malaria.

To mitigate the problem, we use Singa to implement a machine learning model to help with Malaria diagnosis. The dataset is from Kaggle https://www.kaggle.com/datasets/miracle9to9/files1?resource=download. Please download the dataset before running the scripts.

## Structure

* `data` includes the scripts for preprocessing Malaria image datasets.

* `model` includes the CNN model construction codes by creating
  a subclass of `Module` to wrap the neural network operations
  of each model.

* `train_cnn.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

## Command
```bash
python train_cnn.py cnn malaria -dir pathToDataset
```
