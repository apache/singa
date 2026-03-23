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

# Singa for Cerebrovascular Disease Prediction Task

## Cerebrovascular Disease

Cerebrovascular disease refers to a group of conditions that affect the blood vessels and blood supply to the brain. This includes conditions such as stroke, transient ischemic attack (TIA), and other disorders that affect the brain's blood circulation. Early detection and prediction of cerebrovascular disease risk is crucial for effective treatment and prevention of complications.

To address this issue, we use Singa to implement a machine learning model for predicting cerebrovascular disease. The model uses tabular data with various clinical features to predict the likelihood of cerebrovascular disease.

The dataset used in this task is MIMIC-III after preprocessed. Before starting to use this model for cerebrovascular disease prediction, download the sample dataset for cerebrovascular disease prediction: https://github.com/lzjpaul/singa-healthcare/tree/main/data/cerebrovascular

## Structure

* `data` includes the scripts for preprocessing Cerebrovascular datasets.

* `model` includes the MLP model construction codes by creating
  a subclass of `Module` to wrap the neural network operations 
  of each model.

* `train.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

## Command
```bash
python train.py cerebrovascularnet -dir pathToDataset
```
