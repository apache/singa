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

# Singa for Diabetic Readmission Prediction task

## Diabetic Readmission

Diabetic readmission is a significant concern in healthcare, with a substantial number of patients being readmitted to the hospital within a short period after discharge. This not only leads to increased healthcare costs but also poses a risk to patient well-being.

Although diabetes is a manageable condition, early identification of patients at high risk of readmission remains a challenge. A reliable and efficient predictive model can help identify these patients, enabling healthcare providers to intervene early and prevent unnecessary readmissions.

To address this issue, we use Singa to implement a machine learning model for predicting diabetic readmission. The dataset is from [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). Please download the dataset before running the scripts.


## Structure

* `data` includes the scripts for preprocessing Diabetic Readmission datasets.

* `model` includes the MLP model construction codes by creating
  a subclass of `Module` to wrap the neural network operations 
  of each model.

* `train_mlp.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

## Command
```bash
python train.py mlp diabetic
```
