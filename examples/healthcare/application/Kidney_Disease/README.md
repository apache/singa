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

# Singa for Kidney Disease Prediction

## Kidney Disease Prediction Task

Kidney disease prediction is an important tool that uses data science and machine learning techniques to predict the likelihood of a patient suffering from Kidney disease. The goal is to judge whether a patient suffers from kidney disease by analyzing multiple data such as a patient’s medical history, physiological indicators, diagnostic information, treatment options, and socioeconomic factors, so as to take appropriate interventions in advance to provide treatment.

The dataset used in this task is MIMIC-III. The features are data containing 6 visit windows, with 2549 frequent diagnoses, procedures and drugs for each window. These features are encoded by one-hot. The labels are corresponding flags to mark whether the patient suffered from kidney disease, where the label equals "1" if the patient had kidney  disease, and the label equals "0" if not.


## Structure

* `data` includes the load of mimic-iii data to be utilized.

* `model` includes the MLP model construction codes by creating
  a subclass of `Module` to wrap the neural network operations 
  of each model.

* `train_kidney_mlp.py` is the training script, which controls the training flow by
  doing BackPropagation and the SGD update.

## Command
```bash
python train_kidney_mlp.py mlp kidney-disease -dir pathToDataset
```