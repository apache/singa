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

## Kidney disease Prediction Task

Kidney disease prediction is an important tool that uses data science and machine learning techniques to predict the likelihood of a patient suffering from Kidney disease. The core goal of this technology is to judge whether a patient suffers from kidney disease by analyzing multiple data such as a patient’s medical history, physiological indicators, diagnostic information, treatment options, and socioeconomic factors, so as to take appropriate interventions in advance to provide treatment.

The dataset used in this task is MIMIC-III after preprocessed. The features are data containing 6 visit windows, with 2549 frequent diagnoses, procedures and drugs for each window. Each item in features are data for one patient, and these features are encoded by one-hot code. The labels are corresponding flags to mark whether the patient suffered from kidney disease, where the label equals "1" if the patient had kidn  disease, the label equals "0" if not.



## Structure

* `kidney.py` in floder `healthcare/data` includes the load of pre-processed kidney data to be utilized.

* `kidney_net.py` in folder `healthcare/models` includes the construction codes of the KidneyNet model to be applied for kidney disease prediction.

* `train.py` is the training script, which controls the training flow bydoing BackPropagation and SGD update.

## Instruction
Before starting to use this model for kidney disease prediction, download the sample dataset for kidney disease prediction: https://github.com/lzjpaul/singa-healthcare/tree/main/data/kidney

The provided dataset is from MIMIC-III, which has been pre-processed. The dataset contains 100 samples for model testing.

Please download the dataset to the folder (pathToDataset), and then pass the path to run the codes using the following command:
```bash
python train.py kidneynet -dir pathToDataset
```
