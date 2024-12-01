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

# Singa for Diabetic Retinopathy Classification

## Diabetic Retinopathy

Diabetic Retinopathy (DR) is a progressive eye disease caused by long-term diabetes, which damages the blood vessels in the retina, the light-sensitive tissue at the back of the eye. It typically develops in stages, starting with non-proliferative diabetic retinopathy (NPDR), where weakened blood vessels leak fluid or blood, causing swelling or the formation of deposits. If untreated, it can progress to proliferative diabetic retinopathy (PDR), characterized by the growth of abnormal blood vessels that can lead to severe vision loss or blindness. Symptoms may include blurred vision, dark spots, or difficulty seeing at night, although it is often asymptomatic in the early stages. Early diagnosis through regular eye exams and timely treatment, such as laser therapy or anti-VEGF injections, can help manage the condition and prevent vision impairment.

The dataset has 5 groups characterized by the severity of Diabetic Retinopathy (DR).

- 0: No DR
- 1: Mild Non-Proliferative DR
- 2: Moderate Non-Proliferative DR
- 3: Severe Non-Proliferative DR
- 4: Proliferative DR


To mitigate the problem, we use Singa to implement a machine learning model to help with Diabetic Retinopathy  diagnosis. The dataset is from Kaggle https://www.kaggle.com/datasets/mohammadasimbluemoon/diabeticretinopathy-messidor-eyepac-preprocessed. Please download the dataset before running the scripts.

## Structure

* `data` includes the scripts for preprocessing DR image datasets.

* `model` includes the CNN model construction codes by creating
  a subclass of `Module` to wrap the neural network operations 
  of each model.

* `train_cnn.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

## Command
```bash
python train_cnn.py cnn diaret -dir pathToDataset
```
