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

# Singa for Cardiovascular Disease Detection Task

## Cardiovascular Disease

Cardiovascular disease is primarily caused by risk factors like high blood pressure, unhealthy diet, and physical inactivity. As the leading cause of death globally, it accounts for approximately 17.9 million fatalities annually, representing 31% of all global deaths. This makes cardiovascular disease the most significant threat to human health worldwide.

Although early detection can significantly improve outcomes, insufficient screening methods and delayed diagnosis often lead to preventable complications. Therefore, developing rapid and accurate diagnostic tools is crucial for effective prevention and treatment of cardiovascular conditions.

To address this challenge, we utilize Singa to develop a machine learning model for cardiovascular disease risk prediction. The training dataset is sourced from Kaggle https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset. Please download the dataset before running the scripts.

## Structure

* `cardiovascular.py` in the `healthcare/data` directory is the scripts for preprocessing Cardiovascular Disease datasets.

* `cardiovascular_net.py` in the `healthcare/models` directory includes the MLP model construction codes.

* `train_cnn.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

## Command
```bash
python train_cnn.py mlp cardiovascular
```
