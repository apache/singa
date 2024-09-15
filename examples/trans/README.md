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

# Machine translation model using Transformer Example
This example trains a Transformer model on a machine translation task. By default, the training script uses the anki dataset, provided.
You can download from http://www.manythings.org/anki/. This example uses the Chinese and English sentence pairs provided by this dataset
to complete the translation task. The dataset contains 29909 sentence pairs in both English and Chinese.

Data format: English + TAB + Chinese + TAB + Attribution.

Example:
```
Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
Hi.	你好。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #4857568 (musclegirlxyp)
```

The train.py script accepts the following arguments:
```
optional arguments:
  [arg]              [type]     [desc]                                  [default]
  --dataset          string     location of the dataset
  --max-epoch        int        maximum epochs                          default 100           
  --batch_size       int        batch size                              default 64
  --shuffle          bool       shuffle the dataset                     default True
  --lr               float      learning rate                           default 0.005
  --seed             int        random seed                             default 0
  --d_model          int        transformer model d_model               default 512
  --n_head           int        transformer model n_head                default 8
  --dim_feedforward  int        transformer model dim_feedforward       default 2048
  --n_layers         int        transformer model n_layers              default 6
```

**run the example**

step 1: Download the dataset to the cmn-eng directory.

step 2: Run the following script.

```
python train.py --dataset cmn-eng/cmn-2000.txt --max-epoch 100 --batch-size 32 --lr 0.01
```
