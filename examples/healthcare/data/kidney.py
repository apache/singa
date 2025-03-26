#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy  as np

import pickle
import sys
import os

def load_dataset(dir_path="/tmp/kidney"):
    dir_path = check_dataset_exist(dir_path=dir_path)
    feature_path = os.path.join(dir_path, "kidney_features.pkl")
    label_path = os.path.join(dir_path, "kidney_labels.pkl")
    with open(feature_path,'rb') as f: 
        features = pickle.load(f)
    with open(label_path,'rb') as f:  
        labels = pickle.load(f)


    split_train_point = int(len(features) * 8/ 10)
    train_x, train_y = features[:split_train_point], labels[:split_train_point]
    val_x, val_y = features[split_train_point:], labels[split_train_point:]

    return train_x,train_y,val_x,val_y

def check_dataset_exist(dir_path):
    if not os.path.exists(dir_path):
        print(
            'Please download the kidney dataset first'
        )
        sys.exit(0)
    return dir_path


def load(dir_path):
    train_x,train_y,val_x,val_y = load_dataset(dir_path)

    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    val_y = val_y.astype(np.int32)
    
    return train_x,train_y,val_x,val_y

