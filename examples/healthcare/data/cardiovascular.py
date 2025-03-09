#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import numpy as np
import os
import sys

def load_cardiovascular_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')  

    
    X = data[:, :-1]  
    y = data[:, -1]   
    

    # Split the data into training and validation sets
    train_size = int(0.8 * data.shape[0])
    train_x, val_x = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]

    # Normalize the data
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    return train_x, train_y, val_x, val_y

def load():
    file_path = 'cardio_train.csv'  #need to change

    train_x, train_y, val_x, val_y = load_cardiovascular_data(file_path)

    train_x = np.array(train_x, dtype=np.float32)
    val_x = np.array(val_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    val_y = np.array(val_y, dtype=np.int32)

    return train_x, train_y, val_x, val_y

if __name__ == '__main__':
    train_x, train_y, val_x, val_y = load()
    print("Training data shape:", train_x.shape)
    print("Training labels shape:", train_y.shape)
    print("Validation data shape:", val_x.shape)
    print("Validation labels shape:", val_y.shape) 

