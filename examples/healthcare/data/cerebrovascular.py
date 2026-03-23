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

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split

def load_cerebrovascular_data(dir_path):
    import os
    data_file = os.path.join(dir_path, 'cerebrovascular_data.csv')
    data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    
    X = data[:, :-1]
    y = data[:, -1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    X_train_processed = X_train.astype(np.float32)
    X_val_processed = X_val.astype(np.float32)
    
    return X_train_processed, y_train, X_val_processed, y_val
   
def load(dir_path):
    try:
        X_train, y_train, X_val, y_val = load_cerebrovascular_data(dir_path)
    except FileNotFoundError:
        raise SystemExit(f"Error：Directory {dir_path} or data file is not found.")
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    
    return X_train, y_train, X_val, y_val
