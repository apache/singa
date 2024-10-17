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

# https://github.com/zhengzangw/Fed-SINGA/blob/main/src/client/data/bank.py

import pandas as pd
import numpy as np
import sys
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def encode(df):
    res = pd.DataFrame()
    for col in df.columns.values:
        if not is_numeric_dtype(df[col]):
            tmp = pd.get_dummies(df[col], prefix=col)
        else:
            tmp = df[col]
        res = pd.concat([res, tmp], axis=1)
    return res


def load(device_id):
    fn_train = "data/bank_train_" + str(device_id) + ".csv"
    fn_test = "data/bank_test_" + str(device_id) + ".csv"

    train = pd.read_csv(fn_train, sep=',')
    test = pd.read_csv(fn_test, sep=',')

    train_x = train.drop(['y'], axis=1)
    train_y = train['y']
    val_x = test.drop(['y'], axis=1)
    val_y = test['y']

    train_x = np.array((train_x), dtype=np.float32)
    val_x = np.array((val_x), dtype=np.float32)
    train_y = np.array((train_y), dtype=np.int32)
    val_y = np.array((val_y), dtype=np.int32)

    train_x, val_x = normalize(train_x, val_x)
    num_classes = 2

    return train_x, train_y, val_x, val_y, num_classes


def normalize(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def split(num):
    filepath = "../data/bank-additional-full.csv"
    df = pd.read_csv(filepath, sep=';')
    df['y'] = (df['y'] == 'yes').astype(int)
    data = encode(df)
    data = shuffle(data)
    train, test = train_test_split(data, test_size=0.2)

    train.to_csv("data/bank_train_.csv", index=False)
    test.to_csv("data/bank_test_.csv", index=False)

    train_per_client = len(train) // num
    test_per_client = len(test) // num

    print("train_per_client:", train_per_client)
    print("test_per_client:", test_per_client)
    for i in range(num):
        sub_train = train[i * train_per_client:(i + 1) * train_per_client]
        sub_test = test[i * test_per_client:(i + 1) * test_per_client]
        sub_train.to_csv("data/bank_train_" + str(i) + ".csv", index=False)
        sub_test.to_csv("data/bank_test_" + str(i) + ".csv", index=False)


if __name__ == "__main__":
    split(int(sys.argv[1]))
