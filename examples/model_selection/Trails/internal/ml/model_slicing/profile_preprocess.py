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


from typing import List, Tuple
import time


def decode_libsvm(columns):
    map_func = lambda pair: (int(pair[0]), float(pair[1]))
    # 0 is id, 1 is label
    id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
    sample = {'id': list(id),
              'value': list(value),
              'y': int(columns[1])}
    return sample

def decode_libsvm(columns):
    # Decode without additional mapping or zipping, directly processing the splits.
    ids = []
    values = []
    for col in columns[2:]:
        id, value = col.split(':')
        ids.append(int(id))
        values.append(float(value))
    return {'id': ids, 'value': values, 'y': int(columns[1])}


def pre_processing(mini_batch_data: List[Tuple]):
    # Prepare storage for the results.
    all_feat_ids = []
    all_feat_values = []
    all_ys = []

    for row_value in mini_batch_data:
        # Decode and extract data directly without additional unpacking.
        sample = decode_libsvm(list(row_value))
        all_feat_ids.append(sample['id'])
        all_feat_values.append(sample['value'])
        all_ys.append(sample['y'])

    return {'id': all_feat_ids, 'value': all_feat_values, 'y': all_ys}


mini_batch = [
    ('4801', '0', '2:1', '4656:1', '5042:1', '5051:1', '5054:1', '5055:1', '5058:1', '5061:1', '5070:1', '5150:1'),
]

mini_batch = mini_batch * 100000
print(len(mini_batch))

begin = time.time()
pre_processing(mini_batch)
end = time.time()
print(end-begin)




