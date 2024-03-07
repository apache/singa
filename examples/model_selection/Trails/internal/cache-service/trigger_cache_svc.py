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


import requests

url = 'http://localhost:8093/'
columns = ['label', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
response = requests.post(
    url, json={'columns': columns,
               'name_space': "train",
               'table_name': "frappe_train",
               "batch_size": 32})
print(response.json())

response = requests.post(
    url, json={'columns': columns,
               'name_space': "valid",
               'table_name': "frappe_valid",
               "batch_size": 1024})
print(response.json())

