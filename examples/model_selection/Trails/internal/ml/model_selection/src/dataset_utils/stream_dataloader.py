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


import queue
import threading
import requests
import time
import torch
from src.logger import logger


class StreamingDataLoader:
    """
    This will preoritically query data from cache-service
    """

    def __init__(self, cache_svc_url, table_name, name_space):
        self.last_fetch_time = 0
        self.table_name = table_name
        # train, valid, test
        self.name_space = name_space
        self.end_signal = "end_position"
        self.cache_svc_url = cache_svc_url
        self.data_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.fetch_data, daemon=True)
        self.thread.start()

    def fetch_data(self):
        while not self.stop_event.is_set():
            response = requests.get(
                f'{self.cache_svc_url}/',
                params={
                    'table_name': self.table_name,
                    'name_space': self.name_space})

            if response.status_code == 200:
                batch = response.json()

                # in trianing, we use iteraiton-per-epoch to control the end
                if batch == self.end_signal:
                    if self.name_space == "valid":
                        # end_signal in inference, stop !
                        logger.info("[StreamingDataLoader]: last iteration in valid is meet!")
                        self.data_queue.put({self.end_signal: True})
                    else:
                        # end_signal in trianing, then keep training
                        continue
                else:
                    # convert to tensor again
                    id_tensor = torch.LongTensor(batch['id'])
                    value_tensor = torch.FloatTensor(batch['value'])
                    y_tensor = torch.FloatTensor(batch['y'])
                    data_tensor = {'id': id_tensor, 'value': value_tensor, 'y': y_tensor}
                    self.data_queue.put(data_tensor)
            else:
                print(response.json())
                time.sleep(5)

    def __iter__(self):
        return self

    def __next__(self):
        print("compute time = ", time.time() - self.last_fetch_time)
        self.last_fetch_time = time.time()
        if self.data_queue.empty() and not self.thread.is_alive():
            raise StopIteration
        else:
            data = self.data_queue.get(block=True)
            if self.end_signal in data:
                raise StopIteration
            else:
                return data

    def __len__(self):
        return self.data_queue.qsize()

    def stop(self):
        self.stop_event.set()
        self.thread.join()



