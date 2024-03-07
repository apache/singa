
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


import time
import threading
import queue
import psycopg2
from typing import Any, List, Dict, Tuple
from sanic import Sanic
from sanic.response import json
import calendar
import os
import logging

log_logger_folder_name = "log_cache_service"
if not os.path.exists(f"./{log_logger_folder_name}"):
    os.makedirs(f"./{log_logger_folder_name}")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=f"./{log_logger_folder_name}/log_{str(calendar.timegm(time.gmtime()))}", filemode='w')

USER = "postgres"
HOST = "127.0.0.1"
PORT = "28814"
DB_NAME = "pg_extension"
CACHE_SIZE = 10


class CacheService:
    def __init__(self, name_space: str, database: str, table: str, columns: List, batch_size: int, max_size: int = CACHE_SIZE):
        """
        name_space: train, valid, test
        database: database to use
        table: which table
        columns: selected cols
        max_size: max batches to cache
        """
        self.name_space = name_space
        self.batch_size = batch_size
        self.last_id = -1
        self.database = database
        self.table = table
        self.columns = columns
        self.queue = queue.Queue(maxsize=max_size)
        self.thread = threading.Thread(target=self.fetch_data, daemon=True)
        self.thread.start()

    def decode_libsvm(self, columns):
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        # 0 is id, 1 is label
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
        sample = {'id': list(id),
                  'value': list(value),
                  'y': int(columns[1])}
        return sample

    def pre_processing(self, mini_batch_data: List[Tuple]):
        """
        mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
        """
        sample_lines = len(mini_batch_data)
        feat_id = []
        feat_value = []
        y = []

        for i in range(sample_lines):
            row_value = mini_batch_data[i]
            sample = self.decode_libsvm(row_value)
            feat_id.append(sample['id'])
            feat_value.append(sample['value'])
            y.append(sample['y'])
        return {'id': feat_id, 'value': feat_value, 'y': y}

    def fetch_data(self):
        with psycopg2.connect(database=self.database, user=USER, host=HOST, port=PORT) as conn:
            while True:
                try:
                    # fetch and preprocess data from PostgreSQL
                    batch, time_usg = self.fetch_and_preprocess(conn)
                    self.queue.put(batch)
                    print(f"Data is fetched, {self.name_space} queue_size={self.queue.qsize()}, time_usg={time_usg}")
                    logger.info(f"Data is fetched, queue_size={self.queue.qsize()}, time_usg={time_usg}")
                    # block until a free slot is available
                    time.sleep(0.1)
                except psycopg2.OperationalError:
                    logger.exception("Lost connection to the database, trying to reconnect...")
                    time.sleep(5)  # wait before trying to establish a new connection
                    conn = psycopg2.connect(database=self.database, user=USER, host=HOST, port=PORT)

    def fetch_and_preprocess(self, conn):
        begin_time = time.time()
        cur = conn.cursor()
        # Assuming you want to get the latest 100 rows
        columns_str = ', '.join(self.columns)
        # Select rows greater than last_id
        cur.execute(f"SELECT id, {columns_str} FROM {self.table} "
                    f"WHERE id > {self.last_id} ORDER BY id ASC LIMIT {self.batch_size}")
        rows = cur.fetchall()

        if rows:
            # Update last_id with max id of fetched rows
            self.last_id = max(row[0] for row in rows)  # assuming 'id' is at index 0
        else:
            # If no more new rows, reset last_id to start over scan and return 'end_position'
            self.last_id = -1
            return "end_position", time.time() - begin_time

        batch = self.pre_processing(rows)
        return batch, time.time() - begin_time

    def get(self):
        return self.queue.get()

    def is_empty(self):
        return self.queue.empty()


app = Sanic("CacheServiceApp")


# start the server， this is from pg_ingerface
@app.route("/", methods=["POST"])
async def start_service(request):
    try:
        columns = request.json.get('columns')
        # can only be train or valid
        name_space = request.json.get('name_space')
        table_name = request.json.get('table_name')
        batch_size = request.json.get('batch_size')

        if columns is None:
            return json({"error": "No columns specified"}, status=400)
        if name_space not in ["train", "valid", "test"]:
            return json({"error": name_space + " is not correct"}, status=400)

        print(f"columns are {columns}, name_space = {name_space}")

        if not hasattr(app.ctx, f'{table_name}_{name_space}_cache'):
            setattr(app.ctx, f'{table_name}_{name_space}_cache',
                    CacheService(name_space, DB_NAME, table_name, columns, batch_size, CACHE_SIZE))

        return json("OK")
    except Exception as e:
        return json({"error": str(e)}, status=500)


# serve the data retrieve request from eva_service.py
@app.route("/", methods=["GET"])
async def serve_get_request(request):
    name_space = request.args.get('name_space')
    table_name = request.args.get('table_name')

    # check if exist
    if not hasattr(app.ctx, f'{table_name}_{name_space}_cache'):
        return json({"error": f"{table_name}_{name_space}_cache not start yet"}, status=404)

    # get data
    data = getattr(app.ctx, f'{table_name}_{name_space}_cache').get()

    # return
    if data is None:
        return json({"error": "No data available"}, status=404)
    else:
        return json(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8093)
