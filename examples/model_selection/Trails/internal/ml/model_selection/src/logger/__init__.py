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

import logging
import os

if os.environ.get("log_logger_folder_name") == None:
    log_logger_folder_name = "logs_default"
    if not os.path.exists(f"./{log_logger_folder_name}"):
        os.makedirs(f"./{log_logger_folder_name}")
else:
    log_logger_folder_name = os.environ.get("log_logger_folder_name")
    if not os.path.exists(log_logger_folder_name):
        os.makedirs(log_logger_folder_name)

logger = logging.getLogger(__name__)

if os.environ.get("log_file_name") == None:
    log_name = f"{log_logger_folder_name}/test.log"
else:
    log_name = f"{log_logger_folder_name}/" + os.environ.get("log_file_name")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=log_name, filemode='w')
