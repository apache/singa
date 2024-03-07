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

import json


class ModelEvaData:
    """
    Eva worker send score to search strategy
    """

    def __init__(self, model_id: str = None, model_score: dict = None):
        if model_score is None:
            model_score = {}
        self.model_id = model_id
        self.model_score = model_score

    def serialize_model(self) -> str:
        data = {"model_id": self.model_id,
                "model_score": self.model_score}
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str):
        data = json.loads(data_str)
        res = cls(
            data["model_id"],
            data["model_score"])
        return res


class ModelAcquireData:
    """
    Eva worker get model from search strategy
    The serialize/deserialize is for good scalability. The project can be decouple into multiple service
    """

    def __init__(self, model_id: str, model_encoding: str, is_last: bool = False,
                 spi_seconds=None, spi_mini_batch=None, batch_size=32):
        self.is_last = is_last
        self.model_id = model_id
        self.model_encoding = model_encoding

        # this is when using spi
        self.spi_seconds = spi_seconds
        self.spi_mini_batch = spi_mini_batch
        self.batch_size = batch_size

    def serialize_model(self) -> dict:
        data = {"is_last": self.is_last,
                "model_id": self.model_id,
                "model_encoding": self.model_encoding,
                "spi_seconds": self.spi_seconds,
                "preprocess_seconds": self.spi_seconds,
                "batch_size": self.batch_size,
                "spi_mini_batch": self.spi_mini_batch}

        return data

    @classmethod
    def deserialize(cls, data: dict):
        res = cls(
            model_id=data["model_id"],
            model_encoding=data["model_encoding"],
            is_last=data["is_last"],
            spi_mini_batch=data["spi_mini_batch"],
            batch_size=data["batch_size"],
            spi_seconds=data["spi_seconds"])
        return res


class ClientStruct:
    """
    Client get data
    """

    def __init__(self, budget: float, dataset: str):
        self.budget = budget
        self.dataset = dataset

    @classmethod
    def deserialize(cls, data_str: str):
        data = json.loads(data_str)
        res = cls(
            data["budget"],
            data["dataset"]
        )
        return res


if __name__ == "__main__":
    data = ModelEvaData("1", {"a": 1, "b": 2})
    data_str = data.serialize_model()
    res = ModelEvaData.deserialize(data_str)
    print(res)
