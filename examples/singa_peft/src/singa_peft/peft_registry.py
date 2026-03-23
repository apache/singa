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

class PeftRegistry:
    """
    PeftRegistry: the registry class for peft method
    """

    _tuners = {}

    @classmethod
    def register(cls, tuner_name):
        r"""
        Register the Tuner class decorator
        Args:
            tuner_name: the name of the Tuner

        Returns: the class of decorator
        """
        def decorator(tuner_class):
            cls._tuners[tuner_name] = tuner_class
            return tuner_class
        return decorator

    @classmethod
    def get_tuner(cls, tuner_name):
        r"""
        Get the Tuner class by name
        Args:
            tuner_name: the name of the Tuner

        Returns: the class of the Tuner
        """
        if tuner_name not in cls._tuners:
            raise ValueError(f"Unsupported peft method: {tuner_name}")
        return cls._tuners[tuner_name]
