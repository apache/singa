<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# How to format code

This guide is for singa devoloper who should sanitize the code
before merging into the main branch.

## tools to auto formating code

Install clang-format for C++:

Ubuntu 16.04: `sudo apt install clang-format`

Ubuntu 18.04: `sudo apt install clang-format-6.0`


Install yapf for Python:

`pip install yapf`

## Formating a single file

- C++: `clang-format -i path/to/file`

- Python: `yapf -i path/to/file`

## Formating the whole project
usage: `bash tool/code-format/format.sh`

## Configuration:
Currently the configuration are customized to respect google style.
Update of configuration could be done at `.clang-format` and `.style.yapf`
