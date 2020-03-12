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

# Linting check

This guide is for singa devoloper who should sanitize the code
before merging into the main branch.

## linting tools

Install cpplint for C++:
`pip install cpplint`

Install pylint for Python:
`pip install pylint`

## Linting a single file

For C++ code:
`cpplint path/to/file`

For Python Code:
`pylint path/to/file`

## Linting the whole project

usage: `bash tool/linting/py.sh`
usage: `bash tool/linting/cpp.sh`

## Configuration
Currently the configuration are customized to respect google style.
Update of configuration could be done at `.pylintrc` and `CPPLINT.cfg`
