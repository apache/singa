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

# Auto tagging for release

Usage: `$ ./tool/release/release.sh [patch|minor|major]`

Example 1: releasing patch will update from 2.1.1 to 2.1.2

run `$ ./tool/release/release.sh patch`

Example 2: releasing minor will update from 2.1.1 to 2.2.0

run `$ ./tool/release/release.sh minor`

Example 3: releasing major will update from 2.1.1 to 3.0.0

run `$ ./tool/release/release.sh major`


## In the release.sh

Internally, the script retrieve latest git tag by `git describe`,
and increment the version accroding to semantic versioning,
then push latest tag to remote master.

## Next step

CI will automatically detect the update in master
and build and release conda packages.
