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

### Usage: `$ ./tool/release/release.py [-h] [-y] <type>`

### Option:
  - `[-y]`
    - In interactive mode, it is for user to confirm. Could be used in sript.

### Argument:
  - `<type>`
    Allowed release types are `major`, `minor`, `patch`, `rc`, `stable`.
    - `major` increments major version by 1.
    - `minor` increments minor version by 1.
    - `patch` increments patch version by 1.
    - `rc` increments rc version by 1.
    - `stable` removes rc version.


### Example:

  1. Pre-releasing major will update from 2.1.1 to 3.0.0-rc0

    run `$ ./tool/release/release.py major`

  2. The release candidate needs some revise, from 3.0.0-rc0 to 3.0.0-rc1

    run `$ ./tool/release/release.py rc`

  3. The current version is released as stable, from 3.0.0-rc1 to 3.0.0

    run `$ ./tool/release/release.py stable`


## In the release.py

Internally, the script retrieve latest git tag by `git describe`,
and increment the version accroding to semantic versioning,
then push latest tag to remote master.

## Next step

CI will automatically detect the update in master
and build and release conda packages.
