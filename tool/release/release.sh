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
#!/bin/bash
RELEASE_TYPE=$1

# git fetch --all --force
VERSION=`git describe --abbrev=0 --tags`

VERSION_BITS=(${VERSION//./ })

VNUM1=${VERSION_BITS[0]}
VNUM2=${VERSION_BITS[1]}
VNUM3=${VERSION_BITS[2]}

if [[ "$RELEASE_TYPE" == "major" ]]; then
  VNUM1=$((VNUM1+1))
  VNUM2=0
  VNUM3=0
elif [[ "$RELEASE_TYPE" == "minor" ]]; then
  VNUM2=$((VNUM2+1))
  VNUM3=0
elif [[ "$RELEASE_TYPE" == "patch" ]]; then
  VNUM3=$((VNUM3+1))
else
  echo "Release type is one of [major|minor|patch]"
  exit 1
fi

NEW_VERSION="$VNUM1.$VNUM2.$VNUM3"
echo "Updating $VERSION to $NEW_VERSION"

git tag -a $NEW_VERSION -m "Version: $NEW_VERSION"
git push dcslin -f --tags
echo "Tag created and pushed to github: $NEW_VERSION"


