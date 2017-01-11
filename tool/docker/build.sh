#!/usr/bin/env bash
#/**
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

# build all docker images, must be exected under the root directory, i.e., incubator-singa/
# ./build.sh PUSH would push the images to dockerhub/nusdbsystem and then delete the local image
#   (used by Jenkins to avoid dangling images from multiple building)

echo "build singa:runtime"
docker build tool/docker/runtime/ --force-rm -t nusdbsystem/singa:runtime
if [ $1 = "PUSH" ]; then
  docker push nusdbsystem/singa:runtime
  docker rmi nusdbsystem/singa:runtime
fi

echo "build singa:runtime-cuda"
docker build tool/docker/runtime/cuda --force-rm -t nusdbsystem/singa:runtime-cuda
if [ $1 = "PUSH" ]; then
  docker push nusdbsystem/singa:runtime-cuda
  docker rmi nusdbsystem/singa:runtime-cuda
fi

echo "build singa:devel"
docker build tool/docker/devel/ --force-rm -t nusdbsystem/singa:devel
if [ $1 = "PUSH" ]; then
  docker push nusdbsystem/singa:devel
  docker rmi nusdbsystem/singa:devel
fi

echo "build singa:devel-cuda"
docker build tool/docker/devel/cuda --force-rm -t nusdbsystem/singa:devel-cuda
if [ $1 = "PUSH" ]; then
  docker push nusdbsystem/singa:devel-cuda
  docker rmi nusdbsystem/singa:devel-cuda
fi
