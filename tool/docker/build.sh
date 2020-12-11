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

# build all docker images, must be exected under the root directory, i.e., singa/
# ./build.sh PUSH would push the images to dockerhub/nusdbsystem and then delete the local image
#   (used by Jenkins to avoid dangling images from multiple building)

echo "###################"
echo "build singa:conda-cudax.y"
echo "###################"
# docker build tool/docker/devel/conda/cuda/ --force-rm -t nusdbsystem/singa:conda-cuda9.0-cudnn7.1.2

echo "###################"
echo "build singa:cudax.y"
echo "###################"
docker build tool/docker/devel/native/ubuntu/cuda9 --force-rm -t nusdbsystem/singa:cuda9-cudnn7

if [ $1 = "PUSH" ]; then
  echo "##########################################"
  echo "Push to Dockerhub and delete local images"
  echo "#########################################"

  docker push nusdbsystem/singa
  docker rmi -f `docker images nusdbsystem/singa -q`
fi
