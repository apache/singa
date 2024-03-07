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

FROM ubuntu:20.04

#LABEL maintainer="Naili Xing <xingnaili14@gmai.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Vim, and necessary libraries
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release git sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.6 python3-pip vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for PostgreSQL and Rust
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev libpq-dev libclang-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for pgrx
RUN apt-get update && \
    apt-get install -y bison flex libreadline-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create the postgres user
USER root
RUN adduser --disabled-password --gecos "" postgres && \
    mkdir /project && \
    adduser postgres sudo && \
    chown -R postgres:postgres /project

# Add PostgreSQL's repository
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(. /etc/os-release; echo $VERSION_CODENAME)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Install postgresql client
RUN apt-get update && apt-get install -y \
    postgresql-client-14 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch to the postgres user and Install Rust and init the cargo
USER postgres
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init"

# Set environment variables for Rust and Python
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/project/Trails/internal/ml/model_selection"

# ARG CACHEBUST=1 is to force re-execute the following CMDs at each updates.
ARG CACHEBUST=1

# Clone code to there, install dependences,
WORKDIR /project
RUN git clone https://github.com/NLGithubWP/Trails.git && \
    cd Trails && \
    git checkout trails_singa && \
    cp ./internal/pg_extension/template/Cargo.pg14.toml ./internal/pg_extension/Cargo.toml && \
    cd ./internal/ml/model_selection && \
    pip install -r requirement.txt  && \
    pip install ../../../singa_pkg_code/singa-3.1.0-cp38-cp38-manylinux2014_x86_64.whl

WORKDIR /project
RUN chmod +x ./Trails/init.sh

# Set the entry point to your script
ENTRYPOINT ["/project/Trails/init.sh"]
