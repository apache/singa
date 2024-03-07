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


# Based on PolarDB with PostgreSQL 11.9
FROM polardb/polardb_pg_local_instance:latest

# LABEL maintainer="Naili Xing <xingnaili14@gmai.com>"

# Install Python, Vim, and necessary libraries
# Note: The 'pip' package might not be directly available like this, usually python3-pip is the package name.
USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    bzip2 \
    libbz2-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

USER postgres
# Install pyenv and Python 3.8
RUN curl https://pyenv.run | bash \
    && export PYENV_ROOT="$HOME/.pyenv" \
    && export PATH="$PYENV_ROOT/bin:$PATH" \
    && eval "$(pyenv init --path)" \
    && eval "$(pyenv init -)" \
    && env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8


# Switch to the postgres user, install Rust, init the cargo
# polarDB uses the pg 11.9
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init --pg11 /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config"


# Clone code to there, install dependences,
WORKDIR /home/postgres
RUN git clone https://github.com/NLGithubWP/Trails.git && \
    cd Trails && \
    git checkout trails_singa && \
    cp ./internal/pg_extension/template/Cargo.pg11.toml ./internal/pg_extension/Cargo.toml && \
    cd ./internal/ml/model_selection && \
    pip install -r requirement.txt  && \
    pip install ../../../singa_pkg_code/singa-3.1.0-cp38-cp38-manylinux2014_x86_64.whl


WORKDIR /home/postgres/Trails/internal/pg_extension
RUN /bin/bash -c "source $HOME/.cargo/env && cargo pgrx install --pg-config /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config"

WORKDIR /home/postgres
RUN chmod +x ./Trails/init_polardb.sh
# here we run the default script in /home/postgres
