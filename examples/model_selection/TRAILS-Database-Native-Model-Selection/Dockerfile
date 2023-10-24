FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Vim, and necessary libraries
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 lsb-release git && \
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

# Switch to the postgres user andInstall Rust and init the cargo
USER postgres
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.9.7' --locked" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo pgrx init"

# Set environment variables for Rust and Python
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/project/TRAILS/internal/ml/model_selection"

WORKDIR /project
COPY ./internal/ml/model_selection/requirement.txt ./requirement.txt
RUN pip install -r requirement.txt

RUN pip install https://www.comp.nus.edu.sg/~zhaojing/files/singa-3.1.0-cp38-cp38-manylinux2014_x86_64.whl

# appendix
USER root
RUN apt-get update && apt-get install -y \
    postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER postgres

CMD ["tail", "-f", "/dev/null"]
