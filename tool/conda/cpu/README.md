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
The conda package specification includes the package name (i.e. singa), version and build string (could be very long).
To install a certain SINGA package, we run

    conda install -c nusdbsystem singa=<version>=<build string>

It is inconvenient to type all 3 parts when running the installation commands.
The meta.yml file in this folder is to create a conda package `singa-cpu` as
an alias of one specific SINGA package.
It does nothing except creating a dummy conda package that depends on one real
cpu version SINGA package.  For example, the following line in meta.yml indicates
that singa-cpu depends on SINGA with version 1.1.1, python version=3.6

    - singa 1.1.1 py36_cpu

Therefore, when we run

    conda install -c nusdbsystem singa-cpu

The dependent SINGA package will be installed.
By default, singa-cpu depends on the latest SINGA (py3.6).
When we have a new SINGA version available, we need to update the meta.yml file to
change the dependency.

To build this package and upload it

    conda config --add channels nusdbsystem
    conda-build .  --python 3.6
    anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u nusdbsystem -l main <path to the singa-cpu package>

where $ANACONDA_UPLOAD_TOKEN is the upload token associated with nusdbsystem account on anaconda cloud.
