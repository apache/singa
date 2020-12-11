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
# Jenkins CI Support

## Introduction
This documentation is to guide Singa developers to setup Jenkins service for continuous integration of Singa. After each commit,
1. Singa should be compiled and tested automatically under different settings (e.g.,OS, python version and hardware).
2. Binary packages should be generated automatically and archived.

Continuous integration for CPU systems is enabled via [Travis](../travis).
Hence, Jenkins is mainly used for CI on GPUs.

## Install Jenkins
[Jenkins Official Wiki](https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins)
The slave nodes for running different building environments are configured under 'Manage Jenkins'->'Manage nodes'.

Change Jenkins time zone by executing the following code in 'Mange jenkins' -> 'Script Console':

    System.setProperty('org.apache.commons.jelly.tags.fmt.timeZone', 'Asia/Singapore')

## Configure Jenkins for Unit Testing and Binary Package Generation
Create a multi-configuration project and configure project as follows:

### Description
This job automatically pulls latest commits from Apache singa github repository, then for different environments

* compile and test Singa on GPUs
* generate conda package of Singa with CUDA enabled
* invoke the CPU test and packaging on Travis
* (optional) create Debian GPU packages

### General
  * Discard old builds - Max # of builds to keep - 50
  * GitHub project - ``https://github.com/apache/singa``

### Source Code Management
  * Git - Repository URL - ``https://github.com/apache/singa``
  * Git - Branch Specifier - ``*/master``

### Build Triggers
  * Poll SCM - Schedule - ``H/30 * * * *`` (pull every 30 minutes)

### Configuration Matrix
  * User-defined Axis - name ``lang`` values ``CUDA`` (Add CPP if you want to test CPU code)
  * Slave - name ``env`` Node/label: tick available nodes

### Build
The building script can do the following tasks:

  * compile and do unit test on GPU
    Execute shell - command - ``bash -ex tool/jenkins/test.sh``

  * update another github repo with the new commits to invoke travis (for cpu test and conda package generation)
    Execute shell - command - ``git push https://<username:token>@github.com/nusdbsystem/singa.git -f``

  * create conda package and upload it to anaconda cloud
    Execute shell - command

        /root/miniconda/bin/conda-build tool/conda/singa
        /root/miniconda/bin/anaconda -t <ANACONDA_UPLOAD_TOKEN> upload -u nusdbsystem -l main /root/miniconda/linux-64/singa-*.so.*.tar.bz2 --force

  * (optional) create Debian package
    Execute shell - command - ``bash -ex tool/debian/build.sh --python --$lang``

### Post-build Actions
  * Publish JUnit test result report - Test report XMLs - ``**/gtest.xml, **/unittest.xml``
  * (optional) Archive the artifacts - ``build/debian/**.deb``
  * Send build artifacts (Debian package) over SSH for wheel
    * ../debian/build.sh packages the .deb file into $BUILD_ID.tar.gz. Inside the tar file,
      the folder layout is `build_id/commit_hash/os_lang/*.deb`, where `os_lang` is the combination of os version, device programming language (cuda/cpp/opencl) and cudnn version.
    * In `Manage Jenkins`-`Configure System`, configure the SSH for connecting to the remote public server and set the target folder location
    * Source files - `build/debian/*.tar.gz`
    * Remove prefix - `build/debian
    * Remote directory - `debian`
    * Exec a command on the remote server to decompress the package and add a symlink to the latest build. E.g., on a Solaris server the command is

            cd <target_folder>/debian && gunzip $BUILD_ID.tar.gz && tar xf $BUILD_ID.tar && chmod -R 755 $BUILD_ID && /bin/rm -f $BUILD_ID.tar && /bin/rm -f latest && ln -s $BUILD_ID/* latest

    * The file links on the remote public server would be like

            debian/32/84d56b7/ubuntu14.04-cpp/singa-1.0.1.deb
            debian/32/84d56b7/ubuntu14.04-cuda8.0-cudnn5/singa-1.0.1.deb

### Jenkins Nodes

We provide different Singa [Dockerfiles](../docker/README.md) for Jenkins to use as working nodes.

To run the docker images,

    nvidia-docker run --name <node name> -P -d <Image ID>

To add the container into a network for easy access

    docker network create <network name>
    docker network connect <network name> <node name>

After connecting both the jenkins and node contaniners into the same network, we can ssh to the node from jenkins container like


    # inside jenkins container
    ssh root@<node name>

You need execute the above command manually for the first ssh login.

In the Jenkins node configuration page, the container name is used to configure the `Host` field.
Notice that Oracle username and account are required to luanch the node by Jenkins.

The working nodes (or Docker containers) are configured in Jenkins-Manage Jenkins-Mange Nodes.
Each node should configure the following environment variable

    export CUDA=<cuda version, e.g., 9.0>

[Dockerfiles](../conda/docker) are provided to create the working nodes.

## Configure Jenkins for Singa Website Updates

### Description and Configuration

This job is triggered upon any changes to the files of the `doc/` folder.
It does the following tasks,

1. installs the latest Singa
2. pull the latest source code
3. generate the html files for the documentation
4. update the Singa website

The Jenkins job configuration is similar as above except the following fields,

* Source Code Management - Git - Additional Behaviors - Include Region `doc/*`
* Build - Execute Shell - Command

      bash -ex tool/jenkins/gen_doc.sh

* No `Post-build Actions`

### Jenkins Node

The docker images used for testing also be used for document generation.
We have to manually configure something inside the docker container.
First, we start the container

    $docker run --name singa-doc -d <docker image>
    # docker network connect jenkins singa-doc
    $ docker exec -it singa-doc /bin/bash

Next, we do the first commit to the svn repo.

    $ svn co https://svn.apache.org/repos/asf/singa/site/trunk
    # update ~/.subversion/config to set 'store-password=yes'
    # to set password free commit, we have to do a manual commit at first.
    # change any file (add spaces) inside trunk/ to commit a message
    $ svn commit -m "test" --username <committer id> --password <passwd>

## Access Control
Use `Role Strategy Plugin` to give read access for anonymous users.
