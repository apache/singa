# Jenkins CI Support

## Introduction
This documentation is to guide SINGA developers to setup Jenkins service.

We use jenkins to support continuous integration.
After each commit, we want to automatically compile and test SINGA
under different OS and settings.
Those built binaries need to be archived for users to download.

## Install Jenkins
[Jenkins Official Wiki](https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins)

## Configure Jenkins Multi-configuration Project
Create a multi-configuration project and configure project as follows:

### Description
  This job automatically pulls latest commits from apache SINGA github repository.
  It compiles and tests SINGA in different environments and creates PySINGA wheel distribution accordingly.

### General
  * Discard old builds - Max # of builds to keep - 50
  * GitHub project - ``https://github.com/apache/incubator-singa``

### Source Code Management
  * Git - Repository URL - ``https://github.com/apache/incubator-singa``
  * Git - Branch Specifier - ``*/master``

### Build Triggers
  * Poll SCM - Schedule - ``H/30 * * * *`` (pull every 30 minutes)

### Configuration Matrix
  * User-defined Axis - name ``lang`` values ``CPP CUDA``
  * Slave - name ``env`` Node/label ``tick all nodes``

### Build
  * Execute shell - command - ``bash -ex tool/jenkins/jenkins_wheel.sh $lang``

### Post-build Actions
  * Archive the artifacts - ``build/python/dist/**.whl``
  * Send build artifacts over SSH - source files ``build/python/dist/*.tar.gz``, Remove prefix `build/python/dist`, Exec command `tar xf *.tar.gz && rm *.tar.gz`
  * Publish JUnit test result report - Test report XMLs - ``**/gtest.xml, **/unittest.xml``

## Docker Images
We provide in `docker` a number of singa docker images for Jenkins to use as slaves.

## Access Control
Use `Role Strategy Plugin` to give read access for anonymous users.
