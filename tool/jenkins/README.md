# Jenkins CI Support

## Introduction
This documentation is to guide SINGA developers to setup Jenkins service.

We use Jenkins to support continuous integration.
After each commit, we want to automatically compile and test SINGA
under different OS and settings.
Those built binaries need to be archived for users to download.

## Install Jenkins
[Jenkins Official Wiki](https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins)
The slave nodes for running different building environments are configured under 'Manage Jenkins'->'Manage nodes'.

## Configure Jenkins for Unit Testing and Binary Package Generation
Create a multi-configuration project and configure project as follows:

### Description
This job automatically pulls latest commits from Apache incubator-singa github repository, then for different environments
* compile and test SINGA
* create PySINGA wheel files
* create Debian packages

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
  * Slave - name ``env`` Node/label: tick available nodes

### Build
  * Execute shell - command - ``bash -ex tool/jenkins/jenkins_wheel.sh $lang``

### Post-build Actions
  * Publish JUnit test result report - Test report XMLs - ``**/gtest.xml, **/unittest.xml``
  * (optional) Archive the artifacts - ``build/python/dist/**.whl, build/debian/**.deb``
  * Send build artifacts (wheel) over SSH
    * jenkins_wheel.sh packages the .whl file into $BUILD_ID.tar.gz. Inside the tar file,
      the folder layout is `build_id/commit_hash/os_lang/*.whl`, where `os_lang` is the combination of os version, device programming language (cuda/cpp/opencl) and cudnn version.
    * In `Manage Jenkins`-`Configure System`, configure the SSH for connecting to the remote public server and set the target folder location
    * Source files - `build/python/dist/*.tar.gz`
    * Remove prefix - `build/python/dist`
    * Remote directory - `wheel/linux`
    * Exec a command on the remote server to decompress the package and add a symlink to the latest build. E.g., on a Solaris server the command is

            cd <target_folder>/wheel/linux && gunzip $BUILD_ID.tar.gz && tar xf $BUILD_ID.tar && chmod -R 755 $BUILD_ID && /bin/rm -f $BUILD_ID.tar && /bin/rm -f latest && ln -s $BUILD_ID/* latest

    * The file links on the remote public server would be like

            wheel/Linux/32/84d56b7/ubuntu16.04-cpp/singa-1.0.1-py2-none-any.whl
            wheel/Linux/32/84d56b7/ubuntu16.04-cuda8.0-cudnn5/singa-1.0.1-py2-none-any.whl

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

            debian/32/84d56b7/ubuntu16.04-cpp/singa-1.0.1-py2-none-any.whl
            debian/32/84d56b7/ubuntu16.04-cuda8.0-cudnn5/singa-1.0.1-py2-none-any.whl

### Notes for Mac OS X
The Job configuration is the same as stated above except,
* Machines (or docker containers) with Mac OS X should be configured as jenkin nodes; the following environmental variables should be exported in the .bashrc or in the node configuration

    # 10.11 or 10.12
    export OS_VERSION=macosx10.11
    # if using Homebrew
    export PATH=/usr/local/bin:$PATH
    # for numpy.i
    export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Set User-defined Axis to None and tick only Mac OS X nodes

* Send build artifacts (wheel) over SSH
  * The remote directly -- `wheel/macosx`
  * The shell command is

            cd <target_folder>/wheel/macosx && gunzip $BUILD_ID.tar.gz && tar xf $BUILD_ID.tar && chmod -R 755 $BUILD_ID && /bin/rm -f $BUILD_ID.tar && /bin/rm -f latest && ln -s $BUILD_ID/* latest
### Docker Images
We provide in `docker` a number of singa docker images for Jenkins to use as slaves.
To run the docker images,

    nvidia-docker run --name <jenkins-slaveXX> -d <Image ID>

## Configure Jenkins for SINGA Website Updates

### Description and Configuration

This job is triggered upon any changes to the files of the `doc/` folder.
It does the following tasks,

1. installs the latest PySINGA
2. pull the latest source code
3. generate the html files for the documentation
4. update the SINGA website

The Jenkins job configuration is similar as above except the following fields,

* Source Code Management - Git - Additional Behaviors - Include Region `doc/*`
* Build - Execute Shell - Command `bash -ex tool/jenkins/jenkins_doc.sh`
* No `Post-build Actions`

### Docker Images

The Docker image for the Jenkins slave node is at `docker/ubuntu16.04/runtime/Dockerfile`.
To build the docker image,

    # under the docker/ubuntu16.04/runtime/ folder
    $ docker built -t singa:doc .

To start the slave node

    $ docker run --name singa-doc -d singa:doc
    $ docker exec -it singa-doc /bin/bash
    $ svn co https://svn.apache.org/repos/asf/incubator/singa/site/trunk
    # update ~/.subversion/config to set 'store-password=yes'
    # to set password free commit, we have to do a manual commit at first.
    # change any file (add spaces) inside trunk/ to commit a message
    $ svn commit -m "test" --username <committer id> --password <passwd>

## Access Control
Use `Role Strategy Plugin` to give read access for anonymous users.
