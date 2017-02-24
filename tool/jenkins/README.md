# Jenkins CI Support

## Introduction
This documentation is to guide SINGA developers to setup Jenkins service to support continuous integration on GPU systems. After each commit,
1. SINGA should be compiled and tested automatically under different settings (e.g., OS and hardware).
2. Convenient binaries should be generated automatically and archived.

Continuous integration for CPU systems is enabled via [Travis](../travis).

## Install Jenkins
[Jenkins Official Wiki](https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins)
The slave nodes for running different building environments are configured under 'Manage Jenkins'->'Manage nodes'.

## Configure Jenkins for Unit Testing and Binary Package Generation
Create a multi-configuration project and configure project as follows:

### Description
This job automatically pulls latest commits from Apache incubator-singa github repository, then for different environments

* compile and test SINGA on GPUs
* create Debian GPU packages
* create anaconda GPU packages

The working nodes (or Docker containers) are configured in Jenkins-Manage Jenkins-Mange Nodes.
Each node should configure the following environment variable
1. CUDA_VERSION, e.g., 7.5
2. CUDNN_VERSION e.g, 5
3. ANACONDA_UPLOAD_TOKEN
4. SINGA_NAME=singa-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}
5. OS_VERSION, e.g., ubuntu14.04

### General
  * Discard old builds - Max # of builds to keep - 50
  * GitHub project - ``https://github.com/apache/incubator-singa``

### Source Code Management
  * Git - Repository URL - ``https://github.com/apache/incubator-singa``
  * Git - Branch Specifier - ``*/master``

### Build Triggers
  * Poll SCM - Schedule - ``H/30 * * * *`` (pull every 30 minutes)

### Configuration Matrix
  * User-defined Axis - name ``lang`` values ``CUDA`` (Add CPP if you want to test CPU code)
  * Slave - name ``env`` Node/label: tick available nodes

### Build
  * compile and do unit test on GPU
    Execute shell - command - ``bash -ex tool/jenkins/test.sh $lang``
    `$lang` is set in **Configuration Matrix* section

  * create Debian package
    Execute shell - command - ``bash -ex tool/debian/build.sh --python --$lang``

  * create conda package
    Execute shell - command -

        git push https://username:token@github.com/nusdbsystem/incubator-singa.git -f
        bash -ex tool/jenkins/jenkins_test.sh $lang
        export CONDA_BLD_PATH=/root/conda-bld-$BUILD_NUMBER
        mkdir $CONDA_BLD_PATH
        /root/miniconda/bin/conda-build tool/conda
        /root/miniconda/bin/anaconda -t ANACONDA_UPLOAD_TOKEN upload -u nusdbsystem -l main $CONDA_BLD_PATH/linux-64/singa-*.tar.bz2 --force


    It first pushes to a mirror site to invoke travis-ci for CPU package creation;
    Then it compiles and runs unit tests;
    Finally it creates the conda package for GPU and upload it.

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

### Docker Images
We provide in a number of singa docker [images](./docker) for Jenkins to use as slaves.
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
