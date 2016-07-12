# Building SINGA Docker container 
 
This guide explains how to set up a development environment for SINGA using Docker. It requires only Docker to be installed. The resulting image contains the complete working environment for SINGA. The image can then be used to set up cluster environment over one or multiple physical nodes.  

1. [Build SINGA base](#build_base)
2. [Build GPU-enabled SINGA](#build_gpu)
3. [Build SINGA with Mesos and Hadoop](#build_mesos)
4. [Pre-built images](#pre_built)
5. [Launch and stop SINGA (stand alone mode)](#launch_stand_alone)
6. [Launch pseudo-distributed SINGA on one node](#launch_pseudo)
7. [Launch fully distributed SINGA on multiple nodes](#launch_distributed)

---

<a name="build_base"></a>
#### Build SINGA base image
 
````
$ cd $SINGA_HOME/..
$ sudo docker build -t singa/base -f incubator-singa/tool/docker/singa/Dockerfile . 
$ sudo docker images
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
singa/base             latest              XXXX                XXX                 XXX GB
````

The result is the image containing a built version of SINGA. 

   ![singa/base](http://www.comp.nus.edu.sg/~dinhtta/files/images_base.png)

   *Figure 1. singa/base Docker image, containing library dependencies and SINGA built from source.*

---

<a name="build_gpu"></a>
#### Build SINGA with GPU support 
 
````
$ cd $SINGA_HOME/..
$ sudo docker build -t singa/gpu -f incubator-singa/tool/docker/singa/Dockerfile_gpu . 
$ sudo docker images
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
singa/gpu             latest              XXXX                XXX                 XXX GB
````

---

<a name="build_mesos"></a>
#### Build SINGA with Mesos and Hadoop
````
$ cd $SINGA_HOME/.. 
$ sudo docker build -t singa/mesos -f incubator-singa/tool/docker/mesos/Dockerfile .
$ sudo docker images
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
singa/mesos             latest              XXXX                XXX                 XXX GB
````
   ![singa/mesos](http://www.comp.nus.edu.sg/~dinhtta/files/images_mesos.png#1)
   
   *Figure 2. singa/mesos Docker image, containing Hadoop and Mesos built on
top of SINGA. The default namenode address for Hadoop is `node0:9000`*

**Notes** A common failure observed during the build process is caused by network failure occuring when downloading dependencies. Simply re-run the build command. 

---

<a name="pre_built"></a>
#### Pre-built images on epiC cluster
For users with access to the `epiC` cluster, there are pre-built and loaded Docker images at the following nodes:

      ciidaa-c18
      ciidaa-c19

The available images at those nodes are:

````
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
singa/base             latest              XXXX                XXX                 2.01 GB
singa/mesos            latest              XXXX                XXX                 4.935 GB
weaveworks/weaveexec   1.1.1               XXXX                11 days ago         57.8 MB
weaveworks/weave       1.1.1               XXXX                11 days ago         17.56 MB
````

---

<a name="launch_stand_alone"></a>
#### Launch and stop SINGA in stand-alone mode
To launch a test environment for a single-node SINGA training, simply start a container from `singa/base` image. The following starts a container called
`XYZ`, then launches a shell in the container: 

````
$ sudo docker run -dt --name XYZ singa/base /usr/bin/supervisord
$ sudo docker exec -it XYZ /bin/bash
````

![Nothing](http://www.comp.nus.edu.sg/~dinhtta/files/images_standalone.png#1)

   *Figure 3. Launch SINGA in stand-alone mode: single node training*

Inside the launched container, the SINGA source directory can be found at `/root/incubator-singa`. 

**Launching GPU-enabled container**
First, make sure that the host GPUs are up and running. The list of `NVIDIA` devices should be listed at
`/dev/nvidiaYYY`.

Next, start a new container, passing it all the devices

````
$ sudo docker run -dt --device /dev/nvidiaYYY --device /dev/nvidiaYYY ... --name XYZ singa/gpu /usr/bin/supervisord
$ sudo docker exec -it XYZ /bin/bash
````

**Stopping the container**

````
$ sudo docker stop XYZ
$ sudo docker rm ZYZ
````

---

<a name="launch_pseudo"></a>
#### Launch SINGA on pseudo-distributed mode (single node)
To simulate a distributed environment on a single node, one can repeat the
previous step multiple times, each time giving a different name to the
container.  Network connections between these containers are already supported,
thus SINGA instances/nodes in these container can readily communicate with each
other. 

The previous approach requires the user to start SINGA instances individually
at each container. Although there's a bash script for that, we provide a better
way. In particular, multiple containers can be started from `singa/mesos` image
which already bundles Mesos and Hadoop with SINGA. Using Mesos makes it easy to
launch, stop and monitor the distributed execution from a single container.
Figure 4 shows `N+1` containers running concurrently at the local host. 

````
$ sudo docker run -dt --name node0 singa/mesos /usr/bin/supervisord
$ sudo docker run -dt --name node1 singa/mesos /usr/bin/supervisord
...
````

![Nothing](http://www.comp.nus.edu.sg/~dinhtta/files/images_pseudo.png#1)
   
*Figure 4. Launch SINGA in pseudo-distributed mode : multiple SINGA nodes over one single machine*

**Starting SINGA distributed training**

Refer to the [Mesos
guide](mesos.html)
for details of how to start training with multiple SINGA instances. 

**Important:** the container that assumes the role of Hadoop's namenode (and often Mesos's and Zookeeper's mater node as well) **must** be named `node0`. Otherwise, the user must log in to individual containers and change the Hadoop configuration separately. 

**Notes on Docker version >=1.9** Newer version of Docker adopted a built-in DNS server at the deamon. As a consequence,
name resolution inside containers now **cannot** depend on the automatically updated `/etc/hosts` files as in version
1.8 and earlier. Here we recommend two ways to make pseudo-distributed and distributed SINGA containers work as before

1. Downgrade to docker version 1.8 and earlier

         $ sudo apt-get install docker-engine=1.8.3-0~trusty

2. Manually log in to each running container, by `sudo exec -it <name> /bin/bash`, and edit the `/etc/hosts` with the
assigned IP addresses of all other running containers. 

         node0 <ip0>
         node1 <ip1>
         ...

---

<a name="launch_distributed"></a>
#### Launch SINGA on fully distributed mode (multiple nodes)
The previous section has explained how to start a distributed environment on a
single node. But running many containers on one node does not scale. When there
are multiple physical hosts available, it is better to distribute the
containers over them. 

The only extra requirement for the fully distributed mode, as compared with the
pseudo distributed mode, is that the containers from different hosts are able
to transparently communicate with each other. In the pseudo distributed mode,
the local docker engine takes care of such communication. Here, we rely on
[Weave](http://weave.works/guides/weave-docker-ubuntu-simple.html) to make the
communication transparent. The resulting architecture is shown below.  

![Nothing](http://www.comp.nus.edu.sg/~dinhtta/files/images_full.png#1)
   
*Figure 5. Launch SINGA in fully distributed mode: multiple SINGA nodes over multiple machines*

**Install Weave at all hosts**

```
$ curl -L git.io/weave -o /usr/local/bin/weave
$ chmod a+x /usr/local/bin/weave
```

**Starting Weave**

Suppose `node0` will be launched at host with IP `111.222.111.222`.

+ At host `111.222.111.222`:

          $ weave launch
          $ eval "$(weave env)"  //if there's error, do `sudo -s` and try again

+ At other hosts:

          $ weave launch 111.222.111.222
          $ eval "$(weave env)" //if there's error, do `sudo -s` and try again

**Starting containers**

The user logs in to each host and starts the container (same as in [pseudo-distributed](#launch_pseudo) mode). Note that container acting as the head node of the cluster must be named `node0` (and be running at the host with IP `111.222.111.222`, for example). 

**_Important_:** when there are other containers sharing the same host as `node0`, say `node1` and `node2` for example,
there're additional changes to be made to `node1` and `node2`. Particularly, log in to each container and edit
`/etc/hosts` file:

````
# modified by weave
...
X.Y.Z	node0 node0.bridge  //<- REMOVE this line
..
````
This is to ensure that name resolutions (of `node0`'s address) from `node1` and `node2` are correct. By default,
containers of the same host resolves each other's addresses via the Docker bridge. Instead, we want they to use
addressed given by Weave.  


**Starting SINGA distributed training**

Refer to the [Mesos guide](mesos.html)
for details of how to start training with multiple SINGA instances. 

