#Distributed Training on Mesos

This guide explains how to start SINGA distributed training on a Mesos cluster. It assumes that both Mesos and HDFS are already running, and every node has SINGA installed.
We assume the architecture depicted below, in which a cluster nodes are Docker container. Refer to [Docker guide](docker.html) for details of how to start individual nodes and set up network connection between them (make sure [weave](http://weave.works/guides/weave-docker-ubuntu-simple.html) is running at each node, and the cluster's headnode is running in container `node0`)

![Nothing](http://www.comp.nus.edu.sg/~dinhtta/files/singa_mesos.png)

---

## Start HDFS and Mesos
Go inside each container, using:
````
docker exec -it nodeX /bin/bash
````
and configure it as follows:

* On container `node0`

        hadoop namenode -format
        hadoop-daemon.sh start namenode
        /opt/mesos-0.22.0/build/bin/mesos-master.sh --work_dir=/opt --log_dir=/opt --quiet > /dev/null &
        zk-service.sh start

* On container `node1, node2, ...`

        hadoop-daemon.sh start datanode
        /opt/mesos-0.22.0/build/bin/mesos-slave.sh --master=node0:5050 --log_dir=/opt --quiet > /dev/null &

To check if the setup has been successful, check that HDFS namenode has registered `N` datanodes, via:

````
hadoop dfsadmin -report
````

#### Mesos logs
Mesos logs are stored at `/opt/lt-mesos-master.INFO` on `node0` and `/opt/lt-mesos-slave.INFO` at other nodes.

---

## Starting SINGA training on Mesos
Assumed that Mesos and HDFS are already started, SINGA job can be launched at **any** container.

#### Launching job

1. Log in to any container, then
        cd incubator-singa/tool/mesos
<a name="job_start"></a>
2. Check that configuration files are correct:
    + `scheduler.conf` contains information about the master nodes
    + `singa.conf` contains information about Zookeeper node0
    + Job configuration file `job.conf` **contains full path to the examples directories (NO RELATIVE PATH!).**
3. Start the job:
    + If starting for the first time:

	          ./scheduler <job config file> -scheduler_conf <scheduler config file> -singa_conf <SINGA config file>
    + If not the first time:

	          ./scheduler <job config file>

**Notes.** Each running job is given a `frameworkID`. Look for the log message of the form:

             Framework registered with XXX-XXX-XXX-XXX-XXX-XXX

#### Monitoring and Debugging

Each Mesos job is given a `frameworkID` and a *sandbox* directory is created for each job.
The directory is in the specified `work_dir` (or `/tmp/mesos`) by default. For example, the error
during SINGA execution can be found at:

            /tmp/mesos/slaves/xxxxx-Sx/frameworks/xxxxx/executors/SINGA_x/runs/latest/stderr

Other artifacts, like files downloaded from HDFS (`job.conf`) and `stdout` can be found in the same
directory.

#### Stopping

There are two way to kill the running job:

1. If the scheduler is running in the foreground, simply kill it (using `Ctrl-C`, for example).

2. If the scheduler is running in the background, kill it using Mesos's REST API:

          curl -d "frameworkId=XXX-XXX-XXX-XXX-XXX-XXX" -X POST http://<master>/master/shutdown

