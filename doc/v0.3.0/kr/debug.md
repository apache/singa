# How to Debug

---

Since SINGA is developed on Linux using C++, GDB is the preferred debugging
tool. To use GDB, the code must be compiled with `-g` flag. This is enabled by

    ./configure --enable-debug
    make

## Debugging for single process job

If your job launches only one process, then use the default *conf/singa.conf*
for debugging. The process will be launched locally.

To debug, first start zookeeper if it is not started yet, and launch GDB

    # do this for only once
    ./bin/zk-service.sh start
    # do this every time
    gdb .libs/singa

Then set the command line arguments

    set args -conf JOBCONF

Now you can set your breakpoints and start running.

## Debugging for jobs with multiple processes
