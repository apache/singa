#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <iostream>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils/cluster.h"
#include "utils/common.h"
#include "proto/model.pb.h"
#include "proto/cluster.pb.h"
#include "server/server.h"
#include "server/pm_server.h"
#include "worker/pm_client.h"
#include "worker/worker.h"
#include "proto/topology.pb.h"
#include <string.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace google::protobuf::io;
using google::protobuf::TextFormat;

using std::ifstream;

/**
 * Testing put/get/update performance of the new zeromq-based parameter
 * servers.
 */
DEFINE_int32(procsID, 0, "global process ID");
DEFINE_string(hostfile, "examples/imagenet12/hostfile", "hostfile");
DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf",
    "configuration file for the cluster");
DEFINE_string(model_conf, "examples/imagenet12/model.conf",
    "Deep learning model configuration file");

DEFINE_string(topology_config,"examples/imagenet12/topology.conf", "Network of servers");
DEFINE_int32(server_threads,1,"Number of server's worker threads per process");
DEFINE_int32(client_threads,1,"Number of client's worker threads per process");

DEFINE_string(mode, "client", "client or server mode");
DEFINE_int32(node_id, 0, "ID of the node, client or server");
DEFINE_int32(primary_set, 0, "ID of the primary server set (for client mode only)");

/**
 *
 * Read the topology file in, and start the Client or server respectively.
 *
 * test_pm --node_id <id>
 */


#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif

int main(int argc, char **argv) {
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_logtostderr = 1;


	//Read in the topology file
	int fd = open(FLAGS_topology_config.c_str(), O_RDONLY);
	assert(fd != -1);
	singa::Topology topology;
	TextFormat::Parse(new FileInputStream(fd), &topology);


	//read host file
	ifstream hostfile(FLAGS_hostfile.c_str());
	string host;
	vector<string> hosts;
	while (getline(hostfile, host))
		hosts.push_back(host);
	
	if (FLAGS_node_id < topology.nservers()) {
		singa::SingaServer *server = new singa::SingaServer(FLAGS_node_id, topology, hosts);
		server->StartServer();
	} else {
		singa::SingaClient *client = new singa::SingaClient(FLAGS_node_id, topology, hosts);
		client->StartClient();
	}
	
	return 0;
}
