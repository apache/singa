// Copyright 2015 Anh Dinh

#include <stdio.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdlib.h>
#include <mesos/scheduler.hpp>
#include <string>
#include <iostream>
#include <fstream>

using std::string;
using mesos::SchedulerDriver;
using std::vector;
using std::map;

/**
 * Mesos scheduler for Singa. When one of the task/node fails, restart the entire scheduler.
 *
 * Usage: scheduler --singa_home --singa_workdir --nhosts --ncpus_per_host --nmem_per_host
 *
 * + singa_home: contains the singa binary
 * + singa_workdir (for example, singa_home/examples/cifra10) contains the files necessary for training a model:
 * 		- singa_workdir/hostfile
 * 		- singa_workdir/cluster.conf
 * 		- singa_workdir/model.conf
 * + nhosts: number of singa nodes
 * + ncpus_per_host: how many CPUs per singa node
 * + nmem_per_host: how much memory per singa node
 */

DEFINE_string(singa_home, "/home/dinhtta/Research/incubator-singa", "");
DEFINE_string(singa_workdir,
		"/home/dinhtta/Research/incubator-singa/examples/cifar10", "");
DEFINE_int32(nhosts, 1, "number of hosts");
DEFINE_int32(ncpus_per_host, 2, "number of cpus per host");
DEFINE_int32(nmem_per_host, 1024, "memory per host");

class SingaFramework: public mesos::Scheduler {
public:
	SingaFramework() :
			launched_tasks_(0), pending_tasks_(0), has_launched_(false) {
	}

	virtual void registered(SchedulerDriver *driver,
			const mesos::FrameworkID& frameworkId,
			const mesos::MasterInfo& masterInfo) {
		LOG(INFO) << "Registered frameworkId = " << frameworkId.value();
		frameworkId_ = frameworkId.value();
	}

	virtual void reregistered(SchedulerDriver *driver,
			const mesos::MasterInfo& masterInfo) {
		LOG(INFO) << "Reassigned frameworkID = " << frameworkId_;
	}

	virtual void disconnected(SchedulerDriver *driver) {
		LOG(INFO) << "Bye bye";
	}

	/**
	 * Handle resource offers from Mesos nodes.
	 *
	 * 1. Create corresponding tasks for each offer.
	 * 2. Wait until having enough resources as required (nodes, cpus per node, memory).
	 * 3. Write to host file, with IDs (IP addresses) of the nodes offering resources.
	 * 4. Launch tasks.
	 */
	virtual void resourceOffers(SchedulerDriver* driver,
			const std::vector<mesos::Offer>& offers) {
		LOG(INFO) << "Received " << offers.size() << " offers";
		if (has_launched_) {
			LOG(INFO) << "Already launched tasks, skipped this offer";
			return;
		}

		// print out offer
		for (int i = 0; i < offers.size(); i++) {
			const mesos::Offer offer = offers[i];
			string slave_host = offer.slave_id().value();
			LOG(INFO) << "Offer " << offer.id().value() << " from host "
					<< slave_host << " hostname = " << offer.hostname();

			int cpus = 0, mem = 0;
			int nresources = offer.resources().size();

			vector < mesos::TaskInfo > *new_tasks =
					new vector<mesos::TaskInfo>();

			for (int r = 0; r < nresources; r++) {
				const mesos::Resource& resource = offer.resources(r);
				if (resource.name() == "cpus"
						&& resource.type() == mesos::Value::SCALAR)
					cpus = resource.scalar().value();
				else if (resource.name() == "mem"
						&& resource.type() == mesos::Value::SCALAR)
					mem = resource.scalar().value();
			}

			if (cpus < FLAGS_ncpus_per_host || mem < FLAGS_nmem_per_host
					|| pending_tasks_ >= FLAGS_nhosts) {
				LOG(INFO) << "Decline offer, not enough resource";
				driver->declineOffer(offer.id());
				return;
			}

			// wait until having enough resource, then launch
			while (cpus >= FLAGS_ncpus_per_host && mem >= FLAGS_nmem_per_host
					&& pending_tasks_ < FLAGS_nhosts) {
				// create tasks and add to pending tasks
				mesos::TaskInfo task;
				task.set_name("Singa task");
				char string_id[512];
				snprintf(string_id, 256, "%d", pending_tasks_);
				task.mutable_task_id()->set_value(string(string_id));
				task.mutable_slave_id()->MergeFrom(offer.slave_id());

				snprintf(string_id, 256,
						"cd %s/mesos; ./launch_script.sh --model=%s/model.conf --cluster=%s/cluster.conf > out 2>&1",
						FLAGS_singa_home.c_str(), FLAGS_singa_workdir.c_str(),
						FLAGS_singa_workdir.c_str());

				LOG(INFO) << "Command = " << string_id;

				task.mutable_command()->set_value(string(string_id));

				mesos::Resource *resource;
				resource = task.add_resources();
				resource->set_name("cpus");
				resource->set_type(mesos::Value::SCALAR);
				resource->mutable_scalar()->set_value(FLAGS_ncpus_per_host);

				resource = task.add_resources();
				resource->set_name("mem");
				resource->set_type(mesos::Value::SCALAR);
				resource->mutable_scalar()->set_value(FLAGS_nmem_per_host);

				new_tasks->push_back(task);
				pending_tasks_++;
				cpus -= FLAGS_ncpus_per_host;
				mem -= FLAGS_nmem_per_host;

				singa_hosts_.push_back(offer.hostname());
			}
			tasks_[offer.id().value()] = new_tasks;
			// send offer
			if (pending_tasks_ == FLAGS_nhosts) {
				// write to file
				char path[256];
				snprintf(path, 256, "%s/hostfile", FLAGS_singa_workdir.c_str());
				std::ofstream file(path);
				for (int i = 0; i < singa_hosts_.size(); i++)
					file << singa_hosts_[i] << "\n";
				file.close();
				for (map<string, vector<mesos::TaskInfo>*>::iterator it =
						tasks_.begin(); it != tasks_.end(); ++it) {
					mesos::OfferID newId;
					newId.set_value(it->first);
					LOG(INFO) << "Launching task with offer ID " << it->first;
					driver->launchTasks(newId, *(it->second));
				}
				has_launched_ = true;
			}
		}
	}

	virtual void offerRescinded(SchedulerDriver *driver,
			const mesos::OfferID& offerId) {
	}

	virtual void statusUpdate(SchedulerDriver* driver,
			const mesos::TaskStatus& status) {
		LOG(INFO) << " Task status report for task "
				<< status.task_id().value();
		LOG(INFO) << "      Status = " << status.state();

		if (status.state() == mesos::TASK_FINISHED)
			driver->stop();
		else if (status.state() == mesos::TASK_FAILED)
			driver->abort();
	}

	virtual void frameworkMessage(SchedulerDriver* driver,
			const mesos::ExecutorID& executorId, const mesos::SlaveID& slaveId,
			const string& data) {
		LOG(INFO) << "Got a framework message " << data;
	}

	virtual void slaveLost(SchedulerDriver* driver,
			const mesos::SlaveID& slaveId) {
	}

	virtual void executorLost(SchedulerDriver* driver,
			const mesos::ExecutorID& executorId, const mesos::SlaveID& slaveId,
			int status) {
		LOG(INFO) << "Executor lost";
	}

	virtual void error(SchedulerDriver* driver, const string& message) {
		LOG(INFO) << "Got an error " << message;
	}

private:
	string frameworkId_;
	vector<string> singa_hosts_;
	int launched_tasks_;
	int pending_tasks_;
	bool has_launched_;
	map<string, vector<mesos::TaskInfo>*> tasks_;
};

/**
 * <master address> <ninstances> <SINGA HOME>
 */
int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	// google::InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = 1;

	if (argc != 2) {
		std::cerr << "Usage <master address>" << std::endl;
		exit(1);
	}

	SchedulerDriver *driver = NULL;
	int status = mesos::DRIVER_RUNNING;
	while (true) {
		SingaFramework scheduler;

		mesos::FrameworkInfo framework;
		framework.set_user("");
		framework.set_name("Singa framework");

		driver = new mesos::MesosSchedulerDriver(&scheduler, framework,
				argv[1]);
		LOG(INFO) << "Starting Singa framework...";
		status = driver->run();
		if (status == mesos::DRIVER_ABORTED)
			LOG(INFO) << "Singa framework aborted. Restarting ...";
		sleep(2);
		driver->stop();
		delete driver;
	}

	return status == mesos::DRIVER_STOPPED ? 0 : 1;
}

