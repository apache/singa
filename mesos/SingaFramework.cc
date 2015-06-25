#include <mesos/scheduler.hpp>
#include <string>
#include <glog/logging.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std; 
using namespace mesos; 

/**
* Assume 2 CPU per instance
*/
#define CPU_PER_INSTANCE 2
#define MEM_PER_INSTANCE 1024

class SingaFramework: public Scheduler{
public:
	SingaFramework(){}

	SingaFramework(int ninstances, string command): ninstances_(ninstances), 
							command_(command), launched_tasks_(0),
							pending_tasks_(0){}

	virtual void registered(SchedulerDriver *driver, const FrameworkID& frameworkId, 
					const MasterInfo& masterInfo){
		LOG(INFO) << "frameworkId = " << frameworkId.value(); 	
		frameworkId_ = frameworkId.value(); 

	}
		
	virtual void reregistered(SchedulerDriver *driver, const MasterInfo& masterInfo){
		LOG(INFO) << "reassigned frameworkID = " << frameworkId_; 
	}
	virtual void disconnected(SchedulerDriver *driver){
		LOG(INFO) << "Bye bye"; 
	}

	virtual void resourceOffers(SchedulerDriver* driver,	
					const std::vector<Offer>& offers){
		LOG(INFO) << "Number of offers = " <<offers.size(); 
		//print out offer
		for (int i=0; i<offers.size(); i++){
			const Offer offer = offers[i]; 
			LOG(INFO) << "Got offer ID = .." << offer.id().value(); 
			string slave_host = offer.slave_id().value(); 
			LOG(INFO) <<"    From host " << slave_host << " name = " << offer.hostname(); 
			
			int cpus=0, mem=0; 
			int nresources = offer.resources().size(); 

			vector<TaskInfo> *new_tasks = new vector<TaskInfo>();  

			for (int r=0; r<nresources; r++){
				const Resource& resource = offer.resources(r); 
				if (resource.name()=="cpus" && resource.type()==Value::SCALAR)
					cpus=resource.scalar().value(); 
				else if (resource.name()=="mem" && resource.type()==Value::SCALAR)
					mem=resource.scalar().value(); 
			}
			
			if (cpus < CPU_PER_INSTANCE || mem < MEM_PER_INSTANCE || 
				pending_tasks_ >= ninstances_){
				LOG(INFO) << "Decline offer, not enough resource"; 
				driver->declineOffer(offer.id()); 
				return; 
			}

			//wait until having enough resource, then launch
			while (cpus >= CPU_PER_INSTANCE && mem >= MEM_PER_INSTANCE && 
					pending_tasks_<ninstances_){
					
					//create tasks and add to pending tasks
					TaskInfo task; 
					task.set_name("SINGA task"); 
					char string_id[512]; 
					sprintf(string_id,"%d",pending_tasks_); 
					task.mutable_task_id()->set_value(string(string_id));
					task.mutable_slave_id()->MergeFrom(offer.slave_id()); 
					
					sprintf(string_id,"cd %s/mesos; ./pm_run.sh %d",command_.c_str(), pending_tasks_); 
					LOG(INFO) << "Command = " << string_id; 

					task.mutable_command()->set_value(string(string_id)); 		 

					Resource *resource; 
					resource = task.add_resources(); 
					resource->set_name("cpus"); 
					resource->set_type(Value::SCALAR); 
					resource->mutable_scalar()->set_value(CPU_PER_INSTANCE); 

					resource = task.add_resources(); 
					resource->set_name("mem"); 
					resource->set_type(Value::SCALAR); 
					resource->mutable_scalar()->set_value(MEM_PER_INSTANCE); 
			
					new_tasks->push_back(task); 
					pending_tasks_++; 
					cpus-=CPU_PER_INSTANCE; 
					mem-=MEM_PER_INSTANCE; 

					singa_hosts_.push_back(offer.hostname()); 
			}
			tasks_[offer.id().value()] = new_tasks;  		
			//send offer
			if (pending_tasks_==ninstances_){
				//write to file
				char path[256];
				sprintf(path,"%s/examples/mnist/hostfile",command_.c_str()); 
				ofstream file(path); 
				for (int i=0; i<singa_hosts_.size(); i++)
					file << singa_hosts_[i] << "\n"; 
				file.close(); 	
				for (map<string,vector<TaskInfo>*>::iterator it = tasks_.begin(); it!=tasks_.end(); ++it){
					OfferID newId; 
					newId.set_value(it->first); 
					LOG(INFO) << "Launching task with offer ID " << it->first; 
					driver->launchTasks(newId, *(it->second)); 
				}
			}
		}
	}

	virtual void offerRescinded(SchedulerDriver *driver, const OfferID& offerId){}

	virtual void statusUpdate(SchedulerDriver* driver, const TaskStatus& status){
		LOG(INFO) <<" Task status report for task " << status.task_id().value(); 
		LOG(INFO) <<"      Status = " << status.state(); 
		if (status.state()== TASK_FINISHED)
			driver->stop(); 
	}

	virtual void frameworkMessage(SchedulerDriver* driver, const ExecutorID& executorId,
					const SlaveID& slaveId, const std::string& data){}

	virtual void slaveLost(SchedulerDriver* driver, const SlaveID& slaveId){}

	virtual void executorLost(SchedulerDriver* driver, const ExecutorID& executorId, 
							const SlaveID& slaveId, int status){}

	virtual void error(SchedulerDriver* driver, const std::string& message){}
					
private:
	int ninstances_; 
	string command_; 
	string frameworkId_; 
	vector<string> singa_hosts_; 
	int launched_tasks_; 
	int pending_tasks_; 

	map<string, vector<TaskInfo>*> tasks_; 
};

/**
* <master address> <ninstances> <SINGA HOME> 
*/
int main(int argc, char** argv){
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	//google::InitGoogleLogging(argv[0]);
	//FLAGS_logtostderr = 1;

	if (argc!=4){
		std::cerr << "Usage <master address> <ninstances> <SINGA HOME>" << endl; 
		exit(1); 
	}

	SingaFramework scheduler(atoi(argv[2]), argv[3]); 

	FrameworkInfo framework; 
	framework.set_user(""); 
	framework.set_name("Anh's test"); 

	SchedulerDriver *driver = new MesosSchedulerDriver(&scheduler, framework, argv[1]); 
	int status = driver->run() == DRIVER_STOPPED ? 0 : 1; 

	driver->stop(); 
	delete driver; 
	return status; 
}

