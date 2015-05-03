#ifndef INCLUDE_TRAINER_TRAINER_H_
#define INCLUDE_TRAINER_TRAINER_H_
#include "proto/cluster.pb.h"
#include "proto/model.pb.h"
#include "utils/updater.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "neuralnet/neuralnet.h"
#include "trainer/pm_worker.h"
#include "trainer/pm_server.h"
#include "trainer/worker.h"
#include "trainer/server.h"

namespace singa {
/**
 * Every running process has a training object which launches one or more
 * worker (and server) threads.
 *
 * The main thread runs a loop to forward messages between workers and servers.
 */
class Trainer{
 public:
  /**
   * Start the training in one process
   *
   * @param modelproto
   * @param clusterproto
   */
  void Start(const ModelProto& modelproto, const ClusterProto& clusterproto,
    int procs_id);

  // TODO add Resume() function to continue training from a previously stopped
  // point.

 protected:
  void Run();
  /**
   * Register default implementations for all base classes used in the system,
   * e.g., the Updater, BaseMsg, etc.
   *
   * All built-in layer implementations are
   * registered here.
   * For other base classes, use its base class name (string) as the key and the
   * implementation class as the value, e.g., <"Updater" SGDUpdater>.
   */
  void RegisterDefaultClasses(const singa::ModelProto& proto);
};
} /* singa */
#endif // INCLUDE_TRAINER_TRAINER_H_
