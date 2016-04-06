/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

/*interface file for swig */

%module driver
%include "std_vector.i"
%include "std_string.i"
%include "argcargv.i"
%include "carrays.i"
%array_class(float, floatArray);

%apply (int ARGC, char **ARGV) { (int argc, char **argv)  }
%{
#include "singa/driver.h"
#include "singa/worker.h"
#include "singa/neuralnet/layer.h"
#include "singa/neuralnet/neuron_layer.h"
#include "singa/neuralnet/loss_layer.h"
#include "singa/utils/blob.h"
#include "singa/utils/param.h"
#include "singa/utils/updater.h"
#include "singa/proto/job.pb.h"
#include "singa/proto/common.pb.h"
%}

namespace std {
  %template(strVector) vector<string>;
  %template(intVector) vector<int>;
  %template(floatVector) vector<float>;
  %template(layerVector) vector<singa::Layer*>;
  %template(paramVector) vector<singa::Param*>;
}

namespace singa{
  class Driver{
    public:
    void Train(bool resume, const std::string job_conf);
    void Init(int argc, char **argv);
    void InitLog(char* arg);
    void Test(const std::string job_conf);
  };

  %nodefault Worker;
  class Worker{
    public:
      static singa::Worker* CreateWorker(const std::string str);
      void InitNetParams(const std::string& folder, std::vector<singa::Layer*> net);
      void Checkpoint(int step, const std::string& folder, std::vector<singa::Layer*> net);
  };
    
  class DummyLayer{
    public:
      void Setup(const std::string str, const std::vector<singa::Layer*>& srclayers);
      void Feed(int batchsize, std::vector<float>& data, std::vector<int>& aux_data);
      singa::Layer* ToLayer();
  };

  %nodefault Layer;
  class Layer{
    public:
      static singa::Layer* CreateLayer(const std::string str);
      static void SetupLayer(singa::Layer* layer, const std::string str, const std::vector<singa::Layer*>& srclayers);
      virtual void ComputeFeature(int flag, const std::vector<singa::Layer*>& srclayers); 
      virtual void ComputeGradient(int flag, const std::vector<singa::Layer*>& srclayers);
      virtual const singa::Blob<float>& data(const singa::Layer* from); 
      virtual const std::vector<singa::Param*> GetParams();
      virtual const std::string ToString(bool debug, int flag);
      void SetParams(std::vector<singa::Param*> params);
  };

  %nodefault Updater;
  class Updater{
    public:
      static singa::Updater* CreateUpdater(const std::string str);
      virtual void Update(int step, singa::Param* param, float grad_scale);
  };

  template <typename Dtype>
  class Blob{
    public:
      inline int count();
      inline const std::vector<int>& shape();
      inline Dtype* mutable_cpu_data(); 
      inline const Dtype* cpu_data();
  };

  class Param{
    public:
      inline int size();
      inline const std::vector<int>& shape();
      inline float* mutable_cpu_data();
      void FromProto(const std::string str);
      /*void ToProto(singa::BlobProto* blob); 
      */
  };

  %template(floatBlob) Blob<float>;
}
