/*********************************************************
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
************************************************************/
#ifdef USE_TC
#include "./tc_fn.h"

namespace singa {

TcFnHandle::TcFnHandle(std::string tcDefinition, std::string entryFn, const std::vector<Tensor> &inputs)
{
  tc_string = tcDefinition;
  tc_name = entryFn;
  auto naiveOptions = tc::CudaBackend::MappingOptionsType::makeNaiveMappingOptions();
  pExecutor = singa::compileTC<tc::CudaBackend>(tcDefinition, entryFn, inputs, {naiveOptions});
};

Tensor tcExecute(const TcFnHandle &tcFnhandle, const std::vector<Tensor> &inputs)
{
  auto outputs = singa::prepareOutputs(tcFnhandle.tc_string, tcFnhandle.tc_name, inputs);
  singa::runTC(*(tcFnhandle.pExecutor), inputs, outputs);
  return outputs[0];
}

}
#endif // USE_TC
