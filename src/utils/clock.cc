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

#include "singa/utils/clock.h"

#include <glog/logging.h>
#include <cmath>
#include <ctime>
#include "singa/utils/cluster.h"
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"

namespace singa {

Clock::Clock() {
  start_t = clock();
}



void Clock::Start() {
  start_t = clock();
}

double Clock::End() {
  end_t = clock();
  double runtime = (end_t-start_t)/static_cast<double>(CLOCKS_PER_SEC)*1000;
  return runtime;
}

double Clock::End(int iteration_num, std::string content) {
  end_t = clock();
  double runtime = (end_t-start_t);
  runtime = runtime/(static_cast<double>(CLOCKS_PER_SEC)*1000*iteration_num);
  return runtime;
}

void Clock::EndWithLog(std::string content) {
  double runtime = end();
  DLOG(ERROR) << "Running time of " << content<< " is " << runtime << " ms";
}

void Clock::EndWithLog(int iteration_num, std::string content) {
  double runtime = end(iteration_num, content);
  DLOG(ERROR)<< "Average Running time of " << iteration_num;
  DLOG(ERROR)<< " iterations in " << content << " is " << runtime << " ms";
}

double Clock::Elapse() {
  end_t = clock();
  double runtime = (end_t-start_t)/static_cast<double>(CLOCKS_PER_SEC)*1000;
  return runtime;
}

double Clock::Elapse(int iteration_num, std::string content) {
  end_t = clock();
  double runtime = (end_t-start_t);
  runtime = runtime/(static_cast<double>(CLOCKS_PER_SEC)*1000*iteration_num);
  return runtime;
}

void Clock::ElapseWithLog(std::string content) {
  double runtime = end();
  DLOG(ERROR)<< "Running time of " <<content << " is " << runtime << " ms";
}

void Clock::ElapseWithLog(int iteration_num, std::string content) {
  double runtime = end(iteration_num, content);
  DLOG(ERROR) << "Average Running time of " << iteration_num;
  DLOG(ERROR) << " iterations in " << content << " is " << runtime << " ms";
}

}  // namespace singa
