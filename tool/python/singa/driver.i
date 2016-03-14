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
%apply (int ARGC, char **ARGV) { (int argc, char **argv)  }
%{
#include "singa/driver.h"
%}

namespace singa{
using std::vector;
class Driver{
public:
void Train(bool resume, const std::string job_conf);
void Init(int argc, char **argv);
void InitLog(char* arg);
void Test(const std::string job_conf);
};
}

