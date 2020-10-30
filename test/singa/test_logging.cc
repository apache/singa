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

#include "gtest/gtest.h"
#include "singa/utils/logging.h"

TEST(Logging, InfoLogging) {
  int a = 3;
  CHECK_EQ(a, 3);
  LOG(INFO) << "test info logging";
}

TEST(Logging, WarningLogging) {
  int a = 4;
  CHECK_EQ(a, 4);
  LOG(WARNING) << "test warning logging";
}

TEST(Logging, ErrorLogging) {
  int a = 5;
  CHECK_EQ(a, 5);
  LOG(ERROR) << "test error logging";
}

TEST(Logging, FatalLogging) {
  int a = 6;
  CHECK_EQ(a, 6);
  // LOG(FATAL) << "test fatal logging";
}

TEST(Logging, SetLogDestination) {
  int a = 6;
  singa::SetLogDestination(singa::WARNING, "/tmp/test.log");
  CHECK_EQ(a, 6);
  LOG(WARNING) << "test warning logging to file";
}

TEST(Logging, StderrLoggingLevel) {
  int a = 6;
  singa::SetStderrLogging(singa::WARNING);
  CHECK_EQ(a, 6);
  LOG(INFO) << "test info logging to stderr";
  LOG(WARNING) << "test warning logging to stderr and file";
  LOG(ERROR) << "test error logging to stderr and file";
}
