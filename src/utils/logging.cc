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

#include "singa/utils/logging.h"

#include <stdlib.h>
#include <sys/types.h>
#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

namespace singa {

FILE* log_file[NUM_SEVERITIES] = {};
bool not_log_stderr[NUM_SEVERITIES] = {};

void InitLogging(const char* argv) {
#ifdef USE_GLOG
  google::InitGoogleLogging(argv);
#else
  LogToStderr();
#endif
}

void LogToStderr() {
#ifdef USE_GLOG
  google::LogToStderr();
#else
  for (int i = 0; i < NUM_SEVERITIES; ++i) {
    log_file[i] = nullptr;
    not_log_stderr[i] = false;
  }
#endif
}

void SetStderrLogging(int severity) {
#ifdef USE_GLOG
  google::SetStderrLogging(severity);
#else
  for (int i = 0; i < NUM_SEVERITIES; ++i) {
    not_log_stderr[i] = i >= severity ? false : true;
  }
#endif
}

void SetLogDestination(int severity, const char* path) {
#ifdef USE_GLOG
  google::SetLogDestination(severity, path);
#else
  log_file[severity] = fopen(path, "a");
  if (severity < ERROR) not_log_stderr[severity] = true;
#endif
}

#ifndef USE_GLOG
namespace logging {

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

inline pid_t GetPID() { return getpid(); }
inline pid_t GetTID() { return (pid_t)(uintptr_t)pthread_self(); }

void LogMessage::GenerateLogMessage() {
  time_t rw_time = time(nullptr);
  struct tm tm_time;
  localtime_r(&rw_time, &tm_time);
  // log to a file
  for (int i = severity_; i >= 0; --i)
    if (log_file[i]) DoLogging(log_file[i], tm_time);
  // log to stderr
  if (!not_log_stderr[severity_]) DoLogging(stderr, tm_time);
}

void LogMessage::DoLogging(FILE* file, const struct tm& tm_time) {
  fprintf(file, "[%c d%02d%02d t%02d:%02d:%02d p%05d:%03d %s:%d] %s\n",
          "IWEF"[severity_], 1 + tm_time.tm_mon, tm_time.tm_mday,
          tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec, GetPID(),
          static_cast<unsigned>(GetTID() % 1000), fname_, line_, str().c_str());
}

LogMessage::~LogMessage() { GenerateLogMessage(); }

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}
LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return
  GenerateLogMessage();
  abort();
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << (short)v;
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << (short)v;
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << (unsigned short)v;
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& p) {
  (*os) << "nullptr";
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new string(stream_->str());
}

}  // namespace logging
#endif

}  // namespace singa
