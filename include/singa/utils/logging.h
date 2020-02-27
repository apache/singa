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

// adapted from google::tensorflow::logging

#ifndef SINGA_UTILS_LOGGING_H_
#define SINGA_UTILS_LOGGING_H_

#include <stdlib.h>

#include <sstream>
#include <string>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif

namespace singa {

/// Global functions for both glog and built-in log
void InitLogging(const char* argv);
/// Make it so that all log messages go only to stderr
void LogToStderr();
/// Make it so that all log messages of at least a particular severity are
/// logged to stderr (in addtion to logging to the usual log files)
void SetStderrLogging(int severity);
/// Set the file name for logging (and disable logging to stderr)
void SetLogDestination(int severity, const char* path);

using std::string;

const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

#ifndef USE_GLOG
namespace logging {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage();
  void DoLogging(FILE* file, const struct tm& tm_time);

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _SINGA_LOG_INFO \
  ::singa::logging::LogMessage(__FILE__, __LINE__, singa::INFO)
#define _SINGA_LOG_WARNING \
  ::singa::logging::LogMessage(__FILE__, __LINE__, singa::WARNING)
#define _SINGA_LOG_ERROR \
  ::singa::logging::LogMessage(__FILE__, __LINE__, singa::ERROR)
#define _SINGA_LOG_FATAL ::singa::logging::LogMessageFatal(__FILE__, __LINE__)

#define LOG(severity) _SINGA_LOG_##severity

/// CHECK dies with a fatal error if condition is not true.  It is *not*
/// controlled by NDEBUG, so the check will be executed regardless of
/// compilation mode.  Therefore, it is safe to do things like:
///    CHECK(fp->Write(x) == 4)
#define CHECK(condition) \
  if (!(condition)) LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// CHECK* macros. It's not encouraged though.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline short GetReferenceableValue(short t) { return t; }
inline unsigned short GetReferenceableValue(unsigned short t) { return t; }
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline long GetReferenceableValue(long t) { return t; }
inline unsigned long GetReferenceableValue(unsigned long t) { return t; }
inline long long GetReferenceableValue(long long t) { return t; }
inline unsigned long long GetReferenceableValue(unsigned long long t) {
  return t;
}

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v);

// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& p);

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  CheckOpString(string* str) : str_(str) {}
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  operator bool() const { return str_ != NULL; }
  string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext);

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
// statement.  See MakeCheckOpString for sample usage.  Other
// approaches were considered: use of a template method (e.g.,
// base::BuildCheckOpString(exprtext, base::Print<T1>, &v1,
// base::Print<T2>, &v2), however this approach has complications
// related to volatile arguments and function-pointer arguments).
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream* ForVar2();
  // Get the result (inserts the closing ")").
  string* NewString();

 private:
  std::ostringstream* stream_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
#define SINGA_DEFINE_CHECK_OP_IMPL(name, op)                        \
  template <typename T1, typename T2>                               \
  inline string* name##Impl(const T1& v1, const T2& v2,             \
                            const char* exprtext) {                 \
    if (v1 op v2)                                                   \
      return NULL;                                                  \
    else                                                            \
      return ::singa::logging::MakeCheckOpString(v1, v2, exprtext); \
  }                                                                 \
  inline string* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                  \
  }

// We use the full name Check_EQ, Check_NE, etc. in case the file including
// base/logging.h provides its own #defines for the simpler names EQ, NE, etc.
// This happens if, for example, those are used as token names in a
// yacc grammar.
SINGA_DEFINE_CHECK_OP_IMPL(Check_EQ,
                           ==)  // Compilation error with CHECK_EQ(NULL, x)?
SINGA_DEFINE_CHECK_OP_IMPL(Check_NE, !=)  // Use CHECK(x == NULL) instead.
SINGA_DEFINE_CHECK_OP_IMPL(Check_LE, <=)
SINGA_DEFINE_CHECK_OP_IMPL(Check_LT, <)
SINGA_DEFINE_CHECK_OP_IMPL(Check_GE, >=)
SINGA_DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef SINGA_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                      \
  while (::singa::logging::CheckOpString _result =              \
             ::singa::logging::name##Impl(                      \
                 ::singa::logging::GetReferenceableValue(val1), \
                 ::singa::logging::GetReferenceableValue(val2), \
                 #val1 " " #op " " #val2))                      \
  ::singa::logging::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
#define CHECK_NOTNULL(val)                           \
  ::singa::logging::CheckNotNull(__FILE__, __LINE__, \
                                 "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// DCHECK_EQ/NE/...
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) \
  while (false && (condition)) LOG(FATAL)

// NDEBUG is defined, so DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _DCHECK_NOP is a helper, and should not be used outside of this file.
#define _SINGA_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _SINGA_DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _SINGA_DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _SINGA_DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _SINGA_DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _SINGA_DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _SINGA_DCHECK_NOP(x, y)

#endif

// These are for when you don't want a CHECK failure to print a verbose
// stack trace.  The implementation of CHECK* in this file already doesn't.
#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

}  // namespace logging
#endif

}  // namespace singa

#endif  // SINGA_UTILS_LOGGING_H_
