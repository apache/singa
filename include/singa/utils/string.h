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

#ifndef SINGA_UTILS_TOKENIZER_H_
#define SINGA_UTILS_TOKENIZER_H_

#include <algorithm>
#include <string>

#include "singa/utils/logging.h"

namespace singa {
inline bool icasecmp(const string& l, const string& r) {
  return l.size() == r.size() &&
         equal(l.cbegin(), l.cend(), r.cbegin(),
               [](string::value_type l1, string::value_type r1) {
                 return toupper(l1) == toupper(r1);
               });
}

inline string ToLowerCase(const string& input) {
  string out;
  out.resize(input.size());
  std::transform(input.begin(), input.end(), out.begin(), ::tolower);
  return out;
}

inline int ArgPos(int argc, char** arglist, const char* arg) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(arglist[i], arg) == 0) {
      return i;
    }
  }
  return -1;
}

template <typename T>
inline std::string VecToStr(const std::vector<T>& in) {
  std::string out = "(";

  for (auto x : in) {
    out += std::to_string(x) + ", ";
  }
  out += ")";
  return out;
}

/**
 * Tokenize a string.
 *
 * example:
 * Tokenizer t("assa,asf;wes", ",;");
 * string x;
 * t >> x; // x is assa
 * t >> x; // x is asf
 * t >> x; // x is wes
 * cout << (t >> x); // print 0.
 */

class Tokenizer {
 public:
  Tokenizer(const std::string& str, const std::string& sep)
      : start_(0), sep_(sep), buf_(str) {}
  Tokenizer& operator>>(std::string& out) {
    CHECK_LT(start_, buf_.length());
    int start = start_;
    auto pos = buf_.find_first_of(sep_, start);
    if (pos == std::string::npos) pos = buf_.length();
    start_ = (unsigned int)(pos + 1);
    out = buf_.substr(start, pos);
    return *this;
  }
  bool Valid() { return start_ < buf_.length(); }

 private:
  unsigned start_;
  std::string sep_;
  const std::string& buf_;
};

}  // namespace singa

#endif  // SINGA_UTILS_TOKENIZER_H_
