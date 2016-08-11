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

#ifndef SINGA_UTILS_FACTORY_H_
#define SINGA_UTILS_FACTORY_H_

#include <functional>
#include <map>
#include <string>

#include "singa/utils/logging.h"
/**
 * Macro that creats a function which instantiate a subclass instance and
 * returns pointer to the base class.
 */
#define CreateInstance(SubClass, BaseClass) \
  [](void)->BaseClass* {return new SubClass();}

/**
 * Factory template to generate class (or a sub-class) object based on id.
 * 1. register class creation function that generates a class
 * object based on id.
 * 2. call Create() func to call the creation function and return
 * a pointer to the base calss.
 */
template<typename T, typename ID = std::string>
class Factory {
 public:
  /**
   * Register functions to create user defined classes.
   * This function is called by the REGISTER_FACTORY macro.
   *
   * @param id Identifier of the creating function/class
   * @param func a function that creates a layer instance
   */
  static void Register(const ID& id,
                       const std::function<T*(void)>& creator) {
    Registry* reg = GetRegistry();
    // CHECK(reg->find(id) == reg->end())
    //  << "The id " << id << " has been registered";
    (*reg)[id] = creator;
  }

  /**
   * create an instance by providing its id
   *
   * @param id
   */
  static T* Create(const ID& id) {
    Registry* reg = GetRegistry();
    CHECK(reg->find(id) != reg->end())
      << "The creation function for " << id << " has not been registered";
    return (*reg)[id]();
  }

  static const std::vector<ID> GetIDs() {
    std::vector<ID> keys;
    for (const auto entry : *GetRegistry())
      keys.push_back(entry.first);
    return keys;
  }

 private:
  // Map that stores the registered creation functions
  typedef std::map<ID, std::function<T*(void)>> Registry;
  static Registry* GetRegistry() {
    static Registry reg;
    return &reg;
  }
};

template<typename Base, typename Sub, typename ID = std::string>
class Registra {
 public:
  Registra(const ID& id) {
    Factory<Base, ID>::Register(id, [](void) { return new Sub(); });
  }
};
#endif  // SINGA_UTILS_FACTORY_H_
