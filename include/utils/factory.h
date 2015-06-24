#ifndef SINGA_UTILS_FACTORY_H_
#define SINGA_UTILS_FACTORY_H_

#include <glog/logging.h>
#include <functional>
#include <map>
#include <string>

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
template<typename T>
class Factory {
 public:
  /**
   * Register functions to create user defined classes.
   * This function is called by the REGISTER_FACTORY macro.
   *
   * @param id Identifier of the creating function/class
   * @param func a function that creates a layer instance
   */
  inline void Register(const std::string& id,
                       const std::function<T*(void)>& func) {
    CHECK(str2func_.find(id) == str2func_.end())
      << "The id has been registered by another function";
    str2func_[id] = func;
  }
  /**
   * Register functions to create user defined classes.
   * This function is called by the REGISTER_FACTORY macro.
   *
   * @param id Identifier of the creating function/class
   * @param func a function that creates a layer instance
   */
  inline void Register(int id,
                       const std::function<T*(void)>& func) {
    CHECK(id2func_.find(id) == id2func_.end())
      << "The id has been registered by another function";
    id2func_[id] = func;
  }

  /**
   * create an instance by providing its id
   *
   * @param id
   */
  inline T* Create(const std::string& id) {
    CHECK(str2func_.find(id) != str2func_.end())
      << "The creation function for " << id << " has not been registered";
    return str2func_[id]();
  }
  /**
   * create an instance by providing its id
   *
   * @param id
   */
  inline T* Create(int id) {
    CHECK(id2func_.find(id) != id2func_.end())
      << "The creation function for " << id << " has not been registered";
    return id2func_[id]();
  }
 private:
  // Map that stores the registered creation functions
  std::map<std::string, std::function<T*(void)>> str2func_;
  std::map<int, std::function<T*(void)>> id2func_;
};

#endif  // SINGA_UTILS_FACTORY_H_
