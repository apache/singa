#ifndef INCLUDE_UTILS_FACTORY_H_
#define INCLUDE_UTILS_FACTORY_H_
#include <glog/logging.h>

#include <functional>
#include <utility>
#include <map>
/**
 * macro that creats a function which instantiate a subclass instance and
 * returns pointer to the base class.
 */
#define CreateInstance(SubClass, BaseClass) \
  [](void)->BaseClass* {return new SubClass();}

/**
 * factory template to generate class (or a sub-class) object  based on id.
 * 1. register class creation function that generates a class
 * object based on id.
 * 2. call Create() func to call the creation function and return
 * a pointer to the base calss.
 */

template<typename T>
class Factory{
 //template<Factory<T>> friend class Singleton;
 public:
  /**
   * Register functions to create user defined classes.
   * This function is called by the REGISTER_FACTORY macro.
   * @param id identifier of the creating function/class
   * @param create_function a function that creates a layer instance
   */
  void Register(const std::string id, std::function<T*(void)> func);
  /**
   * create a layer  instance by providing its type
   * @param type the identifier of the layer to be created
   */
  T *Create(const std::string id);

 private:
  //<! Map that stores the registered creation functions
  std::map<std::string, std::function<T*(void)>> str2func_;
};

template<typename T>
void Factory<T>::Register(const std::string id,
                                        std::function<T*(void)> func) {
  str2func_[id] = func;
}

template<typename T>
T *Factory<T>::Create(const std::string id) {
  CHECK(str2func_.find(id) != str2func_.end())
      << "The creation function for " << id << " has not been registered";
  return str2func_[id]();
}
#endif // INCLUDE_UTILS_FACTORY_H_
