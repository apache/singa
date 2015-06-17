#ifndef INCLUDE_UTILS_SINGLETON_H_
#define INCLUDE_UTILS_SINGLETON_H_
/**
  * thread-safe implementation for C++11 according to
  * http://stackoverflow.com/questions/2576022/efficient-thread-safe-singleton-in-c
  */
template<typename T>
class Singleton {
 public:

  static T* Instance() {
    static T* data_=new T();
    return data_;
  }
};

/**
 * Thread Specific Singleton
 *
 * Each thread will have its own data_ storage.
 */
template<typename T>
class TSingleton {
 public:
  static T* Instance(){
    static thread_local T* data_=new T();
    return data_;
  }
};

#endif // INCLUDE_UTILS_SINGLETON_H_
