#ifndef INCLUDE_UTILS_SINGLETON_H_
#define INCLUDE_UTILS_SINGLETON_H_

template<typename T>
class Singleton {
 public:
  static T* Instance() {
    if (data_==nullptr) {
      data_ = new T();
    }
    return data_;
  }
 private:
  static T* data_;
};

template<typename T> T* Singleton<T>::data_ = nullptr;


/**
 * Singleton initiated with argument
 */
template<typename T, typename X=int>
class ASingleton {
 public:
  static T* Instance(){
    return data_;
  }
  static T* Instance(X x) {
    if (data_==nullptr) {
      data_ = new T(x);
    }
    return data_;
  }
 private:
  static T* data_;
};

template<typename T, typename X> T* ASingleton<T,X>::data_ = nullptr;

#endif // INCLUDE_UTILS_SINGLETON_H_
