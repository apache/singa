#ifndef SINGA_UTILS_TIMER_H
#define SINGA_UTILS_TIMER_H

#include <chrono>

namespace singa {

/// For benchmarking the time cost of operations.
class Timer {
 public:
  typedef std::chrono::duration<int> Seconds;
  typedef std::chrono::duration<int, std::milli> Milliseconds;
  typedef std::chrono::duration<int, std::ratio<60 * 60>> Hours;

  /// Init the internal time point to the current time
  Timer() { Tick(); }
  /// Reset the internal time point to the current time
  void Tick() { last_ = std::chrono::high_resolution_clock::now(); }
  /// Return the duration since last call to Tick() or since the creation of
  /// Timer. The template arg must be from Second or Millisecond or Hour.
  /// The returned value is the count of the time metric.
  template <typename T = Milliseconds>
  int Elapsed() const {
    static_assert(std::is_same<T, Seconds>::value ||
                      std::is_same<T, Milliseconds>::value ||
                      std::is_same<T, Hours>::value,
                  "Template arg must be Seconds | Milliseconds | Hours");
    auto now  = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<T>(now - last_).count();
  }
  /// Return the string rep of current wall time
  // std::string CurrentTime();

 private:
  std::chrono::high_resolution_clock::time_point last_;
};
}
#endif
