#include "gtest/gtest.h"
#include "singa/utils/timer.h"

#include <chrono>
#include <thread>

TEST(TimerTest, TestTick) {
  singa::Timer t;
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  int time = t.Elapsed<singa::Timer::Milliseconds>();
  EXPECT_GE(time, 1000);
}
