#ifndef SINGA_UTILS_TIMER_H
#define SINGA_UTILS_TIMER_H

#include <chrono>

namespace singa{

    class Timer{
     public:
        double elapsed() const
        {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        }
    };
}

#endif
