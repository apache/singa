/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TIMER_H
#define TIMER_H

#ifdef __cplusplus
/*********************************** C++ ***********************************/
#include <chrono>
#include <string>
#include <unordered_map>


struct Time {
    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;
    std::chrono::duration<double> duration;
};

class Timer {
public:
    void start(const std::string &label);

    double stop(const std::string &label);

    const char *report();

    void reset();

private:
    std::unordered_map<std::string, Time> times_;
    std::string report_;
};

extern "C" {
#endif
/************************************ C ************************************/

typedef struct Timer TimerC;

TimerC *ns_create_timer();

TimerC *ns_get_timer();

void ns_timer_start(TimerC *timer, const char *label);

double ns_timer_stop(TimerC *timer, const char *label);

void ns_timer_destroy(TimerC *timer);

const char *ns_timer_report(TimerC *timer);

void ns_timer_reset(TimerC *timer);

#ifdef __cplusplus
}
#endif
#endif //TIMER_H
