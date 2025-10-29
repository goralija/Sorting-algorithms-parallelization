#pragma once
#include <chrono>

struct Timer {
    std::chrono::high_resolution_clock::time_point start, end;
    void begin() { start = std::chrono::high_resolution_clock::now(); }
    void stop() { end = std::chrono::high_resolution_clock::now(); }
    double ms() { 
        return std::chrono::duration<double, std::milli>(end - start).count(); 
    }
};
