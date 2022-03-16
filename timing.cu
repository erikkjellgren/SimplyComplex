#include "SimplyComplex.h"
#include <iostream>
#include <chrono>

int main(){
    complex_double a = complex_double(1.1, 2.2);
    complex_double b = complex_double(1.1, 2.2);
    complex_double c;
    std::chrono::steady_clock::time_point begin, end;
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = a + b;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Timing + ; " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = a - b;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Timing - ; " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = a * b;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Timing * ; " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = a / b;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Timing / ; " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}
