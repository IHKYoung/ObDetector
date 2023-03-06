//
// Created by young on 23-2-24.
//

#ifndef OBJECT_DETECTION_TICTOC_H
#define OBJECT_DETECTION_TICTOC_H

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc {
public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::system_clock::now();
    }

    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

#endif //OBJECT_DETECTION_TICTOC_H
