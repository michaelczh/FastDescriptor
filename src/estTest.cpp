//
// Created by czh on 4/15/19.
//

//using Eigen's SVD to fastly compute the rigid transformation between two point clouds.
#include <iostream>
#include <ctime>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include "estimator.h"
#include <flann/flann.hpp>
#include <omp.h>
#include <chrono>
#include <unordered_map>
#include <stack>
using namespace flann;

float timeElapsed(std::chrono::steady_clock::time_point start){
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    return (float)duration.count() / 1000;
}

int main(int argc, char** argv)
{
    int nn = 1000;
    unordered_map<int,int> map;
    for (int i = 0; i < nn; i++) map[i] = i;


    for (int j = 0; j < 10; ++j ) {
        int minN = INT_MAX;
        #pragma omp parallel for
        for (int i = 0; i < nn; ++i) {
            if (map[i] < minN) minN = map[i];
        }
        cout << minN << endl;
    }






}


