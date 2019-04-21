//
// Created by czh on 4/17/19.
//

#include <iostream>
#include "gtest/gtest.h"
#include "test.h"
#include "estimator.h"
#include <ctime>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

TEST(ESTIMATOR, TMatrix) {

    typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
    typedef std::vector<Eigen::Vector3d>                PointsType;
    const int POINT_SIZE = 100;
    srand(time(NULL));
    int iterNum = 100;
    while (iterNum-- > 0) {
        PointsType p1s, p2s;
        p1s.resize(POINT_SIZE);
        for (int i=0; i<POINT_SIZE; ++i) {
            p1s[i][0] = rand()%256*1.0 / 512.0;
            p1s[i][1] = rand()%256*1.0 / 512.0;
            p1s[i][2] = rand()%256*1.0 / 512.0;
        }
        TransformType RT;
        RT.first =    AngleAxisd(rand()%180*1.0, Vector3d::UnitZ())
                    * AngleAxisd(rand()%180*1.0, Vector3d::UnitY())
                    * AngleAxisd(rand()%180*1.0, Vector3d::UnitZ());
        RT.second = Eigen::Vector3d(0, 0, 0);
        for (int i=0; i<POINT_SIZE; ++i) {
            p2s.push_back(RT.first*p1s[i] + RT.second);
        }

       // cout << "computing the rigid transformations...\n";
        Eigen::Matrix4d res = Estimator::computeRigidTransform(p1s,p2s);
        Eigen::Matrix3d T_hat;
        T_hat << res(0,0), res(0,1), res(0,2),
                 res(1,0), res(1,1), res(1,2),
                 res(2,0), res(2,1), res(2,2);
        Eigen::Vector3d trans_hat(res(0,3), res(1,3), res(2,3));
        EXPECT_LE((RT.first - T_hat).sum(), 0.0001);
        EXPECT_LE((RT.second-trans_hat).sum(), 0.0001);

    }
}

TEST(ESTIMATOR, RANSAC) {

    typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
    typedef std::vector<Eigen::Vector3d>                PointsType;
    const int POINT_SIZE = 100;
    const int outlier_SIZE = POINT_SIZE*0.05;
    srand(time(NULL));
    int iterNum = 1000;
    while (iterNum-- > 0) {
        PointsType p1s, p2s;
        p1s.resize(POINT_SIZE);
        for (int i=0; i<POINT_SIZE; ++i) {
            p1s[i][0] = rand()%256*1.0 / 512.0;
            p1s[i][1] = rand()%256*1.0 / 512.0;
            p1s[i][2] = rand()%256*1.0 / 512.0;
        }
        TransformType RT;
        RT.first =    AngleAxisd(rand()%180*1.0, Vector3d::UnitZ())
                      * AngleAxisd(rand()%180*1.0, Vector3d::UnitY())
                      * AngleAxisd(rand()%180*1.0, Vector3d::UnitZ());
        RT.second = Eigen::Vector3d(rand() % 5 +1, rand() % 5, rand() % 5);
        for (int i=0; i<POINT_SIZE; ++i) {
            p2s.push_back(RT.first*p1s[i] + RT.second);
        }

        // add outliers
        for (int i = 0; i < outlier_SIZE; ++i) {
            Eigen::Vector3d pt;
            pt[0] = rand()%256*1.0 / 512.0;
            pt[1] = rand()%256*1.0 / 512.0;
            pt[2] = rand()%256*1.0 / 512.0;
            p1s.push_back(pt);
            pt[0] = rand()%256*1.0 / 512.0;
            pt[1] = rand()%256*1.0 / 512.0;
            pt[2] = rand()%256*1.0 / 512.0;
            p2s.push_back(pt);
        }

        // cout << "computing the rigid transformations...\n";
        pair<Eigen::Matrix4d,int> res_est = Estimator::RANSAC(p1s,p2s, 0.1);
//        cout << "original T\n" << RT.first << "\n" << RT.second.transpose() << endl;
//        cout << "estimated \n" << res_est.first << endl;
        Eigen::Matrix4d res = res_est.first;
        Eigen::Matrix3d T_hat;
        T_hat << res(0,0), res(0,1), res(0,2),
                res(1,0), res(1,1), res(1,2),
                res(2,0), res(2,1), res(2,2);
        Eigen::Vector3d trans_hat(res(0,3), res(1,3), res(2,3));
        EXPECT_LE((RT.first - T_hat).sum(), 0.01)  << RT.first << "\n" << T_hat << endl;
        EXPECT_LE((RT.second-trans_hat).sum(), 0.01) << RT.second << "\n" << trans_hat << endl;

    }
}

int main(int argc, char* argv[]){
   // testing::FLAGS_gtest_filter = "TMatrix*";
    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    return 0;
}