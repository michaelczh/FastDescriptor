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

using namespace Eigen;
using namespace std;

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef std::vector<Eigen::Vector3d>                PointsType;

TransformType computeRigidTransform(const PointsType& src, const PointsType& dst)
{
    assert(src.size() == dst.size());
    int pairSize = src.size();
    Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);
    for (int i=0; i<pairSize; ++i)
    {
        center_src += src[i];
        center_dst += dst[i];
    }
    center_src /= (double)pairSize;
    center_dst /= (double)pairSize;

    Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
    for (int i=0; i<pairSize; ++i)
    {
        for (int j=0; j<3; ++j)
            S(i, j) = src[i][j] - center_src[j];
        for (int j=0; j<3; ++j)
            D(i, j) = dst[i][j] - center_dst[j];
    }
    Eigen::MatrixXd Dt = D.transpose();
    Eigen::Matrix3d H = Dt*S;
    Eigen::Matrix3d W, U, V;

    JacobiSVD<Eigen::MatrixXd> svd;
    Eigen::MatrixXd H_(3, 3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) H_(i, j) = H(i, j);
    svd.compute(H_, Eigen::ComputeThinU | Eigen::ComputeThinV );
    if (!svd.computeU() || !svd.computeV()) {
        std::cerr << "decomposition error" << endl;
        return std::make_pair(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    }
   // std::cout << svd.singularValues().transpose() << std::endl;
    Eigen::Matrix3d Vt = svd.matrixV().transpose();
    Eigen::Matrix3d R = svd.matrixU()*Vt;
    Eigen::Vector3d t = center_dst - R*center_src;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T << R(0,0), R(0,1), R(0,2), t(0) ,
         R(1,0), R(1,1), R(1,2), t(1) ,
         R(2,0), R(2,1), R(2,2), t(2);
    cout << T << endl;
    return std::make_pair(R, t);
}

int main() {
    const int POINT_SIZE = 100;
    srand(time(NULL));
    while (1) {
        PointsType p1s, p2s;
        p1s.resize(POINT_SIZE);
        for (int i=0; i<POINT_SIZE; ++i) {
            p1s[i][0] = rand()%256*1.0 / 512.0;
            p1s[i][1] = rand()%256*1.0 / 512.0;
            p1s[i][2] = rand()%256*1.0 / 512.0;
        }
        TransformType RT;
        RT.first =  AngleAxisd(rand()%180*1.0, Vector3d::UnitZ())
                    * AngleAxisd(rand()%180*1.0, Vector3d::UnitY())
                    * AngleAxisd(rand()%180*1.0, Vector3d::UnitZ());
        RT.second = Eigen::Vector3d(3, 4, 1);
        std::cout << RT.first << std::endl;
        std::cout << (RT.second)[0] << "  " << (RT.second)[1] << "  " << (RT.second)[2] << endl;
        for (int i=0; i<POINT_SIZE; ++i) {
            p2s.push_back(RT.first*p1s[i] + RT.second);
        }

        cout << "computing the rigid transformations...\n";
        RT = computeRigidTransform(p1s, p2s);
        Eigen::Matrix4d res = Estimator::computeRigidTransform(p1s,p2s);
        cout << res << endl;
//
//        std::cout << RT.first << endl;
//        std::cout << (RT.second)[0] << "  " << (RT.second)[1] << "  " << (RT.second)[2] << endl;
//        cout << endl;
        getchar();
    }



    return 0;
}