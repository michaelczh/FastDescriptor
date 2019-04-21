//
// Created by czh on 4/9/19.
//

#ifndef FASTDESP_MAIN_H
#define FASTDESP_MAIN_H

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/uniform_sampling.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <queue>
using namespace std;
using namespace Eigen;
typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;

YAML::Node config ;
template<typename T>
T Config(string a)
{
    return config[a].as<T>();
}

template<typename T>
T Config(string a, string b)
{
    YAML::Node aa = config[a];
    return aa[b].as<T>();
}

struct Desp{
    PointT seed;
    vector<Vector3f> N;
    vector<Vector3f> S;
    bool isMatching(Desp& t){
        assert(N.size() == t.N.size());
        assert(S.size() == t.S.size());
        int normalSimilarNum = 0;


    }
};

struct Match{
    Desp* src;
    Desp* tar;
};

std::vector<std::string> split(const std::string& s, char delimiter);
void loadPointCloudData(string filePath, PointCloudT::Ptr output);
void uniformDownSample(PointCloudT::Ptr input, float Rho, PointCloudT::Ptr output);
void computeDescriptor(PointCloudT::Ptr seed, PointCloudT::Ptr source,
                       float radiusMin, float radiusMax, float radiusStep, vector<Desp>& desps);
void svdCov(PointCloudT::Ptr input, PointT seed, vector<int> &othersIdx, Vector3f& s, Vector3f& n);

void trimmedICP(PointCloudT::Ptr tarEst, PointCloudT::Ptr tarData, float overlapRatio);
void estimateRigidTransform(const vector<Match>& matches, const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, Matrix4d & T, float &err);

Eigen::Matrix4d matching(vector<Desp>& srcDesps, vector<Desp>& tarDesps);
double diffOfS(vector<Vector3f> srcS, vector<Vector3f> tarS);

float computeDespDist(Desp& src, Desp& tar);
void computeNormalDiff(Desp& seed, vector<Desp>& allDesps, vector<vector<float>>& res);
void aggMatching(Desp& src, vector<Desp>& srcSeeds, Desp& tar, vector<Desp>& tarSeeds, vector<Match>& matches);
#endif //FASTDESP_MAIN_H
