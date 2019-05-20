//
// Created by czh on 4/9/19.
//

#ifndef FASTDESP_MAIN_H
#define FASTDESP_MAIN_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
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
    Match(Desp* s, Desp* t) : src(s), tar(t){};
};
std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}
void loadPointCloudData(string filePath, PointCloudT::Ptr output);
void uniformDownSample(PointCloudT::Ptr input, float Rho, PointCloudT::Ptr output);
void computeDescriptor(PointCloudT::Ptr seed, PointCloudT::Ptr source,
                       float radiusMin, float radiusMax, float radiusStep, vector<Desp>& desps);
void svdCov(PointCloudT::Ptr input, PointT seed, vector<int> &othersIdx, Vector3f& s, Vector3f& n);

void estimateRigidTransform(const vector<Match>& matches, const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, Matrix4d & T, float &err);
void estimateRigidTransform(const vector<Eigen::Vector3d>& src, const vector<Eigen::Vector3d>& tar, Matrix4d & T, float &err);
Eigen::Matrix4d matching(vector<Desp>& srcDesps, vector<Desp>& tarDesps, vector<Desp>& srcFDesps, vector<Desp>& tarFDesps);
float computeDespDist(Desp& src, Desp& tar);
void computeNormalDiff(Desp& seed, vector<Desp>& allDesps, vector<vector<float>>& res);
void aggMatching(Desp& src, vector<Desp>& srcSeeds, Desp& tar, vector<Desp>& tarSeeds, vector<Match>& matches);
void flannSearch(const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, unordered_map<int,pair<int,float>>& map);
void flannSearch(const vector<Eigen::Vector3d>& srcDesps, const vector<Eigen::Vector3d>& tarDesps, unordered_map<int,pair<int,float>>& map);
void flannSearch(const vector<float>& src, const vector<float>& tar, float radius, vector<vector<int>>& map);
void trimmedICP(const vector<Eigen::Vector3d> &tarEst, const vector<Eigen::Vector3d> &tarData, float overlapRatio);


void trimmedICP(PointCloudT::Ptr tarEst, PointCloudT::Ptr tarData, float overlapRatio){
    vector<Eigen::Vector3d> _tarData, _tarEst;
    for (auto&p : tarData->points) {
        _tarData.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }

    for (auto&p : tarEst->points) {
        _tarEst.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }

    trimmedICP(_tarEst, _tarData, overlapRatio);

};
void extractFeaturePts(PointCloudT::Ptr input, PointCloudT::Ptr output);
void extractFeaturePts_Harris3D(PointCloudT::Ptr input, PointCloudT::Ptr output);

float timeElapsed(std::chrono::steady_clock::time_point start){
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    return (float)duration.count() / 1000;
}
#endif //FASTDESP_MAIN_H
