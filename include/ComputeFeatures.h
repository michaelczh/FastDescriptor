//
// Created by czh on 5/20/19.
//

#ifndef FASTDESP_COMPUTEFEATURES_H
#define FASTDESP_COMPUTEFEATURES_H
#include <iostream>
#include <vector>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/impl/pcl_base.hpp>
#include <main.h>

using namespace std;
class ComputeFeatures {
public:
    ComputeFeatures();
    static void UniformDownSample(PointCloudT::Ptr input, PointCloudT::Ptr output, float Rho);
    static void Harris3D(PointCloudT::Ptr input, PointCloudT::Ptr output, float radius);
    static void ISS(PointCloudT::Ptr input, PointCloudT::Ptr output, float radius, float th32, float th21);

};

void ComputeFeatures::UniformDownSample(PointCloudT::Ptr input, PointCloudT::Ptr output, float Rho){
    int numOri = input->size();
    assert(numOri > 0);
    pcl::UniformSampling<PointT> filter;
    filter.setInputCloud(input);
    filter.setRadiusSearch(Rho);
    filter.filter(*output);
    std::cout << "[uniformDownSample] from " << numOri << " to " << output->size() << std::endl;
}

 void ComputeFeatures::Harris3D(PointCloudT::Ptr input, PointCloudT::Ptr output, float radius){
    pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI> detector;
    detector.setNonMaxSupression (true);
    detector.setRadius (radius);
    detector.setThreshold (1e-6);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto& p : input->points) pc->push_back( pcl::PointXYZ(p.x,p.y,p.z) );
    detector.setInputCloud(pc);
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
    detector.compute(*keypoints);
    std::cout << "[Harris3D] Keypoints detected: " << keypoints->size() << std::endl;
    pcl::PointIndicesConstPtr keypoints_indices = detector.getKeypointsIndices ();
    for (auto& idx : keypoints_indices->indices) {
        PointT p;
        p.x = pc->points[idx].x;
        p.y = pc->points[idx].y;
        p.z = pc->points[idx].z;
        output->push_back(p);
    }
}

void ComputeFeatures::ISS(PointCloudT::Ptr input, PointCloudT::Ptr output, float radius, float th32, float th21){
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud (input);
    vector<float> weights(input->size(),0);

    // compute weights
    for (int i = 0; i < input->size(); ++i) {
        PointT searchPoint = input->points[i];
        vector<int> pointIdxRadiusSearch;
        vector<float> pointRadiusSquaredDistance;
        if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 1 ) {
            weights[i] = (float)1 / (pointIdxRadiusSearch.size()-1);
        }
    }

    for (int i = 0; i < input->size(); ++i) {
        PointT searchPoint = input->points[i];
        vector<int> pointIdxRadiusSearch;
        vector<float> pointRadiusSquaredDistance;
        if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 1 )
        {
            Eigen::Vector3f p_i(searchPoint.x, searchPoint.y, searchPoint.z);
            Eigen::Matrix3f P = Eigen::Matrix3f::Zero();
            float sumWeight = 0;
            for (int& idx : pointIdxRadiusSearch) {
                if (idx == i) continue;
                Eigen::Vector3f p_j(input->points[idx].x, input->points[idx].y, input->points[idx].z);
                Eigen::Vector3f minsV = p_j-p_i;
                P = P + weights[idx] * minsV * minsV.transpose();
                sumWeight += weights[idx];
            }

            P = P / sumWeight;
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto S = svd.singularValues();
            if (S(0) < S(1) && S(0) < S(2) && S(1) < S(2)) cout << "[ISS] error" << endl;
            if (S(1)/S(0) <= th21 && S(2)/S(1) <= th32) output->push_back(input->points[i]);
        }
    }

    cout << "[ISS] num of key points " << output->size() << endl;
}

#endif //FASTDESP_COMPUTEFEATURES_H
