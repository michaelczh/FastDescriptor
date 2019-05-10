//
// Created by czh on 5/10/19.
//
#include <iostream>
#include <vector>
#include <pcl/io/ply_io.h>
#include "main.h"
#include "SimpleView.h"

using namespace std;

std::vector<std::string> split(const std::string& s, char delimiter);
void rotatePointCloud(PointCloudT::Ptr input, Eigen::Matrix4d& T);
int main(){

    string filePath = "/home/czh/Desktop/fastDesp-corrProp/result.txt";
    std::ifstream file(filePath);
    std::string str;
    vector<Eigen::Matrix4d> Ts;
    while (std::getline(file, str))
    {
        vector<string> tokens;
        tokens = split(str, ' ');
        cout << tokens.size() << endl;
        Eigen::Matrix4d T;
        T << stof(tokens[0]), stof(tokens[1]), stof(tokens[2]), stof(tokens[3]),
             stof(tokens[4]), stof(tokens[5]), stof(tokens[6]), stof(tokens[7]),
             stof(tokens[8]), stof(tokens[9]), stof(tokens[10]), stof(tokens[11]),
             stof(tokens[12]), stof(tokens[13]), stof(tokens[14]), stof(tokens[15]);
        cout << T <<"\n";
        Ts.push_back(T);
    }

    SimpleView resultView("result");
    int numPointCloud = Ts.size();
    for (int i = 0; i < numPointCloud; ++i) {
        string ptsFileName = "../data/" + to_string(i+1) + "/1_rgb.ply";
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (int j = i; j < numPointCloud; ++j) T = Ts[j] * T;
        PointCloudT::Ptr cloud(new PointCloudT);
        pcl::io::loadPLYFile<PointT>(ptsFileName, *cloud);
        rotatePointCloud(cloud, T);
        resultView.addPointCloud(cloud);
    }

    return 0;
}


void rotatePointCloud(PointCloudT::Ptr input, Eigen::Matrix4d& T){
    for (auto& p : input->points) {
        Eigen::Vector4d p_v(p.x, p.y, p.z, 1);
        Eigen::Vector4d v = T * p_v;
        p.x = v(0); p.y = v(1); p.z = v(2);
    }
}
