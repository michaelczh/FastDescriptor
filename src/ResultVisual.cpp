//
// Created by czh on 5/6/19.
//

#include <iostream>
#include "SimpleView.h"
#include "pcl/io/ply_io.h"
#include <pcl/common/common.h>
SimpleView viewer("result");
void viewMatch(PointCloudT::Ptr s, PointCloudT::Ptr t);
int main(){

    PointCloudT::Ptr source(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT);
    loadPointCloudData("../data/1/1_rgb.ply", source);
    loadPointCloudData("../data/2/1_rgb.ply", target);
    viewMatch(source, target);
    return 0;
}


void viewMatch(PointCloudT::Ptr s, PointCloudT::Ptr t) {
    PointCloudT::Ptr source(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT);
    pcl::copyPointCloud(*s,*source);
    pcl::copyPointCloud(*t,*target);
    PointT min_s, max_s, min_t, max_t;
    pcl::getMinMax3D(*source, min_s, max_s);
    pcl::getMinMax3D(*target, min_t, max_t);
    Eigen::Vector3f t(max_s.x-min_s.x+1,0,0);
    for (auto& p : source->points) p.x -= t(0);

    string matchPath = "./matchPairs.txt";
    std::ifstream file(matchPath);
    std::string str;
    pair<PointCloudT, PointCloudT> trueMatch;
    pair<PointCloudT, PointCloudT> falseMatch;
    while (std::getline(file, str))
    {
        vector<string> tokens = split(str,' ');
        PointT p_s, p_t;
        p_s.x = stof(tokens[0]) - t(0); p_s.y = stof(tokens[1]) - t(1); p_s.z = stof(tokens[2]) - t(2);
        p_t.x = stof(tokens[3]); p_t.y = stof(tokens[4]); p_t.z = stof(tokens[5]);
        if (tokens[6] == "1") {
            trueMatch.first.push_back(p_s);
            trueMatch.second.push_back(p_t);
        }else{
            falseMatch.first.push_back(p_s);
            falseMatch.second.push_back(p_t);
        }
        // Process str
    }

    viewer.addPointCloud(source);
    viewer.addPointCloud(target);
    viewer.addMatching(trueMatch.first.makeShared(), trueMatch.second.makeShared(), RED);
    viewer.spin();
}

void loadPointCloudData(string filePath, PointCloudT::Ptr output){
    assert(output->size() == 0);
    stringstream ss;
    ss << "Input File: " << filePath << "\n";

    string fileType = filePath.substr(filePath.length() - 3);
    if (!(fileType == "ply" || fileType == "obj" || fileType == "txt" || fileType == "pcd"))
        throw invalid_argument("the file type is not allowed");
    if (fileType == "ply") {
        if (pcl::io::loadPLYFile <PointT>(filePath, *output) == -1) { // the file doesnt exist
            throw invalid_argument("Cannot load the input file, please check and try again!\n");
        }
    }


}