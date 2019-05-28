//
// Created by czh on 5/22/19.
//
#include<iostream>
#include <pcl/io/ply_io.h>
#include "SimpleView.h"
#include "Type.h"
using namespace Type;
int main(){

    Type::PointCloudT::Ptr input(new Type::PointCloudT);
    pcl::io::loadPLYFile <PointT>( "../data/1/1_rgb.ply", *input);
    SimpleView viewer("TEST viewer",4,1);

    viewer.addPointCloud(input,RED, 1,0);
    viewer.addPointCloud(input,YELLOW, 1,1);
    viewer.addPointCloud(input,BLUE, 1,2);
    viewer.addPointCloud(input,GREEN, 1,3);
    viewer.addPointCloud(input,RED, 1,4);
    return 0;

}