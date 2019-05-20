//
// Created by czh on 5/10/19.
//



#include <iostream>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>


int main(){
    pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI> detector;
    detector.setNonMaxSupression (true);
    detector.setRadius (0.05);
//    detector.setThreshold (1e-6);
    //detector.setRadiusSearch (100);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile<pcl::PointXYZ>("../data/1/1.ply",*pc);
    detector.setInputCloud(pc);

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
    detector.compute(*keypoints);

    std::cout << "keypoints detected: " << keypoints->size() << std::endl;


    pcl::PointCloud<pcl::PointXYZ>::Ptr keyFeatures(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndicesConstPtr keypoints_indices = detector.getKeypointsIndices ();
    for (auto& idx : keypoints_indices->indices) keyFeatures->push_back(pc->points[idx]);

    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints3D(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ tmp;
    double max = INT_MIN,min=INT_MAX;

    for(pcl::PointCloud<pcl::PointXYZI>::iterator i = keypoints->begin(); i!= keypoints->end(); i++){
        tmp = pcl::PointXYZ((*i).x,(*i).y,(*i).z);
        if ((*i).intensity>max ){
            std::cout << (*i) << " coords: " << (*i).x << ";" << (*i).y << ";" << (*i).z << std::endl;
            max = (*i).intensity;
        }
        if ((*i).intensity<min){
            min = (*i).intensity;
        }
        keypoints3D->push_back(tmp);
    }

    std::cout << "maximal responce: "<< max << " min responce:  "<< min<<std::endl;

    //show point cloud
    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pccolor(pc, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> kpcolor(keyFeatures, 255, 0, 0);
    viewer.addPointCloud(pc,pccolor,"testimg.png");
    viewer.addPointCloud(keyFeatures,kpcolor,"keypoints.png");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints.png");

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce();
        pcl_sleep (0.01);
    }
    return 0;
}