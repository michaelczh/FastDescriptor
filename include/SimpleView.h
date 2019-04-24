//
// Created by czh on 4/9/19.
//

#ifndef FASTDESP_SIMPLEVIEW_H
#define FASTDESP_SIMPLEVIEW_H
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include "main.h"

using namespace std;
enum Color{
    RED, YELLOW, BLUE, GREEN, CYAN
};

uint32_t getColorValue(Color color){
    int32_t colorValue = 0 << 24; // control transparent
    switch (color) {
        case RED:
            colorValue = 255 <<24 | 255 << 16 | 0   << 8 | 0;   break;
        case YELLOW:
            colorValue = 255 <<24 | 255 << 16 | 255 << 8 | 0;   break;
        case BLUE:
            colorValue = 255 <<24 | 0   << 16 | 0   << 8 | 255; break;
        case GREEN:
            colorValue = 255 <<24 | 0   << 16 | 255 << 8 | 0;   break;
        case CYAN:
            colorValue = 255 <<24 | 0   << 16 | 255 << 8 | 255; break;
    }
    return colorValue;
};

void simpleView(const string &title, const PointCloudRGB::Ptr &cloud ) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "1", 0);
    //viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

template<typename T>
void simpleView(const string &title, const T &cloud, Color color){
    PointCloudRGB::Ptr tmp(new PointCloudRGB);
    uint32_t colorInt = getColorValue(color);
    for(auto& p : cloud->points) {
        PointRGB tmpP;
        tmpP.x = p.x;
        tmpP.y = p.y;
        tmpP.z = p.z;
        tmpP.rgba = colorInt;
        tmp->push_back(tmpP);
    }
    simpleView(title, tmp);
}


class SimpleView{
private:
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    pcl::visualization::PCLVisualizer *viewer;
    int lineIdx = 0;
    int ptsIdx = 0;

public:
    SimpleView(string title) {
        viewer = new pcl::visualization::PCLVisualizer (title);
        //viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer(title));
        viewer->setBackgroundColor(0, 0, 0);
       // viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "1", 0);
        //viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        //viewer->spin();
//        while (!viewer->wasStopped())
//        {
//            viewer->spinOnce(100);
//            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//        }
    }
    template<typename T>
    void addPointCloud(const T &cloud, Color color, int size = 1 ) {
        PointCloudRGB::Ptr tmp(new PointCloudRGB);
        uint32_t colorInt = getColorValue(color);
        for(auto& p : cloud->points) {
            PointRGB tmpP;
            tmpP.x = p.x;
            tmpP.y = p.y;
            tmpP.z = p.z;
            tmpP.rgba = colorInt;
            tmp->push_back(tmpP);
        }
        viewer->addPointCloud(tmp, to_string(lineIdx));
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, to_string(lineIdx++));
        viewer->spin();
    }

    template<typename T>
    void addPointCloud(const vector<T> pts, Color color, int size = 1 ) {
        PointCloudRGB::Ptr tmp(new PointCloudRGB);
        uint32_t colorInt = getColorValue(color);
        for(auto& p : pts) {
            PointRGB tmpP;
            tmpP.x = p(0);
            tmpP.y = p(1);
            tmpP.z = p(2);
            tmpP.rgba = colorInt;
            tmp->push_back(tmpP);
        }
        viewer->addPointCloud(tmp, to_string(lineIdx));
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, to_string(lineIdx++));
        viewer->spin();
    }

    void addMatching(Desp& d1, Desp& d2) {
        viewer->addLine(d1.seed,d2.seed, 255,0,0, to_string(lineIdx++));
        //viewer->spin();
    }

    void spin() {viewer->spin();}
};



#endif //FASTDESP_SIMPLEVIEW_H
