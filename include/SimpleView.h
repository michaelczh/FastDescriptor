//
// Created by czh on 4/9/19.
//

#ifndef FASTDESP_SIMPLEVIEW_H
#define FASTDESP_SIMPLEVIEW_H
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include "Type.h"
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;
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

//void simpleView(const string &title, const PointCloudRGB::Ptr &cloud ) {
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(title));
//    viewer->setBackgroundColor(0, 0, 0);
//    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "1", 0);
//    //viewer->addCoordinateSystem(1.0);
//    viewer->initCameraParameters();
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
//}

//template<typename T>
//void simpleView(const string &title, const T &cloud, Color color){
//    PointCloudRGB::Ptr tmp(new PointCloudRGB);
//    uint32_t colorInt = getColorValue(color);
//    for(auto& p : cloud->points) {
//        PointRGB tmpP;
//        tmpP.x = p.x;
//        tmpP.y = p.y;
//        tmpP.z = p.z;
//        tmpP.rgba = colorInt;
//        tmp->push_back(tmpP);
//    }
//    simpleView(title, tmp);
//}


class SimpleView{
private:
    pcl::visualization::PCLVisualizer *viewer;
    int lineIdx = 0;
    int ptsIdx = 0;
    int _c = 1;
    int _r = 1;
    vector<pair<float,float>> _portCord;

public:
    SimpleView(string title, int r = 1, int c = 1) {
        _c = c;
        _r = r;
        double width =  1/(double)c;
        double height = 1/(double)r;
        for (double i = 0; i <= 1-height; i+= height) {
            for (double j = 0; j <= 1-width; j+= width) _portCord.push_back(make_pair(j,i));
        }

        viewer = new pcl::visualization::PCLVisualizer (title);
        viewer->initCameraParameters();
        for (int i = 0; i < _c*_r; ++i) {
            int vp(i);
            viewer->createViewPort(_portCord[i].first, _portCord[i].second, _portCord[i].first+width,_portCord[i].second+height, vp);
            viewer->setBackgroundColor(0, 0, 0, i);
        }
    }

    template<typename T>
    void addPointCloud(const T &cloud, Color color, int size = 1, int viewport = 0 ) {
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
        viewer->addPointCloud(tmp, to_string(lineIdx), viewport);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, to_string(lineIdx++), viewport);
        viewer->spin();
    }

    template<typename T>
    void addPointCloud(const T &cloud, int size = 1, int viewport = 0 ) {
        PointCloudRGB::Ptr tmp(new PointCloudRGB);
        for(auto& p : cloud->points) {
            PointRGB tmpP;
            tmpP.x = p.x;
            tmpP.y = p.y;
            tmpP.z = p.z;
            tmpP.rgba = p.rgba;
            tmp->push_back(tmpP);
        }
        viewer->addPointCloud(tmp, to_string(lineIdx), viewport);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, to_string(lineIdx++), viewport);
        viewer->spin();
    }

    template<typename T>
    void addPointCloud(const vector<T> pts, Color color, int size = 1, int viewport = 0 ) {
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
        viewer->addPointCloud(tmp, to_string(lineIdx), viewport);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, to_string(lineIdx++), viewport);
        viewer->spin();
    }

    void addMatching(Desp& d1, Desp& d2) {
        viewer->addLine(d1.seed,d2.seed, 255,0,0, to_string(lineIdx++));
    }

    void addMatching(Desp& d1, Desp& d2, Color color, int viewport = 0) {
        uint32_t colorValue = getColorValue(color);
        double r =  colorValue >> 16 & 255;
        double g =  colorValue >> 8 & 255;
        double b =  colorValue & 255;
        viewer->addLine(d1.seed,d2.seed, r,g,b, to_string(lineIdx++), viewport);
    }

    void addMatching(Type::PointCloudT::Ptr d1, Type::PointCloudT::Ptr d2, Color color, int viewport = 0) {
        if (d1->size() != d2->size()) cerr << "error size different " << endl;
        int n = d1->size();
        uint32_t colorValue = getColorValue(color);
        double r =  colorValue >> 16 & 255;
        double g =  colorValue >> 8 & 255;
        double b =  colorValue & 255;
        for (int i = 0; i < n; ++i) {
            viewer->addLine(d1->points[i],d2->points[i], r,g,b, to_string(lineIdx++), viewport);
        }
    }

    void spin() {viewer->spin();}
};



#endif //FASTDESP_SIMPLEVIEW_H
