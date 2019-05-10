#include <gtest/gtest.h>
#include <pcl/pcl_tests.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/iss_3d.h>
#include "SimpleView.h"

using namespace pcl;
using namespace pcl::io;

//
// Main variables
//

SimpleView viewer("view");


int main(){
        PointCloud<PointXYZ>::Ptr keypoints(new PointCloud<PointXYZ>());
        PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ> ());
        pcl::io::loadPLYFile <PointXYZ>("../data/1/1.ply", *cloud);
        search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ> ());
        //
        // Compute the ISS 3D keypoints - Without Boundary Estimation
        //
        ISSKeypoint3D<PointXYZ, PointXYZ> iss_detector;
        iss_detector.setSearchMethod (tree);
        iss_detector.setSalientRadius (10*0.01);
        iss_detector.setNonMaxRadius (2*0.01);

        iss_detector.setThreshold21 (0.975);
        iss_detector.setThreshold32 (0.975);
        iss_detector.setMinNeighbors (5);
        iss_detector.setNumberOfThreads (1);
        iss_detector.setInputCloud (cloud);
        iss_detector.compute (*keypoints);

        cout << "num of pts " << keypoints->size() << endl;
        viewer.addPointCloud(cloud, RED, 1);
        viewer.addPointCloud(keypoints, YELLOW, 5);
        viewer.spin();
        return 0;
};