//
// Created by czh on 5/22/19.
//

#ifndef FASTDESP_TYPE_H
#define FASTDESP_TYPE_H

#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
using namespace std;

namespace Type {
    typedef pcl::PointXYZRGBNormal PointT;
    typedef pcl::PointXYZRGB PointRGB;
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef pcl::PointCloud<PointRGB> PointCloudRGB;
}

struct ColorChannel{
    int r;
    int g;
    int b;
    string to_str() {
        return to_string(r) + " " + to_string(g) + " " + to_string(b);
    }
    int operator-(const ColorChannel& c2){
        return abs(r - c2.r) + abs(g - c2.g) + abs(b - c2.b);
    }
};

struct ColorInfo{
    ColorChannel mean;
    ColorChannel stdv;
    ColorChannel skew;
    int operator-(const ColorInfo& c2){
        return abs(mean - c2.mean) + abs(stdv - c2.stdv) + abs(skew - c2.skew);
    }
};

struct Desp{
    Type::PointT seed;
    std::vector<Eigen::Vector3f> N;
    std::vector<Eigen::Vector3f> S;
    ColorInfo CI;
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

#endif //FASTDESP_TYPE_H
