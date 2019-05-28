//
// Created by czh on 5/22/19.
//

#ifndef FASTDESP_COLORFEATURE_H
#define FASTDESP_COLORFEATURE_H

#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include "Type.h"


using namespace std;
using namespace pcl;
using namespace Type;

class ColorFeature {
public:
    PointCloudT::Ptr meanOutput;
    PointCloudT::Ptr stdvOutput;
    PointCloudT::Ptr skewOutput;
    ColorFeature(PointCloudT::Ptr seeds, PointCloudT::Ptr source, float radius): _radius(radius), _source(source), _seeds(seeds) {
        _kdtree.setInputCloud (source);

        PointCloudT::Ptr tmpA(new PointCloudT); meanOutput = tmpA;
        PointCloudT::Ptr tmpB(new PointCloudT); stdvOutput = tmpB;
        PointCloudT::Ptr tmpC(new PointCloudT); skewOutput = tmpC;
        compute();
    }

    void compute();

private:
    float _radius;
    PointCloudT::Ptr _seeds;
    PointCloudT::Ptr _source;
    KdTreeFLANN<PointT> _kdtree;

};

void ColorFeature::compute() {
    // filter the seeds
    for (int i = 0; i < _seeds->size(); ++i) {
        //cout << i << endl;
        vector<int> idxs;
        vector<float> dists;
        if ( _kdtree.radiusSearch (_seeds->points[i], _radius, idxs, dists) ) {
            int32_t mean_r = 0, mean_g = 0, mean_b = 0;
            int n = idxs.size();
            for (auto& idx : idxs) {
                mean_r += _source->points[idx].r;
                mean_g += _source->points[idx].g;
                mean_b += _source->points[idx].b;
            }
            mean_r = mean_r / n;
            mean_g = mean_g / n;
            mean_b = mean_b / n;

            int32_t stdv_r = 0, stdv_g = 0, stdv_b = 0;
            for (auto& idx : idxs) {
                stdv_r += pow( _source->points[idx].r - mean_r, 2);
                stdv_g += pow( _source->points[idx].g - mean_g, 2);
                stdv_b += pow( _source->points[idx].b - mean_b, 2);
            }
            stdv_r = sqrt( stdv_r/n);
            stdv_g = sqrt( stdv_g/n);
            stdv_b = sqrt( stdv_b/n);

            int32_t skew_r = 0, skew_g = 0, skew_b = 0;
            for (auto& idx : idxs) {
                skew_r += pow( _source->points[idx].r - mean_r, 3);
                skew_g += pow( _source->points[idx].g - mean_g, 3);
                skew_b += pow( _source->points[idx].b - mean_b, 3);
            }
            skew_r = cbrt(skew_r/n);
            skew_g = cbrt(skew_g/n);
            skew_b = cbrt(skew_b/n);

            PointT p_mean = _seeds->points[i]; p_mean.r = mean_r; p_mean.g = mean_g; p_mean.b = mean_b; p_mean.a = 255;
            PointT p_stdv = _seeds->points[i]; p_stdv.r = stdv_r; p_stdv.g = stdv_g; p_stdv.b = stdv_b; p_stdv.a = 255;
            PointT p_skew = _seeds->points[i]; p_skew.r = skew_r; p_skew.g = skew_g; p_skew.b = skew_b; p_skew.a = 255;
            meanOutput->push_back(p_mean);
            stdvOutput->push_back(p_stdv);
            skewOutput->push_back(p_skew);
        }
    }
}


#endif //FASTDESP_COLORFEATURE_H
