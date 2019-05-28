//
// Created by czh on 5/28/19.
//

#ifndef FASTDESP_FLANNSEARCH_H
#define FASTDESP_FLANNSEARCH_H

#include <flann/flann.hpp>
#include <unordered_map>
#include <pcl/search/kdtree.h>
#include <vector>
#include "Type.h"
using namespace Type;
class FlannSearch {

public:
    static void flannSearch(const vector<float>& src, const vector<float>& tar, float radius, vector<vector<int>>& map){
        assert(tar.size() == map.size());
        PointCloudT::Ptr srcCloud(new PointCloudT);
        PointCloudT::Ptr tarCloud(new PointCloudT);
        for (auto d: src) {
            PointT p;
            p.x = d; p.y = 0; p.z = 0;
            srcCloud->push_back(p);
        }
        for (auto d: tar) {
            PointT p;
            p.x = d; p.y = 0; p.z = 0;
            tarCloud->push_back(p);
        }

        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud (srcCloud);
        vector<int> validIndexes;

        // filter the seeds
        for (int i = 0; i < tarCloud->size(); ++i) {
            PointT searchPoint = tarCloud->points[i];
            vector<int> pointIdxRadiusSearch;
            vector<float> pointRadiusSquaredDistance;
            if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) ) {
                map[i] = pointIdxRadiusSearch;
            }
        }
    }

    static void flannSearch(const vector<Eigen::Vector3d>& srcDesps, const vector<Eigen::Vector3d>& tarDesps, unordered_map<int,pair<int,float>>& map){
        int nn = 3;
        int dataSz = srcDesps.size();
        int querySz = tarDesps.size();
        flann::Matrix<float> dataset (new float[nn*dataSz], dataSz, nn);
        flann::Matrix<float> query(new float[nn*querySz], querySz, nn);

        for ( int i = 0; i < dataSz; ++i) {;
            for (int j = 0; j < nn; ++j ) {
                dataset[i][j] = srcDesps[i](j);
            }

        }
        for ( int i = 0; i < querySz; ++i) {;
            for (int j = 0; j < nn; ++j ) {
                query[i][j] = tarDesps[i](j);
            }

        }

        flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
        flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

        // construct an randomized kd-tree index using 4 kd-trees
        flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(8));
        index.buildIndex();

        // do a knn search, using 128 checks
        index.knnSearch(query, indices, dists, nn, flann::SearchParams(64));

        for(int i=0;i<indices.rows;i++){
            map[i] = make_pair(indices[i][0], dists[i][0]);
        }

        delete[] dataset.ptr();
        delete[] query.ptr();
        delete[] indices.ptr();
        delete[] dists.ptr();
    }

    static void flannSearch(const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, unordered_map<int,pair<int,float>>& map){
        int layer = srcDesps[0].S.size();
        int nn = layer*3;
        int dataSz = srcDesps.size();
        int querySz = tarDesps.size();
        flann::Matrix<float> dataset (new float[nn*dataSz], dataSz, nn);
        flann::Matrix<float> query(new float[nn*querySz], querySz, nn);

        for ( int i = 0; i < dataSz; ++i) {
            vector<Vector3f> S = srcDesps[i].S;
            for (int j = 0; j < nn; ++j ) {
                dataset[i][j] = S[j/3](j%3);
            }

        }
        for ( int i = 0; i < querySz; ++i) {
            vector<Vector3f> S = tarDesps[i].S;
            for (int j = 0; j < nn; ++j ) {
                query[i][j] = S[j/3](j%3);
            }
        }

        flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
        flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

        // construct an randomized kd-tree index using 4 kd-trees
        flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(8));
        index.buildIndex();

        // do a knn search, using 128 checks
        index.knnSearch(query, indices, dists, nn, flann::SearchParams(64));

        for(int i=0;i<indices.rows;i++){
            map[i] = make_pair(indices[i][0], dists[i][0]);
        }

        delete[] dataset.ptr();
        delete[] query.ptr();
        delete[] indices.ptr();
        delete[] dists.ptr();
    }

};


#endif //FASTDESP_FLANNSEARCH_H
