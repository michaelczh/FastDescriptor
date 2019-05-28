#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <flann/flann.hpp>
#include <queue>
#include "main.h"
#include "SimpleView.h"
#include "Estimator.h"
#include "DebugExporter.h"
#include "ComputeFeatures.h"
#include "ColorFeature.h"
#include "FlannSearch.h"
#include <omp.h>
#include <main.h>

using namespace pcl;
using namespace std;
using namespace Eigen;
SimpleView viewer("view",1,1);
//DebugFileExporter svdExport("./svdExport.txt");

int main() {

    float Time_downsample, Time_computeDesp, Time_matching;

    string configPath = "../config.yaml";
    config = YAML::LoadFile(configPath);
    string sourcePath = Config<std::string>("srcPath");
    string targetPath = Config<std::string>("tarPath");
    cout << "Input src: " << sourcePath << "\n";
    cout << "Input tar: " << targetPath << "\n";
    PointCloudT::Ptr source(new PointCloudT); PointCloudT::Ptr sourceSeed(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT); PointCloudT::Ptr targetSeed(new PointCloudT);
    loadPointCloudData(sourcePath, source);
    loadPointCloudData(targetPath, target);

    auto start = std::chrono::steady_clock::now();
    ComputeFeatures::UniformDownSample(source, sourceSeed, Config<float>("Downsample","Rho"));
    ComputeFeatures::UniformDownSample(target, targetSeed, Config<float>("Downsample","Rho"));
    Time_downsample = timeElapsed(start);
    if (Config<bool>("Visualization","showInput")) {
        viewer.addPointCloud(source, Color::GREEN);
        viewer.addPointCloud(target, Color::BLUE);
    }else{
        viewer.addPointCloud(sourceSeed, Color::GREEN,3);
        viewer.addPointCloud(targetSeed, Color::BLUE ,3);
    }


    float radiusMin  = Config<float>("Radii", "min") *Config<float>("GridStep");
    float radiusStep = Config<float>("Radii", "step")*Config<float>("GridStep");
    float radiusMax  = Config<float>("Radii", "max") *Config<float>("GridStep");
    vector<Desp> sourceDesp, targetDesp;

    start = std::chrono::steady_clock::now();
    computeDescriptor(sourceSeed, source, radiusMin, radiusMax, radiusStep, sourceDesp);
    computeDescriptor(targetSeed, target, radiusMin, radiusMax, radiusStep, targetDesp);
    Time_computeDesp = timeElapsed(start);
    //svdExport.exportToPath();

    start = std::chrono::steady_clock::now();
    Eigen::Matrix4d bestT = matching(sourceDesp,targetDesp, sourceDesp, targetDesp);
    Time_matching = timeElapsed(start);

    {
    cout << "======================================" << "\n";
    float Time_total = Time_downsample+Time_computeDesp+Time_matching;
    cout << "Total time " << Time_total << " s\n";
    cout << "[Time_downsample]  " << Time_downsample <<  " s | " << Time_downsample/Time_total*100 << "%\n";
    cout << "[Time_computeDesp]  " << Time_computeDesp <<  " s | " << Time_computeDesp/Time_total*100 << "%\n";
    cout << "[Time_matching]  " << Time_matching <<  " s | " << Time_matching/Time_total*100 << "%\n";

    DebugFileExporter resultExporter("../data/ALL_Result.txt", false);
    resultExporter.insertLine("srcPath: " + sourcePath);
    resultExporter.insertLine("tarPath: " + targetPath);
    resultExporter.insertLine("srcSeed: " + to_string(sourceDesp.size()));
    resultExporter.insertLine("tarSeed: " + to_string(targetDesp.size()));
    resultExporter.insertLine("Time: " + to_string(Time_total));
    resultExporter.insertLine("ColorThr: " + to_string(Config<int>("ColorFeature","filterThr")));
    std::stringstream bestTss; bestTss << bestT;
    resultExporter.insertLine("Best T:\n" + bestTss.str());
    resultExporter.insertLine("");
    resultExporter.exportToPath();
    }
    PointCloudT::Ptr tarEst(new PointCloudT);
    pcl::transformPointCloud(*source, *tarEst, bestT);

    if (Config<bool>("Visualization","showFinalResult")) viewer.addPointCloud(tarEst, Color::RED);

    trimmedICP(tarEst, target, Config<float>("overlapRatio"));

    return 0;
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
    else if (fileType == "obj") {
        if (pcl::io::loadOBJFile <PointT>(filePath, *output) == -1) { // the file doesnt exist
            throw invalid_argument("Cannot load the input file, please check and try again!\n");
        }
    }
    else if (fileType == "pcd") {
        if (pcl::io::loadPCDFile <PointT>(filePath, *output) == -1) { // the file doesnt exist
            throw invalid_argument("Cannot load the input file, please check and try again!\n");
        }
    }
    else if (fileType == "txt") {
        std::ifstream file(filePath);
        std::string str;
        while (std::getline(file, str))
        {
            vector<string> tokens;
            tokens = split(str, ',');
            if (tokens.size() != 7) {
                PCL_WARN("@s IMPORT FAILIURE\n", str);
                continue;
            }
            PointT p;
            p.x = stof(tokens[0]); p.y = stof(tokens[1]); p.z = stof(tokens[2]);
            p.r = stof(tokens[4]); p.g = stof(tokens[5]); p.b = stof(tokens[6]);
            p.a = stof(tokens[3]);
            output->push_back(p);
        }
    }


}

void computeDescriptor(PointCloudT::Ptr seed, PointCloudT::Ptr source, float radiusMin, float radiusMax, float radiusStep, vector<Desp>& desps){
    ColorFeature cf(seed, source, Config<float>("ColorFeature","radius"));
    //cf.compute();

    KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud (source);
    vector<int> validIndexes;

    // filter the seeds
    for (int i = 0; i < seed->size(); ++i) {
        PointT searchPoint = seed->points[i];
        vector<int> pointIdxRadiusSearch;
        vector<float> pointRadiusSquaredDistance;
        if ( kdtree.radiusSearch (searchPoint, radiusMin, pointIdxRadiusSearch, pointRadiusSquaredDistance) > Config<int>("minimumSearchTh") )
        {
            validIndexes.push_back(i);
        }
    }

    cout << "[compute Desp] source " << source->size() << "  valid seeds: " << validIndexes.size() << endl;

    // computeSVD
    for (int & idx: validIndexes ) {
        PointT searchPoint = seed->points[idx];
        Desp desp;
        if (searchPoint.x != cf.meanOutput->points[idx].x
            || searchPoint.y != cf.stdvOutput->points[idx].y
            || searchPoint.z != cf.skewOutput->points[idx].z) cout << "[error] wrong calculate of points color feature" << endl;
        desp.CI.mean.r = cf.meanOutput->points[idx].r; desp.CI.mean.g = cf.meanOutput->points[idx].g; desp.CI.mean.b = cf.meanOutput->points[idx].b;
        desp.CI.stdv.r = cf.stdvOutput->points[idx].r; desp.CI.stdv.g = cf.stdvOutput->points[idx].g; desp.CI.stdv.b = cf.stdvOutput->points[idx].b;
        desp.CI.skew.r = cf.skewOutput->points[idx].r; desp.CI.skew.g = cf.skewOutput->points[idx].g; desp.CI.skew.b = cf.skewOutput->points[idx].b;

        desp.seed = searchPoint;
        vector<int> pointIdxRadiusSearch;
        vector<float> pointRadiusSquaredDistance;
        for (double i = radiusMin; i <= radiusMax; i+= radiusStep) {
            vector<int> pointIdxRadiusSearch;
            vector<float> pointRadiusSquaredDistance;
            if ( kdtree.radiusSearch (searchPoint, i, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
                Vector3f s;
                Vector3f n;
                svdCov(source, searchPoint, pointIdxRadiusSearch, s, n);
                desp.S.push_back(s);
                desp.N.push_back(n);
            }
        }

        for (int i = 0; i < desp.S.size()-1; ++i) {
            desp.S[i] = desp.S[i+1]  - desp.S[i];
            //cout << desp.S[i].transpose() << endl;
        }

        desp.S.pop_back();

        assert(desp.S.size() + 1 == desp.N.size());
        desps.push_back(desp);
    }

}

void svdCov(PointCloudT::Ptr input, PointT seed, vector<int> &othersIdx, Vector3f& s, Vector3f& n) {
    MatrixXf C = MatrixXf::Zero(3,3);
    PointT seedP = seed;
    Vector3f x_i(seedP.x, seedP.y, seedP.z);
    for (int &idx:othersIdx) {
        PointT tmp = input->points[idx];
        Vector3f x(tmp.x, tmp.y, tmp.z);
        Vector3f minus = x_i - x;
        C = minus * minus.transpose() + C;
    }
    C = C / othersIdx.size();

    // compute SVD
    JacobiSVD<MatrixXf> svd(C, ComputeThinU |  ComputeThinV);
    auto S = svd.singularValues();
    s =  S / (S(0) + S(1) + S(2));
    auto U = svd.matrixU();
    n = Vector3f(U(0,2),U(1,2),U(2,2));
    Vector3f seed_v(-seed.x, -seed.y, -seed.z);
    if (n.dot(seed_v)<0) n = -n;
//
//    svdExport.insertLine(  to_string(C(0,0)) + " " + to_string(C(0,1)) + " " +to_string(C(0,2)) + " " +
//                           to_string(C(1,0)) + " " + to_string(C(1,1)) + " " +to_string(C(1,2)) + " " +
//                           to_string(C(2,0)) + " " + to_string(C(2,1)) + " " +to_string(C(2,2)) + " " +
//                           to_string(s(0))   + " " + to_string(s(1))   + " " + to_string(s(2)) + " " +
//                           to_string(n(0))   + " " + to_string(n(1))   + " " + to_string(n(2)));
}

Eigen::Matrix4d matching(vector<Desp>& srcDesps, vector<Desp>& tarDesps, vector<Desp>& srcFDesps, vector<Desp>& tarFDesps) {
    cout << "[matching]" << endl;
    int ovNum = tarDesps.size() * Config<float>("overlapRatio");
    float ransacThr = Config<float>("GridStep") * Config<float>("GridStep");

    // 1. Initial matching of Desp
    unordered_map<int,pair<int,float>> map;
    FlannSearch::flannSearch(srcFDesps, tarFDesps, map);

    float minErr = INT_MAX;
    Eigen::Matrix4d bestT = Eigen::Matrix4d::Identity();

    // pair< srcIdx, tarIdx >
    vector<pair<int,int>> rec;
    for (auto& it : map) rec.push_back( make_pair(it.second.first, it.first));


    // visualize the initial matching
    if (Config<bool>("Visualization","showInitMatch")) {
        for (int i = 0; i < rec.size(); ++i) viewer.addMatching(srcFDesps[rec[i].first],tarFDesps[rec[i].second], RED);
        viewer.spin();
    }

    // 2. aggregating matching
    float seedIdx = 0;
    int filteredNum = 0;
    //#pragma omp parallel for
    for (int i = 0; i < rec.size(); ++i) {
        cout << "Matching " << ++seedIdx << "th seed among " << rec.size() << "\n";
        vector<Match> aggMatches;
        if (isFiltered_Color(srcFDesps[rec[i].first],tarFDesps[rec[i].second])) {
            filteredNum++;
            continue;
        }

        Match match( &srcFDesps[rec[i].first],&tarFDesps[rec[i].second] );
        aggMatches.push_back(match);
        aggMatching(*match.src, srcDesps, *match.tar, tarDesps, aggMatches);
        if (aggMatches.size() > 10) {
           Eigen::Matrix4d thisT = Eigen::Matrix4d::Identity();
           float thisErr = INT_MAX;
           Estimator::estimateRigidTransform(aggMatches,srcDesps, tarDesps, thisT, thisErr, ovNum, ransacThr);
           //TimeEstimateRigid = timeElapsed(start);
            //#pragma omp critical
            {
                if (thisErr < minErr) {
                    minErr = thisErr;
                    bestT = thisT;
                }
            };
        }

    }
    cout << "min Err: "<< minErr << endl;
    cout << "best T:\n " << bestT << endl;
    cout << "filtered num: " << filteredNum << endl;

    // show right init match
    int num_usefulMatch = 0;
    DebugFileExporter matchExport("./matchPairs.txt");
    for (auto& it : rec) {
        PointT src = srcDesps[it.first].seed;
        PointT tar = tarDesps[it.second].seed;
        Eigen::Vector4d srcV(src.x, src.y, src.z, 1);
        Eigen::Vector4d tarV(tar.x, tar.y, tar.z, 1);
        Eigen::Vector4d tarV_hat = bestT * srcV;
        string s = to_string(src.x) + " " + to_string(src.y) + " " + to_string(src.z) + " " +
                   to_string(tar.x) + " " + to_string(tar.y) + " " + to_string(tar.z) + " " +
                   srcDesps[it.first].CI.mean.to_str() + " " + srcDesps[it.first].CI.stdv.to_str() + " " + srcDesps[it.first].CI.skew.to_str() + " " +
                   tarDesps[it.second].CI.mean.to_str() + " " + tarDesps[it.second].CI.stdv.to_str() + " " + tarDesps[it.second].CI.skew.to_str() + " ";

        float gridStep = Config<float>("GridStep");
        if ((tarV_hat - tarV).norm() < gridStep) {
            viewer.addMatching(srcDesps[it.first],tarDesps[it.second], YELLOW);
            num_usefulMatch++;
            s += '1';
        }else{
            s += '0';
        }
        matchExport.insertLine(s);
    }
    matchExport.exportToPath();
    cout << "[Analysis] Total " << rec.size() << " init matches, " << num_usefulMatch << " matches contribute to estimation\n";

    return bestT;

}

void aggMatching(Desp& src, vector<Desp>& srcSeeds, Desp& tar, vector<Desp>& tarSeeds, vector<Match>& matches) {
    int n_src = srcSeeds.size();
    float thetaThr = Config<float>("thetaThr");
    float distThr  = Config<float>("distThr");
    float gridStep = Config<float>("GridStep");
    int layerNum = src.N.size();
    vector<float> srcDists;
    vector<float> tarDists;
    PointT s = src.seed;
    PointT t = tar.seed;

    for (auto& seed : srcSeeds) {
        PointT p = seed.seed;
        float dist = pcl::geometry::distance(p, s);
        srcDists.push_back(dist);
    }
    for (auto& seed : tarSeeds) {
        PointT p = seed.seed;
        float dist = pcl::geometry::distance(p, t);
        tarDists.push_back(dist);
    }


    // for each src seed, check each tar seed which dist diff is less than gird/2
    vector<vector<int>> srcNearDist(n_src,  vector<int>());
    FlannSearch::flannSearch(tarDists, srcDists, gridStep/2, srcNearDist);

    // compute the normal difference
    vector<vector<float>> seedAngle;
    vector<vector<float>> tarAngle;
    computeNormalDiff(src, srcSeeds, seedAngle);
    computeNormalDiff(tar, tarSeeds, tarAngle);

    struct Compare {
        bool operator() (pair<int, float>& p1, pair<int, float>& p2) {return p1.second > p2.second;}
    };
    int validnum = 0;
    // reject outlier


    for (int i = 0; i < n_src; ++i) {
        if (srcNearDist[i].size() == 0) continue;
        Desp srcSeed = srcSeeds[i];

        vector<float> s_angle = seedAngle[i];
        priority_queue<pair<int,float>, vector<pair<int,float>>, Compare> thetaDiff;

        int idxValid = 0;
        for (int& idx : srcNearDist[i]) {
            Desp tarSeed = tarSeeds[idx];
            vector<float> t_angle = tarAngle[idx];
            int valid = 0;
            float totalDiff = 0;
            for (int j = 0; j < layerNum; ++j) {
                float diff = abs(s_angle[j] - t_angle[j]);
                if ( diff < thetaThr) valid++;
                totalDiff += diff;
            }
            // all smaller than thed
            totalDiff = valid == layerNum ? (totalDiff/layerNum)  : INT_MAX;
            thetaDiff.push(make_pair(idx,totalDiff));
            if (valid == layerNum) idxValid++;
        }

        if (thetaDiff.top().second < thetaThr) {
            int tarIdx = thetaDiff.top().first;
            float despDist = computeDespDist (srcSeeds[i], tarSeeds[tarIdx]);
            if (despDist < distThr) {
                validnum++;
                Match match(&srcSeeds[i], &tarSeeds[tarIdx]);
                matches.push_back(match);
            }
        }
    }

}

float computeDespDist(Desp& src, Desp& tar) {
    assert(src.S.size() == tar.S.size());
    float totalDist = 0;
    int n = src.S.size();
    for (int i = 0; i < n; ++i) {
        Vector3f diff = src.S[i] - tar.S[i];
        totalDist += pow(diff[0],2) + pow(diff[1],2) + pow(diff[2],2);
    }
    return sqrt(totalDist);

}

// normal diff among L layers
// Unit : degree
void computeNormalDiff(Desp& seed, vector<Desp>& allDesps, vector<vector<float>>& res) {

    assert(res.size() == 0);
    int n = seed.N.size();
    for (auto& tar: allDesps) {
        vector<float> diffoflayers(n);
        for (int i = 0; i < n; ++i) {
            diffoflayers[i] = acos(seed.N[i].dot(tar.N[i])) * 180 / M_PI;

        }
        res.push_back(diffoflayers);
    }

}

void trimmedICP(const vector<Eigen::Vector3d> &tarEst, const vector<Eigen::Vector3d> &tarData, float overlapRatio){
    int ovNum = overlapRatio * tarData.size();
    float ransacThr = Config<float>("GridStep") * Config<float>("GridStep");
    typedef pair<pair<int,int>, float> PAIR;
    struct Compare {
        bool operator() (PAIR& p1, PAIR& p2) {return p1.second < p2.second;}
    };

    unordered_map<int,pair<int,float>> map;
    FlannSearch::flannSearch(tarEst, tarData, map);
    priority_queue<PAIR, vector<PAIR>, Compare> pq;
    for (auto& it:map) {
        pq.push(make_pair( make_pair(it.first,it.second.first), it.second.second));
        if (pq.size() > ovNum) pq.pop();
    }

    int matIter = 100; float dE = INT_MAX; int iter = 0;
    float errThr = 1e-4; float rmseThr = 0.001;
    vector<float> rmsE(matIter+1, 0);

    vector<Eigen::Vector3d> match_srcData;
    vector<Eigen::Vector3d> match_tarData;
    int testIdx = 0;
    while(!pq.empty()) {
        PAIR tmp = pq.top();
        pq.pop();
        if (testIdx++ < 10) cout << tmp.second << endl;
        match_tarData.push_back(tarData[tmp.first.first]);
        match_srcData.push_back(tarEst[tmp.first.second]);
        rmsE[0] += tmp.second;
    }

    rmsE[0] = sqrt (rmsE[0] / ovNum);
    cout << rmsE[0] << endl;
    while(dE > errThr && iter < maxIter && rmsE[iter] > rmseThr) {
        iter++;
        Eigen::Matrix4d T;
        float err;
        Estimator::estimateRigidTransform(match_tarData, match_srcData, T, err, ovNum, ransacThr);
        vector<Eigen::Vector3d> tar_hat;
        for (auto& src : match_srcData) {
            Eigen::Vector4d tmp = T * Eigen::Vector4d(src(0),src(1),src(2),1);
            tar_hat.push_back(Eigen::Vector3d(tmp(0), tmp(1), tmp(2)));
        }

        // search nearest point
        unordered_map<int,pair<int,float>> map;
        FlannSearch::flannSearch(tar_hat, match_tarData, map);
        priority_queue<PAIR, vector<PAIR>, Compare> pq;
        for (auto& it:map) {
            pq.push(make_pair( make_pair(it.first,it.second.first), it.second.second));
            if (pq.size() > ovNum) pq.pop();
        }

        float thisErr = 0;
        vector<Eigen::Vector3d> new_match_srcData;
        vector<Eigen::Vector3d> new_match_tarData;
        while (!pq.empty()) {
            PAIR tmp = pq.top();
            pq.pop();
            thisErr += tmp.second;
            new_match_tarData.push_back(match_tarData[tmp.first.first]);
            new_match_srcData.push_back(match_srcData[tmp.first.second]);
        }
        match_tarData = new_match_tarData;
        match_srcData = new_match_srcData;
        thisErr = thisErr / ovNum;
        rmsE[iter] = sqrt(thisErr);
        dE = rmsE[iter-1] - rmsE[iter];
    }

    cout << "[trimmed-icp] " << iter << endl;

}

bool isFiltered_Color(Desp& d1, Desp& d2){
    return ((d1.CI - d2.CI) > Config<int>("ColorFeature","filterThr"));
}
