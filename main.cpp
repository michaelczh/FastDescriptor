#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/uniform_sampling.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <queue>
#include "main.h"
#include "SimpleView.h"
#include "estimator.h"
#include "DebugExporter.h"
using namespace pcl;
using namespace std;
using namespace Eigen;
SimpleView viewer("view");

DebugFileExporter svdExport("./svdExport.txt");

int main() {
    string configPath = "../config.yaml";
    config = YAML::LoadFile(configPath);
    string sourcePath = Config<std::string>("srcPath");
    string targetPath = Config<std::string>("tarPath");


    PointCloudT::Ptr source(new PointCloudT); PointCloudT::Ptr sourceSeed(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT); PointCloudT::Ptr targetSeed(new PointCloudT);
    loadPointCloudData(sourcePath, source);
    loadPointCloudData(targetPath, target);
    uniformDownSample(source, Config<float>("Downsample","Rho"), sourceSeed);
    uniformDownSample(target, Config<float>("Downsample","Rho"), targetSeed);

    viewer.addPointCloud(source, Color::GREEN);
    viewer.addPointCloud(target, Color::BLUE);
    float radiusMin  = Config<float>("Radii", "min") *Config<float>("GridStep");
    float radiusStep = Config<float>("Radii", "step")*Config<float>("GridStep");
    float radiusMax  = Config<float>("Radii", "max") *Config<float>("GridStep");
    vector<Desp> sourceDesp, targetDesp;
    computeDescriptor(sourceSeed, source, radiusMin, radiusMax, radiusStep, sourceDesp);
    computeDescriptor(targetSeed, target, radiusMin, radiusMax, radiusStep, targetDesp);
    svdExport.exportToPath();

    Eigen::Matrix4d bestT = matching(sourceDesp,targetDesp);

    PointCloudT::Ptr src_hat = PointCloudT::Ptr(new PointCloudT);
    for (auto &p : source->points) {
        Eigen::Vector4d tmp(p.x, p.y, p.z, 1);
        tmp = bestT * tmp;
        PointT tmpP;
        tmpP.x = tmp(0); tmpP.y = tmp(1); tmpP.z = tmp(2);
        src_hat->push_back(tmpP);
    }
    viewer.addPointCloud(src_hat, Color::RED);
    viewer.spin();
    return 0;
}


std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

void loadPointCloudData(string filePath, PointCloudT::Ptr output){
    assert(output->size() == 0);
    stringstream ss;
    ss << "Input File: " << filePath;

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

void uniformDownSample(PointCloudT::Ptr input, float Rho, PointCloudT::Ptr output){
    int numOri = input->size();
    assert(numOri > 0);
    UniformSampling<PointT> filter;
    filter.setInputCloud(input);
    filter.setRadiusSearch(Rho);
    filter.filter(*output);
    cout << "[uniformDownSample] from " << numOri << " to " << output->size() << endl;
}

void computeDescriptor(PointCloudT::Ptr seed, PointCloudT::Ptr source, float radiusMin, float radiusMax, float radiusStep, vector<Desp>& desps){

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

    svdExport.insertLine(  to_string(C(0,0)) + " " + to_string(C(0,1)) + " " +to_string(C(0,2)) + " " +
                           to_string(C(1,0)) + " " + to_string(C(1,1)) + " " +to_string(C(1,2)) + " " +
                           to_string(C(2,0)) + " " + to_string(C(2,1)) + " " +to_string(C(2,2)) + " " +
                           to_string(s(0))   + " " + to_string(s(1))   + " " + to_string(s(2)) + " " +
                           to_string(n(0))   + " " + to_string(n(1))   + " " + to_string(n(2)));
}

void trimmedICP(PointCloudT::Ptr tarEst, PointCloudT::Ptr tarData, float overlapRatio){
    typedef pair<pair<int,int>, float> PAIR;
    KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud (tarData);
    auto cmp = [](const PAIR &a, const PAIR &b) {
        return  a.second < b.second;
    };
    priority_queue<PAIR, vector<PAIR>, decltype( cmp ) > pq(cmp);

    int num = overlapRatio * tarData->size();
    for (int i = 0; i < tarEst->size(); ++i) {
        PointT searchPoint = tarEst->points[i];
        vector<int> pointIdxNKNSearch;
        vector<float> pointNKNSquaredDistance;
        if ( kdtree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
            //unordered_map[i] = pointIdxNKNSearch[0];
            pair<pair<int,int>, float> pair;
            pair.first.first = i;
            pair.first.second = pointIdxNKNSearch[0];
            pair.second = pointNKNSquaredDistance[0];
            if (pq.size() < num) pq.push(pair);
            else if(pq.top().second > pointNKNSquaredDistance[0]) {
                pq.pop();
                pq.push(pair);
            }
        }
    }

}


void estimateRigidTransform(const vector<Match>& matches, const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, Matrix4d & T, float &err) {

    //cout << "[estimateRigidTransform]" << endl;
    int ovNum = tarDesps.size() * Config<float>("overlapRatio");
    vector<Eigen::Vector3d> src;
    vector<Eigen::Vector3d> tar;

    for (auto& match: matches) {
        PointT srcP = match.src->seed;
        PointT tarP = match.tar->seed;
        src.push_back( Eigen::Vector3d(srcP.x, srcP.y, srcP.z));
        tar.push_back( Eigen::Vector3d(tarP.x, tarP.y, tarP.z));
    }

    float ransacThr = Config<float>("GridStep") * Config<float>("GridStep");
    pair<Eigen::Matrix4d, int> estRes = Estimator::RANSAC(src, tar, ransacThr);
    //cout << "int " << estRes.second << endl;
    if (estRes.second < 3) {
        err = INT_MAX;
        T = Eigen::Matrix4d::Identity();
        return;
    }
    Eigen::Matrix4d estimateT = estRes.first;
    T = estimateT;


    vector<Eigen::Vector3d> transSrc;
    for (auto& desp : srcDesps) {
        PointT src = desp.seed;
        Eigen::Vector4d tmp = estimateT * Eigen::Vector4d(src.x, src.y, src.z, 1);
        transSrc.push_back(Eigen::Vector3d(tmp(0), tmp(1), tmp(2)));
    }

    priority_queue<float> pq;
    for (int i = 0; i < tarDesps.size(); ++i) {
        float minDiff = INT_MAX;
        Eigen::Vector3d tar(tarDesps[i].seed.x, tarDesps[i].seed.y, tarDesps[i].seed.z);
        for (int j = 0; j < transSrc.size(); ++j) {
            Eigen::Vector3d diff = tar - transSrc[j];
            float diffValue = diff.squaredNorm();
            pq.push(diffValue);
            if (pq.size() > ovNum) pq.pop();
        }
    }

    err = 0;
    while(!pq.empty()) {
        err += pq.top();
        pq.pop();
    }

}


Eigen::Matrix4d matching(vector<Desp>& srcDesps, vector<Desp>& tarDesps) {
    cout << "[matching]" << endl;
    struct Compare {
        bool operator() (pair<Match, double>& p1, pair<Match, double>& p2) {return p1.second > p2.second;}
    };
    priority_queue<pair<Match, double>, vector<pair<Match, double>>, Compare> pq;

    int n_src = srcDesps.size();
    for (auto& tarDesp : tarDesps) {
        double minDiff = INT_MAX;
        vector<Vector3f> tarS = tarDesp.S;
        Match match;
        match.tar = &tarDesp;
        for (auto& srcDesp : srcDesps) {
            vector<Vector3f> srcS = srcDesp.S;
            double diff = diffOfS(srcS, tarS);
            if (diff < minDiff) {
                minDiff = diff;
                match.src = &srcDesp;
            }
        }
        pq.push(make_pair(match,minDiff));
    }

    cout << "[matching] " << "srcDesp Size: " << srcDesps.size() << " tarDesp size: " <<  tarDesps.size() << "  " << "pq " << pq.size() << endl;

    float minErr = INT_MAX;
    Eigen::Matrix4d bestT = Eigen::Matrix4d::Identity();
    while (!pq.empty()) {
       vector<Match> aggMatches;
       Match match = pq.top().first;
       aggMatches.push_back(match);
       aggMatching(*match.src, srcDesps, *match.tar, tarDesps, aggMatches);

       if (aggMatches.size() > 10) {
           Eigen::Matrix4d thisT = Eigen::Matrix4d::Identity();
           float thisErr = INT_MAX;
           estimateRigidTransform(aggMatches,srcDesps, tarDesps, thisT, thisErr);
           if (thisErr < minErr) {
               minErr = thisErr;
               bestT = thisT;
           }
       }
//
       pq.pop();
    }
    cout << "min Err: "<< minErr << endl;
    cout << "best T: " << bestT << endl;

    return bestT;



}

double diffOfS(vector<Vector3f> srcS, vector<Vector3f> tarS) {
    assert(srcS.size() == tarS.size());
    int n = srcS.size();
    double totalDiff = 0;
    for (int i = 0; i < n; ++i) {
        Vector3f diff = srcS[i] - tarS[i];
        totalDiff+= (pow(diff[0],2) + pow(diff[1],2) + pow(diff[2],2)) ;
    }
    return totalDiff;
}

void aggMatching(Desp& src, vector<Desp>& srcSeeds, Desp& tar, vector<Desp>& tarSeeds, vector<Match>& matches) {
   // cout << "[agg match]" << endl;
    int n_src = srcSeeds.size();
    int n_tar = tarSeeds.size();
    float thetaThr = Config<float>("thetaThr");
    float distThr  = Config<float>("distThr");
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
    float gridStep = Config<float>("GridStep");
    for (int i = 0 ; i < n_src; ++i) {
        for (int j = 0; j < n_tar; ++j) {
            float distDiff = abs (srcDists[i] - tarDists[j]);
            if (distDiff < gridStep/2) srcNearDist[i].push_back(j);
        }
    }

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
            for (int j = 0; j < seedAngle[0].size(); ++j) {
                float diff = abs(s_angle[j] - t_angle[j]);
                //cout << "theta diff " << diff << endl;
                if ( diff < thetaThr) valid++;
                totalDiff += diff;
            }
            // all smaller than thed
            totalDiff = valid == seedAngle[0].size() ? (totalDiff/seedAngle[0].size())  : INT_MAX;
            thetaDiff.push(make_pair(idx,totalDiff));
            if (valid == seedAngle[0].size()) idxValid++;
        }



        if (thetaDiff.top().second < thetaThr) {

            int tarIdx = thetaDiff.top().first;
            float despDist = computeDespDist (srcSeeds[i], tarSeeds[tarIdx]);
            if (despDist < distThr) {
                validnum++;
                Match match;
                match.src = &srcSeeds[i];
                match.tar = &tarSeeds[tarIdx];
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
void computeNormalDiff(Desp& seed, vector<Desp>& allDesps, vector<vector<float>>& res) {

    assert(res.size() == 0);
    int n = seed.N.size();
    for (auto& tar: allDesps) {
        vector<float> diffoflayers(n);
        for (int i = 0; i < n; ++i) {
            diffoflayers[i] = acos(seed.N[i].dot(tar.N[i])) * 180 / M_PI;

        }

        //abort();

        res.push_back(diffoflayers);
    }

}