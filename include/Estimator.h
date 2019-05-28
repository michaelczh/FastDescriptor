// ransac.cpp : Defines the initialization routines for the DLL.
//

#include <iostream>
#include <time.h>
#include <vector>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include "Type.h"
#include "FlannSearch.h"

#define M       3
#define maxIter 1000

using namespace std;


struct EstRes {
    Eigen::Matrix4d T;
    int bestSz;
    float err;
};

class Estimator {
private:
    static Estimator instance;
    // generate random indices for the minimal sample set
    vector<int> generate(int N){
        vector<int> index(N); //the whole indices
        for (int i = 0; i < N; i++) index[i] = i;
        vector<int> vektor(M);
        int in, im = 0;
        for (in = N; in > N - M; in--) {
            int r = rand() % in; /* generate a random number 'r' */
            vektor[im++] = index[r]; /* the range begins from 0 */
            index.erase(index.begin() + r);
        }

        return vektor;
    }
public:
    static Eigen::Matrix4d computeRigidTransform(const vector<Eigen::Vector3d>& src, const vector<Eigen::Vector3d>& dst) {

		assert(src.size()  == dst.size());
		int pairSize = src.size();
		Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);
		for (int i=0; i<pairSize; ++i)
		{
			center_src += src[i];
			center_dst += dst[i];
		}
		center_src /= (double)pairSize;
		center_dst /= (double)pairSize;

		Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
		for (int i=0; i<pairSize; ++i)
		{
			for (int j=0; j<3; ++j)
				S(i, j) = src[i][j] - center_src[j];
			for (int j=0; j<3; ++j)
				D(i, j) = dst[i][j] - center_dst[j];
		}
		Eigen::MatrixXd Dt = D.transpose();
		Eigen::Matrix3d H = Dt*S;
		Eigen::Matrix3d W, U, V;

		Eigen::JacobiSVD<Eigen::MatrixXd> svd;
		Eigen::MatrixXd H_(3, 3);
		for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) H_(i, j) = H(i, j);
		svd.compute(H_, Eigen::ComputeThinU | Eigen::ComputeThinV );
		if (!svd.computeU() || !svd.computeV()) {
			std::cerr << "decomposition error" << endl;
			return Eigen::Matrix4d::Identity();
		}
		Eigen::Matrix3d Vt = svd.matrixV().transpose();
		Eigen::Matrix3d R = svd.matrixU()*Vt;
		Eigen::Vector3d t = center_dst - R*center_src;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T << R(0,0), R(0,1), R(0,2), t(0),
             R(1,0), R(1,1), R(1,2), t(1),
             R(2,0), R(2,1), R(2,2), t(2),
             0, 0, 0, 1;
		return T;
	}

	// return T and num of matched pts
    static pair<Eigen::Matrix4d, int> RANSAC(const vector<Eigen::Vector3d>& src, const vector<Eigen::Vector3d>& dst, double threshold) {
        //cout << src.size() << " " << dst.size() << endl;
        assert(src.size() == dst.size());
        // initializations
        int N = src.size();
        int iter = 0; //number of iterations
        int bestSz = 0; //initial threshold for inlier size of a better model
        double minErr = INT_MAX;
        vector<int> randIdx(M, 0);
        vector<double> x(3, 0), y_hat(3, 0), y(3, 0);
        vector<bool> CS(N, false);
        vector<bool> thisCS(N, false);

        while(iter++ <= maxIter) {

            randIdx = instance.generate(N);
            vector<Eigen::Vector3d> randSrc;
            vector<Eigen::Vector3d> randDst;
            for (auto& idx : randIdx) {
                randSrc.push_back(src[idx]);
                randDst.push_back(dst[idx]);
            }
            Eigen::Matrix4d T = computeRigidTransform(randSrc, randDst);

            int inlierSz = 0;
            for (int i = 0; i < N; i++) {
                Eigen::Vector4d src_v(src[i](0),src[i](1),src[i](2),1);
                Eigen::Vector4d tar_v(dst[i](0),dst[i](1),dst[i](2),1);
                Eigen::Vector4d tar_hat = T * src_v;
                Eigen::Vector4d err_v = (tar_hat - tar_v);
                double thisErr = err_v.transpose()*err_v;

                minErr = min(minErr, thisErr);
                thisCS[i] = false;
                if (thisErr < threshold) {
                    inlierSz++;
                    thisCS[i] = true;
                }
            }

            if (inlierSz > bestSz) {
                bestSz = inlierSz;
                for (int i = 0; i < N; ++i) CS[i] = thisCS[i];
            }

            if (bestSz == N) break;
        }
        //cout << "min err: " << minErr << endl;

        vector<Eigen::Vector3d> match_src;
        vector<Eigen::Vector3d> match_tar;
        for (int i = 0; i < N; ++i) {
            if (CS[i]) {
                match_src.push_back(src[i]);
                match_tar.push_back(dst[i]);
            }
        }

        Eigen::Matrix4d estimateT = Eigen::Matrix4d::Identity();
        if (bestSz >= 3) {
            estimateT = computeRigidTransform(match_src,match_tar);
        }
        return make_pair(estimateT,bestSz);
    };


    static void estimateRigidTransform(const vector<Eigen::Vector3d>& src, const vector<Eigen::Vector3d>& tar, Matrix4d & T, float &err, int ovNum, float ransacThr) {
        pair<Eigen::Matrix4d, int> estRes = Estimator::RANSAC(src, tar, ransacThr);
        if (estRes.second < 3) {
            err = INT_MAX;
            T = Eigen::Matrix4d::Identity();
            return;
        }
        Eigen::Matrix4d estimateT = estRes.first;
        T = estimateT;


        vector<Eigen::Vector3d> transSrc;
        for (auto& p : src) {
            Eigen::Vector4d tmp = estimateT * Eigen::Vector4d(p(0), p(1), p(2), 1);
            transSrc.push_back(Eigen::Vector3d(tmp(0), tmp(1), tmp(2)));
        }

        unordered_map<int,pair<int,float>> map;
        FlannSearch::flannSearch(transSrc, tar, map);

        priority_queue<float> pq;
        for (auto& it : map) {
            pq.push(it.second.second);
            if (pq.size() > ovNum) pq.pop();
        }
        err = 0;
        while(!pq.empty()) {
            err += pq.top();
            pq.pop();
        }
    }

    static void estimateRigidTransform(const vector<Match>& matches, const vector<Desp>& srcDesps, const vector<Desp>& tarDesps, Matrix4d & T, float &err, int ovNum, float ransacThr) {
        vector<Eigen::Vector3d> src;
        vector<Eigen::Vector3d> tar;

        for (auto& match: matches) {
            PointT srcP = match.src->seed;
            PointT tarP = match.tar->seed;
            src.push_back( Eigen::Vector3d(srcP.x, srcP.y, srcP.z));
            tar.push_back( Eigen::Vector3d(tarP.x, tarP.y, tarP.z));
        }

        estimateRigidTransform(src, tar, T, err, ovNum, ransacThr);
    }

};