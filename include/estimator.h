// ransac.cpp : Defines the initialization routines for the DLL.
//

#include <iostream>
#include <time.h>
#include <vector>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

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
             R(2,0), R(2,1), R(2,2), t(2);
		return T;
	}

	// return T and num of matched pts
    static pair<Eigen::Matrix4d, int> RANSAC(const vector<Eigen::Vector3d>& src, const vector<Eigen::Vector3d>& dst, double threshold) {
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

};

/*// generate random indices for the minimal sample set which is 3
vector<int> generate(int N)
{
	vector<int> index(N); //the whole indices
	for (int i = 0; i < N; i++)
	{
		index[i] = i;
	}

	vector<int> vektor(M);

	int in, im = 0;
	for (in = N; in > N - M; in--) 
	{
		int r = rand() % in; *//* generate a random number 'r' *//*
		vektor[im++] = index[r]; *//* the range begins from 0 *//*
		index.erase(index.begin() + r);
	}

	return vektor;
}



double* estimateTform(double* srcPts, double* tarPts, vector<int> &Idx)
{
	mxArray *rhs[2], *lhs[2];
	rhs[0] = mxCreateDoubleMatrix(3, Idx.size(), mxREAL);
	rhs[1] = mxCreateDoubleMatrix(3, Idx.size(), mxREAL);

	double *X, *Y;
	X = mxGetPr(rhs[0]);
	Y = mxGetPr(rhs[1]);	

	for (int j = 0; j < Idx.size(); j++)
	{
		X[3 * j] = tarPts[3 * Idx[j]];
		X[3 * j + 1] = tarPts[3 * Idx[j] + 1];
		X[3 * j + 2] = tarPts[3 * Idx[j] + 2];

		Y[3 * j] = srcPts[3 * Idx[j]];
		Y[3 * j + 1] = srcPts[3 * Idx[j] + 1];
		Y[3 * j + 2] = srcPts[3 * Idx[j] + 2];
	}	

	lhs[0] = mxCreateDoubleMatrix(4, 4, mxREAL);
	lhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	mexCallMATLAB(2,lhs,2,rhs,"estimateRigidTransform");

	double* T = mxGetPr(lhs[0]);
	return T;
}



void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	double *srcPts, *tarPts, threshold;

	if (mxGetM(prhs[0]) != 3 || mxGetM(prhs[1]) != 3 || mxGetN(prhs[0]) != mxGetN(prhs[1]))
		mexErrMsgTxt("The input point matrix should be with size 3-by-N!");

	srcPts = mxGetPr(prhs[0]); 
	tarPts = mxGetPr(prhs[1]);
	threshold = mxGetScalar(prhs[2]);
	int N = mxGetN(prhs[0]);
	
	plhs[0] = mxCreateLogicalMatrix(1,N); // indicate either a match is inlier or not
	bool* CS = (bool*)mxGetData(plhs[0]);

	// Main loop
	//---------------------------------------------------------------
	// initializations
	int iter = 0; //number of iterations
	int bestSz = 3; //initial threshold for inlier size of a better model
	vector<int> randIdx(M, 0);
	vector<double> x(3, 0), y_hat(3, 0), y(3, 0);
	vector<bool> thisCS(N, false);

	srand((unsigned)time(NULL)); //set the seed to the current time
	//srand((unsigned)time(0)); //set the seed to 0
	while (iter <= maxIter)
	{
		randIdx = generate(N);
		double* T = estimateTform(srcPts, tarPts, randIdx);

		// to get size of the consensus set
		int inlierSz = 0;
		for (int i = 0; i < N; i++)
		{
			x[0] = srcPts[3 * i]; x[1] = srcPts[3 * i + 1]; x[2] = srcPts[3 * i + 2];
			y[0] = tarPts[3 * i]; y[1] = tarPts[3 * i + 1]; y[2] = tarPts[3 * i + 2];
			
			y_hat[0] = T[0] * x[0] + T[4] * x[1] + T[8] * x[2] + T[12];
			y_hat[1] = T[1] * x[0] + T[5] * x[1] + T[9] * x[2] + T[13];
			y_hat[2] = T[2] * x[0] + T[6] * x[1] + T[10] * x[2] + T[14];

			double thisErr = (y[0] - y_hat[0])*(y[0] - y_hat[0]) + 
						     (y[1] - y_hat[1])*(y[1] - y_hat[1]) + 
							 (y[2] - y_hat[2])*(y[2] - y_hat[2]);

			thisCS[i] = false;
			if (thisErr < threshold)
			{
				inlierSz++;
				thisCS[i] = true;
			}
		}

		if (inlierSz>bestSz)
		{
			bestSz = inlierSz; // update the best model size

			//update the consensus set
			for (int i = 0; i < N; i++)
			{
				CS[i] = thisCS[i];
			}
		}			

		if (bestSz == N)
			break;

		iter++;
	}
	//--------------------------------------------------------------	

	return;
}*/


