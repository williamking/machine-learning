#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>


void move(char* &p) {
    while (*p != ',' && *p != '\0') {
        if (*p == '\n') return;
        ++p;
     }
    ++p;
}

using namespace std;
using namespace cv;

double datas[25002 * 400];
double result[25002];
double predit[25002];
double paras[400] = {0}, learningRate = 0.5, randomRate = 0.004;
double s[400][400], y[400][400];
double lastG[400];
double alpha = 0.05;
int numOfData = 0, num = 0;

void setMat(Mat &M, double *d) {
	for (int i = 0; i < M.rows; ++i) {
		float *it = M.ptr<float>(i);
		for (int j = 0; j < M.cols; ++j) {
			it[j] = d[i * M.rows + j];
		}
	}
}

void getGK(double *g) {
    memset(g, 0, sizeof(double) * num);
    for (int i = 0; i < numOfData; ++i) {
        double val = 0;
        for (int j = 0; j < num; ++j) {
            val += paras[j] * datas[i * num + j];
        }
        for (int j = 0; j < num; ++j) {
            g[j] += (val - result[i]) * datas[i * num + j];
        }
    }

    for (int j = 0; j < num; ++j) {
        g[j] /= numOfData;
    }
}

double bias(float lamda, double *d, double *data) {
    Mat th(1, num, CV_32FC1);
	setMat(th, paras);
    Mat D(1, num, CV_32FC1);
	setMat(D, d);
    Mat theta =  th - D.mul(lamda);
	cout << theta << endl;
	cout << th << endl;
    for (int i = 0; i < numOfData; ++i) {
        Mat X(1, num, CV_32FC1);
		setMat(X, data + i * num);
        X = X.mul(theta);
        Scalar s = sum(X);
        predit[i] = s.val[0];
    }
    Mat preditMat(1, numOfData, CV_32FC1);
	setMat(preditMat, predit);
    Mat resultMat(1, numOfData, CV_32FC1);
	setMat(resultMat, result);
    preditMat -= resultMat;
    preditMat = preditMat.mul(preditMat);
    Scalar s = sum(preditMat);
    return s.val[0] / (2 * numOfData);
}
        

void calculateGradient(int k, int m, double *d) {
    int l, b;
    if (k > m) l = m; else l = k;
    if (k > m) b = k - m; else b = 0;
    double a[1000];
    double *q = new double[num];
    if (k == 0) {
        getGK(q);
        for (int i = 0; i < num; ++i) {
			lastG[i] = q[i];
            q[i] *= alpha;
            d[i] = q[i];
        }
        return;
    }
    getGK(q);
    Mat Q(num, 1, CV_32FC1, q);
    for (int i = 0; i < num; ++i) {
        y[(k - 1) % m][i] = q[i] - lastG[i];
        lastG[i] = q[i];
    }

    Mat P(1, 1, CV_32FC1);

    //后向循环
    for (int i = l - 1; i >= 0; --i) {
        int j = i + b;
        Mat Y(num, 1, CV_32FC1, y[j % m]);
        Mat S(num, 1, CV_32FC1, s[j % m]);

        P = Y.t() * S;
        double p = 1 / P.ptr<float>(0)[0];
        P = S.t() * Q;
        a[i] = p * P.ptr<float>(0)[0];
		//cout << Q << endl;
		//cout << Y << endl;
        Q = Q - a[i] * Y;
    }

    Mat R(num, 1, CV_32FC1);
    Mat O = Mat::ones(num, num, CV_32FC1);
    R = O * Q;

    double bj;

    //前向循环
    for (int i = 0; i < l; ++i) {
        int j = i + b;
        Mat Y(num, 1, CV_32FC1, s[j % m]);
        Mat S(num, 1, CV_32FC1, y[j % m]);

        P = Y.t() * S;
        double p = 1 / P.ptr<float>(0)[0];

        P = Y.t() * R;
        bj = p * P.ptr<float>(0)[0];

        R += (a[i] - bj) * S; 
    }

    free(q);

    float *val = R.ptr<float>(0);

    for (int i = 0; i < num; ++i) {
        d[i] = val[i];
    }

}


int main() {
    ifstream trainningData("test.csv");
    stringstream dataReader;
    char header[4000], line[2500];
    double *lineData;
    memset(datas, 0, sizeof(datas[0]) * 25002 * 400);

    trainningData.getline(header, 4000);
    char *p;
	char *sa;
    const char *split = ",";
    p = strtok_s(header, split, &sa);
    while (p != NULL) {
        paras[num] = 0; 
        ++num;
        p = strtok_s(NULL, split, &sa);
    }
    num = num - 2;

    cout << "num: " << num << endl;

	//delete sa;
    lineData = datas;

    while (true) {
        trainningData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        for (int j = 0; j < num; ++j) {
            lineData[j] = atof(p);
            move(p);
        }
        lineData += num;
        result[numOfData] = atof(p); 
        ++numOfData;
    }

    memset(paras, 0, sizeof(double) * num);

    Mat B = Mat::ones(1, num, CV_32FC1);
    double *d = new double[num];

    for (int i = 0; i < 2000; ++i) {
        //cout << "i " << i << endl;
        calculateGradient(i, 50, d);
		cout << bias(0, d, datas) << endl;
        double min = bias(0.04, d, datas);
        double minLamda = 0.04;
		double val;
        for (int j = 1; j <= 5; ++j) {
            float lamda = 0.01 * j + 0.04;
            val = bias(lamda, d, datas);
            //cout << "j " << j << endl;
            if (val < min) {
                min = val;
                minLamda = lamda;
            }
        }
		for (int j = 0; j < num; ++j) {
			paras[j] -= minLamda * d[j];
			s[i % 50][j] = -minLamda * d[j];
			cout << ' ' << paras[j];
		}
		cout << endl;
    }

    for (int i = 0; i < num; ++i) {
        cout << paras[i] << ' ' << endl;
    }
        
    delete d;
    return 0;
}
        

