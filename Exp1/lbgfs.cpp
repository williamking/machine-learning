#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


void move(char* &p) {
    while (*p != ',' && *p != '\0') {
        if (*p == '\n') return;
        ++p;
     }
    ++p;
}

using namespace std;
using namespace cv;

double data[25002 * 400];
double result[25002];
double predit[25002];
double paras[400] = {0}, learningRate = 0.5, randomRate = 0.004;
double s[400][400], y[400][400];
double lastG[400];
double alpha = 0.05;
int numOfData = 0, num = 0;

void getGK(double *g) {
    memset(g, 0, sizeof(double) * num);
    for (int i = 0; i < numOfData; ++i) {
        double val = 0;
        for (int j = 0; j < num; ++j) {
            val += paras[j] * data[i * num + j];
        }
        for (int j = 0; j < num; ++j) {
            g[j] += (val - result[i]) * data[i * num + j];
        }
    }

    for (int j = 0; j < num; ++j) {
        g[j] /= numOfData;
    }
}

double bias(float lamda, double *d) {
    Mat theta(1, num, CV_32FC1, paras);
    Mat D(1, num, CV_32FC1, d);
    theta -= D;
    for (int i = 0; i < numOfData; ++i) {
        Mat X(1, num, CV_32FC1, data + i * num);
        X.mul(theta);
        Scalar s = sum(X);
        predit[i] = s.val[0];
    }
    Mat preditMat(1, numOfData, CV_32FC1, predit);
    Mat resultMat(1, numOfData, CV_32FC1, result);
    preditMat -= resultMat;
    preditMat.mul(preditMat);
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
            q[i] *= alpha;
            lastG[i] = d[i] = q[i];
        }
        return;
    }
    getGK(q);
    Mat Q(1, num, CV_32FC1, q);
    for (int i = 0; i < num; ++i) {
        s[k % m][i] = data[k * num + i] - data[(k - 1) * num + i];
        y[k % m][i] = q[i] - lastG[i];
        lastG[i] = q[i];
    }

    Mat P(1, 1, CV_32FC1);

    //后向循环
    for (int i = l - 1; i >= 0; --i) {
        int j = i + b;
        Mat Y(1, num, CV_32FC1, y[j % m]);
        Mat S(1, num, CV_32FC1, s[j % m]);

        P = Y.t() * S;
        double p = 1 / P.ptr<float>(0)[0];
        P = S.t() * Q;
        a[i] = p * P.ptr<float>(0)[0];
        Q = Q - a[i] * Y;
    }

    Mat R = Mat::ones(num, num, CV_32FC1) * Q;

    double bj;

    //前向循环
    for (int i = 0; i < l; ++i) {
        int j = i + b;
        Mat Y(1, num, CV_32FC1, s[j % m]);
        Mat S(1, num, CV_32FC1, y[i % m]);

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
    ifstream trainningData("../../data-of-machine-learning/train.csv");
    stringstream dataReader;
    char header[4000], line[2500];
    double max[400] = {0}, min[400] = {0};
    double *lineData;
    memset(data, 0, sizeof(data[0]) * 25002 * 400);

    trainningData.getline(header, 4000);
    char *p;
    const char *split = ",";
    p = strtok(header, split);
    while (p != NULL) {
        paras[num] = 0; 
        ++num;
        p = strtok(NULL, split);
    }
    num = num - 2;

    cout << "num: " << num << endl;

    lineData = data;

    while (true) {
        trainningData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        for (int j = 0; j < num; ++j) {
            data[j] = atof(p);
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
        calculateGradient(i, 50, d);
        double min = bias(0.4, d);
        double minLamda = 0.4;
        for (int j = 1; j <= 5; ++j) {
            float lamda = 0.1 * j;
            double val = bias(lamda, d);
            if (val < min) {
                min = val;
                minLamda = lamda;
            }
        }
        for (int j = 0; j < num; ++j) paras[j] -= minLamda * d[j];
    }

    for (int i = 0; i < num; ++i) {
        cout << paras[i] << ' ' << endl;
    }
        
    delete d;
    return 0;
}
        

