#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <math.h>

using namespace std;

void move(char* &p) {
    while (*p != ',' && *p != '\0') {
        if (*p == '\n') return;
        ++p;
     }
    ++p;
}

double data[25002 * 400];

double h(double y) {
    return 1.0 / (1 + exp(y));
}

int main() {
    ifstream trainningData("../../data-of-machine-learning/train.csv");
    stringstream dataReader;
    char header[4000], line[2500];
    double paras[400], learningRate = 0.5, randomRate = 0.004;
    double max[400] = {0}, min[400] = {0};
    int numOfData = 0;
    double *lineData;
    memset(data, 0, sizeof(data[0]) * 25002 * 400);
    
    trainningData.getline(header, 4000);
    char *p;
    const char *split = ",";
    int num = 0;
    p = strtok(header, split);
    while (p != NULL) {
        paras[num] = 0; 
        ++num;
        p = strtok(NULL, split);
    }
    num = num - 2;

    cout << "num: " << num << endl;

    bool init = true;
    int ct = 0; 

        while (!trainningData.eof()) {
            trainningData.getline(line, 2500);
            if (strlen(line) == 0) break;
            p = line;
            move(p);
            double preditVal = 0;
            for (int j = 0; j < num + 1; ++j) {
                double value = atof(p);
                if (value > max[j] || init) max[j] = value;
                if (value < min[j] || init) min[j] = value;
                move(p);
                data[numOfData * (num + 1) + j] = value;
            }
            init = false;
            ++numOfData;
        }

        cout << numOfData << endl;

    for (int i = 0; i < numOfData; ++i) {
        lineData = data + i * (num + 1);
        for (int j = 0; j < num; ++j) {
            if (max[j] - min[j] != 0 && false) {
                //lineData[j] = (lineData[j] - min[j]) / (max[j] - min[j]);
            }
            //cout << lineData[j] << endl;
        }
    }


    ofstream lossData("../../data-of-machine-learning/loss_0.04.csv");

    for (int k = 0; k < 30; ++k) {

    for (int i = 0; i < numOfData; ++i) {
        double preditVal = 0;
        lineData = data + i * (num + 1);
        for (int j = 0; j < num; ++j) {
            preditVal += paras[j] * lineData[j] * lineData[j];
        }
        double weight = h(preditVal) - lineData[num];
        //cout << weight << endl;
        for (int j = 0; j < num; ++j) {
            paras[j] -= weight * lineData[j] * randomRate;
        }
    }
    }

    double lastLoss = 1000;

    for (int k = 0; k < 300000; ++k) {

        cout << k << endl;
        if (k % 20000 == 0) learningRate *= 0.7;

        double regresses[400] = {0}; 

        int count = 0;

        double loss = 0;

        for (int i = 0; i < numOfData; ++i) {
            lineData = data + i * (num + 1);
            double preditVal = 0;
            for (int j = 0; j < num; ++j) {
                preditVal += paras[j] * lineData[j] * lineData[j];
            }
            double weight = h(preditVal) - lineData[num];
            //cout << weight << endl;
            loss += weight * weight;
            for (int j = 0; j < num; ++j) {
                regresses[j] += weight * lineData[j];
            }
        }

        loss /= (1 * numOfData);
        loss = sqrt(loss);
        lossData << loss << endl;
        cout << loss << endl;
        lastLoss = loss;

        for (int j = 0; j < num; ++j) {
            //if (j != 0) paras[j] *= (1 - learningRate * 1 / numOfData);
            if (min[j] != 0 || max[j] != 0) paras[j] -= regresses[j] * learningRate / numOfData;
            //lossData << regresses[j] * learningRate / numOfData;
            //if (j != num - 1) lossData << ",";
            //else lossData << endl;
        }

    }
    

    //free(data);

    ifstream testData("../../data-of-machine-learning/test2.csv");
    ofstream preditData("../../data-of-machine-learning/submission.csv");
    ofstream resultData("../../data-of-machine-learning/reuslt_0.04.csv");

    testData.getline(header, 4000);
    init = true;
    int count = 0;

    preditData << "Id,reference" << endl;

    while (!testData.eof()) {
        testData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        double preditVal = 0;
        double value;
        for (int j = 0; j < num; ++j) {
            value = atof(p);
            move(p);
            preditVal += paras[j] * value * value;
        }
        preditData << count << ',' << preditVal << endl;
        init = false;
        ++count;
    }

    int n = 0;


    double loss = 0;
    count = 0;

    for (int i = 0; i < numOfData; ++i) {
        lineData = data + i * (num + 1);
        double preditVal = 0;
        for (int j = 0; j < num; ++j) {
            preditVal += paras[j] * lineData[j] * lineData[j];
        }
        double result = lineData[num];
        loss += (preditVal - result) * (preditVal - result);
        //preditData << n << ',' << preditVal << endl;
        ++n;
    }  

    loss /= numOfData;
    loss = sqrt(loss);


    for (int i = 0; i < num; ++i) {
        printf("%.4f,", paras[i]);
        resultData << setprecision(2) << paras[i] << ',';
    }

    printf("\n%.4f\n", loss);

    return 0;
}


