#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <iomanip>

using namespace std;

void move(char* &p) {
    while (*p != ',' && *p != '\0') {
        if (*p == '\n') return;
        ++p;
     }
    ++p;
}

int main() {
    ifstream trainningData("../../data-of-machine-learning/train.csv");
    stringstream dataReader;
    char header[4000], line[2500];
    double paras[400], learningRate = 0.003;
    double lineData[400], max[400] = {0}, min[400] = {0};
    int numOfData = 0;
    
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
            //if (ct++ >= 10) break;
            trainningData.getline(line, 2500);
            if (strlen(line) == 0) break;
            ++numOfData;
            p = line;
            move(p);
            double preditVal = 0;
            for (int j = 0; j < num; ++j) {
                double value = atof(p);
                if (value > max[j] || init) max[j] = value;
                if (value < min[j] || init) min[j] = value;
                move(p);
            }
            init = false;
        }

        cout << numOfData << endl;

    for (int i = 0; i < 1000; ++i) {
        cout << i << endl;
    trainningData.clear();
    trainningData.seekg(0);

    trainningData.getline(header, 4000);

    ct = 0;

    while (!trainningData.eof()) {
        //if (ct++ > 30) break;
        trainningData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        double preditVal = 0;
        for (int j = 0; j < num; ++j) {
            lineData[j] = atof(p);
            move(p);
            if (j > 0 && max[j] - min[j] != 0) lineData[j] = (lineData[j] - min[j]) / (max[j] - min[j]);
            preditVal += paras[j] * lineData[j];
        }
        double weight = preditVal - atof(p);
        for (int j = 0; j < num; ++j) {
            paras[j] -= weight * lineData[j] * learningRate;
        }
    }
    }


    ifstream testData("../../data-of-machine-learning/train.csv");
    ofstream preditData("../../data-of-machine-learning/submission.csv");
    ofstream resultData("../../data-of-machine-learning/reuslt_0.05.csv");

    testData.getline(header, 4000);
    init = true;
    int count = 0;

    while (!testData.eof()) {
        //if (count++ >= 10) break;
        testData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        double preditVal = 0;
        double value;
        for (int j = 0; j < num; ++j) {
            value = atof(p);
            move(p);
            if (value > max[j] || init) max[j] = value;
            if (value < min[j] || init) min[j] = value;
        }
        init = false;
    }
    
    testData.clear();
    testData.seekg(0);

    testData.getline(header, 4000);
    int n = 0;

    preditData << "Id,reference" << endl;

    double loss = 0;
    count = 0;

    while (!testData.eof()) {
        //if (count++ >= 10) break;
        testData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        double preditVal = 0;
        for (int j = 0; j < num; ++j) {
            lineData[j] = atof(p);
            if (j > 0 && max[j] - min[j] != 0) lineData[j] = (lineData[j] - min[j]) / (max[j] - min[j]);
            move(p);
            preditVal += paras[j] * lineData[j];
        }
        double result = atof(p);
        loss += (preditVal - result) * (preditVal - result);
        preditData << n << ',' << preditVal << endl;
        ++n;
    }  

    loss /= (2 * numOfData);


    for (int i = 0; i < num; ++i) {
        printf("%.4f,", paras[i]);
        resultData << setprecision(2) << paras[i] << ',';
    }

    printf("\n%.4f\n", loss);

    return 0;
}


