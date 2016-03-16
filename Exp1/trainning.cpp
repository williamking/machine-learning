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
    double paras[400], learningRate = 0.03;
    double lineData[400], max[400] = {0}, min[400] = {0};
    int numOfData = 0;
    
    trainningData.getline(header, 4000);
    char *p;
    const char *split = ",";
    int num = 0;
    p = strtok(header, split);
    while (p != NULL) {
        paras[num] = 1; 
        ++num;
        p = strtok(NULL, split);
    }
    num = num - 2;

    cout << "num: " << num << endl;

        while (!trainningData.eof()) {
            trainningData.getline(line, 2500);
            if (strlen(line) == 0) break;
            ++numOfData;
            p = line;
            move(p);
            double preditVal = 0;
            for (int j = 0; j < num; ++j) {
                double value = atof(p);
                if (value > max[j] || j == 0) max[j] = value;
                if (value < min[j] || j == 0) min[j] = value;
                move(p);
            }
        }

    for (int i = 0; i < 30000; ++i) {

        cout << i << endl;
         if (i % 5000 == 0) learningRate *= 0.1;

        trainningData.clear();
        trainningData.seekg(0);

        trainningData.getline(header, 4000);

        double regresses[400] = {0}; 

        while (!trainningData.eof()) {
            trainningData.getline(line, 2500);
            if (strlen(line) == 0) break;
            p = line;
            move(p);
            double preditVal = 0;
            for (int j = 0; j < num; ++j) {
                lineData[j] = atof(p);
                move(p);
                preditVal += paras[j] * lineData[j];
                if (j > 0) lineData[j] = (lineData[j] - min[j]) / (max[j] - min[j]);
            }
            double result = atof(p);
            double weight = preditVal - result;
            for (int j = 0; j < num; ++j) {
                regresses[j] += weight * lineData[j];
            }
        }

        for (int j = 0; j < num; ++j) {
            //if (j != 0) paras[j] *= (1 - 0.03 * 1000 / numOfData);
            paras[j] -= regresses[j] * learningRate / numOfData;
        }

    }

    ifstream testData("../../data-of-machine-learning/test2.csv");
    ofstream preditData("../../data-of-machine-learning/submission8.csv");
    ofstream resultData("../../data-of-machine-learning/reuslt8.csv");

    preditData << "Id,reference" << endl;

    testData.getline(header, 4000);

    int n = 0;

    while (!testData.eof()) {
        testData.getline(line, 2500);
        if (strlen(line) == 0) break;
        p = line;
        move(p);
        double preditVal = 0;
        for (int j = 0; j < num; ++j) {
            lineData[j] = atof(p);
            move(p);
            preditVal += paras[j] * lineData[j];
        }
        preditData << n << ',' << preditVal << endl;
        ++n;
    }  

    for (int i = 0; i < num; ++i) {
        printf("%.4f,", paras[i]);
        resultData << setprecision(2) << paras[i] << ',';
    }

    return 0;
}


