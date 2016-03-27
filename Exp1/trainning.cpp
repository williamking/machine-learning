#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>

using namespace std;

void move(char* &p) {
    while (*p != ',' && *p != '\n') {
        ++p;
     }
    ++p;
}

int main() {
    ifstream trainningData("train.csv");
    stringstream dataReader;
    char header[4000], line[2500];
    double paras[400], learningRate = 0.03;
    double lineData[400];
    int numOfData = 0;
    
    trainningData.getline(header, 4000);
    char *p;
    const char *split = ",";
    int num = 0;
	p = header;
    while (p != NULL) {
        paras[num] = 1; 
        ++num;
		move(p);
    }
    num = num - 2;

    cout << "num: " << num << endl;

    for (int i = 0; i < 100; ++i) {

        trainningData.seekg(0);

        trainningData.getline(header, 4000);

        double regresses[400] = {0}; 

        numOfData = 0;

        while (!trainningData.eof()) {
            ++numOfData;
            trainningData.getline(line, 2500);
            p = line;
            move(p);
            double preditVal = 0;
            for (int j = 0; j < num; ++j) {
                lineData[j] = atof(p);
                //cout << lineData[j] << ',';
                move(p);
                preditVal += paras[j] * lineData[j];
            }
            double result = atof(p);
            double weight = preditVal - result;
            for (int j = 0; j < num; ++j) {
                regresses[j] += weight * lineData[j];
            }
        }

        for (int j = 0; j < num; ++j) {
            paras[j] -= regresses[j] * learningRate / numOfData;
        }

    }

    for (int i = 0; i < num; ++i) {
        cout << paras[i] << ',';
    }

    return 0;
}


