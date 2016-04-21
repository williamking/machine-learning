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

//int data[21777020][580];

double h(double y) {
    return 1.0 / (1 + exp(y));
}

int main() {
    char *buffer;
    long fileLength;
    FILE* file = fopen("../../data-of-machine-learning/train.txt", "rb");
    char header[4000], line[2500];
    double paras[11400], learningRate = 0.5, randomRate = 0.004;
    int numOfData = 0, length = 0;
    double *lineData;
    bool* lab = new bool[21777020];
    char* lines[217777020];

    if (file == NULL) {
        fputs("File error!", stderr);
        exit(3);
    }
    
    fseek(file, 0, SEEK_END);
    fileLength = ftell(file);
    rewind(file);
    buffer = (char*)malloc(sizeof(char*) * fileLength);

    if (buffer == NULL) {
        fputs("Memory error!", stderr);
        exit(2);
    }

    size_t r = fread(buffer, 1, fileLength, file);
    if (r != fileLength) {
        fputs("Reading error!", stderr);
        exit(3);
    }

    fclose(file);

    cout << "asa" << endl;

    char *end = buffer + fileLength - 1;

    lines[0] = buffer;
    numOfData = 1;

    for (long i = 0; i < fileLength; ++i) {
       if (buffer[i] == '\n' && i + 1 < fileLength) lines[numOfData++] = buffer + i + 1;  
    }

    for (long k = 0; k < numOfData; ++k) {
        char *line = lines[k];
        lab[numOfData] = line[0] - '0';
        //cout << i - 1 << endl;
            int l = 0, j = 0;
            int vals[60];
            while (line + j <= end && line[j] != '\n') {
                char index[10];
                int p = 0;
                while (line[j] != ':') {
                    index[p++] = line[j++];
                }
                index[p] = '\0';
                int value = atof(index);
                if (k == 0 && value > length) length = value;
                vals[l++] = value;

                while (line[j] != ' ') ++j;
                ++j;
            }
        for (int j = 0; j < l; ++j) {
            cout << vals[j] << ' ';
        }
        cout << endl;
    }

    cout << numOfData << endl;

    exit(0);

    for (int k = 0; k < 30; ++k) {
        for (int i = 0; i < numOfData; ++i) {
            double preditVal = 0;
            char *line = lines[i];
            int j = 2, it = 0;
            int l = 0;
            int vals[60];
            while (line + j <= end && line[j] != '\n') {
                char index[10];
                int p = 0;
                while (line[j] != ':') {
                    index[p++] = line[j++];
                }
                index[p] = '\0';
                int value = atof(index);
                if (k == 0 && value > length) length = value;
                vals[l++] = value;

                preditVal += paras[value];

                while (line[j] != ' ') ++j;
                ++j;
            }
            double weight = h(preditVal) - lab[i];
            //cout << weight << endl;
            for (j = 0; j < l; ++j) {
                paras[vals[j]] -= weight * 1 * randomRate;
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
            double preditVal = 0;
            int j = 0;
            char *line = lines[i];
            int vals[60], l = 0;
            while (line + j <= end && line[j] != '\n') {
                char index[10];
                int p = 0;
                while (line[j] != ':') {
                    index[p++] = line[j++];
                }
                index[p] = '\0';
                vals[l++] = atof(index);
                while (line[j] != ' ') ++j;
                ++j;
            }
            for (int j = 0; j < l; ++l) {
                preditVal += paras[vals[j]];
            }
            preditVal = h(preditVal);
            double weight = preditVal - lab[i];
            //cout << weight << endl;
            if (lab[i]) {
                loss += -log(preditVal);
            } else {
                loss += -log(1 - preditVal);
            }
            j = 0;
            for (j = 0; j < l; ++l) {
                regresses[vals[j]] += weight;
            }
        }

        loss /= (1 * numOfData);
        cout << loss << endl;

        for (int j = 0; j < length; ++j) {
            //if (j != 0) paras[j] *= (1 - learningRate * 1 / numOfData);
            //if (min[j] != 0 || max[j] != 0) 
            paras[j] -= regresses[j] * learningRate / numOfData;
            //lossData << regresses[j] * learningRate / numOfData;
            //if (j != num - 1) lossData << ",";
            //else lossData << endl;
        }

    }
    

    //free(data);

    ifstream testData("../../data-of-machine-learning/test.txt");
    ofstream resultData("../../data-of-machine-learning/result.csv");

    testData.getline(header, 4000);
    int count = 0;

    resultData << "id,label" << endl;

    int i = 0;
    while (!testData.eof()) {
        testData.getline(line, 2500);
        if (strlen(line) == 0) break;
        int j = 2, i = 0, preditVal = 0;
        int it = 0;
        while (j < strlen(line)) {
            char index[10];
            int p = 0;
            while (line[j] != ':') {
                index[p++] = line[j++];
            }
            line[j] = '\0';
            it = (int)atof(index); 
            preditVal += paras[it];
            while (line[j] != ' ') ++j;
            ++j;
        }
        preditVal = h(preditVal);
        if (preditVal >= 0.5) resultData << i++ << ',' << 1 << endl;
        else resultData << i++ << ',' << 0 << endl;
    }
    delete lab;

    return 0;
}

