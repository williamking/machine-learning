#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <math.h>

using namespace std;

const char filename[40] = "";

int main() {
	
	FILE* file;
	long fileLength;
	char* buffer;
	size_t result;

	file = fopen("train.csv", "rb");
	if (file != NULL) {
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

	result = fread(buffer, 1, fileLength, file);
	if (result != fileLength) {
		fputs("Reading error!", stderr);
		exit(3);
	}

	flose(file);

}
