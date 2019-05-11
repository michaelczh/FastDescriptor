//
// Created by czh on 3/11/19.
//

#ifndef RECONSTRUCTION_DEBUGFILEEXPORTER_H
#define RECONSTRUCTION_DEBUGFILEEXPORTER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace std;

class DebugFileExporter {

public:
    DebugFileExporter(string path, bool overWrite);
    void insertLine(string line);
    void exportToPath();

private:
    bool isFileExist(string path);
    string _path;
    std::ofstream output;
    bool _overWrite;


};


DebugFileExporter::DebugFileExporter(string path, bool overWrite = true) {

    if (!isFileExist(path) || overWrite) {
        this->output.open(path);
        this->output.close();
        cerr << "[DebugFileExporter] file does not exist, create file\n";
    }

    //this->_overWrite = overWrite;
    this->_path = path;
    this->output.open(path, std::ios::app);
}

void DebugFileExporter::insertLine(string line) {
    this->output << line << "\n";
}

void DebugFileExporter::exportToPath() {
    this->output.close();
}

bool DebugFileExporter::isFileExist(string path) {
    std::ifstream infile(path);
    return infile.good();
}

#endif //RECONSTRUCTION_DEBUGFILEEXPORTER_H
