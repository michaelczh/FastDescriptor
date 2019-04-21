//
// Created by czh on 3/11/19.
//

#ifndef RECONSTRUCTION_DEBUGFILEEXPORTER_H
#define RECONSTRUCTION_DEBUGFILEEXPORTER_H

#include <iostream>
#include <vector>
#include <string>
using namespace std;

class DebugFileExporter {

public:
    DebugFileExporter(string path);
    void insertLine(string line);
    void exportToPath();

private:
    string _path;
    std::ofstream output;
};


DebugFileExporter::DebugFileExporter(string path) {
    this->_path = path;
    this->output = ofstream(path);
}

void DebugFileExporter::insertLine(string line) {
    this->output << line << "\n";
}

void DebugFileExporter::exportToPath() {
    this->output.close();
}

#endif //RECONSTRUCTION_DEBUGFILEEXPORTER_H
