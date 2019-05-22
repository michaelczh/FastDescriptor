//
// Created by czh on 5/11/19.
//

#include <DebugExporter.h>
#include <iostream>

int main(){
    DebugFileExporter exporter("./test.txt", false);
    exporter.insertLine("hello1");
    exporter.exportToPath();

    getchar();
    DebugFileExporter exporter2("./test.txt",false);
    exporter2.insertLine("hello2");
    exporter2.exportToPath();
    return 0;
}