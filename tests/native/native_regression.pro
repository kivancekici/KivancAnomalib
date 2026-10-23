QT += widgets testlib
CONFIG += c++17 testcase console
TARGET = native_regression
DEFINES += INSPECTION_TESTING
INCLUDEPATH += ../..
SOURCES += native_regression.cpp ../../mainwindow.cpp ../../anomalibprocess.cpp ../../cameraservice.cpp
HEADERS += ../../mainwindow.h ../../anomalibprocess.h ../../cameraservice.h
RESOURCES += ../../resources.qrc

# Deliberately omit camera SDK defines and linkage. Tests use synthetic devices.
