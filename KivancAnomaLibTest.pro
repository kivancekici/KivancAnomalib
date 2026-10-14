QT += widgets concurrent

CONFIG += c++17

TARGET = KivancAnomaLibTest
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060500

SOURCES += \
    anomalibprocess.cpp \
    cameraservice.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    anomalibprocess.h \
    cameraservice.h \
    mainwindow.h

RESOURCES += resources.qrc

# Optional native camera pipeline. The app still builds without these SDKs,
# but enables Basler discovery/capture and OpenCV image writing when present.
PYLON_ROOT = "$$(PYLON_DEV_DIR)"
isEmpty(PYLON_ROOT): PYLON_ROOT = "C:/Program Files/Basler/pylon/Development"
OPENCV_ROOT = C:/opencv/build

win32-msvc* {
    exists($$PYLON_ROOT/include/pylon/PylonIncludes.h):exists($$OPENCV_ROOT/include/opencv2/opencv.hpp) {
        DEFINES += HAVE_PYLON_OPENCV
        INCLUDEPATH += "$$PYLON_ROOT/include" "$$OPENCV_ROOT/include"
        LIBS += -L"$$PYLON_ROOT/lib/x64" \
            -lPylonBase_v12 \
            -lPylonUtility_v12 \
            -lGCBase_MD_VC142_v3_5_Basler_pylon_v1 \
            -lGenApi_MD_VC142_v3_5_Basler_pylon_v1

        CONFIG(debug, debug|release) {
            LIBS += -L"$$OPENCV_ROOT/x64/vc16/lib" -lopencv_world4120d
        } else {
            LIBS += -L"$$OPENCV_ROOT/x64/vc16/lib" -lopencv_world4120
        }
    }
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
