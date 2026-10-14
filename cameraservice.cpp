#include "cameraservice.h"

#ifdef HAVE_PYLON_OPENCV
#include <pylon/PylonIncludes.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

CameraService::CameraService()
{
#ifdef HAVE_PYLON_OPENCV
    Pylon::PylonInitialize();
    m_initialized = true;
#endif
}

CameraService::~CameraService()
{
#ifdef HAVE_PYLON_OPENCV
    if (m_initialized) {
        Pylon::PylonTerminate();
    }
#endif
}

bool CameraService::isAvailable() const
{
    return m_initialized;
}

QString CameraService::availabilityText() const
{
#ifdef HAVE_PYLON_OPENCV
    return QStringLiteral("Basler Pylon + OpenCV ready");
#else
    return QStringLiteral("Camera SDKs were not linked in this build");
#endif
}

QStringList CameraService::discover(QString *errorMessage) const
{
    QStringList devices;
#ifdef HAVE_PYLON_OPENCV
    try {
        Pylon::DeviceInfoList_t info;
        Pylon::CTlFactory::GetInstance().EnumerateDevices(info);
        for (size_t index = 0; index < info.size(); ++index) {
            const QString model = QString::fromUtf8(info[index].GetModelName().c_str());
            const QString serial = QString::fromUtf8(info[index].GetSerialNumber().c_str());
            devices << QStringLiteral("%1  ·  %2").arg(model, serial);
        }
    } catch (const Pylon::GenericException &exception) {
        if (errorMessage) {
            *errorMessage = QString::fromUtf8(exception.GetDescription());
        }
    }
#else
    if (errorMessage) {
        *errorMessage = availabilityText();
    }
#endif
    return devices;
}

bool CameraService::capture(int deviceIndex, const QString &outputPath,
                            QString *errorMessage) const
{
#ifdef HAVE_PYLON_OPENCV
    try {
        Pylon::DeviceInfoList_t info;
        Pylon::CTlFactory::GetInstance().EnumerateDevices(info);
        if (deviceIndex < 0 || static_cast<size_t>(deviceIndex) >= info.size()) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("Select an available Basler camera.");
            }
            return false;
        }

        Pylon::CInstantCamera camera(
            Pylon::CTlFactory::GetInstance().CreateDevice(info[deviceIndex]));
        camera.Open();
        camera.StartGrabbing(1, Pylon::GrabStrategy_OneByOne,
                             Pylon::GrabLoop_ProvidedByUser);

        Pylon::CGrabResultPtr result;
        camera.RetrieveResult(5000, result, Pylon::TimeoutHandling_ThrowException);
        if (!result || !result->GrabSucceeded()) {
            if (errorMessage) {
                *errorMessage = result
                    ? QString::fromUtf8(result->GetErrorDescription().c_str())
                    : QStringLiteral("The camera did not return an image.");
            }
            return false;
        }

        Pylon::CImageFormatConverter converter;
        converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
        converter.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
        Pylon::CPylonImage converted;
        converter.Convert(converted, result);

        cv::Mat frame(static_cast<int>(converted.GetHeight()),
                      static_cast<int>(converted.GetWidth()),
                      CV_8UC3, converted.GetBuffer());
        if (!cv::imwrite(outputPath.toStdString(), frame)) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("OpenCV could not write the captured image.");
            }
            return false;
        }
        return true;
    } catch (const Pylon::GenericException &exception) {
        if (errorMessage) {
            *errorMessage = QString::fromUtf8(exception.GetDescription());
        }
        return false;
    } catch (const cv::Exception &exception) {
        if (errorMessage) {
            *errorMessage = QString::fromUtf8(exception.what());
        }
        return false;
    }
#else
    Q_UNUSED(deviceIndex)
    Q_UNUSED(outputPath)
    if (errorMessage) {
        *errorMessage = availabilityText();
    }
    return false;
#endif
}
