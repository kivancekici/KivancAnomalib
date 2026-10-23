#ifndef CAMERASERVICE_H
#define CAMERASERVICE_H

#include <QString>
#include <QStringList>
#include <QList>
#ifdef INSPECTION_TESTING
#include <functional>
#endif

class CameraService final
{
public:
    struct Device {
        QString id;
        QString label;
    };
    CameraService();
    ~CameraService();

    bool isAvailable() const;
    QString availabilityText() const;
    QList<Device> discover(QString *errorMessage = nullptr) const;
    bool capture(const QString &deviceId, const QString &outputPath,
                 QString *errorMessage = nullptr) const;

private:
    friend class NativeRegressionTest;
    bool m_initialized = false;
#ifdef INSPECTION_TESTING
    std::function<QList<Device>()> m_testDiscover;
    std::function<bool(const QString &, const QString &, QString *)> m_testCapture;
#endif
};

#endif // CAMERASERVICE_H
