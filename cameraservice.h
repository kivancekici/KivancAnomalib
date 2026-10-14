#ifndef CAMERASERVICE_H
#define CAMERASERVICE_H

#include <QString>
#include <QStringList>

class CameraService final
{
public:
    CameraService();
    ~CameraService();

    bool isAvailable() const;
    QString availabilityText() const;
    QStringList discover(QString *errorMessage = nullptr) const;
    bool capture(int deviceIndex, const QString &outputPath,
                 QString *errorMessage = nullptr) const;

private:
    bool m_initialized = false;
};

#endif // CAMERASERVICE_H
