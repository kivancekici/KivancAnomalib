#ifndef ANOMALIBPROCESS_H
#define ANOMALIBPROCESS_H

#include <QJsonObject>
#include <QObject>
#include <QProcess>

class AnomalibProcess final : public QObject
{
    Q_OBJECT

public:
    enum class NetworkPolicy {
        Offline,
        AssetDownload,
    };

    explicit AnomalibProcess(QObject *parent = nullptr);

    bool isRunning() const;
    void startJob(const QString &pythonExecutable,
                  const QString &workerScript,
                  const QString &jobFile,
                  NetworkPolicy networkPolicy,
                  const QString &assetRoot);
    void startSession(const QString &pythonExecutable,
                      const QString &workerScript,
                      const QString &assetRoot,
                      const QString &sessionRoot);
    bool sendCommand(const QJsonObject &command);
    void stop();

signals:
    void eventReceived(const QJsonObject &event);
    void outputReceived(const QString &text);
    void jobStarted();
    void jobFinished(int exitCode, QProcess::ExitStatus status);
    void jobFailed(const QString &message);

private slots:
    void readStandardOutput();
    void readStandardError();
    void processFinished(int exitCode, QProcess::ExitStatus status);
    void processError(QProcess::ProcessError error);

private:
    void consumeOutputLines();
    void configureEnvironment(NetworkPolicy networkPolicy,
                              const QString &assetRoot);

    QProcess m_process;
    QByteArray m_stdoutBuffer;
    bool m_sessionMode = false;
};

#endif // ANOMALIBPROCESS_H
