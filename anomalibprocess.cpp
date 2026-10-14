#include "anomalibprocess.h"

#include <QDir>
#include <QJsonDocument>
#include <QProcessEnvironment>
#include <QRegularExpression>

AnomalibProcess::AnomalibProcess(QObject *parent)
    : QObject(parent)
{
    connect(&m_process, &QProcess::readyReadStandardOutput,
            this, &AnomalibProcess::readStandardOutput);
    connect(&m_process, &QProcess::readyReadStandardError,
            this, &AnomalibProcess::readStandardError);
    connect(&m_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this, &AnomalibProcess::processFinished);
    connect(&m_process, &QProcess::errorOccurred,
            this, &AnomalibProcess::processError);
}

bool AnomalibProcess::isRunning() const
{
    return m_process.state() != QProcess::NotRunning;
}

void AnomalibProcess::startJob(const QString &pythonExecutable,
                               const QString &workerScript,
                               const QString &jobFile,
                               NetworkPolicy networkPolicy,
                               const QString &assetRoot)
{
    if (isRunning()) {
        emit jobFailed(tr("An Anomalib job is already running."));
        return;
    }

    m_stdoutBuffer.clear();
    m_sessionMode = false;
    configureEnvironment(networkPolicy, assetRoot);
    m_process.setProgram(pythonExecutable);
    m_process.setArguments({QStringLiteral("-u"), workerScript,
                            QStringLiteral("--job"), jobFile});
    m_process.setProcessChannelMode(QProcess::SeparateChannels);
    m_process.start();

    if (!m_process.waitForStarted(3000)) {
        emit jobFailed(tr("Could not start Python: %1").arg(m_process.errorString()));
        return;
    }
    emit jobStarted();
}

void AnomalibProcess::configureEnvironment(NetworkPolicy networkPolicy,
                                            const QString &assetRoot)
{
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    if (!environment.contains(QStringLiteral("INSPECTION_SHARED_HF_CACHE"))) {
        QString sharedCache =
            environment.value(QStringLiteral("HF_HUB_CACHE")).trimmed();
        if (sharedCache.isEmpty()) {
            const QString sharedHome =
                environment.value(QStringLiteral("HF_HOME")).trimmed();
            if (!sharedHome.isEmpty()) {
                sharedCache =
                    QDir(sharedHome).filePath(QStringLiteral("hub"));
            }
        }
        if (!sharedCache.isEmpty()) {
            environment.insert(
                QStringLiteral("INSPECTION_SHARED_HF_CACHE"),
                QDir::cleanPath(sharedCache));
        }
    }
    environment.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("HF_HUB_DISABLE_TELEMETRY"),
                       QStringLiteral("1"));
    environment.insert(QStringLiteral("HF_HUB_DISABLE_IMPLICIT_TOKEN"),
                       QStringLiteral("1"));
    environment.insert(QStringLiteral("DISABLE_TELEMETRY"), QStringLiteral("1"));
    environment.insert(QStringLiteral("DO_NOT_TRACK"), QStringLiteral("1"));
    environment.insert(QStringLiteral("MODEL_ASSET_ROOT"), assetRoot);
    const QDir root(assetRoot);
    environment.insert(QStringLiteral("HF_HOME"),
                       root.filePath(QStringLiteral("huggingface")));
    environment.insert(QStringLiteral("HF_HUB_CACHE"),
                       root.filePath(QStringLiteral("huggingface/hub")));
    environment.insert(QStringLiteral("TORCH_HOME"),
                       root.filePath(QStringLiteral("torch")));
    if (networkPolicy == NetworkPolicy::Offline) {
        environment.insert(QStringLiteral("HF_HUB_OFFLINE"), QStringLiteral("1"));
        environment.insert(QStringLiteral("HF_DATASETS_OFFLINE"),
                           QStringLiteral("1"));
        environment.insert(QStringLiteral("TRANSFORMERS_OFFLINE"),
                           QStringLiteral("1"));
    } else {
        environment.remove(QStringLiteral("HF_HUB_OFFLINE"));
        environment.remove(QStringLiteral("HF_DATASETS_OFFLINE"));
        environment.remove(QStringLiteral("TRANSFORMERS_OFFLINE"));
    }

    m_process.setProcessEnvironment(environment);
}

void AnomalibProcess::startSession(const QString &pythonExecutable,
                                   const QString &workerScript,
                                   const QString &assetRoot,
                                   const QString &sessionRoot)
{
    if (isRunning()) {
        return;
    }

    m_stdoutBuffer.clear();
    m_sessionMode = true;
    configureEnvironment(NetworkPolicy::Offline, assetRoot);
    m_process.setProgram(pythonExecutable);
    m_process.setArguments({QStringLiteral("-u"), workerScript,
                            QStringLiteral("--serve"),
                            QStringLiteral("--asset-root"), assetRoot,
                            QStringLiteral("--session-root"), sessionRoot});
    m_process.setProcessChannelMode(QProcess::SeparateChannels);
    m_process.start();
    if (!m_process.waitForStarted(3000)) {
        m_sessionMode = false;
        emit jobFailed(tr("Could not start Python: %1").arg(m_process.errorString()));
        return;
    }
    emit jobStarted();
}

bool AnomalibProcess::sendCommand(const QJsonObject &command)
{
    if (!isRunning() || !m_sessionMode) {
        emit jobFailed(tr("The inference session is not running."));
        return false;
    }
    QByteArray payload = QJsonDocument(command).toJson(QJsonDocument::Compact);
    payload.append('\n');
    return m_process.write(payload) == payload.size();
}

void AnomalibProcess::stop()
{
    if (!isRunning()) {
        return;
    }
    if (m_sessionMode) {
        QJsonObject shutdown {
            {QStringLiteral("request_id"), QStringLiteral("shutdown")},
            {QStringLiteral("command"), QStringLiteral("shutdown")},
        };
        QByteArray payload = QJsonDocument(shutdown).toJson(QJsonDocument::Compact);
        payload.append('\n');
        m_process.write(payload);
        m_process.waitForBytesWritten(500);
        if (m_process.waitForFinished(1500)) {
            m_sessionMode = false;
            return;
        }
    }
    m_process.terminate();
    if (!m_process.waitForFinished(2000)) {
        m_process.kill();
    }
    m_sessionMode = false;
}

void AnomalibProcess::readStandardOutput()
{
    m_stdoutBuffer += m_process.readAllStandardOutput();
    consumeOutputLines();
}

void AnomalibProcess::readStandardError()
{
    QByteArray bytes = m_process.readAllStandardError();
    bytes.replace(QByteArray(1, '\0'), QByteArray());
    QString text = QString::fromUtf8(bytes).trimmed();
    text.remove(QRegularExpression(QStringLiteral("\\x1b\\[[0-9;]*m")));
    if (!text.isEmpty()) {
        emit outputReceived(text);
    }
}

void AnomalibProcess::processFinished(int exitCode, QProcess::ExitStatus status)
{
    consumeOutputLines();
    m_sessionMode = false;
    emit jobFinished(exitCode, status);
}

void AnomalibProcess::processError(QProcess::ProcessError error)
{
    Q_UNUSED(error)
    emit jobFailed(m_process.errorString());
}

void AnomalibProcess::consumeOutputLines()
{
    qsizetype newline = -1;
    while ((newline = m_stdoutBuffer.indexOf('\n')) >= 0) {
        QByteArray line = m_stdoutBuffer.left(newline).trimmed();
        m_stdoutBuffer.remove(0, newline + 1);
        if (line.isEmpty()) {
            continue;
        }

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
        if (parseError.error == QJsonParseError::NoError && document.isObject()) {
            emit eventReceived(document.object());
        } else {
            emit outputReceived(QString::fromUtf8(line));
        }
    }
}
