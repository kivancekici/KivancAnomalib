#include "mainwindow.h"
#include "anomalibprocess.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>
#include <QtEndian>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
bool writeFile(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

QByteArray readFile(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

QString fingerprint(const QString &path)
{
    return QString::fromLatin1(QCryptographicHash::hash(readFile(path), QCryptographicHash::Sha256).toHex());
}

void emitEvent(QJsonObject event)
{
    std::cout << QJsonDocument(event).toJson(QJsonDocument::Compact).constData() << std::endl;
}

// The test executable also acts as a tiny line-protocol worker. It never imports
// Python, accesses a camera, or loads the fixture checkpoint.
int fakeWorker(int argc, char **argv)
{
    bool serve = false;
    for (int index = 0; index < argc; ++index) {
        serve = serve || QByteArray(argv[index]) == "--serve";
    }
    if (!serve) {
        emitEvent({{"event", "error"}, {"message", "Unexpected batch/demo worker"}});
        return 2;
    }
    emitEvent({{"event", "session_ready"}, {"pid", 1}});
    std::string line;
    while (std::getline(std::cin, line)) {
        const QJsonObject command = QJsonDocument::fromJson(QByteArray::fromStdString(line)).object();
        const QString action = command.value("command").toString();
        if (action == "shutdown") {
            return 0;
        }
        if (action == "load_model") {
            emitEvent({{"event", "model_loaded"}, {"request_id", command.value("request_id")},
                       {"selected_path", command.value("path")}, {"fingerprint", "fixture"},
                       {"model", "PatchCore"}, {"backend", "Fixture runtime"},
                       {"device_name", "Fixture CPU"}, {"format", "checkpoint"},
                       {"saved_sigma", 4}, {"sigma_override_supported", true},
                       {"model_parameters", QJsonObject{{"PatchCore", QJsonObject{
                            {"image_size", QJsonObject{{"width", 32}, {"height", 32}}}, {"sigma", 4}}}}}});
        } else if (action == "predict") {
            emitEvent({{"event", "inference"}, {"request_id", command.value("request_id")},
                       {"model", "PatchCore"}, {"demo", false}, {"label", "normal"},
                       {"anomaly_score", 0.123}, {"image_path", command.value("image_path")},
                       {"total_latency_ms", 1.0}, {"model_latency_ms", 0.5}});
        }
    }
    return 0;
}
}

class NativeRegressionTest : public QObject
{
    Q_OBJECT
private:
    QTemporaryDir m_storage;

    void configure(MainWindow &window, const QString &name)
    {
        window.m_testDataDirectory = QDir(m_storage.path()).filePath(name);
        QDir().mkpath(window.m_testDataDirectory);
        window.m_pythonEdit->setText(QCoreApplication::applicationFilePath());
    }

    QJsonObject realResult(const QString &checkpoint)
    {
        return {{"model", "PatchCore"}, {"demo", false}, {"completed", true},
                {"checkpoint", checkpoint}, {"checkpoint_sha256", fingerprint(checkpoint)},
                {"dataset_root", "original-training-dataset"}, {"run_id", "experiment-123"},
                {"split_seed", 42}, {"image_auroc", 0.9}};
    }

private slots:
    void initTestCase()
    {
        QVERIFY(m_storage.isValid());
        QCoreApplication::setOrganizationName("InspectionNativeRegression");
        QCoreApplication::setApplicationName(QUuid::createUuid().toString(QUuid::Id128));
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, m_storage.path());
        QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope, m_storage.path());
    }

    void init()
    {
        QSettings().clear();
    }

    void realLoadedModelIgnoresGlobalDemoToggle()
    {
        MainWindow window;
        configure(window, "runtime-mode");
        const QString model = QDir(window.dataDirectory()).filePath("fixture.ckpt");
        const QString image = QDir(window.dataDirectory()).filePath("input.png");
        QVERIFY(writeFile(model, "fixture model"));
        QImage input(8, 8, QImage::Format_RGB32);
        input.fill(Qt::black);
        QVERIFY(input.save(image));
        window.m_demoMode->setChecked(true);
        window.loadInferenceModel(model);
        QTRY_VERIFY_WITH_TIMEOUT(window.m_inferenceModelReady, 5000);
        QCOMPARE(window.m_inferenceMode, MainWindow::InferenceMode::Runtime);
        QVERIFY(window.m_inferenceModelDetails->text().contains("Fixture runtime"));
        window.m_inferenceImageEdit->setText(image);
        window.runInference();
        QTRY_COMPARE_WITH_TIMEOUT(window.m_inferenceScore->text(), QString("0.123"), 5000);
        QCOMPARE(window.m_inferenceStatus->text(), QString("NORMAL"));
        QVERIFY(!window.m_worker->isRunning());
        QVERIFY(!QDir(QDir(window.dataDirectory()).filePath("jobs")).exists());
        window.m_demoMode->setChecked(false);
        window.m_demoMode->setChecked(true);
        QCOMPARE(window.m_inferenceMode, MainWindow::InferenceMode::Runtime);
        window.loadInferenceModel(QDir(window.dataDirectory()).filePath("missing.ckpt"));
        QCOMPARE(window.m_inferenceMode, MainWindow::InferenceMode::Runtime);
    }

    void demonstrationResultsAreLabelled()
    {
        MainWindow window;
        configure(window, "demo-label");
        window.updateInferenceResult({{"demo", true}, {"label", "normal"},
                                      {"model", "PatchCore"}, {"anomaly_score", 0.2}});
        QVERIFY(window.m_inferenceStatus->text().contains("DEMONSTRATION"));
        QVERIFY(window.m_statusMode->text().contains("Simulated"));
    }

    void promotionRejectsInvalidArtifactsWithoutTouchingManifest()
    {
        MainWindow window;
        configure(window, "promotion-invalid");
        const QString manifest = QDir(window.dataDirectory()).filePath("production_model.json");
        const QByteArray original = "existing production selection";
        QVERIFY(writeFile(manifest, original));
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
        const QString checkpoint = QDir(window.dataDirectory()).filePath("model.ckpt");
        QVERIFY(writeFile(checkpoint, "original model"));
        QJsonObject result = realResult(checkpoint);
        result.insert("demo", true);
        window.updateResultRow(result);
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
        result.remove("demo");
        window.updateResultRow(result);
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
        result = realResult(checkpoint);
        result.insert("completed", false);
        window.updateResultRow(result);
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
        result = realResult(checkpoint);
        window.updateResultRow(result);
        QVERIFY(writeFile(checkpoint, "modified model"));
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
        QVERIFY(QFile::remove(checkpoint));
        window.promoteWinner();
        QCOMPARE(readFile(manifest), original);
    }

    void promotionUsesImmutableResultProvenance()
    {
        MainWindow window;
        configure(window, "promotion-valid");
        const QString checkpoint = QDir(window.dataDirectory()).filePath("model.ckpt");
        QVERIFY(writeFile(checkpoint, "verified model"));
        window.updateResultRow(realResult(checkpoint));
        window.m_datasetRoot = "a-different-current-dataset";
        window.promoteWinner();
        const QJsonObject manifest = QJsonDocument::fromJson(readFile(
            QDir(window.dataDirectory()).filePath("production_model.json"))).object();
        QCOMPARE(manifest.value("dataset_root").toString(), QString("original-training-dataset"));
        QCOMPARE(manifest.value("run_id").toString(), QString("experiment-123"));
        QCOMPARE(manifest.value("split_seed").toInt(), 42);
        QCOMPARE(manifest.value("artifact").toObject().value("sha256").toString(), fingerprint(checkpoint));
        window.updateWinner("PatchCore");
        QVERIFY(window.m_promoteButton->isEnabled());
        window.handleWorkerEvent({{"event", "finished"}, {"winner", ""},
                                  {"comparison_available", false}});
        QVERIFY(window.m_winnerModel.isEmpty());
        QVERIFY(!window.m_promoteButton->isEnabled());
    }

    void concurrentJobsPreserveOwnershipAndStopControl()
    {
        MainWindow window;
        configure(window, "job-ownership");
        window.m_demoMode->setChecked(true);
        window.m_worker->startSession(QCoreApplication::applicationFilePath(), "fixture",
                                      window.dataDirectory(), window.dataDirectory());
        QVERIFY(window.m_worker->isRunning());
        window.m_activeJob = "train";
        window.setJobRunning(true);
        window.exportWinner("ONNX");
        QCOMPARE(window.m_activeJob, QString("train"));
        QVERIFY(window.m_worker->isRunning());
        QVERIFY(!window.m_stopButton->isHidden());
        QVERIFY(!window.m_trainButton->isEnabled());
        window.runWorkerJob({{"mode", "check"}}, "check");
        QCOMPARE(window.m_activeJob, QString("train"));
        QVERIFY(!QDir(QDir(window.dataDirectory()).filePath("jobs")).exists());
        QSignalSpy failures(window.m_worker, &AnomalibProcess::jobFailed);
        window.m_worker->startJob("unused", "unused", "unused",
                                 AnomalibProcess::NetworkPolicy::Offline, window.dataDirectory());
        QCOMPARE(failures.count(), 0);
        QCOMPARE(window.m_activeJob, QString("train"));
    }

    void capturePreservesExistingClassAliasesEvenWhenCameraFails()
    {
        MainWindow window;
        configure(window, "capture-aliases");
        const QString dataset = QDir(window.dataDirectory()).filePath("dataset");
        QVERIFY(QDir().mkpath(QDir(dataset).filePath("good")));
        QVERIFY(QDir().mkpath(QDir(dataset).filePath("anomalous")));
        window.m_captureDatasetEdit->setText(dataset);
        window.m_cameraService.m_testDiscover = [] { return QList<CameraService::Device>{}; };
        window.captureNormal();
        window.captureAnomalous();
        QVERIFY(!QDir(QDir(dataset).filePath("normal")).exists());
        QVERIFY(!QDir(QDir(dataset).filePath("abnormal")).exists());
        QCOMPARE(window.detectedNormalFolder(dataset), QString("good"));
        QCOMPARE(window.detectedAbnormalFolder(dataset), QString("anomalous"));
        QCOMPARE(QFileInfo(window.nextCapturePath("normal")).dir().dirName(), QString("good"));
        QCOMPARE(QFileInfo(window.nextCapturePath("abnormal")).dir().dirName(), QString("anomalous"));
    }

    void cameraSelectionSurvivesReorderingAndRejectsDisappearance()
    {
        MainWindow window;
        configure(window, "camera-identities");
        QList<CameraService::Device> devices{{"serial-A", "Camera A"}, {"serial-B", "Camera B"}};
        QString captured;
        window.m_cameraService.m_testDiscover = [&] { return devices; };
        window.m_cameraService.m_testCapture = [&](const QString &id, const QString &, QString *) {
            captured = id;
            return true;
        };
        window.discoverCameras();
        QCOMPARE(window.m_cameraCombo->currentData().toString(), QString("serial-A"));
        devices = {{"serial-B", "Camera B"}, {"serial-A", "Camera A"}};
        QString error;
        QVERIFY(window.m_cameraService.capture(window.m_cameraCombo->currentData().toString(), "unused", &error));
        QCOMPARE(captured, QString("serial-A"));
        window.discoverCameras();
        QCOMPARE(window.m_cameraCombo->currentIndex(), 1);
        captured.clear();
        devices = {{"serial-B", "Camera B"}};
        QVERIFY(!window.m_cameraService.capture(window.m_cameraCombo->currentData().toString(), "unused", &error));
        QVERIFY(captured.isEmpty());
        QVERIFY(error.contains("unavailable"));
        window.discoverCameras();
        QCOMPARE(window.m_cameraCombo->currentIndex(), -1);
    }

    void thresholdUsesCalibratedFloatValuesWithoutQuantization()
    {
        MainWindow window;
        configure(window, "float-map");
        const QString path = QDir(window.dataDirectory()).filePath("anomaly-map.f32");
        const QList<float> values{0.1f, 0.2f, 0.59999f, 0.60001f, 0.9f, 0.9f};
        QByteArray bytes(values.size() * 4, '\0');
        for (int index = 0; index < values.size(); ++index) {
            quint32 bits;
            const float value = values[index];
            std::memcpy(&bits, &value, sizeof(bits));
            qToLittleEndian<quint32>(bits, bytes.data() + index * 4);
        }
        QVERIFY(writeFile(path, bytes));
        const QJsonObject metadata{{"anomaly_map_float_path", path},
                                   {"anomaly_map_width", 3}, {"anomaly_map_height", 2}};
        QVERIFY(window.loadAnomalyMap(metadata));
        QImage mask = window.thresholdedAnomalyMap(60);
        QCOMPARE(mask.size(), QSize(3, 2));
        for (int index = 0; index < values.size(); ++index) {
            QCOMPARE(mask.pixel(index % 3, index / 3), index < 3 ? qRgb(0, 0, 0) : qRgb(239, 55, 52));
        }
        QVERIFY(writeFile(path, bytes.left(bytes.size() - 1)));
        QVERIFY(!window.loadAnomalyMap(metadata));
        QVERIFY(window.thresholdedAnomalyMap(60).isNull());
        quint32 nonFinite = 0x7fc00000;
        qToLittleEndian<quint32>(nonFinite, bytes.data());
        QVERIFY(writeFile(path, bytes));
        QVERIFY(!window.loadAnomalyMap(metadata));
    }

    void modelConstraintsMatchSupportedRuntimeConfigurations()
    {
        MainWindow window;
        configure(window, "validation");
        QJsonObject patchcore = window.m_modelParameters.value("PatchCore");
        patchcore.insert("image_size", QJsonObject{{"width", 32}, {"height", 32}});
        patchcore.insert("sigma", 7);
        QVERIFY(window.validateModelParameters("PatchCore", patchcore));
        patchcore.insert("sigma", 8);
        QVERIFY(!window.validateModelParameters("PatchCore", patchcore));
        QJsonObject padim = window.m_modelParameters.value("PaDiM");
        padim.insert("layers", QJsonArray{"layer1"});
        QVERIFY(!window.validateModelParameters("PaDiM", padim));
        padim.insert("n_features", 64);
        QVERIFY(window.validateModelParameters("PaDiM", padim));
        padim.insert("n_features", 65);
        QVERIFY(!window.validateModelParameters("PaDiM", padim));
        QJsonObject efficient = window.m_modelParameters.value("EfficientAD");
        efficient.insert("teacher_out_channels", 383);
        QVERIFY(!window.validateModelParameters("EfficientAD", efficient));
        efficient.insert("teacher_out_channels", 384);
        QVERIFY(window.validateModelParameters("EfficientAD", efficient));
    }
};

int main(int argc, char **argv)
{
    if (argc > 1 && QByteArray(argv[1]) == "-u") {
        return fakeWorker(argc, argv);
    }
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication application(argc, argv);
    NativeRegressionTest test;
    return QTest::qExec(&test, argc, argv);
}

#include "native_regression.moc"
