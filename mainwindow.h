#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QJsonObject>
#include <QMainWindow>
#include <QMap>
#include <QProcess>

#include "cameraservice.h"

class AnomalibProcess;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QProgressBar;
class QSlider;
class QSpinBox;
class QStackedWidget;
class QTableWidget;
class QToolButton;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void navigateTo(int index);
    void chooseDataset();
    void refreshDataset();
    void startTraining();
    void stopTraining();
    void handleWorkerEvent(const QJsonObject &event);
    void handleWorkerOutput(const QString &text);
    void handleWorkerFinished(int exitCode, QProcess::ExitStatus status);
    void handleWorkerFailure(const QString &message);
    void exportWinner(const QString &format);
    void promoteWinner();
    void browsePython();
    void verifyEnvironment();
    void refreshAssetStatus();
    void downloadMissingAssets();
    void discoverCameras();
    void captureNormal();
    void captureAnomalous();
    void updateThreshold(int value);
    void chooseInferenceModel();
    void useWinnerForInference();
    void chooseInferenceImage();
    void runInference();
    void captureAndRunInference();
    void handleInferenceEvent(const QJsonObject &event);
    void handleInferenceOutput(const QString &text);
    void handleInferenceFinished(int exitCode, QProcess::ExitStatus status);
    void handleInferenceFailure(const QString &message);

private:
    QWidget *createTrainPage();
    QWidget *createCapturePage();
    QWidget *createDatasetsPage();
    QWidget *createModelsPage();
    QWidget *createInspectPage();
    QWidget *createSettingsPage();
    QWidget *createPlaceholderPage(const QString &title, const QString &description);
    QWidget *createSummaryItem(const QIcon &icon, const QString &label,
                               QLabel **valueLabel);
    QToolButton *createNavButton(const QString &text, const QIcon &icon, int index);
    QWidget *createModelCard(const QString &title, const QString &subtitle);
    QLabel *createImageLabel(const QString &resourcePath, const QSize &minimumSize);

    void buildUi();
    void buildMenus();
    void applyTheme();
    void populateReferenceState();
    void setJobRunning(bool running);
    void updateWinner(const QString &modelName);
    void updateResultRow(const QJsonObject &result);
    void updateDatasetCounts(const QString &rootPath);
    QString detectedNormalFolder(const QString &rootPath) const;
    QString detectedAbnormalFolder(const QString &rootPath) const;
    QString detectedMaskFolder(const QString &rootPath) const;
    QStringList imageFiles(const QString &path) const;
    QString matchingMaskForImage(const QString &imagePath) const;
    int countImages(const QString &path) const;
    void updateInspectionFromDataset();
    void updateInspectionFromResult(const QJsonObject &result);
    void updateInferenceResult(const QJsonObject &result);
    bool startInferenceSession();
    void stopInferenceSession();
    void loadInferenceModel(const QString &path,
                            const QString &trustedFingerprint = {});
    void restoreInferenceModel();
    void setInferenceModelState(const QString &status,
                                bool ready,
                                bool error = false);
    QString nextInferenceRequestId(const QString &prefix);
    void setInspectionImage(QLabel *label, const QString &path,
                             const QString &emptyText);
    void clearPredictionImages();
    void showModelParameterDialog(const QString &modelName);
    void loadParameterSettings();
    void saveSharedParameterSettings();
    void saveModelParameterSettings(const QString &modelName);
    void updateModelCardStatus(const QString &modelName);
    void updateSharedSettingsSummary();
    bool validateModelParameters(const QString &modelName,
                                 const QJsonObject &parameters,
                                 QString *errorMessage = nullptr,
                                 bool requireExistingPaths = false) const;
    bool validateTrainingConfiguration(const QStringList &models, bool demo,
                                       QString *errorMessage = nullptr) const;
    QJsonObject modelParameterMap(const QStringList &models) const;
    QJsonObject checkpointModelParameters(const QString &modelName) const;
    QString dataDirectory() const;
    QString workspaceDirectory() const;
    QString modelAssetsDirectory() const;
    QString ensureWorkerScript(QString *errorMessage = nullptr);
    QString writeJobFile(const QJsonObject &job, const QString &prefix,
                         QString *errorMessage = nullptr);
    QString defaultPythonExecutable() const;
    QString selectedWinnerCheckpoint() const;
    QString nextCapturePath(const QString &classFolder);
    void runWorkerJob(const QJsonObject &job, const QString &prefix);
    void updateAssetRow(const QJsonObject &event);
    void showStatus(const QString &message, bool error = false);

    AnomalibProcess *m_worker = nullptr;
    AnomalibProcess *m_inferenceWorker = nullptr;
    CameraService m_cameraService;
    QStackedWidget *m_pages = nullptr;
    QList<QToolButton *> m_navButtons;

    QLabel *m_normalCount = nullptr;
    QLabel *m_anomalousCount = nullptr;
    QLabel *m_maskCount = nullptr;
    QLabel *m_totalCount = nullptr;
    QLabel *m_datasetDetails = nullptr;
    QLabel *m_datasetPathLabel = nullptr;
    QLineEdit *m_datasetPathEdit = nullptr;
    QLineEdit *m_captureDatasetEdit = nullptr;
    QComboBox *m_cameraCombo = nullptr;
    QLabel *m_cameraStatus = nullptr;
    QLabel *m_capturePreview = nullptr;

    QMap<QString, QCheckBox *> m_modelChecks;
    QMap<QString, QPushButton *> m_modelConfigureButtons;
    QMap<QString, QLabel *> m_modelConfigurationLabels;
    QMap<QString, QJsonObject> m_modelParameters;
    QToolButton *m_sharedSettingsToggle = nullptr;
    QPushButton *m_sharedResetButton = nullptr;
    QFrame *m_sharedSettingsPanel = nullptr;
    QPushButton *m_trainButton = nullptr;
    QPushButton *m_stopButton = nullptr;
    QSpinBox *m_epochSpin = nullptr;
    QSpinBox *m_trainBatchSpin = nullptr;
    QSpinBox *m_evalBatchSpin = nullptr;
    QSpinBox *m_workerSpin = nullptr;
    QComboBox *m_acceleratorCombo = nullptr;
    QSpinBox *m_devicesSpin = nullptr;
    QProgressBar *m_progress = nullptr;
    QLabel *m_trainingState = nullptr;
    QTableWidget *m_resultsTable = nullptr;
    QPlainTextEdit *m_trainingLog = nullptr;

    QLabel *m_winnerLabel = nullptr;
    QLabel *m_testImageLabel = nullptr;
    QLabel *m_inputImage = nullptr;
    QLabel *m_maskImage = nullptr;
    QLabel *m_heatmapTitle = nullptr;
    QLabel *m_heatmapImage = nullptr;
    QLabel *m_predictedMaskImage = nullptr;
    QSlider *m_thresholdSlider = nullptr;
    QSpinBox *m_thresholdSpin = nullptr;
    QPushButton *m_promoteButton = nullptr;

    QLabel *m_inferenceModelName = nullptr;
    QLabel *m_inferenceModelDetails = nullptr;
    QPushButton *m_inferenceLoadModelButton = nullptr;
    QPushButton *m_inferenceUseWinnerButton = nullptr;
    QLineEdit *m_inferenceImageEdit = nullptr;
    QSpinBox *m_inferenceThresholdSpin = nullptr;
    QCheckBox *m_inferenceSigmaOverrideCheck = nullptr;
    QSpinBox *m_inferenceSigmaSpin = nullptr;
    QPushButton *m_inferenceRunButton = nullptr;
    QPushButton *m_inferenceCaptureButton = nullptr;
    QLabel *m_inferenceStatus = nullptr;
    QLabel *m_inferenceScore = nullptr;
    QLabel *m_inferenceLatency = nullptr;
    QLabel *m_inferenceInputImage = nullptr;
    QLabel *m_inferenceHeatmapImage = nullptr;
    QLabel *m_inferenceMaskImage = nullptr;

    QLineEdit *m_pythonEdit = nullptr;
    QCheckBox *m_demoMode = nullptr;
    QLabel *m_environmentStatus = nullptr;
    QLabel *m_assetRootLabel = nullptr;
    QTableWidget *m_assetTable = nullptr;
    QPushButton *m_refreshAssetsButton = nullptr;
    QPushButton *m_downloadAssetsButton = nullptr;
    QLabel *m_sdkStatus = nullptr;
    QLabel *m_statusMode = nullptr;

    QString m_datasetRoot;
    QString m_anomalyMapPath;
    QString m_winnerModel = QStringLiteral("PatchCore");
    QString m_activeJob;
    QString m_activeInferencePath;
    QString m_activeInferenceFingerprint;
    QString m_pendingInferencePath;
    QString m_pendingInferenceFingerprint;
    QString m_activeInferenceModel;
    QString m_activeInferenceBackend;
    QString m_activeInferenceDevice;
    int m_activeInferenceSavedSigma = 4;
    bool m_activeInferenceSigmaOverrideSupported = false;
    QMap<QString, QJsonObject> m_results;
    QMap<QString, QJsonObject> m_assetStatuses;
    bool m_assetStatusRefreshRequested = false;
    bool m_inferenceModelReady = false;
    bool m_inferenceHadReadyModelBeforeLoad = false;
    bool m_inferenceRestoreAttempted = false;
    quint64 m_inferenceRequestCounter = 0;
};

#endif // MAINWINDOW_H
