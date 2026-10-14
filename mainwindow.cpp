#include "mainwindow.h"

#include "anomalibprocess.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QPixmap>
#include <QSaveFile>
#include <QScreen>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyle>
#include <QTableWidget>
#include <QTime>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>
#include <utility>

namespace {

QLabel *makeLabel(const QString &text, const char *role = nullptr)
{
    auto *label = new QLabel(text);
    if (role) {
        label->setProperty("role", role);
    }
    return label;
}

QFrame *divider(Qt::Orientation orientation)
{
    auto *line = new QFrame;
    line->setFrameShape(orientation == Qt::Horizontal ? QFrame::HLine : QFrame::VLine);
    line->setFrameShadow(QFrame::Plain);
    line->setObjectName(QStringLiteral("divider"));
    return line;
}

QString metricText(const QJsonObject &result, const QString &key,
                   int precision = 3, const QString &suffix = {})
{
    const QJsonValue value = result.value(key);
    if (!value.isDouble()) {
        return QStringLiteral("—");
    }
    return QString::number(value.toDouble(), 'f', precision) + suffix;
}

QJsonObject defaultImageSize()
{
    return {
        {QStringLiteral("width"), 256},
        {QStringLiteral("height"), 256},
    };
}

QSize configuredImageSize(const QJsonObject &parameters)
{
    const QJsonObject imageSize =
        parameters.value(QStringLiteral("image_size")).toObject();
    return QSize(imageSize.value(QStringLiteral("width")).toInt(256),
                 imageSize.value(QStringLiteral("height")).toInt(256));
}

QString imageSizeText(const QJsonObject &parameters)
{
    const QSize size = configuredImageSize(parameters);
    return QStringLiteral("%1×%2").arg(size.width()).arg(size.height());
}

QTableWidgetItem *tableItem(const QString &text, Qt::Alignment alignment = Qt::AlignCenter)
{
    auto *item = new QTableWidgetItem(text);
    item->setTextAlignment(alignment);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}

QJsonObject defaultModelParameters(const QString &modelName)
{
    if (modelName == QStringLiteral("PatchCore")) {
        return {
            {QStringLiteral("backbone"), QStringLiteral("wide_resnet50_2")},
            {QStringLiteral("layers"),
             QJsonArray {QStringLiteral("layer2"), QStringLiteral("layer3")}},
            {QStringLiteral("pre_trained"), true},
            {QStringLiteral("coreset_sampling_ratio"), 0.1},
            {QStringLiteral("num_neighbors"), 9},
            {QStringLiteral("precision"), QStringLiteral("float32")},
             {QStringLiteral("sigma"), 4},
             {QStringLiteral("patch_size"), 3},
             {QStringLiteral("patch_stride"), 1},
             {QStringLiteral("image_size"), defaultImageSize()},
         };
    }
    if (modelName == QStringLiteral("PaDiM")) {
        return {
            {QStringLiteral("backbone"), QStringLiteral("resnet18")},
            {QStringLiteral("layers"),
             QJsonArray {QStringLiteral("layer1"), QStringLiteral("layer2"),
                         QStringLiteral("layer3")}},
             {QStringLiteral("pre_trained"), true},
             {QStringLiteral("n_features"), QJsonValue::Null},
             {QStringLiteral("image_size"), defaultImageSize()},
         };
    }
    return {
        {QStringLiteral("imagenet_dir"), QStringLiteral("./datasets/imagenette")},
        {QStringLiteral("teacher_out_channels"), 384},
        {QStringLiteral("model_size"), QStringLiteral("s")},
        {QStringLiteral("lr"), 0.0001},
        {QStringLiteral("weight_decay"), 0.00001},
         {QStringLiteral("padding"), false},
         {QStringLiteral("pad_maps"), true},
         {QStringLiteral("image_size"), defaultImageSize()},
     };
}

const QStringList &supportedModelNames()
{
    static const QStringList names = {
        QStringLiteral("PatchCore"),
        QStringLiteral("PaDiM"),
        QStringLiteral("EfficientAD"),
    };
    return names;
}

const QStringList &supportedLayers()
{
    static const QStringList layers = {
        QStringLiteral("layer1"),
        QStringLiteral("layer2"),
        QStringLiteral("layer3"),
        QStringLiteral("layer4"),
    };
    return layers;
}

const QStringList &supportedBackbones()
{
    static const QStringList backbones = {
        QStringLiteral("wide_resnet50_2"),
        QStringLiteral("resnet18"),
    };
    return backbones;
}

const QList<QPair<QString, QString>> &managedModelAssets()
{
    static const QList<QPair<QString, QString>> assets = {
        {QStringLiteral("patchcore_backbone"),
         QStringLiteral("PatchCore · wide_resnet50_2")},
        {QStringLiteral("padim_backbone"),
         QStringLiteral("PaDiM · resnet18")},
        {QStringLiteral("efficientad_teacher"),
         QStringLiteral("EfficientAD · teacher weights")},
        {QStringLiteral("efficientad_imagenette"),
         QStringLiteral("EfficientAD · ImageNette data")},
    };
    return assets;
}

QString formatAssetSize(qint64 bytes)
{
    if (bytes <= 0) {
        return QObject::tr("size unknown");
    }
    constexpr double megabyte = 1024.0 * 1024.0;
    constexpr double gigabyte = 1024.0 * megabyte;
    if (bytes >= gigabyte) {
        return QObject::tr("%1 GB").arg(bytes / gigabyte, 0, 'f', 2);
    }
    return QObject::tr("%1 MB").arg(bytes / megabyte, 0, 'f', 0);
}

QString sha256ForFile(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        digest.addData(file.read(1024 * 1024));
    }
    return QString::fromLatin1(digest.result().toHex());
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_worker(new AnomalibProcess(this))
    , m_inferenceWorker(new AnomalibProcess(this))
{
    setWindowTitle(tr("Inspection Workbench"));
    setMinimumSize(1180, 760);
    resize(1488, 1000);

    for (const QString &modelName : supportedModelNames()) {
        m_modelParameters.insert(modelName, defaultModelParameters(modelName));
    }
    buildMenus();
    buildUi();
    loadParameterSettings();
    applyTheme();
    populateReferenceState();

    connect(m_worker, &AnomalibProcess::eventReceived,
            this, &MainWindow::handleWorkerEvent);
    connect(m_worker, &AnomalibProcess::outputReceived,
            this, &MainWindow::handleWorkerOutput);
    connect(m_worker, &AnomalibProcess::jobFinished,
            this, &MainWindow::handleWorkerFinished);
    connect(m_worker, &AnomalibProcess::jobFailed,
            this, &MainWindow::handleWorkerFailure);
    connect(m_inferenceWorker, &AnomalibProcess::eventReceived,
            this, &MainWindow::handleInferenceEvent);
    connect(m_inferenceWorker, &AnomalibProcess::outputReceived,
            this, &MainWindow::handleInferenceOutput);
    connect(m_inferenceWorker, &AnomalibProcess::jobFinished,
            this, &MainWindow::handleInferenceFinished);
    connect(m_inferenceWorker, &AnomalibProcess::jobFailed,
            this, &MainWindow::handleInferenceFailure);

    if (QScreen *screen = this->screen()) {
        const QRect available = screen->availableGeometry();
        resize(qMin(width(), available.width() - 24),
               qMin(height(), available.height() - 24));
    }
}

MainWindow::~MainWindow()
{
    m_inferenceWorker->stop();
    m_worker->stop();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    m_inferenceWorker->stop();
    m_worker->stop();
    event->accept();
}

void MainWindow::buildMenus()
{
    QMenu *file = menuBar()->addMenu(tr("&File"));
    QAction *openDataset = file->addAction(tr("Open Dataset…"));
    connect(openDataset, &QAction::triggered, this, &MainWindow::chooseDataset);
    file->addSeparator();
    QAction *exit = file->addAction(tr("Exit"));
    connect(exit, &QAction::triggered, this, &QWidget::close);

    QMenu *view = menuBar()->addMenu(tr("&View"));
    const QStringList pages = {tr("Capture"), tr("Datasets"), tr("Train"),
                               tr("Models"), tr("Inspect"), tr("Settings")};
    for (int index = 0; index < pages.size(); ++index) {
        QAction *action = view->addAction(pages[index]);
        connect(action, &QAction::triggered, this, [this, index] { navigateTo(index); });
    }

    QMenu *tools = menuBar()->addMenu(tr("&Tools"));
    QAction *verify = tools->addAction(tr("Verify Anomalib Environment"));
    connect(verify, &QAction::triggered, this, &MainWindow::verifyEnvironment);
    QAction *cameras = tools->addAction(tr("Discover Basler Cameras"));
    connect(cameras, &QAction::triggered, this, &MainWindow::discoverCameras);

    QMenu *help = menuBar()->addMenu(tr("&Help"));
    QAction *about = help->addAction(tr("About"));
    connect(about, &QAction::triggered, this, [this] {
        QMessageBox::about(
            this, tr("Inspection Workbench"),
            tr("A native Qt workbench for collecting Basler images, training and "
               "testing Anomalib models, and promoting a production model."));
    });
}

void MainWindow::buildUi()
{
    auto *root = new QWidget;
    root->setObjectName(QStringLiteral("appRoot"));
    auto *rootLayout = new QHBoxLayout(root);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    auto *navigation = new QFrame;
    navigation->setObjectName(QStringLiteral("navigation"));
    navigation->setFixedWidth(170);
    auto *navigationLayout = new QVBoxLayout(navigation);
    navigationLayout->setContentsMargins(8, 18, 8, 18);
    navigationLayout->setSpacing(8);

    auto *brandRow = new QHBoxLayout;
    auto *brandIcon = new QLabel;
    brandIcon->setPixmap(style()->standardIcon(QStyle::SP_ComputerIcon)
                             .pixmap(QSize(24, 24)));
    brandRow->addWidget(brandIcon);
    brandRow->addWidget(makeLabel(tr("Inspection\nWorkbench"), "brand"));
    brandRow->addStretch();
    navigationLayout->addLayout(brandRow);
    navigationLayout->addSpacing(16);

    const QList<QPair<QString, QStyle::StandardPixmap>> navItems = {
        {tr("Capture"), QStyle::SP_DialogSaveButton},
        {tr("Datasets"), QStyle::SP_DirIcon},
        {tr("Train"), QStyle::SP_MediaPlay},
        {tr("Models"), QStyle::SP_ComputerIcon},
        {tr("Inspect"), QStyle::SP_FileDialogDetailedView},
        {tr("Settings"), QStyle::SP_FileDialogContentsView},
    };
    for (int index = 0; index < navItems.size(); ++index) {
        QToolButton *button = createNavButton(
            navItems[index].first,
            style()->standardIcon(navItems[index].second), index);
        navigationLayout->addWidget(button);
        if (index == 4) {
            navigationLayout->addStretch();
        }
    }

    m_pages = new QStackedWidget;
    m_pages->addWidget(createCapturePage());
    m_pages->addWidget(createDatasetsPage());
    m_pages->addWidget(createTrainPage());
    m_pages->addWidget(createModelsPage());
    m_pages->addWidget(createInspectPage());
    m_pages->addWidget(createSettingsPage());

    rootLayout->addWidget(navigation);
    rootLayout->addWidget(m_pages, 1);
    setCentralWidget(root);
    navigateTo(2);

    auto *statusMode = new QLabel(tr("Demo data · configure Anomalib Python in Settings"));
    statusMode->setObjectName(QStringLiteral("statusMode"));
    m_statusMode = statusMode;
    statusBar()->addWidget(statusMode, 1);
    statusBar()->addPermanentWidget(
        makeLabel(m_cameraService.availabilityText(), "statusDetail"));
}

QToolButton *MainWindow::createNavButton(const QString &text, const QIcon &icon, int index)
{
    auto *button = new QToolButton;
    button->setText(text);
    button->setIcon(icon);
    button->setIconSize(QSize(22, 22));
    button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    button->setCheckable(true);
    button->setAutoExclusive(true);
    button->setProperty("nav", true);
    button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    button->setFixedHeight(50);
    connect(button, &QToolButton::clicked, this, [this, index] { navigateTo(index); });
    m_navButtons.append(button);
    return button;
}

QWidget *MainWindow::createTrainPage()
{
    auto *page = new QWidget;
    auto *pageLayout = new QHBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);
    pageLayout->setSpacing(0);

    auto *centerScroll = new QScrollArea;
    centerScroll->setWidgetResizable(true);
    centerScroll->setFrameShape(QFrame::NoFrame);
    centerScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto *center = new QWidget;
    center->setObjectName(QStringLiteral("trainCenter"));
    auto *content = new QVBoxLayout(center);
    content->setContentsMargins(26, 18, 26, 20);
    content->setSpacing(12);

    auto *titleRow = new QHBoxLayout;
    titleRow->addWidget(makeLabel(tr("Housing Seal"), "pageTitle"));
    titleRow->addStretch();
    m_datasetPathLabel = makeLabel(tr("Reference dataset"), "muted");
    titleRow->addWidget(m_datasetPathLabel);
    content->addLayout(titleRow);

    content->addWidget(makeLabel(tr("Dataset summary"), "sectionTitle"));
    auto *summary = new QFrame;
    summary->setObjectName(QStringLiteral("summaryBar"));
    auto *summaryLayout = new QHBoxLayout(summary);
    summaryLayout->setContentsMargins(14, 8, 14, 8);
    summaryLayout->setSpacing(0);
    summaryLayout->addWidget(createSummaryItem(
        style()->standardIcon(QStyle::SP_DialogApplyButton),
        tr("Normal images"), &m_normalCount), 1);
    summaryLayout->addWidget(divider(Qt::Vertical));
    summaryLayout->addWidget(createSummaryItem(
        style()->standardIcon(QStyle::SP_MessageBoxWarning),
        tr("Anomalous images"), &m_anomalousCount), 1);
    summaryLayout->addWidget(divider(Qt::Vertical));
    summaryLayout->addWidget(createSummaryItem(
        style()->standardIcon(QStyle::SP_FileIcon),
        tr("Mask images"), &m_maskCount), 1);
    summaryLayout->addWidget(divider(Qt::Vertical));
    summaryLayout->addWidget(createSummaryItem(
        style()->standardIcon(QStyle::SP_DirIcon),
        tr("Total images"), &m_totalCount), 1);
    content->addWidget(summary);

    auto *datasetRow = new QHBoxLayout;
    m_datasetDetails = makeLabel(
        tr("Resolution: 2048 × 1536  ·  Color: Grayscale  ·  "
           "Split: 80% / 10% / 10% (train/val/test)  ·  Normalization: Per-image"),
        "muted");
    m_datasetDetails->setWordWrap(true);
    datasetRow->addWidget(m_datasetDetails, 1);
    auto *editDataset = new QPushButton(tr("Edit dataset"));
    editDataset->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    editDataset->setProperty("secondary", true);
    connect(editDataset, &QPushButton::clicked, this, &MainWindow::chooseDataset);
    datasetRow->addWidget(editDataset);
    content->addLayout(datasetRow);

    content->addSpacing(4);
    content->addWidget(makeLabel(tr("Models"), "sectionTitle"));
    content->addWidget(makeLabel(tr("Select models to train and compare."), "muted"));

    auto *modelsRow = new QHBoxLayout;
    modelsRow->setSpacing(12);
    modelsRow->addWidget(
        createModelCard(QStringLiteral("PatchCore"), tr("Memory-bank baseline")), 1);
    modelsRow->addWidget(
        createModelCard(QStringLiteral("PaDiM"), tr("Gaussian patch model")), 1);
    modelsRow->addWidget(
        createModelCard(QStringLiteral("EfficientAD"), tr("Student–teacher model")), 1);
    content->addLayout(modelsRow);

    m_sharedSettingsToggle = new QToolButton;
    m_sharedSettingsToggle->setCheckable(true);
    m_sharedSettingsToggle->setChecked(false);
    m_sharedSettingsToggle->setArrowType(Qt::RightArrow);
    m_sharedSettingsToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_sharedSettingsToggle->setProperty("secondary", true);
    content->addWidget(m_sharedSettingsToggle, 0, Qt::AlignLeft);

    m_sharedSettingsPanel = new QFrame;
    m_sharedSettingsPanel->setObjectName(QStringLiteral("parameterPanel"));
    auto *settingsGrid = new QGridLayout(m_sharedSettingsPanel);
    settingsGrid->setContentsMargins(18, 16, 18, 16);
    settingsGrid->setHorizontalSpacing(12);
    settingsGrid->setVerticalSpacing(10);

    m_epochSpin = new QSpinBox;
    m_epochSpin->setRange(1, 2000000000);
    m_epochSpin->setValue(5);
    m_epochSpin->setToolTip(
        tr("Maximum epochs (up to 2 billion). Memory-bank models may enforce a single collection pass."));
    settingsGrid->addWidget(makeLabel(tr("Epochs"), "fieldLabel"), 0, 0);
    settingsGrid->addWidget(m_epochSpin, 0, 1);

    m_trainBatchSpin = new QSpinBox;
    m_trainBatchSpin->setRange(1, 1024);
    m_trainBatchSpin->setValue(8);
    settingsGrid->addWidget(makeLabel(tr("Train batch"), "fieldLabel"), 0, 2);
    settingsGrid->addWidget(m_trainBatchSpin, 0, 3);

    m_evalBatchSpin = new QSpinBox;
    m_evalBatchSpin->setRange(1, 1024);
    m_evalBatchSpin->setValue(8);
    settingsGrid->addWidget(makeLabel(tr("Evaluation batch"), "fieldLabel"), 0, 4);
    settingsGrid->addWidget(m_evalBatchSpin, 0, 5);

    m_workerSpin = new QSpinBox;
    m_workerSpin->setRange(0, 64);
    m_workerSpin->setValue(2);
    settingsGrid->addWidget(makeLabel(tr("Data workers"), "fieldLabel"), 1, 0);
    settingsGrid->addWidget(m_workerSpin, 1, 1);

    m_acceleratorCombo = new QComboBox;
    m_acceleratorCombo->addItem(tr("Auto"), QStringLiteral("auto"));
    m_acceleratorCombo->addItem(tr("CPU"), QStringLiteral("cpu"));
    m_acceleratorCombo->addItem(tr("GPU"), QStringLiteral("gpu"));
    settingsGrid->addWidget(makeLabel(tr("Accelerator"), "fieldLabel"), 1, 2);
    settingsGrid->addWidget(m_acceleratorCombo, 1, 3);

    m_devicesSpin = new QSpinBox;
    m_devicesSpin->setRange(1, 16);
    m_devicesSpin->setValue(1);
    settingsGrid->addWidget(makeLabel(tr("Devices"), "fieldLabel"), 1, 4);
    settingsGrid->addWidget(m_devicesSpin, 1, 5);

    auto *efficientNote = makeLabel(
        tr("EfficientAD requires train batch 1. The worker applies that value only "
           "to EfficientAD and preserves the shared batch for other models."),
        "muted");
    efficientNote->setWordWrap(true);
    settingsGrid->addWidget(efficientNote, 2, 0, 1, 5);
    m_sharedResetButton = new QPushButton(tr("Restore defaults"));
    m_sharedResetButton->setProperty("secondary", true);
    settingsGrid->addWidget(m_sharedResetButton, 2, 5);
    m_sharedSettingsPanel->hide();
    content->addWidget(m_sharedSettingsPanel);

    connect(m_sharedSettingsToggle, &QToolButton::toggled, this,
            [this](bool expanded) {
                m_sharedSettingsPanel->setVisible(expanded);
                m_sharedSettingsToggle->setArrowType(
                    expanded ? Qt::DownArrow : Qt::RightArrow);
            });
    const auto sharedSettingsChanged = [this] {
        updateSharedSettingsSummary();
        saveSharedParameterSettings();
    };
    connect(m_epochSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, sharedSettingsChanged);
    connect(m_trainBatchSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, sharedSettingsChanged);
    connect(m_evalBatchSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, sharedSettingsChanged);
    connect(m_workerSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, sharedSettingsChanged);
    connect(m_devicesSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, sharedSettingsChanged);
    connect(m_acceleratorCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, sharedSettingsChanged);
    connect(m_sharedResetButton, &QPushButton::clicked, this, [this] {
        m_epochSpin->setValue(5);
        m_trainBatchSpin->setValue(8);
        m_evalBatchSpin->setValue(8);
        m_workerSpin->setValue(2);
        m_acceleratorCombo->setCurrentIndex(0);
        m_devicesSpin->setValue(1);
        updateSharedSettingsSummary();
        saveSharedParameterSettings();
    });
    updateSharedSettingsSummary();

    auto *actions = new QHBoxLayout;
    m_trainButton = new QPushButton(tr("Train selected models"));
    m_trainButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_trainButton->setProperty("primary", true);
    connect(m_trainButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    actions->addWidget(m_trainButton);

    m_stopButton = new QPushButton(tr("Stop"));
    m_stopButton->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    m_stopButton->setProperty("danger", true);
    m_stopButton->hide();
    connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::stopTraining);
    actions->addWidget(m_stopButton);

    actions->addStretch();
    m_trainingState = makeLabel(tr("Training completed"), "successText");
    actions->addWidget(m_trainingState);
    content->addLayout(actions);

    m_progress = new QProgressBar;
    m_progress->setRange(0, 100);
    m_progress->setValue(100);
    m_progress->setTextVisible(false);
    m_progress->setFixedHeight(4);
    m_progress->hide();
    content->addWidget(m_progress);

    auto *resultsTitle = new QHBoxLayout;
    resultsTitle->addWidget(makeLabel(tr("Results"), "sectionTitle"));
    resultsTitle->addWidget(makeLabel(tr("(sorted by Image AUROC)"), "muted"));
    resultsTitle->addStretch();
    resultsTitle->addWidget(makeLabel(tr("Metrics: Test set"), "muted"));
    content->addLayout(resultsTitle);

    m_resultsTable = new QTableWidget(3, 6);
    m_resultsTable->setObjectName(QStringLiteral("resultsTable"));
    m_resultsTable->setHorizontalHeaderLabels(
        {tr("Model"), tr("Image AUROC"), tr("Pixel AUROC"),
         tr("F1 (image)"), tr("Latency (ms)"), tr("Model size")});
    m_resultsTable->verticalHeader()->hide();
    m_resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_resultsTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_resultsTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_resultsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultsTable->setShowGrid(true);
    m_resultsTable->setAlternatingRowColors(false);
    m_resultsTable->setMinimumHeight(190);
    m_resultsTable->setMaximumHeight(230);
    for (int row = 0; row < 3; ++row) {
        m_resultsTable->setRowHeight(row, 52);
    }
    connect(m_resultsTable, &QTableWidget::cellClicked, this,
            [this](int row, int) {
                if (QTableWidgetItem *item = m_resultsTable->item(row, 0)) {
                    updateWinner(item->data(Qt::UserRole).toString());
                }
            });
    content->addWidget(m_resultsTable);
    content->addWidget(makeLabel(
        tr("Higher is better for AUROC and F1. Lower is better for latency and model size."),
        "muted"));

    auto *logHeader = new QHBoxLayout;
    logHeader->addWidget(makeLabel(tr("Training log"), "sectionTitle"));
    logHeader->addStretch();
    auto *clearLog = new QPushButton(tr("Clear log"));
    clearLog->setProperty("secondary", true);
    connect(clearLog, &QPushButton::clicked, this, [this] { m_trainingLog->clear(); });
    logHeader->addWidget(clearLog);
    content->addLayout(logHeader);

    m_trainingLog = new QPlainTextEdit;
    m_trainingLog->setReadOnly(true);
    m_trainingLog->setMinimumHeight(135);
    m_trainingLog->setMaximumHeight(190);
    m_trainingLog->setLineWrapMode(QPlainTextEdit::NoWrap);
    content->addWidget(m_trainingLog);
    content->addStretch();
    centerScroll->setWidget(center);

    auto *inspection = new QFrame;
    inspection->setObjectName(QStringLiteral("inspectionPanel"));
    inspection->setMinimumWidth(360);
    inspection->setMaximumWidth(410);
    auto *right = new QVBoxLayout(inspection);
    right->setContentsMargins(22, 18, 22, 18);
    right->setSpacing(9);

    auto *winnerRow = new QHBoxLayout;
    m_winnerLabel = makeLabel(tr("Winner: PatchCore"), "winner");
    winnerRow->addWidget(m_winnerLabel);
    winnerRow->addStretch();
    QLabel *winnerIcon = new QLabel;
    winnerIcon->setPixmap(style()->standardIcon(QStyle::SP_DialogApplyButton)
                              .pixmap(QSize(20, 20)));
    winnerRow->addWidget(winnerIcon);
    right->addLayout(winnerRow);
    m_testImageLabel = makeLabel(tr("No dataset sample selected"), "muted");
    m_testImageLabel->setWordWrap(true);
    right->addWidget(m_testImageLabel);

    right->addWidget(makeLabel(tr("Input image"), "imageTitle"));
    m_inputImage = createImageLabel({}, QSize(330, 112));
    right->addWidget(m_inputImage);

    right->addWidget(makeLabel(tr("Ground truth mask"), "imageTitle"));
    m_maskImage = createImageLabel({}, QSize(330, 96));
    right->addWidget(m_maskImage);

    m_heatmapTitle = makeLabel(tr("Predicted heatmap (PatchCore)"), "imageTitle");
    right->addWidget(m_heatmapTitle);
    m_heatmapImage = createImageLabel({}, QSize(330, 96));
    right->addWidget(m_heatmapImage);

    auto *thresholdRow = new QHBoxLayout;
    thresholdRow->addWidget(makeLabel(tr("Threshold"), "muted"));
    m_thresholdSlider = new QSlider(Qt::Horizontal);
    m_thresholdSlider->setRange(0, 100);
    m_thresholdSlider->setValue(60);
    thresholdRow->addWidget(m_thresholdSlider, 1);
    m_thresholdSpin = new QSpinBox;
    m_thresholdSpin->setRange(0, 100);
    m_thresholdSpin->setSuffix(QStringLiteral("%"));
    m_thresholdSpin->setValue(60);
    thresholdRow->addWidget(m_thresholdSpin);
    connect(m_thresholdSlider, &QSlider::valueChanged,
            this, &MainWindow::updateThreshold);
    connect(m_thresholdSpin, qOverload<int>(&QSpinBox::valueChanged),
            m_thresholdSlider, &QSlider::setValue);
    right->addLayout(thresholdRow);

    right->addWidget(makeLabel(tr("Predicted mask (thresholded)"), "imageTitle"));
    m_predictedMaskImage = createImageLabel({}, QSize(330, 74));
    right->addWidget(m_predictedMaskImage);

    setInspectionImage(m_inputImage, {}, tr("Choose a dataset to preview a sample"));
    setInspectionImage(m_maskImage, {}, tr("No ground-truth mask selected"));
    clearPredictionImages();

    right->addStretch();
    m_promoteButton = new QPushButton(tr("Set as production model"));
    m_promoteButton->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));
    m_promoteButton->setProperty("success", true);
    connect(m_promoteButton, &QPushButton::clicked, this, &MainWindow::promoteWinner);
    right->addWidget(m_promoteButton);

    auto *more = new QToolButton;
    more->setText(tr("Export winner"));
    more->setPopupMode(QToolButton::InstantPopup);
    more->setProperty("secondary", true);
    QMenu *exportMenu = new QMenu(more);
    QAction *openVino = exportMenu->addAction(tr("Export OpenVINO"));
    connect(openVino, &QAction::triggered, this,
            [this] { exportWinner(QStringLiteral("OPENVINO")); });
    QAction *onnx = exportMenu->addAction(tr("Export ONNX"));
    connect(onnx, &QAction::triggered, this,
            [this] { exportWinner(QStringLiteral("ONNX")); });
    QAction *torch = exportMenu->addAction(tr("Export Torch (.pt)"));
    connect(torch, &QAction::triggered, this,
            [this] { exportWinner(QStringLiteral("TORCH")); });
    exportMenu->addSeparator();
    QAction *openResults = exportMenu->addAction(tr("Open results folder"));
    connect(openResults, &QAction::triggered, this, [this] {
        QDesktopServices::openUrl(QUrl::fromLocalFile(
            QDir(dataDirectory()).filePath(QStringLiteral("results"))));
    });
    more->setMenu(exportMenu);
    right->addWidget(more);

    pageLayout->addWidget(centerScroll, 1);
    pageLayout->addWidget(inspection);
    return page;
}

QWidget *MainWindow::createSummaryItem(const QIcon &icon, const QString &label,
                                       QLabel **valueLabel)
{
    auto *widget = new QWidget;
    auto *layout = new QHBoxLayout(widget);
    layout->setContentsMargins(10, 3, 10, 3);
    layout->setSpacing(10);
    auto *iconLabel = new QLabel;
    iconLabel->setPixmap(icon.pixmap(QSize(26, 26)));
    layout->addWidget(iconLabel);
    auto *text = new QVBoxLayout;
    text->setSpacing(1);
    text->addWidget(makeLabel(label, "muted"));
    *valueLabel = makeLabel(QStringLiteral("0"), "summaryValue");
    text->addWidget(*valueLabel);
    layout->addLayout(text);
    layout->addStretch();
    return widget;
}

QWidget *MainWindow::createModelCard(const QString &title, const QString &subtitle)
{
    auto *card = new QFrame;
    card->setObjectName(QStringLiteral("modelCard"));
    card->setProperty("selected", true);
    card->setMinimumHeight(98);
    auto *layout = new QGridLayout(card);
    layout->setContentsMargins(14, 11, 12, 11);
    layout->setHorizontalSpacing(8);
    layout->setVerticalSpacing(4);

    auto *check = new QCheckBox(title);
    check->setChecked(true);
    check->setProperty("modelSelector", true);
    check->setToolTip(tr("Include %1 in the next comparison run.").arg(title));
    layout->addWidget(check, 0, 0, 1, 2);

    auto *description = makeLabel(subtitle, "muted");
    layout->addWidget(description, 1, 0, 1, 2);

    auto *configuration = makeLabel(tr("Default settings"), "muted");
    layout->addWidget(configuration, 2, 0);
    auto *configure = new QPushButton(tr("Configure…"));
    configure->setProperty("secondary", true);
    configure->setMinimumWidth(96);
    layout->addWidget(configure, 2, 1, Qt::AlignRight);

    m_modelChecks.insert(title, check);
    m_modelConfigureButtons.insert(title, configure);
    m_modelConfigurationLabels.insert(title, configuration);

    connect(configure, &QPushButton::clicked, this,
            [this, title] { showModelParameterDialog(title); });
    connect(check, &QCheckBox::toggled, card, [card](bool checked) {
        card->setProperty("selected", checked);
        card->style()->unpolish(card);
        card->style()->polish(card);
    });
    return card;
}

void MainWindow::showModelParameterDialog(const QString &modelName)
{
    if (!supportedModelNames().contains(modelName)) {
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Configure %1").arg(modelName));
    dialog.setMinimumWidth(520);
    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(22, 20, 22, 18);
    layout->setSpacing(14);
    auto *lead = makeLabel(
        tr("These values are saved locally and captured with every trained checkpoint."),
        "muted");
    lead->setWordWrap(true);
    layout->addWidget(lead);
    auto *form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setHorizontalSpacing(16);
    form->setVerticalSpacing(11);
    layout->addLayout(form);

    auto *resolutionPreset = new QComboBox;
    const QList<QSize> resolutionPresets = {
        QSize(256, 256), QSize(384, 384), QSize(512, 512),
        QSize(512, 384), QSize(640, 480), QSize(1024, 768),
    };
    for (const QSize &size : resolutionPresets) {
        const QString value = QStringLiteral("%1x%2")
                                  .arg(size.width()).arg(size.height());
        resolutionPreset->addItem(
            QStringLiteral("%1 × %2").arg(size.width()).arg(size.height()),
            value);
    }
    resolutionPreset->addItem(tr("Custom"), QString());
    form->addRow(tr("Input-size preset"), resolutionPreset);

    const int minimumImageDimension =
        modelName == QStringLiteral("EfficientAD") ? 256 : 32;
    auto *imageSizeWidget = new QWidget;
    auto *imageSizeLayout = new QHBoxLayout(imageSizeWidget);
    imageSizeLayout->setContentsMargins(0, 0, 0, 0);
    imageSizeLayout->setSpacing(8);
    auto *imageWidth = new QSpinBox;
    imageWidth->setRange(minimumImageDimension, 2048);
    imageWidth->setSuffix(tr(" px"));
    imageWidth->setToolTip(tr(
        "Model input width. Changing the input size requires retraining."));
    auto *imageHeight = new QSpinBox;
    imageHeight->setRange(minimumImageDimension, 2048);
    imageHeight->setSuffix(tr(" px"));
    imageHeight->setToolTip(tr(
        "Model input height. Source images are resized exactly to this size."));
    imageSizeLayout->addWidget(makeLabel(tr("Width"), "muted"));
    imageSizeLayout->addWidget(imageWidth, 1);
    imageSizeLayout->addWidget(makeLabel(tr("Height"), "muted"));
    imageSizeLayout->addWidget(imageHeight, 1);
    form->addRow(tr("Input dimensions"), imageSizeWidget);

    auto *resolutionWarning = makeLabel(
        tr("Large inputs can substantially increase training time, inference "
           "latency, and accelerator memory use."),
        "warningText");
    resolutionWarning->setWordWrap(true);
    resolutionWarning->hide();
    form->addRow(QString(), resolutionWarning);

    const auto updateResolutionWarning = [=] {
        const qint64 pixels = qint64(imageWidth->value()) * imageHeight->value();
        resolutionWarning->setVisible(pixels > 1000000LL);
    };
    const auto syncPresetFromFields = [=] {
        const QString value = QStringLiteral("%1x%2")
                                  .arg(imageWidth->value())
                                  .arg(imageHeight->value());
        int index = resolutionPreset->findData(value);
        if (index < 0) {
            index = resolutionPreset->count() - 1;
        }
        const QSignalBlocker blocker(resolutionPreset);
        resolutionPreset->setCurrentIndex(index);
        updateResolutionWarning();
    };
    connect(resolutionPreset, qOverload<int>(&QComboBox::currentIndexChanged),
            &dialog, [=](int index) {
                const QString value = resolutionPreset->itemData(index).toString();
                if (value.isEmpty()) {
                    return;
                }
                const QStringList parts = value.split(QLatin1Char('x'));
                if (parts.size() != 2) {
                    return;
                }
                const QSignalBlocker widthBlocker(imageWidth);
                const QSignalBlocker heightBlocker(imageHeight);
                imageWidth->setValue(parts.at(0).toInt());
                imageHeight->setValue(parts.at(1).toInt());
                updateResolutionWarning();
            });
    connect(imageWidth, qOverload<int>(&QSpinBox::valueChanged),
            &dialog, [=] { syncPresetFromFields(); });
    connect(imageHeight, qOverload<int>(&QSpinBox::valueChanged),
            &dialog, [=] { syncPresetFromFields(); });

    std::function<QJsonObject()> collectParameters;
    std::function<void(const QJsonObject &)> setControls;

    if (modelName == QStringLiteral("PatchCore")) {
        auto *backbone = new QComboBox;
        backbone->addItems(supportedBackbones());
        form->addRow(tr("Backbone"), backbone);

        auto *layersWidget = new QWidget;
        auto *layersLayout = new QHBoxLayout(layersWidget);
        layersLayout->setContentsMargins(0, 0, 0, 0);
        layersLayout->setSpacing(12);
        QList<QCheckBox *> layerChecks;
        for (const QString &layerName : supportedLayers()) {
            auto *check = new QCheckBox(layerName);
            check->setProperty("layerName", layerName);
            layerChecks.append(check);
            layersLayout->addWidget(check);
        }
        layersLayout->addStretch();
        form->addRow(tr("Feature layers"), layersWidget);

        auto *pretrained = new QCheckBox(tr("Use pretrained backbone weights"));
        form->addRow(tr("Weights"), pretrained);
        auto *coreset = new QDoubleSpinBox;
        coreset->setRange(0.001, 1.0);
        coreset->setSingleStep(0.01);
        coreset->setDecimals(3);
        form->addRow(tr("Coreset ratio"), coreset);
        auto *neighbors = new QSpinBox;
        neighbors->setRange(1, 1024);
        form->addRow(tr("Nearest neighbors"), neighbors);
        auto *sigma = new QSpinBox;
        sigma->setRange(1, 32);
        sigma->setToolTip(tr(
            "Gaussian standard deviation used to smooth pixel anomaly maps."));
        form->addRow(tr("Map smoothing sigma"), sigma);
        auto *patchSize = new QSpinBox;
        patchSize->setRange(1, 31);
        patchSize->setSingleStep(2);
        patchSize->setToolTip(tr(
            "Odd aggregation-kernel size in backbone feature-map positions. "
            "Changing it requires rebuilding the PatchCore memory bank."));
        form->addRow(tr("Patch size"), patchSize);
        auto *patchStride = new QSpinBox;
        patchStride->setRange(1, 3);
        patchStride->setToolTip(tr(
            "Distance between patch embeddings in feature-map positions, not "
            "input-image pixels. Changing it requires retraining."));
        form->addRow(tr("Patch stride"), patchStride);
        connect(patchSize, qOverload<int>(&QSpinBox::valueChanged),
                patchStride, [patchStride](int value) {
                    patchStride->setMaximum(value);
                });
        auto *precision = new QComboBox;
        precision->addItems({QStringLiteral("float32"), QStringLiteral("float16")});
        form->addRow(tr("Precision"), precision);

        setControls = [=](const QJsonObject &parameters) {
            backbone->setCurrentText(
                parameters.value(QStringLiteral("backbone")).toString());
            const QJsonArray selected =
                parameters.value(QStringLiteral("layers")).toArray();
            for (QCheckBox *check : layerChecks) {
                check->setChecked(
                    selected.contains(check->property("layerName").toString()));
            }
            pretrained->setChecked(
                parameters.value(QStringLiteral("pre_trained")).toBool());
            coreset->setValue(
                parameters.value(QStringLiteral("coreset_sampling_ratio")).toDouble());
            neighbors->setValue(
                parameters.value(QStringLiteral("num_neighbors")).toInt());
            sigma->setValue(
                parameters.value(QStringLiteral("sigma")).toInt(4));
            patchSize->setValue(
                parameters.value(QStringLiteral("patch_size")).toInt(3));
            patchStride->setMaximum(patchSize->value());
            patchStride->setValue(
                parameters.value(QStringLiteral("patch_stride")).toInt(1));
            precision->setCurrentText(
                parameters.value(QStringLiteral("precision")).toString());
        };
        collectParameters = [=] {
            QJsonArray layers;
            for (QCheckBox *check : layerChecks) {
                if (check->isChecked()) {
                    layers.append(check->property("layerName").toString());
                }
            }
            return QJsonObject {
                {QStringLiteral("backbone"), backbone->currentText()},
                {QStringLiteral("layers"), layers},
                {QStringLiteral("pre_trained"), pretrained->isChecked()},
                {QStringLiteral("coreset_sampling_ratio"), coreset->value()},
                {QStringLiteral("num_neighbors"), neighbors->value()},
                {QStringLiteral("sigma"), sigma->value()},
                {QStringLiteral("patch_size"), patchSize->value()},
                {QStringLiteral("patch_stride"), patchStride->value()},
                {QStringLiteral("precision"), precision->currentText()},
            };
        };
    } else if (modelName == QStringLiteral("PaDiM")) {
        auto *backbone = new QComboBox;
        backbone->addItems(supportedBackbones());
        form->addRow(tr("Backbone"), backbone);

        auto *layersWidget = new QWidget;
        auto *layersLayout = new QHBoxLayout(layersWidget);
        layersLayout->setContentsMargins(0, 0, 0, 0);
        layersLayout->setSpacing(12);
        QList<QCheckBox *> layerChecks;
        for (const QString &layerName : supportedLayers()) {
            auto *check = new QCheckBox(layerName);
            check->setProperty("layerName", layerName);
            layerChecks.append(check);
            layersLayout->addWidget(check);
        }
        layersLayout->addStretch();
        form->addRow(tr("Feature layers"), layersWidget);

        auto *pretrained = new QCheckBox(tr("Use pretrained backbone weights"));
        form->addRow(tr("Weights"), pretrained);
        auto *featuresWidget = new QWidget;
        auto *featuresLayout = new QHBoxLayout(featuresWidget);
        featuresLayout->setContentsMargins(0, 0, 0, 0);
        auto *automaticFeatures = new QCheckBox(tr("Automatic"));
        auto *features = new QSpinBox;
        features->setRange(1, 65536);
        featuresLayout->addWidget(automaticFeatures);
        featuresLayout->addWidget(features, 1);
        form->addRow(tr("Retained features"), featuresWidget);
        connect(automaticFeatures, &QCheckBox::toggled, features,
                [features](bool automatic) { features->setEnabled(!automatic); });

        setControls = [=](const QJsonObject &parameters) {
            backbone->setCurrentText(
                parameters.value(QStringLiteral("backbone")).toString());
            const QJsonArray selected =
                parameters.value(QStringLiteral("layers")).toArray();
            for (QCheckBox *check : layerChecks) {
                check->setChecked(
                    selected.contains(check->property("layerName").toString()));
            }
            pretrained->setChecked(
                parameters.value(QStringLiteral("pre_trained")).toBool());
            const QJsonValue nFeatures =
                parameters.value(QStringLiteral("n_features"));
            automaticFeatures->setChecked(nFeatures.isNull()
                                          || nFeatures.isUndefined());
            features->setValue(nFeatures.isDouble() ? nFeatures.toInt() : 100);
        };
        collectParameters = [=] {
            QJsonArray layers;
            for (QCheckBox *check : layerChecks) {
                if (check->isChecked()) {
                    layers.append(check->property("layerName").toString());
                }
            }
            return QJsonObject {
                {QStringLiteral("backbone"), backbone->currentText()},
                {QStringLiteral("layers"), layers},
                {QStringLiteral("pre_trained"), pretrained->isChecked()},
                {QStringLiteral("n_features"),
                 automaticFeatures->isChecked()
                     ? QJsonValue(QJsonValue::Null)
                     : QJsonValue(features->value())},
            };
        };
    } else {
        auto *imagenetWidget = new QWidget;
        auto *imagenetLayout = new QHBoxLayout(imagenetWidget);
        imagenetLayout->setContentsMargins(0, 0, 0, 0);
        imagenetLayout->setSpacing(8);
        auto *imagenetDir = new QLineEdit;
        auto *browse = new QPushButton(tr("Browse…"));
        browse->setProperty("secondary", true);
        imagenetLayout->addWidget(imagenetDir, 1);
        imagenetLayout->addWidget(browse);
        form->addRow(tr("ImageNette directory"), imagenetWidget);
        connect(browse, &QPushButton::clicked, &dialog, [&, imagenetDir] {
            const QString selected = QFileDialog::getExistingDirectory(
                &dialog, tr("Choose ImageNette directory"),
                imagenetDir->text().trimmed());
            if (!selected.isEmpty()) {
                imagenetDir->setText(QDir::cleanPath(selected));
            }
        });

        auto *teacherChannels = new QSpinBox;
        teacherChannels->setRange(1, 65536);
        form->addRow(tr("Teacher channels"), teacherChannels);
        auto *modelSize = new QComboBox;
        modelSize->addItem(tr("Small (S)"), QStringLiteral("s"));
        modelSize->addItem(tr("Medium (M)"), QStringLiteral("m"));
        form->addRow(tr("Model size"), modelSize);
        auto *learningRate = new QDoubleSpinBox;
        learningRate->setRange(0.00000001, 1.0);
        learningRate->setDecimals(8);
        learningRate->setSingleStep(0.0001);
        form->addRow(tr("Learning rate"), learningRate);
        auto *weightDecay = new QDoubleSpinBox;
        weightDecay->setRange(0.0, 1.0);
        weightDecay->setDecimals(8);
        weightDecay->setSingleStep(0.00001);
        form->addRow(tr("Weight decay"), weightDecay);
        auto *padding = new QCheckBox(tr("Use convolution padding"));
        form->addRow(tr("Padding"), padding);
        auto *padMaps = new QCheckBox(tr("Pad anomaly maps when padding is off"));
        form->addRow(tr("Map padding"), padMaps);
        connect(padding, &QCheckBox::toggled, padMaps,
                [padMaps](bool enabled) { padMaps->setEnabled(!enabled); });

        setControls = [=](const QJsonObject &parameters) {
            imagenetDir->setText(
                parameters.value(QStringLiteral("imagenet_dir")).toString());
            teacherChannels->setValue(
                parameters.value(QStringLiteral("teacher_out_channels")).toInt());
            const int sizeIndex = modelSize->findData(
                parameters.value(QStringLiteral("model_size")).toString());
            modelSize->setCurrentIndex(qMax(0, sizeIndex));
            learningRate->setValue(
                parameters.value(QStringLiteral("lr")).toDouble());
            weightDecay->setValue(
                parameters.value(QStringLiteral("weight_decay")).toDouble());
            padding->setChecked(
                parameters.value(QStringLiteral("padding")).toBool());
            padMaps->setChecked(
                parameters.value(QStringLiteral("pad_maps")).toBool());
            padMaps->setEnabled(!padding->isChecked());
        };
        collectParameters = [=] {
            return QJsonObject {
                {QStringLiteral("imagenet_dir"), imagenetDir->text().trimmed()},
                {QStringLiteral("teacher_out_channels"), teacherChannels->value()},
                {QStringLiteral("model_size"), modelSize->currentData().toString()},
                {QStringLiteral("lr"), learningRate->value()},
                {QStringLiteral("weight_decay"), weightDecay->value()},
                {QStringLiteral("padding"), padding->isChecked()},
                {QStringLiteral("pad_maps"), padMaps->isChecked()},
            };
        };
    }

    const auto setModelControls = setControls;
    setControls = [=](const QJsonObject &parameters) {
        const QSize size = configuredImageSize(parameters);
        {
            const QSignalBlocker widthBlocker(imageWidth);
            const QSignalBlocker heightBlocker(imageHeight);
            imageWidth->setValue(size.width());
            imageHeight->setValue(size.height());
        }
        syncPresetFromFields();
        setModelControls(parameters);
    };
    const auto collectModelParameters = collectParameters;
    collectParameters = [=] {
        QJsonObject parameters = collectModelParameters();
        parameters.insert(
            QStringLiteral("image_size"),
            QJsonObject {
                {QStringLiteral("width"), imageWidth->value()},
                {QStringLiteral("height"), imageHeight->value()},
            });
        return parameters;
    };

    setControls(m_modelParameters.value(modelName,
                                        defaultModelParameters(modelName)));
    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok
            | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText(tr("Apply"));
    layout->addWidget(buttons);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults),
            &QPushButton::clicked, &dialog,
            [=] { setControls(defaultModelParameters(modelName)); });
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&, modelName] {
        const QJsonObject parameters = collectParameters();
        QString error;
        if (!validateModelParameters(modelName, parameters, &error, false)) {
            QMessageBox::warning(&dialog, tr("Invalid model parameters"), error);
            return;
        }
        m_modelParameters.insert(modelName, parameters);
        saveModelParameterSettings(modelName);
        updateModelCardStatus(modelName);
        dialog.accept();
    });
    dialog.exec();
}

bool MainWindow::validateModelParameters(const QString &modelName,
                                         const QJsonObject &parameters,
                                         QString *errorMessage,
                                         bool requireExistingPaths) const
{
    const auto fail = [errorMessage](const QString &message) {
        if (errorMessage) {
            *errorMessage = message;
        }
        return false;
    };
    if (!supportedModelNames().contains(modelName)) {
        return fail(tr("Unsupported model: %1").arg(modelName));
    }

    QStringList expected;
    if (modelName == QStringLiteral("PatchCore")) {
        expected = {QStringLiteral("backbone"), QStringLiteral("layers"),
                    QStringLiteral("pre_trained"),
                    QStringLiteral("coreset_sampling_ratio"),
                    QStringLiteral("num_neighbors"), QStringLiteral("precision"),
                    QStringLiteral("sigma"), QStringLiteral("patch_size"),
                    QStringLiteral("patch_stride")};
    } else if (modelName == QStringLiteral("PaDiM")) {
        expected = {QStringLiteral("backbone"), QStringLiteral("layers"),
                    QStringLiteral("pre_trained"), QStringLiteral("n_features")};
    } else {
        expected = {QStringLiteral("imagenet_dir"),
                    QStringLiteral("teacher_out_channels"),
                    QStringLiteral("model_size"), QStringLiteral("lr"),
                    QStringLiteral("weight_decay"), QStringLiteral("padding"),
                    QStringLiteral("pad_maps")};
    }
    expected.append(QStringLiteral("image_size"));
    for (const QString &key : parameters.keys()) {
        if (!expected.contains(key)) {
            return fail(tr("%1 contains an unsupported setting: %2")
                            .arg(modelName, key));
        }
    }
    for (const QString &key : expected) {
        if (!parameters.contains(key)) {
            return fail(tr("%1 is missing the %2 setting.").arg(modelName, key));
        }
    }

    const QJsonValue imageSizeValue =
        parameters.value(QStringLiteral("image_size"));
    if (!imageSizeValue.isObject()) {
        return fail(tr("%1 image size must contain width and height.")
                        .arg(modelName));
    }
    const QJsonObject imageSize = imageSizeValue.toObject();
    const QStringList imageSizeKeys = imageSize.keys();
    if (imageSizeKeys.size() != 2
        || !imageSize.contains(QStringLiteral("width"))
        || !imageSize.contains(QStringLiteral("height"))) {
        return fail(tr("%1 image size must contain only width and height.")
                        .arg(modelName));
    }
    const QJsonValue widthValue = imageSize.value(QStringLiteral("width"));
    const QJsonValue heightValue = imageSize.value(QStringLiteral("height"));
    const int width = widthValue.toInt(-1);
    const int height = heightValue.toInt(-1);
    if (!widthValue.isDouble() || !heightValue.isDouble()
        || widthValue.toDouble() != width || heightValue.toDouble() != height) {
        return fail(tr("%1 image width and height must be whole numbers.")
                        .arg(modelName));
    }
    const int minimumImageDimension =
        modelName == QStringLiteral("EfficientAD") ? 256 : 32;
    if (width < minimumImageDimension || width > 2048
        || height < minimumImageDimension || height > 2048) {
        return fail(
            tr("%1 image width and height must be between %2 and 2048 pixels.")
                .arg(modelName)
                .arg(minimumImageDimension));
    }

    if (modelName == QStringLiteral("PatchCore")
        || modelName == QStringLiteral("PaDiM")) {
        const QString backbone =
            parameters.value(QStringLiteral("backbone")).toString();
        if (!supportedBackbones().contains(backbone)) {
            return fail(tr("%1 has an unsupported backbone.").arg(modelName));
        }
        const QJsonArray layers =
            parameters.value(QStringLiteral("layers")).toArray();
        if (layers.isEmpty()) {
            return fail(tr("Select at least one %1 feature layer.").arg(modelName));
        }
        QStringList uniqueLayers;
        for (const QJsonValue &layerValue : layers) {
            const QString layer = layerValue.toString();
            if (!supportedLayers().contains(layer)
                || uniqueLayers.contains(layer)) {
                return fail(tr("%1 contains an invalid feature-layer selection.")
                                .arg(modelName));
            }
            uniqueLayers.append(layer);
        }
        if (!parameters.value(QStringLiteral("pre_trained")).isBool()) {
            return fail(tr("%1 pretrained weights must be enabled or disabled.")
                            .arg(modelName));
        }
    }

    if (modelName == QStringLiteral("PatchCore")) {
        const double ratio =
            parameters.value(QStringLiteral("coreset_sampling_ratio")).toDouble(-1.0);
        const int neighbors =
            parameters.value(QStringLiteral("num_neighbors")).toInt(-1);
        const QString precision =
            parameters.value(QStringLiteral("precision")).toString();
        const int sigma = parameters.value(QStringLiteral("sigma")).toInt(-1);
        const int patchSize =
            parameters.value(QStringLiteral("patch_size")).toInt(-1);
        const int patchStride =
            parameters.value(QStringLiteral("patch_stride")).toInt(-1);
        if (ratio <= 0.0 || ratio > 1.0) {
            return fail(tr("PatchCore coreset ratio must be greater than 0 and at most 1."));
        }
        if (neighbors < 1 || neighbors > 1024) {
            return fail(tr("PatchCore nearest neighbors must be between 1 and 1024."));
        }
        if (precision != QStringLiteral("float32")
            && precision != QStringLiteral("float16")) {
            return fail(tr("PatchCore precision must be float32 or float16."));
        }
        if (sigma < 1 || sigma > 32) {
            return fail(tr("PatchCore sigma must be between 1 and 32."));
        }
        if (patchSize < 1 || patchSize > 31 || patchSize % 2 == 0) {
            return fail(tr("PatchCore patch size must be an odd value between 1 and 31."));
        }
        if (patchStride < 1 || patchStride > patchSize) {
            return fail(tr("PatchCore patch stride must be between 1 and the patch size."));
        }
    } else if (modelName == QStringLiteral("PaDiM")) {
        const QJsonValue features =
            parameters.value(QStringLiteral("n_features"));
        if (!features.isNull()
            && (features.toInt(-1) < 1 || features.toInt(-1) > 65536)) {
            return fail(tr("PaDiM retained features must be automatic or between 1 and 65,536."));
        }
    } else {
        const QString imagenetDir =
            parameters.value(QStringLiteral("imagenet_dir")).toString().trimmed();
        if (imagenetDir.isEmpty()) {
            return fail(tr("Choose an ImageNette directory for EfficientAD."));
        }
        if (requireExistingPaths && !QDir(imagenetDir).exists()) {
            return fail(tr("The EfficientAD ImageNette directory does not exist: %1")
                            .arg(QDir::toNativeSeparators(imagenetDir)));
        }
        const int channels =
            parameters.value(QStringLiteral("teacher_out_channels")).toInt(-1);
        const QString size =
            parameters.value(QStringLiteral("model_size")).toString();
        const double learningRate =
            parameters.value(QStringLiteral("lr")).toDouble(-1.0);
        const double weightDecay =
            parameters.value(QStringLiteral("weight_decay")).toDouble(-1.0);
        if (channels < 1 || channels > 65536) {
            return fail(tr("EfficientAD teacher channels must be between 1 and 65,536."));
        }
        if (size != QStringLiteral("s") && size != QStringLiteral("m")) {
            return fail(tr("EfficientAD model size must be S or M."));
        }
        if (learningRate <= 0.0 || learningRate > 1.0) {
            return fail(tr("EfficientAD learning rate must be greater than 0 and at most 1."));
        }
        if (weightDecay < 0.0 || weightDecay > 1.0) {
            return fail(tr("EfficientAD weight decay must be between 0 and 1."));
        }
        if (!parameters.value(QStringLiteral("padding")).isBool()
            || !parameters.value(QStringLiteral("pad_maps")).isBool()) {
            return fail(tr("EfficientAD padding settings must be enabled or disabled."));
        }
    }
    return true;
}

bool MainWindow::validateTrainingConfiguration(const QStringList &models,
                                               bool demo,
                                               QString *errorMessage) const
{
    Q_UNUSED(demo)
    for (const QString &modelName : models) {
        const QJsonObject parameters =
            m_modelParameters.value(modelName, defaultModelParameters(modelName));
        if (!validateModelParameters(modelName, parameters, errorMessage, false)) {
            return false;
        }
    }
    return true;
}

QJsonObject MainWindow::modelParameterMap(const QStringList &models) const
{
    QJsonObject result;
    for (const QString &modelName : models) {
        result.insert(modelName,
                      m_modelParameters.value(modelName,
                                              defaultModelParameters(modelName)));
    }
    return result;
}

QJsonObject MainWindow::checkpointModelParameters(const QString &modelName) const
{
    const QJsonObject stored = m_results.value(modelName)
                                   .value(QStringLiteral("model_parameters"))
                                   .toObject();
    if (stored.value(modelName).isObject()) {
        return QJsonObject {{modelName, stored.value(modelName).toObject()}};
    }
    return QJsonObject {{modelName, defaultModelParameters(modelName)}};
}

void MainWindow::loadParameterSettings()
{
    QSettings settings;
    settings.beginGroup(QStringLiteral("TrainingParameters/v1"));
    const auto boundedSetting =
        [&settings](const QString &key, int defaultValue, int minimum,
                    int maximum) {
            bool ok = false;
            const int value = settings.value(key, defaultValue).toInt(&ok);
            return ok && value >= minimum && value <= maximum
                ? value
                : defaultValue;
        };
    m_epochSpin->setValue(
        boundedSetting(QStringLiteral("shared/max_epochs"), 5, 1, 2000000000));
    m_trainBatchSpin->setValue(
        boundedSetting(QStringLiteral("shared/train_batch_size"), 8, 1, 1024));
    m_evalBatchSpin->setValue(
        boundedSetting(QStringLiteral("shared/eval_batch_size"), 8, 1, 1024));
    m_workerSpin->setValue(
        boundedSetting(QStringLiteral("shared/num_workers"), 2, 0, 64));
    m_devicesSpin->setValue(
        boundedSetting(QStringLiteral("shared/devices"), 1, 1, 16));
    const QString accelerator =
        settings.value(QStringLiteral("shared/accelerator"),
                       QStringLiteral("auto")).toString();
    const int acceleratorIndex = m_acceleratorCombo->findData(accelerator);
    m_acceleratorCombo->setCurrentIndex(qMax(0, acceleratorIndex));

    for (const QString &modelName : supportedModelNames()) {
        const QByteArray json =
            settings.value(QStringLiteral("models/") + modelName).toByteArray();
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(json, &parseError);
        QString validationError;
        QJsonObject migrated = defaultModelParameters(modelName);
        if (parseError.error == QJsonParseError::NoError && document.isObject()) {
            const QJsonObject stored = document.object();
            for (auto item = stored.constBegin(); item != stored.constEnd(); ++item) {
                migrated.insert(item.key(), item.value());
            }
        }
        if (parseError.error == QJsonParseError::NoError && document.isObject()
            && validateModelParameters(modelName, migrated,
                                       &validationError, false)) {
            m_modelParameters.insert(modelName, migrated);
        } else {
            m_modelParameters.insert(modelName, defaultModelParameters(modelName));
        }
    }
    settings.endGroup();
    updateSharedSettingsSummary();
    for (const QString &modelName : supportedModelNames()) {
        updateModelCardStatus(modelName);
    }
}

void MainWindow::saveSharedParameterSettings()
{
    if (!m_epochSpin || !m_trainBatchSpin || !m_evalBatchSpin
        || !m_workerSpin || !m_acceleratorCombo || !m_devicesSpin) {
        return;
    }
    QSettings settings;
    settings.beginGroup(QStringLiteral("TrainingParameters/v1"));
    settings.setValue(QStringLiteral("shared/max_epochs"), m_epochSpin->value());
    settings.setValue(QStringLiteral("shared/train_batch_size"),
                      m_trainBatchSpin->value());
    settings.setValue(QStringLiteral("shared/eval_batch_size"),
                      m_evalBatchSpin->value());
    settings.setValue(QStringLiteral("shared/num_workers"), m_workerSpin->value());
    settings.setValue(QStringLiteral("shared/accelerator"),
                      m_acceleratorCombo->currentData().toString());
    settings.setValue(QStringLiteral("shared/devices"), m_devicesSpin->value());
    settings.endGroup();
}

void MainWindow::saveModelParameterSettings(const QString &modelName)
{
    QSettings settings;
    settings.beginGroup(QStringLiteral("TrainingParameters/v1"));
    settings.setValue(
        QStringLiteral("models/") + modelName,
        QJsonDocument(m_modelParameters.value(
                          modelName, defaultModelParameters(modelName)))
            .toJson(QJsonDocument::Compact));
    settings.endGroup();
}

void MainWindow::updateModelCardStatus(const QString &modelName)
{
    QLabel *label = m_modelConfigurationLabels.value(modelName);
    if (!label) {
        return;
    }
    const QJsonObject parameters =
        m_modelParameters.value(modelName, defaultModelParameters(modelName));
    const bool customized = parameters != defaultModelParameters(modelName);
    label->setText(
        customized
            ? tr("%1 · Customized").arg(imageSizeText(parameters))
            : tr("%1 · Default settings").arg(imageSizeText(parameters)));
    label->setProperty("role", customized ? "winner" : "muted");
    label->setToolTip(QString::fromUtf8(
        QJsonDocument(parameters).toJson(QJsonDocument::Indented)));
    label->style()->unpolish(label);
    label->style()->polish(label);
}

void MainWindow::updateSharedSettingsSummary()
{
    if (!m_sharedSettingsToggle || !m_epochSpin || !m_trainBatchSpin) {
        return;
    }
    m_sharedSettingsToggle->setText(
        tr("Training settings · %1 epochs · batch %2")
            .arg(m_epochSpin->value())
            .arg(m_trainBatchSpin->value()));
}

QLabel *MainWindow::createImageLabel(const QString &resourcePath, const QSize &minimumSize)
{
    auto *label = new QLabel;
    label->setObjectName(QStringLiteral("inspectionImage"));
    label->setMinimumWidth(minimumSize.width());
    label->setFixedHeight(minimumSize.height());
    label->setAlignment(Qt::AlignCenter);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setInspectionImage(label, resourcePath, tr("Preview unavailable"));
    return label;
}

void MainWindow::setInspectionImage(QLabel *label, const QString &path,
                                    const QString &emptyText)
{
    if (!label) {
        return;
    }

    const QPixmap source(path);
    if (path.isEmpty() || source.isNull()) {
        label->setPixmap(QPixmap());
        label->setText(emptyText);
        label->setToolTip(path);
        return;
    }

    label->setText({});
    label->setPixmap(source.scaled(
        QSize(label->minimumWidth(), label->height()),
        Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
    label->setToolTip(QDir::toNativeSeparators(path));
}

void MainWindow::clearPredictionImages()
{
    m_anomalyMapPath.clear();
    setInspectionImage(m_heatmapImage, {},
                       tr("Run real training to generate a heatmap"));
    setInspectionImage(m_predictedMaskImage, {},
                       tr("No prediction for this sample yet"));
    if (m_thresholdSlider) {
        m_thresholdSlider->setEnabled(false);
    }
    if (m_thresholdSpin) {
        m_thresholdSpin->setEnabled(false);
    }
}

void MainWindow::updateInspectionFromDataset()
{
    if (!m_testImageLabel || m_datasetRoot.isEmpty()) {
        return;
    }

    QString sampleFolder = detectedAbnormalFolder(m_datasetRoot);
    if (sampleFolder.isEmpty()) {
        sampleFolder = detectedNormalFolder(m_datasetRoot);
    }
    const QStringList samples = sampleFolder.isEmpty()
        ? QStringList()
        : imageFiles(QDir(m_datasetRoot).filePath(sampleFolder));
    if (samples.isEmpty()) {
        m_testImageLabel->setText(tr("No dataset image found"));
        setInspectionImage(m_inputImage, {}, tr("No image found in the selected dataset"));
        setInspectionImage(m_maskImage, {}, tr("No matching ground-truth mask"));
        clearPredictionImages();
        return;
    }

    const QString samplePath = samples.first();
    m_testImageLabel->setText(
        tr("Dataset sample: %1        1 / %2")
            .arg(QFileInfo(samplePath).fileName(),
                 QLocale().toString(samples.size())));
    m_testImageLabel->setToolTip(QDir::toNativeSeparators(samplePath));
    setInspectionImage(m_inputImage, samplePath, tr("Dataset image unavailable"));
    setInspectionImage(m_maskImage, matchingMaskForImage(samplePath),
                       tr("No matching ground-truth mask"));
    clearPredictionImages();
}

void MainWindow::updateInspectionFromResult(const QJsonObject &result)
{
    const QJsonObject inspection =
        result.value(QStringLiteral("inspection")).toObject();
    if (inspection.isEmpty()) {
        return;
    }

    const QString imagePath =
        inspection.value(QStringLiteral("image_path")).toString();
    const QString maskPath =
        inspection.value(QStringLiteral("mask_path")).toString();
    const int sampleIndex =
        qMax(1, inspection.value(QStringLiteral("sample_index")).toInt(1));
    const int sampleCount =
        qMax(sampleIndex, inspection.value(QStringLiteral("sample_count")).toInt(1));
    if (m_testImageLabel && !imagePath.isEmpty()) {
        m_testImageLabel->setText(
            tr("Test image: %1        %2 / %3")
                .arg(QFileInfo(imagePath).fileName(),
                     QLocale().toString(sampleIndex),
                     QLocale().toString(sampleCount)));
        m_testImageLabel->setToolTip(QDir::toNativeSeparators(imagePath));
    }

    setInspectionImage(m_inputImage, imagePath, tr("Test image unavailable"));
    setInspectionImage(m_maskImage, maskPath, tr("No ground-truth mask for this image"));
    setInspectionImage(
        m_heatmapImage,
        inspection.value(QStringLiteral("heatmap_path")).toString(),
        tr("No heatmap was produced"));

    m_anomalyMapPath =
        inspection.value(QStringLiteral("anomaly_map_path")).toString();
    if (!m_anomalyMapPath.isEmpty() && QFileInfo::exists(m_anomalyMapPath)) {
        m_thresholdSlider->setEnabled(true);
        m_thresholdSpin->setEnabled(true);
        updateThreshold(m_thresholdSlider->value());
    } else {
        setInspectionImage(
            m_predictedMaskImage,
            inspection.value(QStringLiteral("predicted_mask_path")).toString(),
            tr("No predicted mask was produced"));
        m_thresholdSlider->setEnabled(false);
        m_thresholdSpin->setEnabled(false);
    }
}

QWidget *MainWindow::createCapturePage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->setSpacing(16);
    layout->addWidget(makeLabel(tr("Capture dataset"), "pageTitle"));
    layout->addWidget(makeLabel(
        tr("Acquire normal and anomalous samples directly from a Basler camera. "
           "Frames are converted and written by OpenCV."),
        "pageLead"));

    auto *cameraFrame = new QFrame;
    cameraFrame->setObjectName(QStringLiteral("panel"));
    auto *cameraLayout = new QGridLayout(cameraFrame);
    cameraLayout->setContentsMargins(22, 20, 22, 20);
    cameraLayout->setHorizontalSpacing(12);
    cameraLayout->setVerticalSpacing(12);
    cameraLayout->addWidget(makeLabel(tr("Camera"), "fieldLabel"), 0, 0);
    m_cameraCombo = new QComboBox;
    m_cameraCombo->setMinimumWidth(380);
    cameraLayout->addWidget(m_cameraCombo, 0, 1);
    auto *refresh = new QPushButton(tr("Discover"));
    refresh->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
    refresh->setProperty("secondary", true);
    connect(refresh, &QPushButton::clicked, this, &MainWindow::discoverCameras);
    cameraLayout->addWidget(refresh, 0, 2);

    cameraLayout->addWidget(makeLabel(tr("Dataset root"), "fieldLabel"), 1, 0);
    m_captureDatasetEdit = new QLineEdit;
    m_captureDatasetEdit->setPlaceholderText(tr("Choose a dataset folder…"));
    cameraLayout->addWidget(m_captureDatasetEdit, 1, 1);
    auto *browse = new QPushButton(tr("Browse…"));
    browse->setProperty("secondary", true);
    connect(browse, &QPushButton::clicked, this, &MainWindow::chooseDataset);
    cameraLayout->addWidget(browse, 1, 2);

    m_cameraStatus = makeLabel(m_cameraService.availabilityText(), "muted");
    cameraLayout->addWidget(m_cameraStatus, 2, 1, 1, 2);
    layout->addWidget(cameraFrame);

    auto *captureArea = new QHBoxLayout;
    m_capturePreview = createImageLabel(QStringLiteral(":/assets/housing-seal-input.png"),
                                        QSize(620, 300));
    captureArea->addWidget(m_capturePreview, 2);

    auto *captureControls = new QFrame;
    captureControls->setObjectName(QStringLiteral("panel"));
    auto *controls = new QVBoxLayout(captureControls);
    controls->setContentsMargins(24, 24, 24, 24);
    controls->addWidget(makeLabel(tr("Save next frame as"), "sectionTitle"));
    controls->addSpacing(8);
    auto *normal = new QPushButton(tr("Capture normal sample"));
    normal->setProperty("primary", true);
    normal->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    connect(normal, &QPushButton::clicked, this, &MainWindow::captureNormal);
    controls->addWidget(normal);
    auto *anomalous = new QPushButton(tr("Capture anomalous sample"));
    anomalous->setProperty("danger", true);
    anomalous->setIcon(style()->standardIcon(QStyle::SP_MessageBoxWarning));
    connect(anomalous, &QPushButton::clicked, this, &MainWindow::captureAnomalous);
    controls->addWidget(anomalous);
    controls->addSpacing(14);
    controls->addWidget(makeLabel(
        tr("Folder layout\n"
           "normal/      training references\n"
           "abnormal/    validation and test defects\n"
           "masks/       optional pixel masks"),
        "codeHint"));
    controls->addStretch();
    captureArea->addWidget(captureControls, 1);
    layout->addLayout(captureArea, 1);
    return page;
}

QWidget *MainWindow::createDatasetsPage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->setSpacing(16);
    layout->addWidget(makeLabel(tr("Datasets"), "pageTitle"));
    layout->addWidget(makeLabel(
        tr("Point the workbench at a Folder-style Anomalib dataset. "
           "The app recognizes normal/good, abnormal/anomalous/defect, and masks."),
        "pageLead"));

    auto *panel = new QFrame;
    panel->setObjectName(QStringLiteral("panel"));
    auto *form = new QGridLayout(panel);
    form->setContentsMargins(24, 24, 24, 24);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(14);
    form->addWidget(makeLabel(tr("Dataset root"), "fieldLabel"), 0, 0);
    m_datasetPathEdit = new QLineEdit;
    m_datasetPathEdit->setPlaceholderText(tr("Choose normal/abnormal image folders…"));
    form->addWidget(m_datasetPathEdit, 0, 1);
    auto *browse = new QPushButton(tr("Browse…"));
    browse->setProperty("primary", true);
    connect(browse, &QPushButton::clicked, this, &MainWindow::chooseDataset);
    form->addWidget(browse, 0, 2);
    auto *rescan = new QPushButton(tr("Rescan"));
    rescan->setProperty("secondary", true);
    connect(rescan, &QPushButton::clicked, this, &MainWindow::refreshDataset);
    form->addWidget(rescan, 1, 2);
    layout->addWidget(panel);

    auto *hint = new QFrame;
    hint->setObjectName(QStringLiteral("panel"));
    auto *hintLayout = new QVBoxLayout(hint);
    hintLayout->setContentsMargins(24, 24, 24, 24);
    hintLayout->addWidget(makeLabel(tr("Recommended structure"), "sectionTitle"));
    hintLayout->addWidget(makeLabel(
        tr("dataset/\n"
           "├── normal/       defect-free training images\n"
           "├── abnormal/     known anomalous test images\n"
           "└── masks/        optional binary masks matching abnormal images"),
        "codeHint"));
    layout->addWidget(hint);
    layout->addStretch();
    return page;
}

QWidget *MainWindow::createModelsPage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->setSpacing(16);
    layout->addWidget(makeLabel(tr("Model catalog"), "pageTitle"));
    layout->addWidget(makeLabel(
        tr("Start with three complementary Anomalib models. "
           "The experiment bench records the exact checkpoint and metrics for each run."),
        "pageLead"));

    auto *table = new QTableWidget(3, 4);
    table->setHorizontalHeaderLabels(
        {tr("Model"), tr("Best fit"), tr("Training style"), tr("Default status")});
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    const QList<QStringList> rows = {
        {tr("PatchCore"), tr("Strong general baseline"), tr("Memory bank"), tr("Enabled")},
        {tr("PaDiM"), tr("Fast classical baseline"), tr("Feature statistics"), tr("Enabled")},
        {tr("EfficientAD"), tr("Low-latency deployment"), tr("Student–teacher"), tr("Enabled")},
    };
    for (int row = 0; row < rows.size(); ++row) {
        for (int column = 0; column < rows[row].size(); ++column) {
            table->setItem(row, column, tableItem(rows[row][column], Qt::AlignLeft | Qt::AlignVCenter));
        }
        table->setRowHeight(row, 58);
    }
    table->setMaximumHeight(250);
    layout->addWidget(table);
    layout->addStretch();
    return page;
}

QWidget *MainWindow::createInspectPage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->setSpacing(14);
    layout->addWidget(makeLabel(tr("Test a trained model"), "pageTitle"));
    layout->addWidget(makeLabel(
        tr("Choose an image or acquire one from the selected Basler camera. "
           "The workbench loads the chosen training checkpoint and returns an "
           "anomaly decision, score, heatmap, and mask."),
        "pageLead"));

    auto *controlsPanel = new QFrame;
    controlsPanel->setObjectName(QStringLiteral("panel"));
    auto *controls = new QGridLayout(controlsPanel);
    controls->setContentsMargins(20, 18, 20, 18);
    controls->setHorizontalSpacing(10);
    controls->setVerticalSpacing(10);

    controls->addWidget(makeLabel(tr("Active model"), "fieldLabel"), 0, 0);
    m_inferenceModelName = makeLabel(tr("No model loaded"), "summaryValue");
    controls->addWidget(m_inferenceModelName, 0, 1);

    controls->addWidget(makeLabel(tr("Decision threshold"), "fieldLabel"), 0, 2);
    m_inferenceThresholdSpin = new QSpinBox;
    m_inferenceThresholdSpin->setRange(1, 99);
    m_inferenceThresholdSpin->setValue(60);
    m_inferenceThresholdSpin->setSuffix(QStringLiteral("%"));
    controls->addWidget(m_inferenceThresholdSpin, 0, 3);

    controls->addWidget(makeLabel(tr("Map smoothing"), "fieldLabel"), 1, 0);
    auto *sigmaControl = new QWidget;
    auto *sigmaLayout = new QHBoxLayout(sigmaControl);
    sigmaLayout->setContentsMargins(0, 0, 0, 0);
    sigmaLayout->setSpacing(8);
    m_inferenceSigmaOverrideCheck = new QCheckBox(tr("Override model sigma"));
    m_inferenceSigmaOverrideCheck->setToolTip(tr(
        "Temporarily changes PatchCore heatmap smoothing for this loaded model. "
        "Image-level scoring is unchanged."));
    m_inferenceSigmaSpin = new QSpinBox;
    m_inferenceSigmaSpin->setRange(1, 32);
    m_inferenceSigmaSpin->setValue(4);
    m_inferenceSigmaSpin->setPrefix(tr("σ "));
    m_inferenceSigmaSpin->setEnabled(false);
    connect(m_inferenceSigmaOverrideCheck, &QCheckBox::toggled,
            m_inferenceSigmaSpin, [this](bool checked) {
                m_inferenceSigmaSpin->setEnabled(
                    checked && m_inferenceSigmaOverrideCheck->isEnabled());
            });
    sigmaLayout->addWidget(m_inferenceSigmaOverrideCheck);
    sigmaLayout->addWidget(m_inferenceSigmaSpin);
    sigmaLayout->addStretch();
    controls->addWidget(sigmaControl, 1, 1, 1, 3);

    m_inferenceModelDetails = makeLabel(
        tr("Load an Anomalib checkpoint or exported model to begin."), "muted");
    m_inferenceModelDetails->setWordWrap(true);
    controls->addWidget(m_inferenceModelDetails, 2, 1, 1, 3);

    auto *modelActions = new QHBoxLayout;
    m_inferenceLoadModelButton = new QPushButton(tr("Load model…"));
    m_inferenceLoadModelButton->setProperty("secondary", true);
    connect(m_inferenceLoadModelButton, &QPushButton::clicked,
            this, &MainWindow::chooseInferenceModel);
    modelActions->addWidget(m_inferenceLoadModelButton);
    m_inferenceUseWinnerButton = new QPushButton(tr("Use trained winner"));
    m_inferenceUseWinnerButton->setProperty("secondary", true);
    connect(m_inferenceUseWinnerButton, &QPushButton::clicked,
            this, &MainWindow::useWinnerForInference);
    modelActions->addWidget(m_inferenceUseWinnerButton);
    modelActions->addStretch();
    controls->addLayout(modelActions, 3, 1, 1, 3);

    controls->addWidget(makeLabel(tr("Test image"), "fieldLabel"), 4, 0);
    m_inferenceImageEdit = new QLineEdit;
    m_inferenceImageEdit->setPlaceholderText(
        tr("Choose a PNG, JPEG, BMP, or TIFF image…"));
    controls->addWidget(m_inferenceImageEdit, 4, 1, 1, 2);
    auto *browse = new QPushButton(tr("Browse…"));
    browse->setProperty("secondary", true);
    connect(browse, &QPushButton::clicked, this, &MainWindow::chooseInferenceImage);
    controls->addWidget(browse, 4, 3);

    auto *actions = new QHBoxLayout;
    m_inferenceRunButton = new QPushButton(tr("Test selected image"));
    m_inferenceRunButton->setProperty("primary", true);
    m_inferenceRunButton->setIcon(
        style()->standardIcon(QStyle::SP_MediaPlay));
    connect(m_inferenceRunButton, &QPushButton::clicked,
            this, &MainWindow::runInference);
    actions->addWidget(m_inferenceRunButton);
    m_inferenceCaptureButton = new QPushButton(tr("Capture && test"));
    m_inferenceCaptureButton->setProperty("secondary", true);
    m_inferenceCaptureButton->setIcon(
        style()->standardIcon(QStyle::SP_DialogSaveButton));
    connect(m_inferenceCaptureButton, &QPushButton::clicked,
            this, &MainWindow::captureAndRunInference);
    actions->addWidget(m_inferenceCaptureButton);
    actions->addWidget(makeLabel(
        tr("Camera selection is shared with the Capture page."), "muted"));
    actions->addStretch();
    controls->addLayout(actions, 5, 1, 1, 3);
    m_inferenceRunButton->setEnabled(false);
    m_inferenceCaptureButton->setEnabled(false);
    layout->addWidget(controlsPanel);

    auto *resultPanel = new QFrame;
    resultPanel->setObjectName(QStringLiteral("panel"));
    auto *result = new QHBoxLayout(resultPanel);
    result->setContentsMargins(20, 12, 20, 12);
    m_inferenceStatus = makeLabel(tr("Ready for an image"), "winner");
    result->addWidget(m_inferenceStatus, 1);
    result->addWidget(makeLabel(tr("Anomaly score"), "muted"));
    m_inferenceScore = makeLabel(QStringLiteral("—"), "summaryValue");
    result->addWidget(m_inferenceScore);
    result->addSpacing(20);
    result->addWidget(makeLabel(tr("Latency"), "muted"));
    m_inferenceLatency = makeLabel(QStringLiteral("—"), "summaryValue");
    result->addWidget(m_inferenceLatency);
    layout->addWidget(resultPanel);

    auto *images = new QHBoxLayout;
    images->setSpacing(12);
    const QList<QPair<QString, QLabel **>> imageColumns = {
        {tr("Input image"), &m_inferenceInputImage},
        {tr("Anomaly heatmap"), &m_inferenceHeatmapImage},
        {tr("Predicted mask"), &m_inferenceMaskImage},
    };
    for (const auto &column : imageColumns) {
        auto *container = new QVBoxLayout;
        container->setSpacing(6);
        container->addWidget(makeLabel(column.first, "imageTitle"));
        *column.second = createImageLabel({}, QSize(230, 230));
        container->addWidget(*column.second);
        images->addLayout(container, 1);
    }
    layout->addLayout(images, 1);
    return page;
}

QWidget *MainWindow::createSettingsPage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->setSpacing(16);
    layout->addWidget(makeLabel(tr("Settings"), "pageTitle"));
    layout->addWidget(makeLabel(
        tr("Use a dedicated Python environment with Anomalib installed. "
           "Training and testing run outside the Qt process."),
        "pageLead"));

    auto *pythonPanel = new QFrame;
    pythonPanel->setObjectName(QStringLiteral("panel"));
    auto *form = new QGridLayout(pythonPanel);
    form->setContentsMargins(24, 24, 24, 24);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(14);
    form->addWidget(makeLabel(tr("Python executable"), "fieldLabel"), 0, 0);
    m_pythonEdit = new QLineEdit(defaultPythonExecutable());
    connect(m_pythonEdit, &QLineEdit::editingFinished, this, [this] {
        QSettings settings;
        const QString executable = m_pythonEdit->text().trimmed();
        if (executable.isEmpty()) {
            settings.remove(QStringLiteral("Environment/pythonExecutable"));
        } else {
            settings.setValue(QStringLiteral("Environment/pythonExecutable"),
                              QDir::cleanPath(executable));
        }
    });
    form->addWidget(m_pythonEdit, 0, 1);
    auto *browse = new QPushButton(tr("Browse…"));
    browse->setProperty("secondary", true);
    connect(browse, &QPushButton::clicked, this, &MainWindow::browsePython);
    form->addWidget(browse, 0, 2);

    m_demoMode = new QCheckBox(tr("Use demonstration results"));
    {
        QSettings settings;
        m_demoMode->setChecked(
            settings.value(QStringLiteral("Environment/demoMode"), true).toBool());
    }
    connect(m_demoMode, &QCheckBox::toggled, this, [](bool enabled) {
        QSettings settings;
        settings.setValue(QStringLiteral("Environment/demoMode"), enabled);
    });
    m_demoMode->setToolTip(tr("Runs the complete UI/worker flow without importing Anomalib."));
    form->addWidget(m_demoMode, 1, 1, 1, 2);

    auto *verify = new QPushButton(tr("Verify environment"));
    verify->setProperty("primary", true);
    connect(verify, &QPushButton::clicked, this, &MainWindow::verifyEnvironment);
    form->addWidget(verify, 2, 1, Qt::AlignLeft);
    m_environmentStatus = makeLabel(tr("Not checked yet"), "muted");
    form->addWidget(m_environmentStatus, 3, 1, 1, 2);
    layout->addWidget(pythonPanel);

    auto *assetPanel = new QFrame;
    assetPanel->setObjectName(QStringLiteral("panel"));
    auto *assetLayout = new QVBoxLayout(assetPanel);
    assetLayout->setContentsMargins(24, 22, 24, 22);
    assetLayout->setSpacing(10);
    assetLayout->addWidget(makeLabel(tr("Local model assets"), "sectionTitle"));
    auto *assetLead = makeLabel(
        tr("Training, inference, verification, and export run offline. "
           "Only the explicit download action below may contact the listed "
           "public sources; no dataset or result is uploaded."),
        "muted");
    assetLead->setWordWrap(true);
    assetLayout->addWidget(assetLead);
    m_assetRootLabel = makeLabel(
        tr("Managed location: %1")
            .arg(QDir::toNativeSeparators(modelAssetsDirectory())),
        "muted");
    m_assetRootLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_assetRootLabel->setWordWrap(true);
    assetLayout->addWidget(m_assetRootLabel);

    m_assetTable = new QTableWidget(managedModelAssets().size(), 4);
    m_assetTable->setHorizontalHeaderLabels(
        {tr("Asset"), tr("Source"), tr("Status"), tr("Local path")});
    m_assetTable->verticalHeader()->hide();
    m_assetTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_assetTable->setSelectionMode(QAbstractItemView::NoSelection);
    m_assetTable->setFocusPolicy(Qt::NoFocus);
    m_assetTable->horizontalHeader()->setSectionResizeMode(
        0, QHeaderView::ResizeToContents);
    m_assetTable->horizontalHeader()->setSectionResizeMode(
        1, QHeaderView::ResizeToContents);
    m_assetTable->horizontalHeader()->setSectionResizeMode(
        2, QHeaderView::ResizeToContents);
    m_assetTable->horizontalHeader()->setSectionResizeMode(
        3, QHeaderView::Stretch);
    m_assetTable->setMinimumHeight(190);
    int assetRow = 0;
    for (const auto &asset : managedModelAssets()) {
        auto *name = new QTableWidgetItem(asset.second);
        name->setData(Qt::UserRole, asset.first);
        m_assetTable->setItem(assetRow, 0, name);
        m_assetTable->setItem(assetRow, 1,
                              new QTableWidgetItem(tr("Not checked")));
        m_assetTable->setItem(assetRow, 2,
                              new QTableWidgetItem(tr("Not checked")));
        m_assetTable->setItem(assetRow, 3, new QTableWidgetItem);
        ++assetRow;
    }
    assetLayout->addWidget(m_assetTable);

    auto *assetActions = new QHBoxLayout;
    m_refreshAssetsButton = new QPushButton(tr("Refresh status"));
    m_refreshAssetsButton->setProperty("secondary", true);
    connect(m_refreshAssetsButton, &QPushButton::clicked,
            this, &MainWindow::refreshAssetStatus);
    assetActions->addWidget(m_refreshAssetsButton);
    m_downloadAssetsButton = new QPushButton(tr("Download missing assets…"));
    m_downloadAssetsButton->setProperty("primary", true);
    m_downloadAssetsButton->setEnabled(false);
    connect(m_downloadAssetsButton, &QPushButton::clicked,
            this, &MainWindow::downloadMissingAssets);
    assetActions->addWidget(m_downloadAssetsButton);
    assetActions->addStretch();
    assetLayout->addLayout(assetActions);

    {
        QList<QJsonObject> storedStatuses;
        QSettings settings;
        settings.beginGroup(QStringLiteral("AssetStatus/v1"));
        for (const auto &asset : managedModelAssets()) {
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(
                settings.value(asset.first).toByteArray(), &parseError);
            if (parseError.error == QJsonParseError::NoError
                && document.isObject()
                && document.object()
                       .value(QStringLiteral("managed_root"))
                       .toString()
                    == modelAssetsDirectory()) {
                storedStatuses.append(document.object());
            }
        }
        settings.endGroup();
        for (const QJsonObject &status : std::as_const(storedStatuses)) {
            updateAssetRow(status);
        }
    }
    layout->addWidget(assetPanel);

    auto *sdkPanel = new QFrame;
    sdkPanel->setObjectName(QStringLiteral("panel"));
    auto *sdkLayout = new QVBoxLayout(sdkPanel);
    sdkLayout->setContentsMargins(24, 24, 24, 24);
    sdkLayout->addWidget(makeLabel(tr("Native image pipeline"), "sectionTitle"));
    m_sdkStatus = makeLabel(m_cameraService.availabilityText(),
                            m_cameraService.isAvailable() ? "successText" : "errorText");
    sdkLayout->addWidget(m_sdkStatus);
    sdkLayout->addWidget(makeLabel(
        tr("Basler Pylon handles camera discovery and grabbing. "
           "OpenCV converts and saves captured frames into the selected dataset."),
        "muted"));
    layout->addWidget(sdkPanel);
    layout->addStretch();
    return page;
}

QWidget *MainWindow::createPlaceholderPage(const QString &title,
                                           const QString &description)
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(34, 28, 34, 28);
    layout->addWidget(makeLabel(title, "pageTitle"));
    layout->addWidget(makeLabel(description, "pageLead"));
    layout->addStretch();
    return page;
}

void MainWindow::applyTheme()
{
    setStyleSheet(QStringLiteral(R"(
        * {
            font-family: "Segoe UI";
            font-size: 14px;
            color: #202328;
        }
        QMainWindow, QWidget#appRoot, QWidget#trainCenter {
            background: #fbfcfd;
        }
        QMenuBar {
            background: #ffffff;
            border-bottom: 1px solid #dfe3e8;
            padding: 3px 8px;
        }
        QMenuBar::item {
            padding: 6px 10px;
            background: transparent;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background: #e9f2ff;
        }
        QMenu {
            background: white;
            border: 1px solid #d5dbe3;
            padding: 5px;
        }
        QMenu::item { padding: 7px 28px 7px 12px; }
        QFrame#navigation {
            background: #f2f5f8;
            border-right: 1px solid #d9dee5;
        }
        QLabel[role="brand"] {
            font-size: 14px;
            font-weight: 650;
            color: #17202a;
        }
        QToolButton[nav="true"] {
            border: 0;
            border-radius: 5px;
            padding: 8px 12px;
            text-align: left;
            color: #2f3944;
            background: transparent;
        }
        QToolButton[nav="true"]:hover { background: #e6ebf1; }
        QToolButton[nav="true"]:checked {
            color: #ffffff;
            background: #1267d5;
        }
        QLabel[role="pageTitle"] {
            font-size: 26px;
            font-weight: 650;
            color: #111418;
        }
        QLabel[role="pageLead"] {
            font-size: 15px;
            color: #56616d;
        }
        QLabel[role="sectionTitle"], QLabel[role="winner"] {
            font-size: 15px;
            font-weight: 650;
            color: #181c21;
        }
        QLabel[role="winner"] { font-size: 16px; }
        QLabel[role="muted"], QLabel[role="statusDetail"] {
            color: #65717d;
            font-size: 13px;
        }
        QLabel[role="fieldLabel"], QLabel[role="imageTitle"] {
            font-weight: 600;
            color: #30363d;
        }
        QLabel[role="summaryValue"] {
            font-size: 16px;
            font-weight: 650;
            color: #15191e;
        }
        QLabel[role="successText"] { color: #178647; font-weight: 600; }
        QLabel[role="errorText"] { color: #c43f3f; font-weight: 600; }
        QLabel[role="warningText"] { color: #9a5b00; font-weight: 600; }
        QLabel[role="codeHint"] {
            font-family: "Cascadia Mono";
            color: #42505e;
            background: #f5f7f9;
            border: 1px solid #e2e6ea;
            padding: 12px;
        }
        QFrame#summaryBar, QFrame#panel, QFrame#parameterPanel {
            background: #ffffff;
            border: 1px solid #d8dde3;
            border-radius: 5px;
        }
        QFrame#parameterPanel {
            background: #f8fafc;
        }
        QFrame#modelCard {
            background: #ffffff;
            border: 1px solid #cfd5dc;
            border-radius: 5px;
        }
        QFrame#modelCard[selected="true"] {
            background: #f0f6ff;
            border: 1px solid #1b70dc;
        }
        QCheckBox[modelSelector="true"] {
            color: #202328;
            font-weight: 600;
        }
        QFrame#inspectionPanel {
            background: #ffffff;
            border-left: 1px solid #d9dee5;
        }
        QFrame#divider { color: #e0e4e8; background: #e0e4e8; }
        QLabel#inspectionImage {
            background: #111417;
            border: 1px solid #cfd5dc;
        }
        QPushButton, QToolButton[secondary="true"], QComboBox, QLineEdit,
        QSpinBox, QDoubleSpinBox {
            min-height: 34px;
            border-radius: 4px;
            border: 1px solid #cfd5dc;
            background: #ffffff;
            padding: 0 12px;
        }
        QPushButton:hover, QToolButton[secondary="true"]:hover {
            border-color: #8ca8ca;
            background: #f6f9fd;
        }
        QPushButton[primary="true"] {
            color: #ffffff;
            border: 1px solid #0d5ec6;
            background: #1267d5;
            font-weight: 600;
        }
        QPushButton[primary="true"]:hover { background: #0d59bd; }
        QPushButton[success="true"] {
            color: #ffffff;
            border: 1px solid #198348;
            background: #1b914f;
            font-weight: 600;
        }
        QPushButton[danger="true"] {
            color: #ffffff;
            border: 1px solid #c44444;
            background: #d54b4b;
            font-weight: 600;
        }
        QPushButton:disabled { color: #98a1aa; background: #eef1f4; }
        QProgressBar {
            border: 0;
            background: #e4e8ec;
        }
        QProgressBar::chunk { background: #238851; }
        QTableWidget, QPlainTextEdit {
            background: #ffffff;
            border: 1px solid #d7dce2;
            gridline-color: #e0e4e8;
            selection-background-color: #eaf4ed;
            selection-color: #202328;
        }
        QHeaderView::section {
            background: #f8f9fb;
            border: 0;
            border-right: 1px solid #dfe3e7;
            border-bottom: 1px solid #d7dce2;
            padding: 10px 8px;
            color: #424a53;
            font-weight: 600;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #d7dce2;
        }
        QSlider::sub-page:horizontal { background: #1267d5; }
        QSlider::handle:horizontal {
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
            background: #70a6ed;
            border: 1px solid #1267d5;
        }
        QStatusBar {
            background: #ffffff;
            border-top: 1px solid #dfe3e8;
        }
        QLabel#statusMode { color: #53606d; padding-left: 8px; }
    )"));
}

void MainWindow::populateReferenceState()
{
    m_normalCount->setText(QStringLiteral("1,200"));
    m_anomalousCount->setText(QStringLiteral("300"));
    m_maskCount->setText(QStringLiteral("300"));
    m_totalCount->setText(QStringLiteral("1,500"));

    const QList<QJsonObject> reference = {
        {{QStringLiteral("model"), QStringLiteral("PatchCore")},
         {QStringLiteral("image_auroc"), 0.978},
         {QStringLiteral("pixel_auroc"), 0.962},
         {QStringLiteral("image_f1"), 0.917},
         {QStringLiteral("latency_ms"), 38.2},
         {QStringLiteral("model_size_mb"), 1270.0},
         {QStringLiteral("checkpoint"), QStringLiteral("demo/patchcore/model.ckpt")}},
        {{QStringLiteral("model"), QStringLiteral("PaDiM")},
         {QStringLiteral("image_auroc"), 0.957},
         {QStringLiteral("pixel_auroc"), 0.929},
         {QStringLiteral("image_f1"), 0.881},
         {QStringLiteral("latency_ms"), 22.7},
         {QStringLiteral("model_size_mb"), 412.0},
         {QStringLiteral("checkpoint"), QStringLiteral("demo/padim/model.ckpt")}},
        {{QStringLiteral("model"), QStringLiteral("EfficientAD")},
         {QStringLiteral("image_auroc"), 0.942},
         {QStringLiteral("pixel_auroc"), 0.909},
         {QStringLiteral("image_f1"), 0.846},
         {QStringLiteral("latency_ms"), 16.4},
         {QStringLiteral("model_size_mb"), 298.0},
         {QStringLiteral("checkpoint"), QStringLiteral("demo/efficientad/model.ckpt")}},
    };
    for (const QJsonObject &result : reference) {
        updateResultRow(result);
    }
    updateWinner(QStringLiteral("PatchCore"));

    m_trainingLog->setPlainText(
        tr("10:21:33   PatchCore    Training completed · best image AUROC 0.978\n"
           "10:19:12   PatchCore    Testing completed · pixel AUROC 0.962\n"
           "10:02:51   PaDiM       Training completed · image AUROC 0.957\n"
           "09:53:39   EfficientAD Training completed · image AUROC 0.942"));
}

void MainWindow::navigateTo(int index)
{
    if (!m_pages || index < 0 || index >= m_pages->count()) {
        return;
    }
    m_pages->setCurrentIndex(index);
    if (index < m_navButtons.size()) {
        m_navButtons[index]->setChecked(true);
    }
    if (index == 5 && !m_assetStatusRefreshRequested
        && m_worker && !m_worker->isRunning()) {
        m_assetStatusRefreshRequested = true;
        QTimer::singleShot(0, this, &MainWindow::refreshAssetStatus);
    }
    if (index == 4 && !m_inferenceModelReady
        && m_inferenceWorker && !m_inferenceWorker->isRunning()) {
        QTimer::singleShot(0, this, &MainWindow::restoreInferenceModel);
    }
}

void MainWindow::chooseDataset()
{
    const QString start = m_datasetRoot.isEmpty()
        ? QStandardPaths::writableLocation(QStandardPaths::PicturesLocation)
        : m_datasetRoot;
    const QString selected = QFileDialog::getExistingDirectory(
        this, tr("Choose Anomalib dataset root"), start);
    if (selected.isEmpty()) {
        return;
    }
    m_datasetRoot = QDir::cleanPath(selected);
    if (m_datasetPathEdit) {
        m_datasetPathEdit->setText(m_datasetRoot);
    }
    if (m_captureDatasetEdit) {
        m_captureDatasetEdit->setText(m_datasetRoot);
    }
    if (m_datasetPathLabel) {
        m_datasetPathLabel->setText(QFileInfo(m_datasetRoot).fileName());
    }
    updateDatasetCounts(m_datasetRoot);
    updateInspectionFromDataset();
    showStatus(tr("Dataset loaded: %1").arg(m_datasetRoot));
}

void MainWindow::refreshDataset()
{
    const QString path = m_datasetPathEdit ? m_datasetPathEdit->text().trimmed()
                                           : m_datasetRoot;
    if (path.isEmpty() || !QDir(path).exists()) {
        showStatus(tr("Choose an existing dataset folder first."), true);
        return;
    }
    m_datasetRoot = QDir::cleanPath(path);
    updateDatasetCounts(m_datasetRoot);
    updateInspectionFromDataset();
    showStatus(tr("Dataset scan completed."));
}

void MainWindow::updateDatasetCounts(const QString &rootPath)
{
    const QString normalName = detectedNormalFolder(rootPath);
    const QString abnormalName = detectedAbnormalFolder(rootPath);
    const QString maskName = detectedMaskFolder(rootPath);
    const int normal = normalName.isEmpty()
        ? 0 : countImages(QDir(rootPath).filePath(normalName));
    const int abnormal = abnormalName.isEmpty()
        ? 0 : countImages(QDir(rootPath).filePath(abnormalName));
    const int masks = maskName.isEmpty()
        ? 0 : countImages(QDir(rootPath).filePath(maskName));
    m_normalCount->setText(QLocale().toString(normal));
    m_anomalousCount->setText(QLocale().toString(abnormal));
    m_maskCount->setText(QLocale().toString(masks));
    m_totalCount->setText(QLocale().toString(normal + abnormal));
    m_datasetDetails->setText(
        tr("Detected: %1  ·  %2  ·  %3  ·  Folder datamodule")
            .arg(normalName.isEmpty() ? tr("no normal folder") : normalName,
                 abnormalName.isEmpty() ? tr("no abnormal folder") : abnormalName,
                 maskName.isEmpty() ? tr("no masks") : maskName));
}

QString MainWindow::detectedNormalFolder(const QString &rootPath) const
{
    for (const QString &name : {QStringLiteral("normal"), QStringLiteral("good")}) {
        if (QDir(QDir(rootPath).filePath(name)).exists()) {
            return name;
        }
    }
    return {};
}

QString MainWindow::detectedAbnormalFolder(const QString &rootPath) const
{
    for (const QString &name : {QStringLiteral("abnormal"), QStringLiteral("anomalous"),
                                QStringLiteral("defect"), QStringLiteral("bad")}) {
        if (QDir(QDir(rootPath).filePath(name)).exists()) {
            return name;
        }
    }
    return {};
}

QString MainWindow::detectedMaskFolder(const QString &rootPath) const
{
    for (const QString &name : {QStringLiteral("masks"), QStringLiteral("mask"),
                                QStringLiteral("ground_truth")}) {
        if (QDir(QDir(rootPath).filePath(name)).exists()) {
            return name;
        }
    }
    return {};
}

QStringList MainWindow::imageFiles(const QString &path) const
{
    static const QStringList filters = {
        QStringLiteral("*.png"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
        QStringLiteral("*.bmp"), QStringLiteral("*.tif"), QStringLiteral("*.tiff")
    };
    QStringList files;
    QDirIterator iterator(path, filters, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        files.append(QDir::cleanPath(iterator.next()));
    }
    files.sort(Qt::CaseInsensitive);
    return files;
}

QString MainWindow::matchingMaskForImage(const QString &imagePath) const
{
    const QString maskName = detectedMaskFolder(m_datasetRoot);
    if (imagePath.isEmpty() || maskName.isEmpty()) {
        return {};
    }

    const QString abnormalName = detectedAbnormalFolder(m_datasetRoot);
    const QString maskRoot = QDir(m_datasetRoot).filePath(maskName);
    const QFileInfo imageInfo(imagePath);
    QString relativePath = imageInfo.fileName();
    if (!abnormalName.isEmpty()) {
        relativePath = QDir(QDir(m_datasetRoot).filePath(abnormalName))
                           .relativeFilePath(imagePath);
    }

    const QFileInfo relativeInfo(relativePath);
    const QString relativeDir = relativeInfo.path() == QStringLiteral(".")
        ? QString()
        : relativeInfo.path();
    const QString candidateDir = relativeDir.isEmpty()
        ? maskRoot
        : QDir(maskRoot).filePath(relativeDir);
    const QStringList stems = {
        relativeInfo.completeBaseName(),
        relativeInfo.completeBaseName() + QStringLiteral("_mask"),
    };
    static const QStringList extensions = {
        QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"),
        QStringLiteral("bmp"), QStringLiteral("tif"), QStringLiteral("tiff")
    };
    for (const QString &stem : stems) {
        for (const QString &extension : extensions) {
            const QString candidate = QDir(candidateDir).filePath(
                stem + QLatin1Char('.') + extension);
            if (QFileInfo::exists(candidate)) {
                return QDir::cleanPath(candidate);
            }
        }
    }

    const QString baseName = imageInfo.completeBaseName();
    const QStringList masks = imageFiles(maskRoot);
    for (const QString &mask : masks) {
        const QString maskBase = QFileInfo(mask).completeBaseName();
        if (maskBase.compare(baseName, Qt::CaseInsensitive) == 0
            || maskBase.compare(baseName + QStringLiteral("_mask"),
                                Qt::CaseInsensitive) == 0) {
            return mask;
        }
    }
    return {};
}

int MainWindow::countImages(const QString &path) const
{
    return imageFiles(path).size();
}

void MainWindow::startTraining()
{
    QStringList models;
    for (const QString &modelName : supportedModelNames()) {
        if (QCheckBox *check = m_modelChecks.value(modelName);
            check && check->isChecked()) {
            models << modelName;
        }
    }
    if (models.isEmpty()) {
        showStatus(tr("Select at least one model."), true);
        return;
    }

    const bool demo = m_demoMode && m_demoMode->isChecked();
    if (!demo) {
        if (m_datasetRoot.isEmpty() || !QDir(m_datasetRoot).exists()) {
            showStatus(tr("Choose a dataset before starting a real Anomalib job."), true);
            navigateTo(1);
            return;
        }
        if (detectedNormalFolder(m_datasetRoot).isEmpty()) {
            showStatus(tr("The dataset needs a normal or good folder."), true);
            return;
        }
    }
    QString parameterError;
    if (!validateTrainingConfiguration(models, demo, &parameterError)) {
        showStatus(parameterError, true);
        return;
    }

    QJsonArray selected;
    for (const QString &model : models) {
        selected.append(model);
    }
    QJsonObject job {
        {QStringLiteral("mode"), QStringLiteral("train")},
        {QStringLiteral("demo"), demo},
        {QStringLiteral("dataset_name"),
         m_datasetRoot.isEmpty() ? QStringLiteral("Housing Seal")
                                 : QFileInfo(m_datasetRoot).fileName()},
        {QStringLiteral("dataset_root"), m_datasetRoot},
        {QStringLiteral("normal_dir"), detectedNormalFolder(m_datasetRoot)},
        {QStringLiteral("abnormal_dir"), detectedAbnormalFolder(m_datasetRoot)},
        {QStringLiteral("mask_dir"), detectedMaskFolder(m_datasetRoot)},
        {QStringLiteral("models"), selected},
        {QStringLiteral("model_parameters"), modelParameterMap(models)},
        {QStringLiteral("max_epochs"), m_epochSpin ? m_epochSpin->value() : 5},
        {QStringLiteral("train_batch_size"),
         m_trainBatchSpin ? m_trainBatchSpin->value() : 8},
        {QStringLiteral("eval_batch_size"),
         m_evalBatchSpin ? m_evalBatchSpin->value() : 8},
        {QStringLiteral("num_workers"),
         m_workerSpin ? m_workerSpin->value() : 2},
        {QStringLiteral("accelerator"),
         m_acceleratorCombo
             ? m_acceleratorCombo->currentData().toString()
             : QStringLiteral("auto")},
        {QStringLiteral("devices"), m_devicesSpin ? m_devicesSpin->value() : 1},
        {QStringLiteral("output_root"),
         QDir(dataDirectory()).filePath(QStringLiteral("results"))},
    };

    m_results.clear();
    m_resultsTable->clearContents();
    m_trainingLog->clear();
    if (models.contains(QStringLiteral("EfficientAD"))
        && m_trainBatchSpin && m_trainBatchSpin->value() != 1) {
        handleWorkerOutput(
            tr("EfficientAD will use train batch 1; other models will use batch %1.")
                .arg(m_trainBatchSpin->value()));
    }
    m_progress->setValue(0);
    m_trainingState->setText(tr("Starting worker…"));
    clearPredictionImages();
    stopInferenceSession();
    runWorkerJob(job, QStringLiteral("train"));
}

void MainWindow::stopTraining()
{
    m_worker->stop();
    setJobRunning(false);
    m_trainingState->setText(tr("Stopped"));
    handleWorkerOutput(tr("Training was stopped by the user."));
}

void MainWindow::runWorkerJob(const QJsonObject &job, const QString &prefix)
{
    const auto reportPreparationFailure =
        [this, &prefix](const QString &message, const QString &trainingSummary) {
            showStatus(message, true);
            handleWorkerOutput(message);
            if (prefix == QStringLiteral("train") && m_trainingState) {
                m_trainingState->setText(trainingSummary);
            } else if (prefix == QStringLiteral("inference")
                       && m_inferenceStatus) {
                m_inferenceStatus->setText(trainingSummary);
                m_inferenceStatus->setProperty("role", "errorText");
                m_inferenceStatus->style()->unpolish(m_inferenceStatus);
                m_inferenceStatus->style()->polish(m_inferenceStatus);
            }
        };

    const QString assetRoot = modelAssetsDirectory();
    if (!QDir().mkpath(assetRoot)) {
        reportPreparationFailure(
            tr("Could not create the managed model asset directory: %1")
                .arg(QDir::toNativeSeparators(assetRoot)),
            tr("Asset directory unavailable"));
        return;
    }

    QJsonObject preparedJob = job;
    preparedJob.insert(QStringLiteral("asset_root"), assetRoot);
    preparedJob.insert(QStringLiteral("workspace_root"), workspaceDirectory());
    if (!preparedJob.contains(QStringLiteral("imagenette_dir"))) {
        const QJsonObject efficientAd = m_modelParameters.value(
            QStringLiteral("EfficientAD"),
            defaultModelParameters(QStringLiteral("EfficientAD")));
        preparedJob.insert(
            QStringLiteral("imagenette_dir"),
            efficientAd.value(QStringLiteral("imagenet_dir")).toString());
    }

    QString preparationError;
    const QString workerScript = ensureWorkerScript(&preparationError);
    if (workerScript.isEmpty()) {
        reportPreparationFailure(
            preparationError.isEmpty()
                ? tr("Could not prepare the Anomalib worker.")
                : preparationError,
            tr("Worker preparation failed"));
        return;
    }

    const QString jobFile = writeJobFile(preparedJob, prefix, &preparationError);
    if (jobFile.isEmpty()) {
        reportPreparationFailure(
            preparationError.isEmpty()
                ? tr("Could not prepare the Anomalib job.")
                : preparationError,
            tr("Job preparation failed"));
        return;
    }

    const QString python = m_pythonEdit ? m_pythonEdit->text().trimmed()
                                       : defaultPythonExecutable();
    if (python.isEmpty()) {
        showStatus(tr("Configure a Python executable in Settings."), true);
        navigateTo(5);
        return;
    }
    m_activeJob = prefix;
    setJobRunning(true);
    const bool assetDownload =
        preparedJob.value(QStringLiteral("mode")).toString()
            == QStringLiteral("assets")
        && preparedJob.value(QStringLiteral("action")).toString()
            == QStringLiteral("download");
    m_worker->startJob(
        python,
        workerScript,
        jobFile,
        assetDownload ? AnomalibProcess::NetworkPolicy::AssetDownload
                      : AnomalibProcess::NetworkPolicy::Offline,
        assetRoot);
}

void MainWindow::handleWorkerEvent(const QJsonObject &event)
{
    const QString kind = event.value(QStringLiteral("event")).toString();
    if (kind == QStringLiteral("hello")) {
        handleWorkerOutput(tr("Worker ready · Anomalib %1")
                               .arg(event.value(QStringLiteral("anomalib_version")).toString()));
    } else if (kind == QStringLiteral("stage")) {
        const QString message = event.value(QStringLiteral("message")).toString();
        m_trainingState->setText(message);
        handleWorkerOutput(message);
    } else if (kind == QStringLiteral("progress")) {
        const int epoch = event.value(QStringLiteral("epoch")).toInt();
        const int maximum = qMax(1, event.value(QStringLiteral("max_epochs")).toInt());
        m_progress->setValue(qRound(100.0 * epoch / maximum));
        m_trainingState->setText(event.value(QStringLiteral("message")).toString());
    } else if (kind == QStringLiteral("result")) {
        updateResultRow(event);
        handleWorkerOutput(
            tr("%1 completed · image AUROC %2")
                .arg(event.value(QStringLiteral("model")).toString(),
                     metricText(event, QStringLiteral("image_auroc"))));
    } else if (kind == QStringLiteral("inference")) {
        updateInferenceResult(event);
        handleWorkerOutput(
            tr("%1 inference completed · anomaly score %2")
                .arg(event.value(QStringLiteral("model")).toString(),
                     metricText(event, QStringLiteral("anomaly_score"))));
    } else if (kind == QStringLiteral("warning")) {
        handleWorkerOutput(
            tr("Warning: %1")
                .arg(event.value(QStringLiteral("message")).toString()));
    } else if (kind == QStringLiteral("finished")) {
        updateWinner(event.value(QStringLiteral("winner")).toString());
        m_progress->setValue(100);
        m_progress->hide();
        m_trainingState->setText(tr("Training completed"));
        showStatus(tr("Experiment completed. Winner: %1").arg(m_winnerModel));
    } else if (kind == QStringLiteral("environment")) {
        const bool ok = event.value(QStringLiteral("ok")).toBool();
        const QString message = event.value(QStringLiteral("message")).toString();
        if (m_environmentStatus) {
            m_environmentStatus->setText(message);
            m_environmentStatus->setProperty("role", ok ? "successText" : "errorText");
            m_environmentStatus->style()->unpolish(m_environmentStatus);
            m_environmentStatus->style()->polish(m_environmentStatus);
        }
        showStatus(message, !ok);
    } else if (kind == QStringLiteral("asset_status")) {
        updateAssetRow(event);
    } else if (kind == QStringLiteral("asset_progress")) {
        const QString assetId =
            event.value(QStringLiteral("asset_id")).toString();
        QJsonObject status = m_assetStatuses.value(assetId);
        if (!status.isEmpty()) {
            const double downloaded =
                event.value(QStringLiteral("downloaded_bytes")).toDouble();
            const double total =
                event.value(QStringLiteral("total_bytes")).toDouble();
            QString message =
                event.value(QStringLiteral("message")).toString();
            if (total > 0.0) {
                message += tr(" (%1%)")
                               .arg(qBound(0, qRound(100.0 * downloaded / total),
                                           100));
            }
            status.insert(QStringLiteral("message"), message);
            status.insert(QStringLiteral("state"), QStringLiteral("working"));
            updateAssetRow(status);
        }
    } else if (kind == QStringLiteral("asset_result")) {
        const QString message =
            event.value(QStringLiteral("message")).toString();
        const bool completed =
            event.value(QStringLiteral("completed")).toBool();
        if (!message.isEmpty()) {
            handleWorkerOutput(message);
            showStatus(message, !completed);
        }
    } else if (kind == QStringLiteral("exported")) {
        const QString path = event.value(QStringLiteral("path")).toString();
        handleWorkerOutput(tr("Exported %1 to %2").arg(
            event.value(QStringLiteral("format")).toString(), path));
        showStatus(tr("Model export completed: %1").arg(path));
    } else if (kind == QStringLiteral("error")) {
        const QString message = event.value(QStringLiteral("message")).toString();
        handleWorkerOutput(message);
        showStatus(message, true);
    }
}

void MainWindow::handleWorkerOutput(const QString &text)
{
    if (!m_trainingLog || text.trimmed().isEmpty()) {
        return;
    }
    m_trainingLog->appendPlainText(
        QStringLiteral("%1   %2")
            .arg(QTime::currentTime().toString(QStringLiteral("HH:mm:ss")),
                 text.trimmed()));
}

void MainWindow::handleWorkerFinished(int exitCode, QProcess::ExitStatus status)
{
    const QString completedJob = m_activeJob;
    m_activeJob.clear();
    setJobRunning(false);
    if (status != QProcess::NormalExit || exitCode != 0) {
        if (completedJob == QStringLiteral("train")) {
            m_trainingState->setText(tr("Job failed"));
        }
        if (completedJob == QStringLiteral("inference") && m_inferenceStatus) {
            m_inferenceStatus->setText(tr("Inference failed"));
            m_inferenceStatus->setProperty("role", "errorText");
            m_inferenceStatus->style()->unpolish(m_inferenceStatus);
            m_inferenceStatus->style()->polish(m_inferenceStatus);
        }
        showStatus(tr("The worker exited with code %1. See the activity log.")
                       .arg(exitCode),
                   true);
    } else if (completedJob == QStringLiteral("train")) {
        m_trainingState->setText(tr("Training completed"));
    }
    if (completedJob == QStringLiteral("assets-download")) {
        QTimer::singleShot(0, this, &MainWindow::refreshAssetStatus);
    }
}

void MainWindow::handleWorkerFailure(const QString &message)
{
    setJobRunning(false);
    m_trainingState->setText(tr("Worker unavailable"));
    if (m_activeJob == QStringLiteral("inference") && m_inferenceStatus) {
        m_inferenceStatus->setText(tr("Worker unavailable"));
        m_inferenceStatus->setProperty("role", "errorText");
        m_inferenceStatus->style()->unpolish(m_inferenceStatus);
        m_inferenceStatus->style()->polish(m_inferenceStatus);
    }
    showStatus(message, true);
    handleWorkerOutput(message);
    m_activeJob.clear();
}

void MainWindow::setJobRunning(bool running)
{
    m_trainButton->setEnabled(!running);
    if (m_epochSpin) {
        m_epochSpin->setEnabled(!running);
    }
    if (m_trainBatchSpin) {
        m_trainBatchSpin->setEnabled(!running);
    }
    if (m_evalBatchSpin) {
        m_evalBatchSpin->setEnabled(!running);
    }
    if (m_workerSpin) {
        m_workerSpin->setEnabled(!running);
    }
    if (m_acceleratorCombo) {
        m_acceleratorCombo->setEnabled(!running);
    }
    if (m_devicesSpin) {
        m_devicesSpin->setEnabled(!running);
    }
    if (m_sharedSettingsToggle) {
        m_sharedSettingsToggle->setEnabled(!running);
    }
    if (m_sharedResetButton) {
        m_sharedResetButton->setEnabled(!running);
    }
    m_stopButton->setVisible(running);
    if (running && m_activeJob == QStringLiteral("train")) {
        m_progress->show();
    } else if (!running) {
        m_progress->hide();
    }
    if (m_inferenceRunButton) {
        m_inferenceRunButton->setEnabled(!running && m_inferenceModelReady);
    }
    if (m_inferenceCaptureButton) {
        m_inferenceCaptureButton->setEnabled(!running && m_inferenceModelReady);
    }
    if (m_inferenceLoadModelButton) {
        m_inferenceLoadModelButton->setEnabled(!running);
    }
    if (m_inferenceUseWinnerButton) {
        m_inferenceUseWinnerButton->setEnabled(!running);
    }
    for (QCheckBox *check : std::as_const(m_modelChecks)) {
        check->setEnabled(!running);
    }
    for (QPushButton *button : std::as_const(m_modelConfigureButtons)) {
        button->setEnabled(!running);
    }
    if (m_refreshAssetsButton) {
        m_refreshAssetsButton->setEnabled(!running);
    }
    if (m_downloadAssetsButton) {
        const bool hasMissing =
            std::any_of(m_assetStatuses.cbegin(), m_assetStatuses.cend(),
                        [](const QJsonObject &status) {
                            return !status.value(QStringLiteral("ready")).toBool();
                        });
        m_downloadAssetsButton->setEnabled(!running && hasMissing);
    }
}

void MainWindow::updateResultRow(const QJsonObject &result)
{
    const QString model = result.value(QStringLiteral("model")).toString();
    if (model.isEmpty()) {
        return;
    }
    QJsonObject storedResult = result;
    QJsonObject storedParameters =
        storedResult.value(QStringLiteral("model_parameters")).toObject();
    if (!storedParameters.value(model).isObject()) {
        storedParameters.insert(model, defaultModelParameters(model));
        storedResult.insert(QStringLiteral("model_parameters"),
                            storedParameters);
    }
    m_results.insert(model, storedResult);
    const QStringList order = {QStringLiteral("PatchCore"), QStringLiteral("PaDiM"),
                               QStringLiteral("EfficientAD")};
    int row = order.indexOf(model);
    if (row < 0) {
        row = qMin(m_resultsTable->rowCount() - 1, m_results.size() - 1);
    }

    auto *modelItem = tableItem(model, Qt::AlignLeft | Qt::AlignVCenter);
    modelItem->setData(Qt::UserRole, model);
    QFont modelFont = modelItem->font();
    modelFont.setBold(model == m_winnerModel);
    modelItem->setFont(modelFont);
    m_resultsTable->setItem(row, 0, modelItem);
    m_resultsTable->setItem(row, 1, tableItem(metricText(result, QStringLiteral("image_auroc"))));
    m_resultsTable->setItem(row, 2, tableItem(metricText(result, QStringLiteral("pixel_auroc"))));
    m_resultsTable->setItem(row, 3, tableItem(metricText(result, QStringLiteral("image_f1"))));
    m_resultsTable->setItem(row, 4, tableItem(metricText(result, QStringLiteral("latency_ms"), 1)));

    const double size = result.value(QStringLiteral("model_size_mb")).toDouble();
    const QString sizeText = size >= 1024.0
        ? tr("%1 GB").arg(QString::number(size / 1024.0, 'f', 2))
        : tr("%1 MB").arg(QString::number(size, 'f', 0));
    m_resultsTable->setItem(row, 5, tableItem(size > 0 ? sizeText : QStringLiteral("—")));
    if (model == m_winnerModel) {
        updateInspectionFromResult(result);
    }
}

void MainWindow::updateWinner(const QString &modelName)
{
    if (modelName.isEmpty()) {
        return;
    }
    m_winnerModel = modelName;
    m_winnerLabel->setText(tr("Winner: %1").arg(modelName));
    if (m_heatmapTitle) {
        m_heatmapTitle->setText(tr("Predicted heatmap (%1)").arg(modelName));
    }
    for (int row = 0; row < m_resultsTable->rowCount(); ++row) {
        QTableWidgetItem *modelItem = m_resultsTable->item(row, 0);
        if (!modelItem) {
            continue;
        }
        const bool winner = modelItem->data(Qt::UserRole).toString() == modelName;
        QFont font = modelItem->font();
        font.setBold(winner);
        modelItem->setFont(font);
        const QColor background = winner ? QColor(QStringLiteral("#eaf6ed"))
                                         : QColor(Qt::transparent);
        for (int column = 0; column < m_resultsTable->columnCount(); ++column) {
            if (QTableWidgetItem *item = m_resultsTable->item(row, column)) {
                item->setBackground(background);
            }
        }
        if (winner) {
            m_resultsTable->selectRow(row);
        }
    }
    updateInspectionFromResult(m_results.value(modelName));
}

void MainWindow::exportWinner(const QString &format)
{
    const QString checkpoint = selectedWinnerCheckpoint();
    const bool demo = m_demoMode && m_demoMode->isChecked();
    if (!demo && checkpoint.isEmpty()) {
        showStatus(tr("Train the selected model before exporting it."), true);
        return;
    }

    QJsonObject job {
        {QStringLiteral("mode"), QStringLiteral("export")},
        {QStringLiteral("demo"), demo},
        {QStringLiteral("model"), m_winnerModel},
        {QStringLiteral("model_parameters"),
         checkpointModelParameters(m_winnerModel)},
        {QStringLiteral("training_parameters"),
         m_results.value(m_winnerModel)
             .value(QStringLiteral("training_parameters")).toObject()},
        {QStringLiteral("metrics"), m_results.value(m_winnerModel)},
        {QStringLiteral("checkpoint"), checkpoint},
        {QStringLiteral("format"), format},
        {QStringLiteral("output_root"),
         QDir(dataDirectory()).filePath(QStringLiteral("exports"))},
    };
    stopInferenceSession();
    runWorkerJob(job, QStringLiteral("export"));
}

void MainWindow::promoteWinner()
{
    const QJsonObject result = m_results.value(m_winnerModel);
    const QString checkpoint = selectedWinnerCheckpoint();
    const QString relativeCheckpoint = QDir(dataDirectory()).relativeFilePath(
        checkpoint);
    QJsonObject manifest {
        {QStringLiteral("schema_version"), 2},
        {QStringLiteral("model"), m_winnerModel},
        {QStringLiteral("checkpoint"), checkpoint},
        {QStringLiteral("artifact"),
         QJsonObject {
             {QStringLiteral("format"), QStringLiteral("checkpoint")},
             {QStringLiteral("path"), relativeCheckpoint},
             {QStringLiteral("sha256"), sha256ForFile(checkpoint)},
         }},
        {QStringLiteral("dataset_root"), m_datasetRoot},
        {QStringLiteral("promoted_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
        {QStringLiteral("model_parameters"),
         checkpointModelParameters(m_winnerModel)},
        {QStringLiteral("training_parameters"),
         result.value(QStringLiteral("training_parameters")).toObject()},
        {QStringLiteral("metrics"), result},
        {QStringLiteral("capabilities"),
         QJsonObject {
             {QStringLiteral("sigma_override"),
              QJsonObject {
                  {QStringLiteral("supported"),
                   m_winnerModel == QStringLiteral("PatchCore")},
                  {QStringLiteral("minimum"), 1},
                  {QStringLiteral("maximum"), 32},
              }},
         }},
    };
    const QString path = QDir(dataDirectory()).filePath(QStringLiteral("production_model.json"));
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0
        || !file.commit()) {
        showStatus(tr("Could not write the production model manifest."), true);
        return;
    }
    m_promoteButton->setText(tr("%1 is the production model").arg(m_winnerModel));
    m_promoteButton->setEnabled(false);
    showStatus(tr("Production model manifest saved to %1").arg(path));
}

QString MainWindow::selectedWinnerCheckpoint() const
{
    return m_results.value(m_winnerModel)
        .value(QStringLiteral("checkpoint")).toString();
}

QString MainWindow::nextInferenceRequestId(const QString &prefix)
{
    ++m_inferenceRequestCounter;
    return QStringLiteral("%1-%2").arg(prefix).arg(m_inferenceRequestCounter);
}

void MainWindow::setInferenceModelState(const QString &status, bool ready,
                                        bool error)
{
    m_inferenceModelReady = ready;
    if (m_inferenceModelDetails) {
        m_inferenceModelDetails->setText(status);
    }
    if (m_inferenceModelName) {
        m_inferenceModelName->setText(
            m_activeInferenceModel.isEmpty()
                ? (ready ? tr("Model ready") : tr("No model loaded"))
                : m_activeInferenceModel);
        m_inferenceModelName->setProperty(
            "role", error ? "errorText" : (ready ? "successText" : "summaryValue"));
        m_inferenceModelName->style()->unpolish(m_inferenceModelName);
        m_inferenceModelName->style()->polish(m_inferenceModelName);
    }
    const bool workerBusy = m_worker && m_worker->isRunning();
    if (m_inferenceRunButton) {
        m_inferenceRunButton->setEnabled(ready && !workerBusy);
    }
    if (m_inferenceCaptureButton) {
        m_inferenceCaptureButton->setEnabled(ready && !workerBusy);
    }
    if (!ready && m_inferenceSigmaOverrideCheck) {
        m_inferenceSigmaOverrideCheck->setChecked(false);
        m_inferenceSigmaOverrideCheck->setEnabled(false);
        if (m_inferenceSigmaSpin) {
            m_inferenceSigmaSpin->setEnabled(false);
        }
    } else if (ready && m_inferenceSigmaOverrideCheck) {
        m_inferenceSigmaOverrideCheck->setEnabled(
            m_activeInferenceSigmaOverrideSupported);
        if (m_inferenceSigmaSpin) {
            m_inferenceSigmaSpin->setEnabled(
                m_activeInferenceSigmaOverrideSupported
                && m_inferenceSigmaOverrideCheck->isChecked());
        }
    }
}

bool MainWindow::startInferenceSession()
{
    if (m_inferenceWorker->isRunning()) {
        return true;
    }
    const QString assetRoot = modelAssetsDirectory();
    const QString sessionRoot =
        QDir(dataDirectory()).filePath(QStringLiteral("inference-session"));
    if (!QDir().mkpath(assetRoot) || !QDir().mkpath(sessionRoot)) {
        setInferenceModelState(tr("Could not create inference data directories."),
                               false, true);
        return false;
    }
    QString error;
    const QString workerScript = ensureWorkerScript(&error);
    if (workerScript.isEmpty()) {
        setInferenceModelState(error, false, true);
        return false;
    }
    const QString python = m_pythonEdit ? m_pythonEdit->text().trimmed()
                                       : defaultPythonExecutable();
    if (python.isEmpty()) {
        setInferenceModelState(
            tr("Configure a Python executable in Settings."), false, true);
        return false;
    }
    setInferenceModelState(tr("Starting persistent inference worker…"), false);
    m_inferenceWorker->startSession(
        python, workerScript, assetRoot, sessionRoot);
    return m_inferenceWorker->isRunning();
}

void MainWindow::stopInferenceSession()
{
    if (m_inferenceWorker) {
        m_inferenceWorker->stop();
    }
    m_inferenceModelReady = false;
    m_inferenceHadReadyModelBeforeLoad = false;
    if (m_inferenceModelDetails && !m_activeInferencePath.isEmpty()) {
        m_inferenceModelDetails->setText(
            tr("Model unloaded to release accelerator memory. Open Inspect to reload it."));
    }
}

void MainWindow::loadInferenceModel(const QString &path,
                                    const QString &trustedFingerprint)
{
    m_inferenceHadReadyModelBeforeLoad =
        m_inferenceHadReadyModelBeforeLoad || m_inferenceModelReady;
    const QString cleanPath = QDir::cleanPath(path.trimmed());
    if (!QFileInfo(cleanPath).isFile()) {
        setInferenceModelState(
            tr("Model file does not exist: %1").arg(cleanPath), false, true);
        return;
    }
    if (!startInferenceSession()) {
        return;
    }
    m_pendingInferencePath = cleanPath;
    m_pendingInferenceFingerprint = trustedFingerprint;
    setInferenceModelState(
        tr("Validating and loading %1…").arg(QFileInfo(cleanPath).fileName()),
        false);
    QJsonObject command {
        {QStringLiteral("request_id"), nextInferenceRequestId(QStringLiteral("load"))},
        {QStringLiteral("command"), QStringLiteral("load_model")},
        {QStringLiteral("path"), cleanPath},
    };
    if (!trustedFingerprint.isEmpty()) {
        command.insert(QStringLiteral("trusted_sha256"), trustedFingerprint);
    }
    m_inferenceWorker->sendCommand(command);
}

void MainWindow::chooseInferenceModel()
{
    QSettings settings;
    const QString saved = settings.value(
        QStringLiteral("Inference/lastArtifactPath")).toString();
    const QString start = saved.isEmpty()
        ? QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
        : QFileInfo(saved).absolutePath();
    const QString selected = QFileDialog::getOpenFileName(
        this, tr("Load Anomalib model"), start,
        tr("Anomalib models (*.ckpt *.pt *.onnx *.xml *.bin *.json);;"
           "Lightning checkpoints (*.ckpt);;Torch exports (*.pt);;"
           "ONNX models (*.onnx);;OpenVINO models (*.xml *.bin);;"
           "Model manifests (*.json)"));
    if (!selected.isEmpty()) {
        loadInferenceModel(selected);
    }
}

void MainWindow::useWinnerForInference()
{
    const bool demo = m_demoMode && m_demoMode->isChecked();
    if (demo) {
        m_activeInferenceModel = m_winnerModel;
        m_activeInferencePath = selectedWinnerCheckpoint();
        m_activeInferenceBackend = tr("Demonstration");
        m_activeInferenceDevice = tr("Simulated");
        const QJsonObject parameters = checkpointModelParameters(m_winnerModel)
            .value(m_winnerModel).toObject();
        const int savedSigma = parameters.value(QStringLiteral("sigma")).toInt(4);
        m_activeInferenceSavedSigma = savedSigma;
        m_activeInferenceSigmaOverrideSupported =
            m_winnerModel == QStringLiteral("PatchCore");
        if (m_inferenceSigmaOverrideCheck) {
            m_inferenceSigmaOverrideCheck->setChecked(false);
            m_inferenceSigmaOverrideCheck->setEnabled(
                m_activeInferenceSigmaOverrideSupported);
        }
        if (m_inferenceSigmaSpin) {
            m_inferenceSigmaSpin->setValue(savedSigma);
            m_inferenceSigmaSpin->setEnabled(false);
        }
        setInferenceModelState(
            tr("Demonstration result · input %1 · no model runtime is loaded")
                .arg(imageSizeText(parameters)),
            true);
        return;
    }
    const QString checkpoint = selectedWinnerCheckpoint();
    if (!QFileInfo(checkpoint).isFile()) {
        showStatus(tr("Train a model before selecting the trained winner."), true);
        navigateTo(2);
        return;
    }
    loadInferenceModel(checkpoint);
}

void MainWindow::restoreInferenceModel()
{
    if (m_inferenceModelReady || m_inferenceWorker->isRunning()) {
        return;
    }
    const bool demo = m_demoMode && m_demoMode->isChecked();
    if (demo) {
        useWinnerForInference();
        return;
    }
    QSettings settings;
    QString path = m_activeInferencePath;
    if (path.isEmpty() && !m_inferenceRestoreAttempted) {
        path = settings.value(QStringLiteral("Inference/lastArtifactPath")).toString();
    }
    m_inferenceRestoreAttempted = true;
    if (path.isEmpty()) {
        return;
    }
    if (!QFileInfo(path).isFile()) {
        setInferenceModelState(
            tr("The remembered model is no longer available: %1").arg(path),
            false, true);
        return;
    }
    const QString fingerprint = settings.value(
        QStringLiteral("Inference/lastFingerprint")).toString();
    const QStringList trusted = settings.value(
        QStringLiteral("Inference/trustedPickleFingerprints")).toStringList();
    loadInferenceModel(path, trusted.contains(fingerprint) ? fingerprint : QString());
}

void MainWindow::handleInferenceEvent(const QJsonObject &event)
{
    const QString kind = event.value(QStringLiteral("event")).toString();
    if (kind == QStringLiteral("session_ready")) {
        handleWorkerOutput(tr("Persistent inference worker ready (PID %1).")
                               .arg(event.value(QStringLiteral("pid")).toInt()));
        return;
    }
    if (kind == QStringLiteral("trust_required")) {
        const QString path = event.value(QStringLiteral("selected_path")).toString();
        const QString fingerprint = event.value(QStringLiteral("fingerprint")).toString();
        const QString warning = tr(
            "Torch .pt files contain Python-pickled objects and may execute code.\n\n"
            "Only continue if you trust this file and its source:\n%1\n\n"
            "SHA-256 fingerprint:\n%2")
            .arg(QDir::toNativeSeparators(path), fingerprint);
        if (QMessageBox::warning(
                this, tr("Trust Torch model?"), warning,
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No)
            == QMessageBox::Yes) {
            QSettings settings;
            QStringList trusted = settings.value(
                QStringLiteral("Inference/trustedPickleFingerprints")).toStringList();
            if (!trusted.contains(fingerprint)) {
                trusted.append(fingerprint);
                settings.setValue(
                    QStringLiteral("Inference/trustedPickleFingerprints"), trusted);
            }
            loadInferenceModel(path, fingerprint);
        } else {
            setInferenceModelState(tr("Torch model loading was cancelled."),
                                   !m_activeInferencePath.isEmpty());
        }
        return;
    }
    if (kind == QStringLiteral("model_loaded")) {
        m_activeInferencePath =
            event.value(QStringLiteral("selected_path")).toString();
        m_activeInferenceFingerprint =
            event.value(QStringLiteral("fingerprint")).toString();
        m_activeInferenceModel = event.value(QStringLiteral("model")).toString();
        m_activeInferenceBackend =
            event.value(QStringLiteral("backend")).toString();
        m_activeInferenceDevice =
            event.value(QStringLiteral("device_name")).toString();
        m_pendingInferencePath.clear();
        m_pendingInferenceFingerprint.clear();
        m_inferenceHadReadyModelBeforeLoad = false;
        const bool sigmaOverrideSupported =
            event.value(QStringLiteral("sigma_override_supported")).toBool();
        const int savedSigma =
            event.value(QStringLiteral("saved_sigma")).toInt(4);
        m_activeInferenceSavedSigma = savedSigma;
        m_activeInferenceSigmaOverrideSupported = sigmaOverrideSupported;
        if (m_inferenceSigmaOverrideCheck) {
            m_inferenceSigmaOverrideCheck->setChecked(false);
            m_inferenceSigmaOverrideCheck->setEnabled(sigmaOverrideSupported);
            m_inferenceSigmaOverrideCheck->setToolTip(
                sigmaOverrideSupported
                    ? tr("Saved sigma is %1. Enable this control to override it "
                         "temporarily for heatmaps and masks.").arg(savedSigma)
                    : tr("This artifact has sigma baked into its inference graph."));
        }
        if (m_inferenceSigmaSpin) {
            m_inferenceSigmaSpin->setValue(savedSigma);
            m_inferenceSigmaSpin->setEnabled(false);
        }
        QSettings settings;
        settings.setValue(QStringLiteral("Inference/lastArtifactPath"),
                          m_activeInferencePath);
        settings.setValue(QStringLiteral("Inference/lastFingerprint"),
                          m_activeInferenceFingerprint);
        const QString sigmaDetails = m_activeInferenceModel == QStringLiteral("PatchCore")
            ? (sigmaOverrideSupported
                   ? tr("sigma %1 · runtime override available").arg(savedSigma)
                   : tr("sigma %1 · baked into artifact").arg(savedSigma))
            : QString();
        const QJsonObject parameterMap =
            event.value(QStringLiteral("model_parameters")).toObject();
        const QJsonObject modelParameters =
            parameterMap.value(m_activeInferenceModel).toObject();
        QStringList configurationDetails {
            tr("input %1").arg(imageSizeText(modelParameters)),
        };
        if (!sigmaDetails.isEmpty()) {
            configurationDetails.append(sigmaDetails);
        }
        const QString details = tr("%1 · %2 · %3 · loaded in %4 ms\n%5\n%6")
            .arg(event.value(QStringLiteral("format")).toString(),
                 m_activeInferenceBackend,
                 m_activeInferenceDevice,
                 metricText(event, QStringLiteral("load_ms"), 0),
                 QDir::toNativeSeparators(m_activeInferencePath),
                 configurationDetails.join(QStringLiteral(" · ")));
        setInferenceModelState(details, true);
        if (m_inferenceStatus) {
            m_inferenceStatus->setText(tr("Model ready for inference"));
        }
        handleWorkerOutput(
            tr("Loaded %1 with %2 on %3.")
                .arg(m_activeInferenceModel, m_activeInferenceBackend,
                     m_activeInferenceDevice));
        return;
    }
    if (kind == QStringLiteral("inference")) {
        updateInferenceResult(event);
        setInferenceModelState(m_inferenceModelDetails->text(), true);
        handleWorkerOutput(
            tr("%1 inference completed · model %2 ms · total %3 ms")
                .arg(event.value(QStringLiteral("model")).toString(),
                     metricText(event, QStringLiteral("model_latency_ms"), 1),
                     metricText(event, QStringLiteral("total_latency_ms"), 1)));
        return;
    }
    if (kind == QStringLiteral("model_unloaded")) {
        setInferenceModelState(tr("Model unloaded."), false);
        return;
    }
    if (kind == QStringLiteral("error")) {
        const QString message = event.value(QStringLiteral("message")).toString();
        const QString requestId =
            event.value(QStringLiteral("request_id")).toString();
        const bool modelStillAvailable =
            requestId.startsWith(QStringLiteral("predict-"))
            || (requestId.startsWith(QStringLiteral("load-"))
                && m_inferenceHadReadyModelBeforeLoad);
        m_inferenceHadReadyModelBeforeLoad = false;
        setInferenceModelState(message, modelStillAvailable, true);
        if (m_inferenceStatus && requestId.startsWith(QStringLiteral("predict-"))) {
            m_inferenceStatus->setText(tr("Inference failed"));
        }
        showStatus(message, true);
        handleWorkerOutput(message);
    }
}

void MainWindow::handleInferenceOutput(const QString &text)
{
    handleWorkerOutput(text);
}

void MainWindow::handleInferenceFinished(int exitCode,
                                         QProcess::ExitStatus status)
{
    m_inferenceModelReady = false;
    if (status != QProcess::NormalExit || exitCode != 0) {
        setInferenceModelState(
            tr("Inference worker exited with code %1.").arg(exitCode), false, true);
    } else if (!m_activeInferencePath.isEmpty() && m_inferenceModelDetails) {
        m_inferenceModelDetails->setText(
            tr("Model unloaded; it will reload when Inspect is opened."));
    }
}

void MainWindow::handleInferenceFailure(const QString &message)
{
    setInferenceModelState(message, false, true);
    showStatus(message, true);
    handleWorkerOutput(message);
}

void MainWindow::chooseInferenceImage()
{
    const QString current = m_inferenceImageEdit
        ? m_inferenceImageEdit->text().trimmed() : QString();
    const QString start = current.isEmpty()
        ? QStandardPaths::writableLocation(QStandardPaths::PicturesLocation)
        : QFileInfo(current).absolutePath();
    const QString selected = QFileDialog::getOpenFileName(
        this, tr("Choose an image to test"), start,
        tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"));
    if (selected.isEmpty()) {
        return;
    }
    m_inferenceImageEdit->setText(QDir::cleanPath(selected));
    setInspectionImage(m_inferenceInputImage, selected, tr("Image unavailable"));
    setInspectionImage(m_inferenceHeatmapImage, {},
                       tr("Run the model to generate a heatmap"));
    setInspectionImage(m_inferenceMaskImage, {},
                       tr("Run the model to generate a mask"));
    m_inferenceStatus->setText(tr("Image ready to test"));
    m_inferenceScore->setText(QStringLiteral("—"));
    m_inferenceLatency->setText(QStringLiteral("—"));
}

void MainWindow::runInference()
{
    if (!m_inferenceImageEdit) {
        return;
    }
    const QString imagePath =
        QDir::cleanPath(m_inferenceImageEdit->text().trimmed());
    if (!QFileInfo(imagePath).isFile() || QImage(imagePath).isNull()) {
        showStatus(tr("Choose a readable image before testing."), true);
        return;
    }

    const bool demo = m_demoMode && m_demoMode->isChecked();
    if (!m_inferenceModelReady) {
        showStatus(tr("Load a model before testing an image."), true);
        return;
    }
    if (m_worker && m_worker->isRunning()) {
        showStatus(tr("Wait for the current training or export job to finish."), true);
        return;
    }

    const int thresholdPercent = m_inferenceThresholdSpin
        ? m_inferenceThresholdSpin->value() : 60;
    const bool overrideSigma = m_inferenceSigmaOverrideCheck
        && m_inferenceSigmaOverrideCheck->isEnabled()
        && m_inferenceSigmaOverrideCheck->isChecked();
    const int sigmaOverride = m_inferenceSigmaSpin
        ? m_inferenceSigmaSpin->value() : 4;
    const QString runFolder = QStringLiteral("run-%1").arg(
        QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz")));
    setInspectionImage(m_inferenceInputImage, imagePath, tr("Image unavailable"));
    setInspectionImage(m_inferenceHeatmapImage, {},
                       tr("Inference is running…"));
    setInspectionImage(m_inferenceMaskImage, {},
                       tr("Inference is running…"));
    m_inferenceStatus->setText(tr("Running %1…").arg(m_activeInferenceModel));
    m_inferenceStatus->setProperty("role", "winner");
    m_inferenceStatus->style()->unpolish(m_inferenceStatus);
    m_inferenceStatus->style()->polish(m_inferenceStatus);
    m_inferenceScore->setText(QStringLiteral("—"));
    m_inferenceLatency->setText(QStringLiteral("—"));
    m_inferenceRunButton->setEnabled(false);
    m_inferenceCaptureButton->setEnabled(false);

    const QString outputRoot = QDir(dataDirectory()).filePath(
        QStringLiteral("inference/") + runFolder);
    if (demo) {
        const QString model = m_activeInferenceModel.isEmpty()
            ? m_winnerModel : m_activeInferenceModel;
        QJsonObject job {
            {QStringLiteral("mode"), QStringLiteral("inference")},
            {QStringLiteral("demo"), true},
            {QStringLiteral("model"), model},
            {QStringLiteral("model_parameters"), checkpointModelParameters(model)},
            {QStringLiteral("checkpoint"), selectedWinnerCheckpoint()},
            {QStringLiteral("image_path"), imagePath},
            {QStringLiteral("threshold"), thresholdPercent / 100.0},
            {QStringLiteral("output_root"), outputRoot},
        };
        if (overrideSigma) {
            job.insert(QStringLiteral("sigma_override"), sigmaOverride);
        }
        runWorkerJob(job, QStringLiteral("inference"));
        return;
    }

    if (!startInferenceSession()) {
        return;
    }
    QJsonObject command {
        {QStringLiteral("request_id"),
         nextInferenceRequestId(QStringLiteral("predict"))},
        {QStringLiteral("command"), QStringLiteral("predict")},
        {QStringLiteral("image_path"), imagePath},
        {QStringLiteral("threshold"), thresholdPercent / 100.0},
        {QStringLiteral("output_root"), outputRoot},
    };
    if (overrideSigma) {
        command.insert(QStringLiteral("sigma_override"), sigmaOverride);
    }
    m_inferenceWorker->sendCommand(command);
}

void MainWindow::captureAndRunInference()
{
    if (!m_cameraCombo) {
        showStatus(tr("The camera controls are unavailable."), true);
        return;
    }
    const QString captureRoot =
        QDir(dataDirectory()).filePath(QStringLiteral("inference-captures"));
    if (!QDir().mkpath(captureRoot)) {
        showStatus(tr("Could not create the inference capture folder."), true);
        return;
    }
    const QString imagePath = QDir(captureRoot).filePath(
        QStringLiteral("capture-%1.png").arg(
            QDateTime::currentDateTime().toString(
                QStringLiteral("yyyyMMdd-HHmmss-zzz"))));
    QString error;
    if (!m_cameraService.capture(m_cameraCombo->currentIndex(), imagePath, &error)) {
        showStatus(error, true);
        return;
    }
    m_inferenceImageEdit->setText(imagePath);
    setInspectionImage(m_inferenceInputImage, imagePath, tr("Captured image unavailable"));
    showStatus(tr("Camera frame captured. Starting inference…"));
    runInference();
}

void MainWindow::updateInferenceResult(const QJsonObject &result)
{
    const QString label = result.value(QStringLiteral("label")).toString();
    const bool anomalous = label == QStringLiteral("anomalous");
    m_inferenceStatus->setText(
        anomalous ? tr("ANOMALY DETECTED") : tr("NORMAL"));
    m_inferenceStatus->setProperty(
        "role", anomalous ? "errorText" : "successText");
    m_inferenceStatus->style()->unpolish(m_inferenceStatus);
    m_inferenceStatus->style()->polish(m_inferenceStatus);
    m_inferenceScore->setText(
        metricText(result, QStringLiteral("anomaly_score")));
    if (result.value(QStringLiteral("total_latency_ms")).isDouble()) {
        m_inferenceLatency->setText(
            tr("%1 ms total · %2 ms model")
                .arg(metricText(result, QStringLiteral("total_latency_ms"), 1),
                     metricText(result, QStringLiteral("model_latency_ms"), 1)));
    } else {
        m_inferenceLatency->setText(
            metricText(result, QStringLiteral("latency_ms"), 1, tr(" ms")));
    }
    setInspectionImage(
        m_inferenceInputImage,
        result.value(QStringLiteral("image_path")).toString(),
        tr("Test image unavailable"));
    setInspectionImage(
        m_inferenceHeatmapImage,
        result.value(QStringLiteral("heatmap_path")).toString(),
        tr("No heatmap was produced"));
    setInspectionImage(
        m_inferenceMaskImage,
        result.value(QStringLiteral("predicted_mask_path")).toString(),
        tr("No mask was produced"));
    showStatus(
        tr("%1 classified the image as %2 with anomaly score %3.")
            .arg(result.value(QStringLiteral("model")).toString(),
                 anomalous ? tr("anomalous") : tr("normal"),
                 metricText(result, QStringLiteral("anomaly_score"))));
}

void MainWindow::browsePython()
{
    const QString selected = QFileDialog::getOpenFileName(
        this, tr("Choose Python executable"),
        QFileInfo(m_pythonEdit->text()).absolutePath(),
        tr("Python (python.exe);;Executable (*.exe);;All files (*)"));
    if (!selected.isEmpty()) {
        m_pythonEdit->setText(QDir::cleanPath(selected));
    }
}

void MainWindow::verifyEnvironment()
{
    if (!m_pythonEdit) {
        navigateTo(5);
        return;
    }
    QJsonObject job {
        {QStringLiteral("mode"), QStringLiteral("check")},
        {QStringLiteral("demo"), m_demoMode->isChecked()},
    };
    runWorkerJob(job, QStringLiteral("check"));
}

void MainWindow::refreshAssetStatus()
{
    if (!m_assetTable || (m_worker && m_worker->isRunning())) {
        return;
    }
    m_assetStatuses.clear();
    for (int row = 0; row < m_assetTable->rowCount(); ++row) {
        m_assetTable->item(row, 1)->setText(tr("Checking…"));
        m_assetTable->item(row, 1)->setToolTip({});
        m_assetTable->item(row, 2)->setText(tr("Checking…"));
        m_assetTable->item(row, 3)->setText({});
    }
    if (m_downloadAssetsButton) {
        m_downloadAssetsButton->setEnabled(false);
    }
    QJsonObject job {
        {QStringLiteral("mode"), QStringLiteral("assets")},
        {QStringLiteral("action"), QStringLiteral("check")},
        {QStringLiteral("demo"), false},
    };
    runWorkerJob(job, QStringLiteral("assets-check"));
}

void MainWindow::downloadMissingAssets()
{
    if (m_worker && m_worker->isRunning()) {
        return;
    }
    if (m_assetStatuses.isEmpty()) {
        refreshAssetStatus();
        return;
    }

    QJsonArray requested;
    QStringList details;
    for (const auto &asset : managedModelAssets()) {
        const QJsonObject status = m_assetStatuses.value(asset.first);
        if (status.value(QStringLiteral("ready")).toBool()) {
            continue;
        }
        requested.append(asset.first);
        const QString sourceUrl =
            status.value(QStringLiteral("source_url")).toString();
        const QString source =
            sourceUrl.isEmpty()
            ? status.value(QStringLiteral("source")).toString()
            : sourceUrl;
        const qint64 estimated =
            static_cast<qint64>(
                status.value(QStringLiteral("estimated_bytes")).toDouble());
        QString detail =
            tr("• %1\n  Source: %2\n  Estimated size: %3")
                .arg(status.value(QStringLiteral("display_name")).toString(),
                     source,
                     formatAssetSize(estimated));
        if (status.value(QStringLiteral("import_available")).toBool()) {
            detail += tr("\n  An existing compatible user cache will be "
                         "imported before any download is attempted.");
        }
        details.append(detail);
    }
    if (requested.isEmpty()) {
        showStatus(tr("All local model assets are ready."));
        return;
    }

    const QString destination = QDir::toNativeSeparators(modelAssetsDirectory());
    const QString prompt =
        tr("The following public model assets will be prepared:\n\n%1\n\n"
           "Destination: %2\n\n"
           "This is the only operation allowed to use the network. Existing "
           "compatible caches are imported locally first. No dataset, result, "
           "local path, or telemetry is uploaded.\n\nContinue?")
            .arg(details.join(QStringLiteral("\n\n")), destination);
    if (QMessageBox::question(
            this,
            tr("Download missing model assets"),
            prompt,
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No)
        != QMessageBox::Yes) {
        return;
    }

    QJsonObject job {
        {QStringLiteral("mode"), QStringLiteral("assets")},
        {QStringLiteral("action"), QStringLiteral("download")},
        {QStringLiteral("demo"), false},
        {QStringLiteral("asset_ids"), requested},
    };
    runWorkerJob(job, QStringLiteral("assets-download"));
}

void MainWindow::updateAssetRow(const QJsonObject &event)
{
    if (!m_assetTable) {
        return;
    }
    const QString assetId =
        event.value(QStringLiteral("asset_id")).toString();
    if (assetId.isEmpty()) {
        return;
    }
    int row = -1;
    for (int candidate = 0; candidate < m_assetTable->rowCount(); ++candidate) {
        if (m_assetTable->item(candidate, 0)->data(Qt::UserRole).toString()
            == assetId) {
            row = candidate;
            break;
        }
    }
    if (row < 0) {
        return;
    }

    QJsonObject stored = event;
    stored.insert(QStringLiteral("managed_root"), modelAssetsDirectory());
    m_assetStatuses.insert(assetId, stored);

    const QString source =
        event.value(QStringLiteral("source")).toString();
    m_assetTable->item(row, 1)->setText(
        source.isEmpty() ? tr("Unknown") : source);
    m_assetTable->item(row, 1)->setToolTip(
        event.value(QStringLiteral("source_url")).toString());

    const QString state =
        event.value(QStringLiteral("state")).toString();
    QString statusText = event.value(QStringLiteral("message")).toString();
    if (statusText.isEmpty()) {
        statusText = state;
    }
    const QString revision =
        event.value(QStringLiteral("revision")).toString();
    if (!revision.isEmpty()) {
        statusText += tr(" · revision %1").arg(revision.left(12));
    }
    m_assetTable->item(row, 2)->setText(statusText);
    m_assetTable->item(row, 3)->setText(
        QDir::toNativeSeparators(
            event.value(QStringLiteral("path")).toString()));

    QSettings settings;
    settings.beginGroup(QStringLiteral("AssetStatus/v1"));
    settings.setValue(
        assetId,
        QJsonDocument(stored).toJson(QJsonDocument::Compact));
    settings.endGroup();

    if (m_downloadAssetsButton && (!m_worker || !m_worker->isRunning())) {
        const bool hasMissing =
            std::any_of(m_assetStatuses.cbegin(), m_assetStatuses.cend(),
                        [](const QJsonObject &status) {
                            return !status.value(QStringLiteral("ready")).toBool();
                        });
        m_downloadAssetsButton->setEnabled(hasMissing);
    }
}

void MainWindow::discoverCameras()
{
    QString error;
    const QStringList devices = m_cameraService.discover(&error);
    if (m_cameraCombo) {
        m_cameraCombo->clear();
        m_cameraCombo->addItems(devices);
        if (devices.isEmpty()) {
            m_cameraCombo->addItem(tr("No Basler camera detected"));
        }
    }
    if (m_cameraStatus) {
        m_cameraStatus->setText(devices.isEmpty()
            ? (error.isEmpty() ? tr("No Basler camera detected.") : error)
            : tr("%1 camera(s) ready.").arg(devices.size()));
    }
    showStatus(devices.isEmpty()
        ? (error.isEmpty() ? tr("No Basler camera detected.") : error)
        : tr("%1 Basler camera(s) detected.").arg(devices.size()),
        !error.isEmpty());
}

void MainWindow::captureNormal()
{
    const QString path = nextCapturePath(QStringLiteral("normal"));
    if (path.isEmpty()) {
        return;
    }
    QString error;
    if (!m_cameraService.capture(m_cameraCombo->currentIndex(), path, &error)) {
        showStatus(error, true);
        return;
    }
    showStatus(tr("Normal sample saved: %1").arg(path));
    const QPixmap captured(path);
    if (!captured.isNull()) {
        m_capturePreview->setPixmap(captured.scaled(
            m_capturePreview->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    updateDatasetCounts(m_datasetRoot);
    updateInspectionFromDataset();
}

void MainWindow::captureAnomalous()
{
    const QString path = nextCapturePath(QStringLiteral("abnormal"));
    if (path.isEmpty()) {
        return;
    }
    QString error;
    if (!m_cameraService.capture(m_cameraCombo->currentIndex(), path, &error)) {
        showStatus(error, true);
        return;
    }
    showStatus(tr("Anomalous sample saved: %1").arg(path));
    const QPixmap captured(path);
    if (!captured.isNull()) {
        m_capturePreview->setPixmap(captured.scaled(
            m_capturePreview->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    updateDatasetCounts(m_datasetRoot);
    updateInspectionFromDataset();
}

QString MainWindow::nextCapturePath(const QString &classFolder)
{
    QString root = m_captureDatasetEdit ? m_captureDatasetEdit->text().trimmed()
                                        : m_datasetRoot;
    if (root.isEmpty()) {
        showStatus(tr("Choose a dataset folder before capturing."), true);
        return {};
    }
    m_datasetRoot = QDir::cleanPath(root);
    if (m_datasetPathEdit) {
        m_datasetPathEdit->setText(m_datasetRoot);
    }
    QDir rootDir(root);
    if (!rootDir.mkpath(classFolder)) {
        showStatus(tr("Could not create the dataset class folder."), true);
        return {};
    }
    const QString fileName = QStringLiteral("%1_%2.png")
        .arg(classFolder,
             QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz")));
    return rootDir.filePath(classFolder + QLatin1Char('/') + fileName);
}

void MainWindow::updateThreshold(int value)
{
    if (m_thresholdSpin && m_thresholdSpin->value() != value) {
        m_thresholdSpin->setValue(value);
    }
    if (!m_anomalyMapPath.isEmpty() && m_predictedMaskImage) {
        QImage anomalyMap(m_anomalyMapPath);
        if (!anomalyMap.isNull()) {
            anomalyMap = anomalyMap.convertToFormat(QImage::Format_Grayscale8);
            QImage thresholded(anomalyMap.size(), QImage::Format_RGB32);
            const int cutoff = qRound(255.0 * value / 100.0);
            for (int y = 0; y < anomalyMap.height(); ++y) {
                const uchar *source = anomalyMap.constScanLine(y);
                auto *target = reinterpret_cast<QRgb *>(thresholded.scanLine(y));
                for (int x = 0; x < anomalyMap.width(); ++x) {
                    target[x] = source[x] >= cutoff
                        ? qRgb(239, 55, 52)
                        : qRgb(0, 0, 0);
                }
            }
            m_predictedMaskImage->setText({});
            m_predictedMaskImage->setPixmap(QPixmap::fromImage(thresholded).scaled(
                QSize(m_predictedMaskImage->minimumWidth(),
                      m_predictedMaskImage->height()),
                Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
            m_predictedMaskImage->setToolTip(
                tr("Generated from %1 at threshold %2")
                    .arg(QDir::toNativeSeparators(m_anomalyMapPath))
                    .arg(value / 100.0, 0, 'f', 2));
        }
    }
    showStatus(tr("Display threshold set to %1.").arg(value / 100.0, 0, 'f', 2));
}

QString MainWindow::dataDirectory() const
{
    const QString path = QDir(QCoreApplication::applicationDirPath())
        .filePath(QStringLiteral(".data"));
    QDir().mkpath(path);
    return path;
}

QString MainWindow::workspaceDirectory() const
{
    const auto discoverRoot = [](QDir directory) {
        while (true) {
            if (QFileInfo::exists(
                    directory.filePath(
                        QStringLiteral("KivancAnomaLibTest.pro")))
                && QFileInfo::exists(
                    directory.filePath(
                        QStringLiteral("worker/anomalib_worker.py")))) {
                return QDir::cleanPath(directory.absolutePath());
            }
            if (!directory.cdUp()) {
                break;
            }
        }
        return QString {};
    };

    // Prefer this checkout even when a Python environment from another
    // checkout remains selected in the user's application settings.
    QList<QDir> candidates {
        QDir(QCoreApplication::applicationDirPath()),
        QDir::current(),
    };
    if (m_pythonEdit) {
        const QFileInfo python(m_pythonEdit->text().trimmed());
        if (python.isAbsolute()) {
            candidates.append(python.absoluteDir());
        }
    }
    for (const QDir &candidate : std::as_const(candidates)) {
        const QString discovered = discoverRoot(candidate);
        if (!discovered.isEmpty()) {
            return discovered;
        }
    }
    return QDir::cleanPath(QCoreApplication::applicationDirPath());
}

QString MainWindow::modelAssetsDirectory() const
{
    return QDir(workspaceDirectory())
        .filePath(QStringLiteral(".data/model-assets"));
}

QString MainWindow::ensureWorkerScript(QString *errorMessage)
{
    if (errorMessage) {
        errorMessage->clear();
    }
    const QString target = QDir(dataDirectory()).filePath(
        QStringLiteral("anomalib_worker.py"));
    QFile source(QStringLiteral(":/worker/anomalib_worker.py"));
    if (!source.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = tr("Could not read the embedded Anomalib worker: %1")
                                .arg(source.errorString());
        }
        return {};
    }
    const QByteArray contents = source.readAll();
    source.close();

    {
        QFile existing(target);
        if (existing.exists()) {
            if (!existing.open(QIODevice::ReadOnly)) {
                if (errorMessage) {
                    *errorMessage =
                        tr("Could not read the existing Anomalib worker at %1: %2")
                            .arg(QDir::toNativeSeparators(target),
                                 existing.errorString());
                }
                return {};
            }
            const bool matches = existing.readAll() == contents;
            existing.close();
            if (matches) {
                return target;
            }
        }
    }

    QSaveFile output(target);
    if (!output.open(QIODevice::WriteOnly)) {
        if (errorMessage) {
            *errorMessage = tr("Could not update the Anomalib worker at %1: %2")
                                .arg(QDir::toNativeSeparators(target),
                                     output.errorString());
        }
        return {};
    }
    if (output.write(contents) != contents.size()) {
        if (errorMessage) {
            *errorMessage = tr("Could not write the Anomalib worker at %1: %2")
                                .arg(QDir::toNativeSeparators(target),
                                     output.errorString());
        }
        output.cancelWriting();
        return {};
    }
    if (!output.commit()) {
        if (errorMessage) {
            *errorMessage = tr("Could not replace the Anomalib worker at %1: %2")
                                .arg(QDir::toNativeSeparators(target),
                                     output.errorString());
        }
        return {};
    }
    return target;
}

QString MainWindow::writeJobFile(const QJsonObject &job, const QString &prefix,
                                 QString *errorMessage)
{
    if (errorMessage) {
        errorMessage->clear();
    }
    const QString jobs = QDir(dataDirectory()).filePath(QStringLiteral("jobs"));
    if (!QDir().mkpath(jobs)) {
        if (errorMessage) {
            *errorMessage = tr("Could not create the Anomalib jobs folder: %1")
                                .arg(QDir::toNativeSeparators(jobs));
        }
        return {};
    }
    const QString path = QDir(jobs).filePath(
        QStringLiteral("%1-%2.json")
            .arg(prefix, QDateTime::currentDateTimeUtc()
                             .toString(QStringLiteral("yyyyMMdd-HHmmss-zzz"))));
    QSaveFile file(path);
    const QByteArray contents = QJsonDocument(job).toJson(QJsonDocument::Indented);
    if (!file.open(QIODevice::WriteOnly)) {
        if (errorMessage) {
            *errorMessage = tr("Could not create the Anomalib job at %1: %2")
                                .arg(QDir::toNativeSeparators(path),
                                     file.errorString());
        }
        return {};
    }
    if (file.write(contents) != contents.size()) {
        if (errorMessage) {
            *errorMessage = tr("Could not write the Anomalib job at %1: %2")
                                .arg(QDir::toNativeSeparators(path),
                                     file.errorString());
        }
        file.cancelWriting();
        return {};
    }
    if (!file.commit()) {
        if (errorMessage) {
            *errorMessage = tr("Could not finalize the Anomalib job at %1: %2")
                                .arg(QDir::toNativeSeparators(path),
                                     file.errorString());
        }
        return {};
    }
    return path;
}

QString MainWindow::defaultPythonExecutable() const
{
    const QString configured = qEnvironmentVariable("ANOMALIB_PYTHON");
    if (!configured.isEmpty()) {
        return QDir::cleanPath(configured);
    }

    {
        QSettings settings;
        const QString saved =
            settings.value(QStringLiteral("Environment/pythonExecutable"))
                .toString()
                .trimmed();
        if (!saved.isEmpty() && QFileInfo::exists(saved)) {
            return QDir::cleanPath(saved);
        }
    }

    const auto findProjectVenv = [](QDir directory) {
        while (true) {
            const QString candidate =
                directory.filePath(QStringLiteral(".venv/Scripts/python.exe"));
            if (QFileInfo::exists(candidate)) {
                return QDir::cleanPath(candidate);
            }
            if (!directory.cdUp()) {
                break;
            }
        }
        return QString {};
    };
    for (const QDir &directory :
         {QDir::current(), QDir(QCoreApplication::applicationDirPath())}) {
        const QString workspaceVenv = findProjectVenv(directory);
        if (!workspaceVenv.isEmpty()) {
            return workspaceVenv;
        }
    }

    const QString bundledDevelopmentPython = QDir(QDir::homePath()).filePath(
        QStringLiteral(".cache/codex-runtimes/codex-primary-runtime/"
                       "dependencies/python/python.exe"));
    if (QFileInfo::exists(bundledDevelopmentPython)) {
        return bundledDevelopmentPython;
    }
    const QString found = QStandardPaths::findExecutable(QStringLiteral("python.exe"));
    return found.isEmpty() ? QStringLiteral("python.exe") : found;
}

void MainWindow::showStatus(const QString &message, bool error)
{
    statusBar()->showMessage(message, error ? 10000 : 5000);
    if (m_statusMode) {
        m_statusMode->setText(message);
        m_statusMode->setStyleSheet(
            error ? QStringLiteral("color:#c43f3f; padding-left:8px;")
                  : QStringLiteral("color:#53606d; padding-left:8px;"));
    }
}
