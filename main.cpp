#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QApplication::setOrganizationName(QStringLiteral("Kivanc Vision"));
    QApplication::setApplicationName(QStringLiteral("Inspection Workbench"));
    QApplication::setApplicationVersion(QStringLiteral("0.1.0"));
    QApplication::setStyle(QStringLiteral("Fusion"));

    MainWindow w;
    w.show();
    return QApplication::exec();
}
