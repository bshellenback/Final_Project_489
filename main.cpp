#include "videosuite.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    VideoSuite w;
    w.show();
    return a.exec();
}
