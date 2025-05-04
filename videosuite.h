#ifndef VIDEOSUITE_H
#define VIDEOSUITE_H

#include <QMainWindow>
#include <QTimer>
#include <QFileDialog>
#include <QMenuBar>
#include <QLabel>
#include <QProgressDialog>
#include <QActionGroup>
#include <QFuture>
#include <QFutureWatcher>
#include <opencv2/opencv.hpp>
#include "cuda_functions.h"

QT_BEGIN_NAMESPACE
namespace Ui { class VideoSuite; }
QT_END_NAMESPACE

class VideoSuite : public QMainWindow
{
    Q_OBJECT

public:
    VideoSuite(QWidget *parent = nullptr);
    ~VideoSuite();

private slots:
    void openFile();
    void updateFrame();
    void updateProcessedFrame();  
    void playPause();
    void playProcessedVideo();   
    void applyFilter();
    void exportVideo();
    void processFinished();
    void updateProcessingProgress(int current, int total);
    

    void updateSliderPosition();
    void seekVideo(int position);
    void videoSliderPressed();
    void videoSliderReleased();

private:
    Ui::VideoSuite *ui;

    QTimer *timer;
    cv::VideoCapture capture;
    cv::VideoCapture processedCapture;  
    cv::Mat currentFrame;
    cv::Mat originalFrame; 
    cv::Mat processedFrame;
    QImage qtImage;

    QString currentVideoPath;
    QString processedVideoPath;

    bool isPlaying;
    bool hasProcessedVideo;
    bool isShowingProcessed;
    bool isSliderPressed;  

    
    int totalFrames;
    double fps;
    double currentTimeInSeconds;
    double totalTimeInSeconds;
   
    QFutureWatcher<bool> *processingWatcher;
    QProgressDialog *progressDialog;
    FilterType currentFilter;

    QMenu *filterMenu;
    QAction *gaussianBlurAction;
    QAction *grayscaleAction;
    QAction *thresholdAction;
    QAction *sepiaAction;
    QAction *roseyAction;
    QAction *invertAction;
    QAction *grayscaleInvertAction;
    QAction *pixelateAction;
    QAction *edgeDetectAction;
    QAction *kuwaharaAction;

    QAction *exportAction;
    QActionGroup *viewModeGroup;
    QAction *showOriginalAction;
    QAction *showProcessedAction;

    void setupMenuBar();
    void setupConnections();
    void displayFrame(const cv::Mat &frame);
    void setViewMode(bool showProcessed);
    void updateViewControls();
    
    QString formatTime(double seconds);
    void updateTimeDisplay();
    void initializeVideoControls();
};
#endif // VIDEOSUITE_H
