#include "videosuite.h"
#include "ui_videosuite.h"
#include <QMessageBox>
#include <QtConcurrent/QtConcurrent>
#include <QFileInfo>
#include <chrono>

VideoSuite::VideoSuite(QWidget *parent)
    :QMainWindow(parent), ui(new Ui::VideoSuite), isPlaying(false), hasProcessedVideo(false), isShowingProcessed(false), 
    isSliderPressed(false), totalFrames(0), fps(0.0), currentTimeInSeconds(0.0), totalTimeInSeconds(0.0)
{
    ui->setupUi(this);

    // Set window title
    setWindowTitle("CUDA Video Processing Suite");

    // Initialize timer for video playback
    timer = new QTimer(this);

    // Initialize processing watcher
    processingWatcher = new QFutureWatcher<bool>(this);

    // Set up the menu bar and connections
    setupMenuBar();
    setupConnections();
    
    // Initialize video slider
    ui->videoSlider->setEnabled(false);
}

VideoSuite::~VideoSuite()
{
    if (capture.isOpened()) {
        capture.release();
    }

    if (processedCapture.isOpened()) {
        processedCapture.release();
    }

    if (timer->isActive()) {
        timer->stop();
    }

    delete timer;
    delete processingWatcher;
    delete ui;
}

void VideoSuite::setupMenuBar()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openAction = new QAction("&Open", this);
    openAction->setShortcut(QKeySequence::Open);
    fileMenu->addAction(openAction);

    exportAction = new QAction("&Export Processed Video", this);
    exportAction->setEnabled(false);
    fileMenu->addAction(exportAction);

    QAction *exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    fileMenu->addAction(exitAction);

    // Filter menu
    filterMenu = menuBar()->addMenu("&Filters");

    gaussianBlurAction = new QAction("Gaussian &Blur", this);
    filterMenu->addAction(gaussianBlurAction);
    
    // Add new filter actions
    grayscaleAction = new QAction("&Grayscale", this);
    filterMenu->addAction(grayscaleAction);
    
    thresholdAction = new QAction("&Threshold", this);
    filterMenu->addAction(thresholdAction);
    
    sepiaAction = new QAction("S&epia", this);
    filterMenu->addAction(sepiaAction);
    
    roseyAction = new QAction("&Rosey", this);
    filterMenu->addAction(roseyAction);
    
    invertAction = new QAction("&Invert", this);
    filterMenu->addAction(invertAction);
    
    grayscaleInvertAction = new QAction("Grayscale In&vert", this);
    filterMenu->addAction(grayscaleInvertAction);
    
    pixelateAction = new QAction("&Pixelate", this);
    filterMenu->addAction(pixelateAction);
    
    edgeDetectAction = new QAction("&Edge Detect", this);
    filterMenu->addAction(edgeDetectAction);

    kuwaharaAction = new QAction("&Kuwahara", this);
    filterMenu->addAction(kuwaharaAction);

    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");

    viewModeGroup = new QActionGroup(this);

    showOriginalAction = new QAction("Show &Original", this);
    showOriginalAction->setCheckable(true);
    showOriginalAction->setChecked(true);
    viewModeGroup->addAction(showOriginalAction);
    viewMenu->addAction(showOriginalAction);

    showProcessedAction = new QAction("Show &Processed", this);
    showProcessedAction->setCheckable(true);
    showProcessedAction->setEnabled(false);
    viewModeGroup->addAction(showProcessedAction);
    viewMenu->addAction(showProcessedAction);

    connect(openAction, &QAction::triggered, this, &VideoSuite::openFile);
    connect(exportAction, &QAction::triggered, this, &VideoSuite::exportVideo);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);
    
    // Connect filter actions
    connect(gaussianBlurAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(grayscaleAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(thresholdAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(sepiaAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(roseyAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(invertAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(grayscaleInvertAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(pixelateAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(edgeDetectAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    connect(kuwaharaAction, &QAction::triggered, this, &VideoSuite::applyFilter);
    
    connect(showOriginalAction, &QAction::triggered, [this]() { setViewMode(false); });
    connect(showProcessedAction, &QAction::triggered, [this]() { setViewMode(true); });
}

void VideoSuite::setupConnections()
{
    connect(timer, &QTimer::timeout, this, &VideoSuite::updateFrame);
    connect(ui->playButton, &QPushButton::clicked, this, &VideoSuite::playPause);
    connect(processingWatcher, &QFutureWatcher<bool>::finished, this, &VideoSuite::processFinished);
    
    // Connect slider signals
    connect(ui->videoSlider, &QSlider::sliderMoved, this, &VideoSuite::seekVideo);
    connect(ui->videoSlider, &QSlider::sliderPressed, this, &VideoSuite::videoSliderPressed);
    connect(ui->videoSlider, &QSlider::sliderReleased, this, &VideoSuite::videoSliderReleased);
    
    // Create a timer for updating the slider position during playback
    QTimer *sliderTimer = new QTimer(this);
    connect(sliderTimer, &QTimer::timeout, this, &VideoSuite::updateSliderPosition);
    sliderTimer->start(100); // Update every 100ms
}

void VideoSuite::openFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Video File", "", "Video Files (*.avi *.mp4 *.mkv *.mov *.wmv)");

    if (fileName.isEmpty()) {
        return;
    }

    // Stop any current playback
    if (timer->isActive()) {
        timer->stop();
    }

    if (capture.isOpened()) {
        capture.release();
    }

    // Close processed video if open
    if (processedCapture.isOpened()) {
        processedCapture.release();
    }

    // Open the video file
    capture.open(fileName.toStdString());

    if (!capture.isOpened()) {
        QMessageBox::critical(this, "Error", "Could not open video file");
        return;
    }

    // Store the current video path
    currentVideoPath = fileName;

    // Reset processed video state
    hasProcessedVideo = false;
    isShowingProcessed = false;
    updateViewControls();

    // Get the first frame to display
    capture.read(currentFrame);
    if (!currentFrame.empty()) {
        originalFrame = currentFrame.clone();
        displayFrame(currentFrame);
    }

    // Update UI
    ui->playButton->setText("Play");
    isPlaying = false;

    // Initialize video controls with proper timing info
    initializeVideoControls();

    // Show information about the video
    ui->statusbar->showMessage(QString("Video loaded: %1x%2, %3 FPS, %4 frames")
                                   .arg(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)))
                                   .arg(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)))
                                   .arg(fps).arg(totalFrames));
}

void VideoSuite::updateFrame()
{
    if (!capture.isOpened()) {
        return;
    }

    // Read the next frame
    if (!capture.read(currentFrame)) {
        // End of video or error
        timer->stop();
        ui->playButton->setText("Play");
        isPlaying = false;

        // Reset to the beginning of the video
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        return;
    }

    // Store original frame
    originalFrame = currentFrame.clone();

    // Display appropriate frame based on view mode
    if (isShowingProcessed && hasProcessedVideo) {
        displayFrame(processedFrame);
    } else {
        displayFrame(currentFrame);
    }
    
    // Update slider position and time display
    updateSliderPosition();
}

void VideoSuite::updateProcessedFrame()
{
    if (!processedCapture.isOpened()) {
        return;
    }

    // Read the next frame
    if (!processedCapture.read(processedFrame)) {
        // End of video or error
        timer->stop();
        ui->playButton->setText("Play");
        isPlaying = false;

        // Reset to the beginning of the video
        processedCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
        return;
    }

    // Display the frame
    displayFrame(processedFrame);
    
    // Update slider position and time display
    updateSliderPosition();
}

void VideoSuite::displayFrame(const cv::Mat &frame)
{
    if (frame.empty()) return;

    // Convert from BGR to RGB format
    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

    // Convert to QImage
    qtImage = QImage(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                     rgbFrame.step, QImage::Format_RGB888);

    // Display in the video label
    ui->videoLabel->setPixmap(QPixmap::fromImage(qtImage).scaled(
        ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void VideoSuite::playPause()
{
    if (!capture.isOpened() && !processedCapture.isOpened()) {
        QMessageBox::information(this, "Information", "No video loaded. Please open a video file first.");
        return;
    }

    if (isPlaying) {
        // Pause the video
        timer->stop();
        ui->playButton->setText("Play");
        isPlaying = false;

    } else {
        // Play the video
        if (isShowingProcessed && hasProcessedVideo) {
            playProcessedVideo();
            
        } else {
            // Original video playback logic
            double fps = capture.get(cv::CAP_PROP_FPS);
            int interval = static_cast<int>(1000.0 / fps);

            // Make sure timer is connected to the right update function
            disconnect(timer, &QTimer::timeout, 0, 0);  // Disconnect all signals
            connect(timer, &QTimer::timeout, this, &VideoSuite::updateFrame);

            timer->start(interval);
            ui->playButton->setText("Pause");
            isPlaying = true;
        }
    }
}

void VideoSuite::playProcessedVideo()
{
    if (!hasProcessedVideo) {
        return;
    }

    // If the original is playing, stop it
    if (isPlaying) {
        timer->stop();
        isPlaying = false;
        ui->playButton->setText("Play");
    }

    // Open the processed video if not already open
    if (!processedCapture.isOpened()) {
        processedCapture.open(processedVideoPath.toStdString());
        if (!processedCapture.isOpened()) {
            QMessageBox::critical(this, "Error", "Could not open processed video");
            return;
        }
    }

    // Get properties
    double fps = processedCapture.get(cv::CAP_PROP_FPS);
    int interval = static_cast<int>(1000.0 / fps);

    // Set up a different timer for processed video
    if (timer->isActive()) {
        timer->stop();
    }

    // Start timer with callback to show processed frames
    disconnect(timer, &QTimer::timeout, 0, 0);  // Disconnect all signals
    connect(timer, &QTimer::timeout, this, &VideoSuite::updateProcessedFrame);
    timer->start(interval);

    isPlaying = true;
    ui->playButton->setText("Pause");
}

void VideoSuite::applyFilter()
{
    if (!capture.isOpened()) {
        QMessageBox::information(this, "Information", "No video loaded. Please open a video file first.");
        return;
    }

    // Pause video if playing
    if (isPlaying) {
        timer->stop();
        ui->playButton->setText("Play");
        isPlaying = false;
    }

    // Determine which filter to apply
    QObject* sender = QObject::sender();
    if (sender == gaussianBlurAction) {
        currentFilter = FilterType::GAUSSIAN_BLUR;
    } else if (sender == grayscaleAction) {
        currentFilter = FilterType::GRAYSCALE;
    } else if (sender == thresholdAction) {
        currentFilter = FilterType::THRESHOLD;
    } else if (sender == sepiaAction) {
        currentFilter = FilterType::SEPIA;
    } else if (sender == roseyAction) {
        currentFilter = FilterType::ROSEY;
    } else if (sender == invertAction) {
        currentFilter = FilterType::INVERT;
    } else if (sender == grayscaleInvertAction) {
        currentFilter = FilterType::GRAYSCALE_INVERT;
    } else if (sender == pixelateAction) {
        currentFilter = FilterType::PIXELATE;
    } else if (sender == edgeDetectAction) {
        currentFilter = FilterType::EDGE_DETECT;
    } else if (sender == kuwaharaAction) {
        currentFilter = FilterType::KUWAHARA;
    }

    // Create temporary output path
    QFileInfo fileInfo(currentVideoPath);
    QString baseName = fileInfo.baseName();
    processedVideoPath = fileInfo.absolutePath() + "/" + baseName + "_processed.mp4";

    // Create progress dialog
    progressDialog = new QProgressDialog("Processing video...", nullptr, 0, 100, this);

    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setMinimumDuration(0);
    progressDialog->setValue(0);

    // Disable the UI during processing
    setEnabled(false);

    // Start the processing in a separate thread
    QFuture<bool> future = QtConcurrent::run([this]() {
        return processVideoWithCUDA(
            currentVideoPath.toStdString(),
            processedVideoPath.toStdString(),
            currentFilter,
            [this](int current, int total) {
                // Update progress from worker thread
                QMetaObject::invokeMethod(this, "updateProcessingProgress", Qt::QueuedConnection, Q_ARG(int, current), 
                    Q_ARG(int, total));
            }
            );
    });

    processingWatcher->setFuture(future);
}

void VideoSuite::updateProcessingProgress(int current, int total)
{
    if (progressDialog) {
        int percent = (total > 0) ? (current * 100 / total) : 0;
        progressDialog->setValue(percent);
    }
}

void VideoSuite::processFinished()
{
    // Re-enable the UI
    setEnabled(true);

    // Clean up progress dialog
    if (progressDialog) {
        progressDialog->close();
        delete progressDialog;
        progressDialog = nullptr;
    }

    bool success = processingWatcher->result();

    if (success) {
        // Store the original video position
        int originalPos = 0;
        if (capture.isOpened()) {
            originalPos = capture.get(cv::CAP_PROP_POS_FRAMES);
        }

        // Close any existing processed capture
        if (processedCapture.isOpened()) {
            processedCapture.release();
        }

        // Open the processed video for viewing
        processedCapture.open(processedVideoPath.toStdString());
        if (processedCapture.isOpened()) {
            // Read first frame for display
            processedCapture.read(processedFrame);

            // Update UI
            hasProcessedVideo = true;
            isShowingProcessed = true;
            updateViewControls();

            // Display processed frame
            displayFrame(processedFrame);
            
            // Reset slider for processed video
            if (isShowingProcessed) {
                ui->videoSlider->setValue(0);
                updateTimeDisplay();
            }

            // Update status
            QFileInfo fileInfo(processedVideoPath);
            ui->statusbar->showMessage(QString("Processing completed: %1") .arg(processedVideoPath));
        }

        // Restore original video position if we want to go back to it
        if (capture.isOpened()) {
            capture.set(cv::CAP_PROP_POS_FRAMES, originalPos);
        }

    } else {
        QMessageBox::critical(this, "Error", "Failed to process video");
    }
}

void VideoSuite::exportVideo()
{
    if (!hasProcessedVideo) {
        QMessageBox::information(this, "Information", "No processed video available to export.");
        return;
    }

    QString saveFileName = QFileDialog::getSaveFileName(this, "Export Processed Video", "", "Video Files (*.mp4)");

    if (saveFileName.isEmpty()) {
        return;
    }

    // Ensure file has .mp4 extension
    if (!saveFileName.endsWith(".mp4", Qt::CaseInsensitive)) {
        saveFileName += ".mp4";
    }

    // Copy processed video to new location
    QFile::copy(processedVideoPath, saveFileName);

    ui->statusbar->showMessage(QString("Video exported to: %1") .arg(saveFileName));
}

void VideoSuite::setViewMode(bool showProcessed)
{
    if (!hasProcessedVideo && showProcessed) {
        QMessageBox::information(this, "Information", "No processed video available to display.");
        showOriginalAction->setChecked(true);
        return;
    }

    isShowingProcessed = showProcessed;

    // Update the display
    if (isShowingProcessed) {
        // If we want to show processed video, open it if needed
        if (!processedCapture.isOpened()) {
            processedCapture.open(processedVideoPath.toStdString());
            if (processedCapture.isOpened()) {
                processedCapture.read(processedFrame);
            }
        }
        displayFrame(processedFrame);

        // Update slider to match processed video position
        if (processedCapture.isOpened()) {
            ui->videoSlider->setValue(static_cast<int>(processedCapture.get(cv::CAP_PROP_POS_FRAMES)));
            updateTimeDisplay();
        }

        // If we're playing, switch to the processed video playback
        if (isPlaying) {
            playProcessedVideo();
        }
    } else {
        // Switch back to original video
        displayFrame(originalFrame);

        // Update slider to match original video position
        if (capture.isOpened()) {
            ui->videoSlider->setValue(static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES)));
            updateTimeDisplay();
        }

        // If we're playing, make sure we're playing the original
        if (isPlaying) {
            // Reconnect timer to the original update function
            disconnect(timer, &QTimer::timeout, 0, 0);  // Disconnect all signals
            connect(timer, &QTimer::timeout, this, &VideoSuite::updateFrame);
        }
    }
}

void VideoSuite::updateViewControls()
{
    showProcessedAction->setEnabled(hasProcessedVideo);
    exportAction->setEnabled(hasProcessedVideo);

    if (!hasProcessedVideo) {
        showOriginalAction->setChecked(true);
    } else {
        if (isShowingProcessed) {
            showProcessedAction->setChecked(true);
        } else {
            showOriginalAction->setChecked(true);
        }
    }
}


void VideoSuite::initializeVideoControls()
{
    // Get video properties
    fps = capture.get(cv::CAP_PROP_FPS);
    totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    totalTimeInSeconds = totalFrames / fps;
    
    // Configure slider
    ui->videoSlider->setEnabled(true);
    ui->videoSlider->setMinimum(0);
    ui->videoSlider->setMaximum(totalFrames > 0 ? totalFrames : 1000);
    ui->videoSlider->setValue(0);
    
    // Update time displays
    currentTimeInSeconds = 0.0;
    ui->currentTimeLabel->setText("00:00");
    ui->totalTimeLabel->setText(formatTime(totalTimeInSeconds));
}

QString VideoSuite::formatTime(double seconds)
{
    int totalSecs = static_cast<int>(seconds);
    int mins = totalSecs / 60;
    int secs = totalSecs % 60;
    return QString("%1:%2").arg(mins, 2, 10, QChar('0')).arg(secs, 2, 10, QChar('0'));
}

void VideoSuite::updateTimeDisplay()
{
    if (!capture.isOpened() && !processedCapture.isOpened()) {
        return;
    }
    
    int currentFrame;
    if (isShowingProcessed && processedCapture.isOpened()) {
        currentFrame = static_cast<int>(processedCapture.get(cv::CAP_PROP_POS_FRAMES));
    } else {
        currentFrame = static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES));
    }
    
    currentTimeInSeconds = currentFrame / fps;
    ui->currentTimeLabel->setText(formatTime(currentTimeInSeconds));
}

void VideoSuite::updateSliderPosition()
{
    if (isSliderPressed || (!capture.isOpened() && !processedCapture.isOpened())) {
        return;
    }
    
    // Get the current frame position
    int currentFrame;
    if (isShowingProcessed && processedCapture.isOpened()) {
        currentFrame = static_cast<int>(processedCapture.get(cv::CAP_PROP_POS_FRAMES));

    } else if (capture.isOpened()) {
        currentFrame = static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES));
        
    } else {
        return;
    }
    
    // Update the slider without triggering the valueChanged signal
    ui->videoSlider->blockSignals(true);
    ui->videoSlider->setValue(currentFrame);
    ui->videoSlider->blockSignals(false);
    
    // Update time display
    updateTimeDisplay();
}

void VideoSuite::seekVideo(int position)
{
    if (!capture.isOpened() && !processedCapture.isOpened()) {
        return;
    }
    
    // Seek to the specified frame
    if (isShowingProcessed && processedCapture.isOpened()) {
        processedCapture.set(cv::CAP_PROP_POS_FRAMES, position);
        if (processedCapture.read(processedFrame)) {
            displayFrame(processedFrame);
        }
    } else if (capture.isOpened()) {
        capture.set(cv::CAP_PROP_POS_FRAMES, position);
        if (capture.read(currentFrame)) {
            originalFrame = currentFrame.clone();
            displayFrame(currentFrame);
        }
    }
    
    // Update the time display
    updateTimeDisplay();
}

void VideoSuite::videoSliderPressed()
{
    isSliderPressed = true;
    if (isPlaying) {
        timer->stop(); // Pause playback while sliding
    }
}

void VideoSuite::videoSliderReleased()
{
    isSliderPressed = false;
    if (isPlaying) {
        // Resume playback if it was playing
        double fps = isShowingProcessed ? processedCapture.get(cv::CAP_PROP_FPS) : capture.get(cv::CAP_PROP_FPS);
        int interval = static_cast<int>(1000.0 / fps);
        timer->start(interval);
    }
    // Update position from slider
    seekVideo(ui->videoSlider->value());
}
