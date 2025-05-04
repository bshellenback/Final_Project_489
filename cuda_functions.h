#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <string>
#include <functional>
#include <vector>

// Define filter types
enum class FilterType {
    GAUSSIAN_BLUR,
    GRAYSCALE,
    THRESHOLD,
    SEPIA,
    ROSEY,
    INVERT,
    GRAYSCALE_INVERT,
    PIXELATE,
    EDGE_DETECT,
    KUWAHARA
};

// Function to process video with CUDA
bool processVideoWithCUDA(const std::string& inputPath, const std::string& outputPath, FilterType filterType, 
    std::function<void(int, int)> progressCallback = nullptr);

// Function to process a batch of frames
bool processBatch(std::vector<cv::Mat>& inputFrames, std::vector<cv::Mat>& outputFrames, std::vector<cv::cuda::GpuMat>& d_inputs,
    std::vector<cv::cuda::GpuMat>& d_outputs, std::vector<cudaStream_t>& streams, std::vector<cv::cuda::Stream>& cv_streams, 
    FilterType filterType);

// Helper function to compute Gaussian kernel
void computeGaussianKernel(float* kernel, int kernelSize, float sigma);

// Add declarations for the new filter functions
void applyGaussianBlurFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyGrayscaleFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyThresholdFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, int threshold = 128, cudaStream_t stream = 0);
void applySepiaFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyRoseyFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyInvertFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyGrayscaleInvertFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyPixelateFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyEdgeDetectFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);
void applyKuwaharaFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream = 0);

#endif // CUDA_FUNCTIONS_H
