#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_functions.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define NUM_STREAMS 8

#define KERNEL_SIZE 5
#define PIXEL_SIZE 8
#define TILE_SIZE 16
#define NUM_CHANNELS 3

// Constant memory for Gaussian kernel weights
__constant__ float d_gaussian_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Helper function to compute Gaussian kernel
void computeGaussianKernel(float* kernel, float sigma)
{
    float sum = 0.0f;
    int radius = KERNEL_SIZE / 2;

    const float inv_2sigma_squared = 1.0f / (2.0f * sigma * sigma);

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float weight = expf(-(x*x + y*y) * inv_2sigma_squared);
            kernel[(y + radius) * KERNEL_SIZE + (x + radius)] = weight;
            sum += weight;
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
        kernel[i] /= sum;
    }
}

// Kernel for Gaussian Blur
__global__ void gaussianBlurSharedMem(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];  

    const int channels = 3;
    int radius = KERNEL_SIZE / 2;
    
    int sharedWidth  = BLOCK_SIZE_X + 2*radius;
    int sharedHeight = BLOCK_SIZE_Y + 2*radius;
    
    int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    
    int tileStartX = blockIdx.x * BLOCK_SIZE_X - radius;
    int tileStartY = blockIdx.y * BLOCK_SIZE_Y - radius;
    
    int tx = threadIdx.x, ty = threadIdx.y;

    
    for (int dy = ty; dy < sharedHeight; dy += BLOCK_SIZE_Y) {
        for (int dx = tx; dx < sharedWidth; dx += BLOCK_SIZE_X) {
            
            int gx = tileStartX + dx;
            int gy = tileStartY + dy;
            gx = min(max(gx, 0), width  - 1);
            gy = min(max(gy, 0), height - 1);
           
            for (int c = 0; c < channels; ++c) {
                sharedMem[(dy*sharedWidth + dx)*channels + c] =
                    input[(gy * pitch) + (gx*channels) + c];
            }
        }
    }
    __syncthreads();

    
    if (x >= width || y >= height) return;

    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                int sx = tx + kx;
                int sy = ty + ky;
                float kernel_val = d_gaussian_kernel[ky * KERNEL_SIZE + kx];
                sum += sharedMem[(sy*sharedWidth + sx)*channels + c] * kernel_val;
            }
        }
        
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[(y * pitch) + (x*channels) + c] = static_cast<unsigned char>(sum);
    }
}

// Kernel for grayscale conversion
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Calculate grayscale value
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            sum += sharedMem[(ty * TILE_SIZE + tx) * channels + c];
        }

        float val = sum / channels;

        // Write grayscale value to all channels
        for (int c = 0; c < channels; c++) {
            output[(y * pitch) + (x * channels) + c] = static_cast<unsigned char>(val);
        }
    }
}

// Kernel for threshold filter
__global__ void thresholdKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch, int threshold)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Calculate grayscale value for thresholding
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            sum += sharedMem[(ty * TILE_SIZE + tx) * channels + c];
        }

        float val = sum / channels;
        unsigned char result = (val > threshold) ? 255 : 0;

        // Write thresholded value to all channels
        for (int c = 0; c < channels; c++) {
            output[(y * pitch) + (x * channels) + c] = result;
        }
    }
}

// Kernel for sepia filter
__global__ void sepiaKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Get original colors (BGR order in OpenCV)
        unsigned char b = sharedMem[(ty * TILE_SIZE + tx) * channels + 0];
        unsigned char g = sharedMem[(ty * TILE_SIZE + tx) * channels + 1];
        unsigned char r = sharedMem[(ty * TILE_SIZE + tx) * channels + 2];

        // Apply sepia transformation
        int outputR = min(255, int(0.393f * r + 0.769f * g + 0.189f * b));
        int outputG = min(255, int(0.349f * r + 0.686f * g + 0.168f * b));
        int outputB = min(255, int(0.272f * r + 0.534f * g + 0.131f * b));

        // Write to output (BGR order)
        output[(y * pitch) + (x * channels) + 0] = outputB;
        output[(y * pitch) + (x * channels) + 1] = outputG;
        output[(y * pitch) + (x * channels) + 2] = outputR;
    }
}

// Kernel for rosey filter
__global__ void roseyKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Get original colors (BGR order in OpenCV)
        unsigned char b = sharedMem[(ty * TILE_SIZE + tx) * channels + 0];
        unsigned char g = sharedMem[(ty * TILE_SIZE + tx) * channels + 1];
        unsigned char r = sharedMem[(ty * TILE_SIZE + tx) * channels + 2];

        // Apply rosey transformation
        int outputR = min(255, int(0.993f * r + 0.369f * g + 0.489f * b));
        int outputG = min(255, int(0.149f * r + 0.286f * g + 0.368f * b));
        int outputB = min(255, int(0.372f * r + 0.234f * g + 0.531f * b));

        // Write to output (BGR order)
        output[(y * pitch) + (x * channels) + 0] = outputB;
        output[(y * pitch) + (x * channels) + 1] = outputG;
        output[(y * pitch) + (x * channels) + 2] = outputR;
    }
}

// Kernel for invert filter
__global__ void invertKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Invert each channel
        for (int c = 0; c < channels; c++) {
            unsigned char val = 255 - sharedMem[(ty * TILE_SIZE + tx) * channels + c];
            output[(y * pitch) + (x * channels) + c] = val;
        }
    }
}

// Kernel for grayscale invert filter
__global__ void grayscaleInvertKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;

    // Global pixel coords
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 1D index of this thread within the block
    int tx = threadIdx.x, ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        for (int c = 0; c < channels; c++) {
            sharedMem[(ty * TILE_SIZE + tx) * channels + c] =
                input[(y * pitch) + (x * channels) + c];
        }

        __syncthreads();

        // Calculate grayscale value
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            sum += sharedMem[(ty * TILE_SIZE + tx) * channels + c];
        }

        // Invert the grayscale value
        unsigned char val = 255 - (sum / channels);

        // Write inverted grayscale value to all channels
        for (int c = 0; c < channels; c++) {
            output[(y * pitch) + (x * channels) + c] = val;
        }
    }
}

// Kernel for pixelate filter
__global__ void pixelateKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    int block_x = blockIdx.x * PIXEL_SIZE;
    int block_y = blockIdx.y * PIXEL_SIZE;
    int x = block_x + threadIdx.x;
    int y = block_y + threadIdx.y;

    if (threadIdx.x >= PIXEL_SIZE || threadIdx.y >= PIXEL_SIZE || x >= width || y >= height) return;

    const int channels = 3;
    float avg[3] = {0.0f, 0.0f, 0.0f};
    int count = 0;

    // Calculate average color for this block
    for (int dy = 0; dy < PIXEL_SIZE; dy++) {
        for (int dx = 0; dx < PIXEL_SIZE; dx++) {
            int px = block_x + dx;
            int py = block_y + dy;

            if (px < width && py < height) {
                for (int c = 0; c < channels; c++) {
                    avg[c] += input[(py * pitch) + (px * channels) + c];
                }
                count++;
            }
        }
    }

    // Normalize
    if (count > 0) {
        for (int c = 0; c < channels; c++) {
            avg[c] /= count;
        }
    }

    // Apply pixelated color to this thread's pixel
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            output[(y * pitch) + (x * channels) + c] = static_cast<unsigned char>(avg[c]);
        }
    }
}

// Kernel for edge detect filter
__global__ void edgeDetectKernel(const unsigned char* input, unsigned char* output, int width, int height, int pitch)
{
    extern __shared__ unsigned char sharedMem[];
    const int channels = 3;
    int radius = 1; // For 3x3 Sobel operator

    // Calculate shared memory dimensions
    int sharedWidth = TILE_SIZE + 2*radius;
    int sharedHeight = TILE_SIZE + 2*radius;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // Origin of tile in input image
    int tileStartX = blockIdx.x * TILE_SIZE - radius;
    int tileStartY = blockIdx.y * TILE_SIZE - radius;

    // Load data into shared memory with halo
    for (int dy = ty; dy < sharedHeight; dy += TILE_SIZE) {
        for (int dx = tx; dx < sharedWidth; dx += TILE_SIZE) {
            // Calculate clamped global coordinates
            int gx = tileStartX + dx;
            int gy = tileStartY + dy;
            gx = min(max(gx, 0), width - 1);
            gy = min(max(gy, 0), height - 1);

            // Copy all channels
            for (int c = 0; c < channels; c++) {
                sharedMem[(dy * sharedWidth + dx) * channels + c] =
                    input[(gy * pitch) + (gx * channels) + c];
            }
        }
    }

    __syncthreads();

    // If outside image bounds, bail out
    if (x >= width || y >= height) return;

    // Apply Sobel operator to each channel
    for (int c = 0; c < channels; c++) {
        // Indices in shared memory
        int sx = tx + radius;
        int sy = ty + radius;

        // Apply Sobel X filter
        float gx =
            -1.0f * sharedMem[((sy-1)*sharedWidth + (sx-1))*channels + c] +
             0.0f * sharedMem[((sy-1)*sharedWidth + (sx  ))*channels + c] +
             1.0f * sharedMem[((sy-1)*sharedWidth + (sx+1))*channels + c] +
            -2.0f * sharedMem[((sy  )*sharedWidth + (sx-1))*channels + c] +
             0.0f * sharedMem[((sy  )*sharedWidth + (sx  ))*channels + c] +
             2.0f * sharedMem[((sy  )*sharedWidth + (sx+1))*channels + c] +
            -1.0f * sharedMem[((sy+1)*sharedWidth + (sx-1))*channels + c] +
             0.0f * sharedMem[((sy+1)*sharedWidth + (sx  ))*channels + c] +
             1.0f * sharedMem[((sy+1)*sharedWidth + (sx+1))*channels + c];

        // Apply Sobel Y filter
        float gy =
            -1.0f * sharedMem[((sy-1)*sharedWidth + (sx-1))*channels + c] +
            -2.0f * sharedMem[((sy-1)*sharedWidth + (sx  ))*channels + c] +
            -1.0f * sharedMem[((sy-1)*sharedWidth + (sx+1))*channels + c] +
             0.0f * sharedMem[((sy  )*sharedWidth + (sx-1))*channels + c] +
             0.0f * sharedMem[((sy  )*sharedWidth + (sx  ))*channels + c] +
             0.0f * sharedMem[((sy  )*sharedWidth + (sx+1))*channels + c] +
             1.0f * sharedMem[((sy+1)*sharedWidth + (sx-1))*channels + c] +
             2.0f * sharedMem[((sy+1)*sharedWidth + (sx  ))*channels + c] +
             1.0f * sharedMem[((sy+1)*sharedWidth + (sx+1))*channels + c];

        // Calculate magnitude
        float mag = sqrtf(gx*gx + gy*gy);
        mag = min(mag, 255.0f);

        // Write result
        output[(y * pitch) + (x * channels) + c] = static_cast<unsigned char>(mag);
    }
}

// Kernel for Kuwahara filter
__global__ void sharedKuwahara(const unsigned char* input, unsigned char* output, int width, int height, int pitch) {
    const int radius = 2;
    const int window_size = radius * 2 + 1; // gives the 5x5 filter that is normally used by the kuwahara filter

    __shared__ unsigned char ds_input[(TILE_SIZE + 4) * (TILE_SIZE + 4) * 3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_width = TILE_SIZE + 2 * radius;
    int shared_x = tx + radius;
    int shared_y = ty + radius;
    const int channels = 3;

    // Load image into shared memory with padding
    for (int c = 0; c < channels; c++) {
        int global_x = min(max(x, 0), width - 1);
        int global_y = min(max(y, 0), height - 1);
        ds_input[(shared_y * shared_width + shared_x) * channels + c] =
            input[(global_y * pitch) + (global_x * channels) + c];

        // Handle borders (expand edges)
        if (tx < radius) {
            int left_x = max(x - radius, 0);
            ds_input[(shared_y * shared_width + (shared_x - radius)) * channels + c] =
                input[(global_y * pitch) + (left_x * channels) + c];

            int right_x = min(x + TILE_SIZE, width - 1);
            ds_input[(shared_y * shared_width + (shared_x + TILE_SIZE)) * channels + c] =
                input[(global_y * pitch) + (right_x * channels) + c];
        }

        if (ty < radius) {
            int top_y = max(y - radius, 0);
            ds_input[((shared_y - radius) * shared_width + shared_x) * channels + c] =
                input[(top_y * pitch) + (global_x * channels) + c];

            int bottom_y = min(y + TILE_SIZE, height - 1);
            ds_input[((shared_y + TILE_SIZE) * shared_width + shared_x) * channels + c] =
                input[(bottom_y * pitch) + (global_x * channels) + c];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float min_var = 1e20f;
    float final_color[3] = {0.0f, 0.0f, 0.0f};

    // Kuwahara filter: divide the window into 4 overlapping regions
    for (int region = 0; region < 4; region++) {
        int x_offset = (region % 2 == 0) ? 0 : radius;
        int y_offset = (region / 2 == 0) ? 0 : radius;

        float mean[3] = {0.0f, 0.0f, 0.0f};
        float mean_sq[3] = {0.0f, 0.0f, 0.0f};
        int count = 0;

        for (int dy = 0; dy <= radius; dy++) {
            for (int dx = 0; dx <= radius; dx++) {
                int sx = shared_x + x_offset + dx - radius;
                int sy = shared_y + y_offset + dy - radius;
                if (sx >= 0 && sx < shared_width && sy >= 0 && sy < shared_width) {
                    for (int c = 0; c < channels; c++) {
                        float val = ds_input[(sy * shared_width + sx) * channels + c];
                        mean[c] += val;
                        mean_sq[c] += val * val;
                    }
                    count++;
                }
            }
        }

        float var_sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            mean[c] /= count;
            mean_sq[c] /= count;
            var_sum += (mean_sq[c] - mean[c] * mean[c]);
        }

        if (var_sum < min_var) {
            min_var = var_sum;
            for (int c = 0; c < channels; c++) {
                final_color[c] = mean[c];
            }
        }
    }

    // Write the pixel
    for (int c = 0; c < channels; c++) {
        output[(y * pitch) + (x * channels) + c] = (unsigned char)final_color[c];
    }
}

// Function to apply Kuwahara filter
void applyKuwaharaFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + TILE_SIZE - 1) / TILE_SIZE,
        (input.rows + TILE_SIZE - 1) / TILE_SIZE
        );

    // Calculate shared memory size
    const int radius = 2;
    int sharedWidth = TILE_SIZE + 2 * radius;
    int sharedHeight = TILE_SIZE + 2 * radius;
    size_t sharedMemSize = sharedWidth * sharedHeight * NUM_CHANNELS * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    sharedKuwahara<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
        );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kuwahara kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Kuwahara: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply Gaussian Blur filter
void applyGaussianBlurFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
        );

    float sigma = 2.0f;

    // Compute Gaussian kernel on CPU
    float* h_gaussian_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    computeGaussianKernel(h_gaussian_kernel, sigma);

    float kernelSum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        kernelSum += h_gaussian_kernel[i];
    }
    
    // Copy kernel to GPU constant memory
    cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    delete[] h_gaussian_kernel;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Calculate shared memory size
    int radius = KERNEL_SIZE / 2;
    int sharedWidth = BLOCK_SIZE_X + 2 * radius;
    int sharedHeight = BLOCK_SIZE_Y + 2 * radius;
    int sharedMemSize = sharedWidth * sharedHeight * 3 * sizeof(unsigned char);

    // Launch the improved kernel
    gaussianBlurSharedMem<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
        );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply grayscale filter
void applyGrayscaleFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    grayscaleKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Grayscale kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Grayscale: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply threshold filter
void applyThresholdFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, int threshold, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    thresholdKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step),
        threshold
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Threshold kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Threshold: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply sepia filter
void applySepiaFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    sepiaKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sepia kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Sepia: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply rosey filter
void applyRoseyFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    roseyKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Rosey kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Rosey: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply invert filter
void applyInvertFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    invertKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Invert kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Invert: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply grayscale invert filter
void applyGrayscaleInvertFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + block.x - 1) / block.x,
        (input.rows + block.y - 1) / block.y
    );

    // Size of shared memory
    size_t sharedMemSize = TILE_SIZE * TILE_SIZE * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    grayscaleInvertKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Grayscale Invert kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Grayscale Invert: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply pixelate filter
void applyPixelateFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(PIXEL_SIZE, PIXEL_SIZE);
    const dim3 grid(
        (input.cols + PIXEL_SIZE - 1) / PIXEL_SIZE,
        (input.rows + PIXEL_SIZE - 1) / PIXEL_SIZE
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    pixelateKernel<<<grid, block, 0, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pixelate kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Pixelate: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to apply edge detect filter
void applyEdgeDetectFilter(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cudaStream_t stream)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        (input.cols + TILE_SIZE - 1) / TILE_SIZE,
        (input.rows + TILE_SIZE - 1) / TILE_SIZE
    );

    // Calculate shared memory size
    int radius = 1; // For 3x3 Sobel operator
    int sharedWidth = TILE_SIZE + 2*radius;
    int sharedHeight = TILE_SIZE + 2*radius;
    size_t sharedMemSize = sharedWidth * sharedHeight * NUM_CHANNELS * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Launch kernel
    edgeDetectKernel<<<grid, block, sharedMemSize, stream>>>(
        input.data,
        output.data,
        input.cols,
        input.rows,
        static_cast<int>(input.step)
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Edge Detect kernel execution time: %.3f ms\n", milliseconds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in Edge Detect: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Batch processing function for multiple frames
bool processBatch(std::vector<cv::Mat>& inputFrames, std::vector<cv::Mat>& outputFrames, std::vector<cv::cuda::GpuMat>& d_inputs,
                  std::vector<cv::cuda::GpuMat>& d_outputs, std::vector<cudaStream_t>& streams, std::vector<cv::cuda::Stream>& cv_streams, FilterType filterType)
{

    int batchSize = static_cast<int>(inputFrames.size());
    if (batchSize == 0) return true;

    // Resize output vector to match input
    outputFrames.resize(batchSize);

    // Make sure GPU buffers are properly sized
    if (d_inputs.size() < static_cast<size_t>(batchSize)) d_inputs.resize(batchSize);
    if (d_outputs.size() < static_cast<size_t>(batchSize)) d_outputs.resize(batchSize);

    // Upload all frames asynchronously
    for (int i = 0; i < batchSize; i++) {
        if (i < static_cast<int>(streams.size()) && i < static_cast<int>(cv_streams.size())) {
            d_inputs[i].upload(inputFrames[i], cv_streams[i]);
        }
    }

    // Process all frames asynchronously
    for (int i = 0; i < batchSize; i++) {
        if (i < static_cast<int>(streams.size())) {
            switch (filterType) {
            case FilterType::GAUSSIAN_BLUR:
                applyGaussianBlurFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::GRAYSCALE:
                applyGrayscaleFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::THRESHOLD:
                applyThresholdFilter(d_inputs[i], d_outputs[i], 128, streams[i]); // Default threshold value of 128
                break;
            case FilterType::SEPIA:
                applySepiaFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::ROSEY:
                applyRoseyFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::INVERT:
                applyInvertFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::GRAYSCALE_INVERT:
                applyGrayscaleInvertFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::PIXELATE:
                applyPixelateFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::EDGE_DETECT:
                applyEdgeDetectFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            case FilterType::KUWAHARA:
                applyKuwaharaFilter(d_inputs[i], d_outputs[i], streams[i]);
                break;
            }
        }
    }

    // Download all results asynchronously
    for (int i = 0; i < batchSize; i++) {
        if (i < static_cast<int>(streams.size()) && i < static_cast<int>(cv_streams.size())) {
            outputFrames[i] = cv::Mat(inputFrames[i].rows, inputFrames[i].cols, inputFrames[i].type());
            d_outputs[i].download(outputFrames[i], cv_streams[i]);
        }
    }

    // Synchronize all streams
    for (size_t i = 0; i < streams.size() && i < static_cast<size_t>(batchSize); i++) {
        cudaStreamSynchronize(streams[i]);
    }

    return true;
}

// Main function to process a video file with CUDA 
bool processVideoWithCUDA(const std::string& inputPath, const std::string& outputPath, FilterType filterType,  
    std::function<void(int, int)> progressCallback)
{
    try {
        // Open video capture
        cv::VideoCapture cap(inputPath.c_str());
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video: " << inputPath << std::endl;
            return false;
        }

        // Get video properties
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // H.264 codec
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Create video writer
        cv::VideoWriter writer(outputPath.c_str(), fourcc, fps, cv::Size(width, height), true);
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create output video: " << outputPath << std::endl;
            return false;
        }

        // Create CUDA streams for parallelization
        std::vector<cudaStream_t> streams(NUM_STREAMS);
        std::vector<cv::cuda::Stream> cv_streams(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamCreate(&streams[i]);
            cv_streams[i] = cv::cuda::Stream();
        }

        // Pre-allocate GPU memory for frames
        std::vector<cv::cuda::GpuMat> d_inputs(NUM_STREAMS), d_outputs(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; ++i) {
            d_inputs[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            d_outputs[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
        }

      
        cudaEvent_t gpuStart, gpuStop;
        cudaEventCreate(&gpuStart);
        cudaEventCreate(&gpuStop);
        cudaEventRecord(gpuStart);

        // Define batch size for processing
        const int BATCH_SIZE = NUM_STREAMS;
        std::vector<cv::Mat> inputBatch, outputBatch;
        cv::Mat frame;
        int frameCount = 0;

        while (cap.read(frame)) {
            // Ensure frame is BGR 8-bit 3-channel
            cv::Mat bgrFrame;
            if (frame.channels() == 1) {
                cv::cvtColor(frame, bgrFrame, cv::COLOR_GRAY2BGR);
            } else if (frame.channels() == 4) {
                cv::cvtColor(frame, bgrFrame, cv::COLOR_BGRA2BGR);
            } else {
                bgrFrame = frame.clone();
            }

            if (bgrFrame.type() != CV_8UC3) {
                bgrFrame.convertTo(bgrFrame, CV_8UC3);
            }

            inputBatch.push_back(bgrFrame);

            if (static_cast<int>(inputBatch.size()) == BATCH_SIZE || frameCount == totalFrames - 1) {
                processBatch(inputBatch, outputBatch, d_inputs, d_outputs, streams, cv_streams, filterType);

                // Write output frames to video
                for (size_t i = 0; i < outputBatch.size(); i++) {
                    writer.write(outputBatch[i]);
                }

                inputBatch.clear();
                outputBatch.clear();

                if (progressCallback) {
                    progressCallback(frameCount, totalFrames);
                }
            }
            frameCount++;
        }

        // Process any remaining frames
        if (!inputBatch.empty()) {
            processBatch(inputBatch, outputBatch, d_inputs, d_outputs, streams, cv_streams, filterType);
            for (size_t i = 0; i < outputBatch.size(); i++) {
                writer.write(outputBatch[i]);
            }
        }

        // Synchronize all streams
        for (size_t i = 0; i < streams.size(); ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        cudaEventRecord(gpuStop);
        cudaEventSynchronize(gpuStop);
        float totalMs = 0.0f;
        cudaEventElapsedTime(&totalMs, gpuStart, gpuStop);
        std::cout << "Total GPU processing time: " << totalMs << " ms for "
                  << frameCount << " frames (" << (totalMs / frameCount)
                  << " ms per frame)" << std::endl;

        // Cleanup
        for (size_t i = 0; i < streams.size(); i++) {
            cudaStreamDestroy(streams[i]);
        }

        writer.release();
        cap.release();
        cudaEventDestroy(gpuStart);
        cudaEventDestroy(gpuStop);

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Error processing video: " << e.what() << std::endl;
        return false;
    }
}
