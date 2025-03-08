// OpenCV Test :)
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

#define KERNEL_SIZE 5
#define NUM_CHANNELS 3
#define TILE_SIZE 16

using namespace cv;

void generateGaussianKernel(float* kernel, int size, float sigma) { // probably doesn't have to be parallel!

    float sum = 0.0f;
    int center = size / 2;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            kernel[y * size + x] = exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma)); // kernel values being set here
            sum += kernel[y * size + x]; //total sum of kernel values
        }
    }
    // Normalize kernel - basically averaging
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

__global__ void basicGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel, int kernelSize) {

   int radius = kernelSize / 2;
   
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int tc = threadIdx.z;
   
   int y = blockIdx.y * blockDim.y + ty;
   int x = blockIdx.x * blockDim.x + tx;
   int c = blockIdx.z * blockDim.z + tc;
   
   x = (x < 0) ? 0 : ((x >= width) ? width - 1 : x);
   y = (y < 0) ? 0 : ((y >= height) ? height - 1 : y);
   
   float sum;

   if((x < width) && (y < height) && (c < channels)){
      
        sum = 0.0f;
   
        for (int ky = -radius; ky <= radius; ky++) {
           for (int kx = -radius; kx <= radius; kx++) {

               int px = x + kx;
               int py = y + ky;
               
               //Check bounds: 
               px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
               py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
               
               int pixel_idx = (py * width + px) * channels + c;
               float kernel_val = kernel[(ky + radius) * kernelSize + (kx + radius)];
               
               sum += input[pixel_idx] * kernel_val;

           }
       }
       
       output[(y * width + x) * channels + c] = (unsigned char)sum;
   }
}

int main(int argc, char** argv) {
    
        //Check valid input
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
        
    int width, height, channels;
        //unsigned char* input = stbi_load(argv[1], &width, &height, &channels, 0);
    Mat input;
    input = imread(argv[1], IMREAD_COLOR); //this is getting right img
    channels = input.channels();
    width = input.cols;
    height = input.rows;

        //Check valid image
    if (input.empty()) {
        printf("Failed to load image: %s\n", argv[1]);
        return 1;
    }

    unsigned char* comp_input = (unsigned char*)malloc(width * height * channels);
    //unsigned char* comp_input;
    for (int i = 0; i < (width * height * channels); i++) {
        comp_input[i] = input.data[i];
    }

    size_t total_size = sizeof(unsigned char) * (width * height * channels);
    unsigned char* output = (unsigned char*)malloc(width * height * channels);
  
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    
    //     // Generate and apply Gaussian kernel
    
        const int kernelSize = 5;
        const float sigma = 1.0f;
        float* kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    
        generateGaussianKernel(kernel, kernelSize, sigma);
        
        //////Gaussian Blur//////////////////////////////////
        
        unsigned char* d_input;
        unsigned char* d_output;
        float* d_kernel;
        
        cudaMalloc((void**)&d_input, total_size);
        cudaMalloc((void**)&d_output, total_size);
        cudaMalloc((void**)&d_kernel, (kernelSize * kernelSize * sizeof(float)));
    
        cudaMemcpy(d_input, comp_input, total_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, output, total_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, (kernelSize * kernelSize * sizeof(float)), cudaMemcpyHostToDevice);
    
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, channels);
        dim3 blocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
       
        cudaEventRecord(event_start, 0);
     
        /*BASIC IMPLEMENTATION*/
        basicGaussianBlur<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels, d_kernel, kernelSize);

        cudaDeviceSynchronize();
        cudaCheckError();
        
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        
        float mlsec;
        cudaEventElapsedTime(&mlsec, event_start, event_stop);
        std::cout << "Execution Time " << mlsec << std::endl;
        
    cudaMemcpy(comp_input, d_input, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, total_size, cudaMemcpyDeviceToHost);
    

    Mat outputimg = Mat(height, width, CV_8UC3, output);

    imshow("Lenna Blurred", outputimg); 
    waitKey(0);
    destroyAllWindows();
        
    // cudaFree(d_input);
    // cudaFree(d_output);
    
    //i have memory leak with comp_input!
    free(output);
    free(kernel);

    return 0;
    
}
    



