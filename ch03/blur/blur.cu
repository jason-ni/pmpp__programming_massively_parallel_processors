#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common.h"
#include "../../utils.h"

__global__ void blur_kernel(unsigned char* input_d, unsigned char* blur_d, int width, int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (outCol < width && outRow < height) {
        int pix_red_val = 0;
        int pix_green_val = 0;
        int pix_blue_val = 0;
        int idx = 0;
        for(int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow) {
            for(int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol) {
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    idx = (inRow * width + inCol)*3;
                    pix_red_val += input_d[idx];
                    pix_green_val += input_d[idx + 1];
                    pix_blue_val += input_d[idx + 2];
                }
            }
        }
        idx = (outRow * width + outCol)*3;
        int pix_cnt = (2*BLUR_SIZE + 1) * (2*BLUR_SIZE + 1);
        blur_d[idx] = (uchar)(pix_red_val / pix_cnt);
        blur_d[idx + 1] = (uchar)(pix_green_val / pix_cnt);
        blur_d[idx + 2] = (uchar)(pix_blue_val / pix_cnt);
    }
}

void blur_gpu(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char *input_d, *blur_d;
    size_t pic_size = width * height * sizeof(unsigned char) * 3;
    cudaDeviceSynchronize();

    HOST_TIC(0);
    cudaMalloc(&input_d, pic_size);
    cudaMalloc(&blur_d, pic_size);
    cudaMemcpy(input_d, input, pic_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    HOST_TOC(0);

    HOST_TIC(1);
    // call kernel
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, blur_d, width, height);

    cudaDeviceSynchronize();
    HOST_TOC(1);

    // Copy data from gpu
    HOST_TIC(2);
    cudaMemcpy(output, blur_d, pic_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    HOST_TOC(2);
    // Free memory
    cudaFree(input_d);
    cudaFree(blur_d);
}

void read_img(
    const char* filename, 
    uchar **input, 
    uchar **output, 
    int *width, 
    int *height) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Error: Could not read image %s\n", filename);
        exit(1);
    }
    printf("img: (%d X %d)\n", img.cols, img.rows);
    int length = img.rows * img.cols * 3;
    *input = (uchar *)malloc(length);
    *output = (uchar *)malloc(length);

    // copy image data to input
    memcpy(*input, img.data, length);
    *width = img.cols;
    *height = img.rows;
    img.release();
}

int main() {
    const char* filename = "cat.png";
    uchar *input = NULL;
    uchar *output = NULL;
    int width = 0;
    int height = 0;
    read_img(filename, &input, &output, &width, &height);


    blur_gpu(input, output, width, height);

    // free input/red/green/blue memory
    free(input);

    // save output image
    cv::Mat dst(height, width, CV_8UC3, output);
    cv::imwrite("cat_blur.png", dst);

    return;
}