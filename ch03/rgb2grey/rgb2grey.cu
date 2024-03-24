#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../utils.h"

__global__ void rgb2grey_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* grey, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x < width && y < height) {
        grey[idx] = (unsigned char)(0.299 * red[idx] + 0.587 * green[idx] + 0.114 * blue[idx]);
    }
}

void rgb2grey_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* grey, int width, int height) {
    unsigned char *red_d, *green_d, *blue_d, *grey_d;
    size_t pic_size = width * height * sizeof(unsigned char);
    cudaDeviceSynchronize();

    HOST_TIC(0);
    cudaMalloc(&red_d, pic_size);
    cudaMalloc(&green_d, pic_size);
    cudaMalloc(&blue_d, pic_size);
    cudaMalloc(&grey_d, pic_size);
    cudaMemcpy(red_d, red, pic_size, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, pic_size, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, pic_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    HOST_TOC(0);

    HOST_TIC(1);
    // call kernel
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    rgb2grey_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, grey_d, width, height);

    cudaDeviceSynchronize();
    HOST_TOC(1);

    // Copy data from gpu
    HOST_TIC(2);
    cudaMemcpy(grey, grey_d, pic_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    HOST_TOC(2);
    // Free memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(grey_d);
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
    int length = img.rows * img.cols;
    *input = (uchar *)malloc(length * 3 * sizeof(uchar));
    *output = (uchar *)malloc(length * sizeof(uchar));

    // copy image data to input
    memcpy(*input, img.data, length * 3 * sizeof(uchar));
    *width = img.cols;
    *height = img.rows;
    img.release();
}

void split_rgb(uchar *input, uchar *red, uchar *green, uchar *blue, int width, int height) {
    int length = width * height;
    for (int i = 0; i < length; i++) {
        red[i] = input[3 * i];
        green[i] = input[3 * i + 1];
        blue[i] = input[3 * i + 2];
    }
}

int main() {
    const char* filename = "rgb2grey_tree.png";
    uchar *input = NULL;
    uchar *output = NULL;
    int width = 0;
    int height = 0;
    read_img(filename, &input, &output, &width, &height);

    uchar *red, *green, *blue;
    red = (uchar *)malloc(width * height * sizeof(uchar));
    green = (uchar *)malloc(width * height * sizeof(uchar));
    blue = (uchar *)malloc(width * height * sizeof(uchar));

    split_rgb(input, red, green, blue, width, height);

    rgb2grey_gpu(red, green, blue, output, width, height);

    // free input/red/green/blue memory
    free(input);
    free(red);
    free(green);
    free(blue);

    // save output image
    cv::Mat dst(height, width, CV_8UC1, output);
    cv::imwrite("rgb2grey_tree_grey.png", dst);

    return;
}