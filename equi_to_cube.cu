#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

#define PI 3.14159265358979323846

using namespace std;

// Helper math functions
__device__ void faceDirection(int faceIdx, float u, float v, float3& dir) {
    float x = 2.0f * u - 1.0f;
    float y = 2.0f * v - 1.0f;
    switch (faceIdx) {
        case 0: dir = make_float3(1, -y, -x); break;    // +X (right)
        case 1: dir = make_float3(-1, -y, x); break;    // -X (left)
        case 2: dir = make_float3(x, 1, y); break;      // +Y (top)
        case 3: dir = make_float3(x, -1, -y); break;    // -Y (bottom)
        case 4: dir = make_float3(x, -y, 1); break;     // +Z (front)
        case 5: dir = make_float3(-x, -y, -1); break;   // -Z (back)
    }
    float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;
}

__device__ void dirToEquirect(float3 dir, int width, int height, int& u, int& v) {
    float theta = atan2f(dir.x, -dir.z);
    float phi = asinf(dir.y);
    float uf = (theta + PI) / (2 * PI);
    float vf = (phi + PI / 2) / PI;
    u = int(uf * width);
    v = int(vf * height);
    u = min(max(u, 0), width - 1);
    v = min(max(v, 0), height - 1);
}

__device__ void rotateCoords(int& x, int& y, int faceSize, int rotation) {
    int tx = x, ty = y;
    switch (rotation) {
        case 90:
            x = ty;
            y = faceSize - 1 - tx;
            break;
        case 180:
            x = faceSize - 1 - tx;
            y = faceSize - 1 - ty;
            break;
        case 270:
            x = faceSize - 1 - ty;
            y = tx;
            break;
        default:
            break;
    }
}

__global__ void EquirectToCubeKernel(uchar3* input, uchar3* output, int eqWidth, int eqHeight, int faceSize) {
    int face = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= faceSize || y >= faceSize || face >= 6) return;

    float u = (x + 0.5f) / faceSize;
    float v = (y + 0.5f) / faceSize;
    float3 dir;
    faceDirection(face, u, v, dir);

    int uE, vE;
    dirToEquirect(dir, eqWidth, eqHeight, uE, vE);
    uchar3 color = input[vE * eqWidth + uE];

    int outX = 0, outY = 0, rot = 0;
    switch (face) {
        case 0: outX = 2 * faceSize; outY = faceSize; rot = 270; break; // +X
        case 1: outX = 0;            outY = faceSize; rot = 90;  break; // -X
        case 2: outX = faceSize;     outY = 0;         rot = 180; break; // +Y (top)
        case 3: outX = faceSize;     outY = 2 * faceSize; rot = 0; break; // -Y (bottom)
        case 4: outX = faceSize;     outY = faceSize;   rot = 0;  break; // +Z
        case 5: outX = 3 * faceSize; outY = faceSize;   rot = 180; break; // -Z
    }

    int tx = x, ty = y;
    rotateCoords(tx, ty, faceSize, rot);
    int outIdx = (outY + ty) * faceSize * 4 + (outX + tx);
    output[outIdx] = color;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./equirect_to_cube <equirectangular_image>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    int eqWidth = img.cols;
    int eqHeight = img.rows;
    int faceSize = eqWidth / 4;

    size_t inputSize = eqWidth * eqHeight * sizeof(uchar3);
    size_t outputSize = (faceSize * 4) * (faceSize * 3) * sizeof(uchar3);

    uchar3* d_input;
    uchar3* d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    cudaMemcpy(d_input, img.ptr<uchar3>(), inputSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((faceSize + 15) / 16, (faceSize + 15) / 16, 6);
    EquirectToCubeKernel<<<grid, block>>>(d_input, d_output, eqWidth, eqHeight, faceSize);
    cudaDeviceSynchronize();

    cv::Mat result(faceSize * 3, faceSize * 4, CV_8UC3);
    cudaMemcpy(result.ptr<uchar3>(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cv::imwrite("cube_map_output.jpg", result);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Saved: cube_map_output.jpg" << std::endl;
    return 0;
}
