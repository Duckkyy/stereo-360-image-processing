#include <cmath>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__device__ float3 direction_from_face_coords(int face, float u, float v) {
    float3 dir;
    float x = 2.0f * u - 1.0f;
    float y = 2.0f * v - 1.0f;

    switch (face) {
        case 0: dir = make_float3(1.0f, -y, -x); break;  // +X
        case 1: dir = make_float3(-1.0f, -y, x); break;  // -X
        case 2: dir = make_float3(x, 1.0f, y); break;    // +Y
        case 3: dir = make_float3(x, -1.0f, -y); break;  // -Y
        case 4: dir = make_float3(x, -y, 1.0f); break;   // +Z
        case 5: dir = make_float3(-x, -y, -1.0f); break; // -Z
    }

    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;
    return dir;
}

__device__ void dir_to_uv(float3 dir, float &u, float &v) {
    float theta = atan2f(dir.y, dir.x);
    float phi = acosf(dir.z);
    u = (theta + M_PI) / (2.0f * M_PI);
    v = phi / M_PI;
}

__global__ void equirect_to_cube_kernel(
    const unsigned char* input,
    int inWidth, int inHeight,
    unsigned char* output,
    int faceSize,
    int face
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= faceSize || y >= faceSize) return;

    float u = (x + 0.5f) / faceSize;
    float v = (y + 0.5f) / faceSize;

    float3 dir = direction_from_face_coords(face, u, v);

    float uf, vf;
    dir_to_uv(dir, uf, vf);
    int px = min((int)(uf * inWidth), inWidth - 1);
    int py = min((int)(vf * inHeight), inHeight - 1);

    int input_idx = (py * inWidth + px) * 3;
    int output_idx = (y * faceSize + x) * 3;

    output[output_idx + 0] = input[input_idx + 0];
    output[output_idx + 1] = input[input_idx + 1];
    output[output_idx + 2] = input[input_idx + 2];
}

void convert_face(const unsigned char* inputImage, int inW, int inH, int faceSize, int face, const std::string& filename) {
    unsigned char* d_input, *d_output, *h_output;
    size_t inSize = inW * inH * 3;
    size_t faceBytes = faceSize * faceSize * 3;

    cudaMalloc(&d_input, inSize);
    cudaMemcpy(d_input, inputImage, inSize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, faceBytes);
    h_output = new unsigned char[faceBytes];

    dim3 block(16, 16);
    dim3 grid((faceSize + 15) / 16, (faceSize + 15) / 16);
    equirect_to_cube_kernel<<<grid, block>>>(d_input, inW, inH, d_output, faceSize, face);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, faceBytes, cudaMemcpyDeviceToHost);

    stbi_write_png(filename.c_str(), faceSize, faceSize, 3, h_output, faceSize * 3);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input_image>\n", argv[0]);
        return -1;
    }

    int width, height, channels;
    unsigned char* inputImage = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!inputImage) {
        fprintf(stderr, "Failed to load image\n");
        return -1;
    }

    int faceSize = width / 4;

    const char* face_names[6] = { "XPOS", "XNEG", "YPOS", "YNEG", "ZPOS", "ZNEG" };
    for (int i = 0; i < 6; ++i) {
        std::string fname = "CUBE_" + std::string(face_names[i]) + ".png";
        convert_face(inputImage, width, height, faceSize, i, fname);
    }

    stbi_image_free(inputImage);
    return 0;
}
