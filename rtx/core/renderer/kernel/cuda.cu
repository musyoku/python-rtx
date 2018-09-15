#include "../../header/enum.h"
#include "../header/cuda.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

void rtx_cuda_malloc(void** gpu_array, size_t size)
{
    assert(size > 0);
    cudaError_t error = cudaMalloc(gpu_array, size);
    // printf("malloc %p\n", *gpu_array);
    cudaError_t status = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMalloc: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_memcpy_host_to_device(void* gpu_array, void* cpu_array, size_t size)
{
    cudaError_t error = cudaMemcpy(gpu_array, cpu_array, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMemcpyHostToDevice: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_memcpy_device_to_host(void* cpu_array, void* gpu_array, size_t size)
{
    cudaError_t error = cudaMemcpy(cpu_array, gpu_array, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_free(void** array)
{
    if (*array != NULL) {
        // printf("free %p\n", *array);
        cudaError_t error = cudaFree(*array);
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error at cudaFree: %s\n", cudaGetErrorString(error));
        }
        *array = NULL;
    }
}
void rtx_cuda_device_reset()
{
    cudaDeviceReset();
}