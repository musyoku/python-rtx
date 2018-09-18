#include "../../header/enum.h"
#include "../header/bridge.h"
#include "../header/cuda_common.h"
#include "../header/cuda_texture.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

cudaTextureObject_t* texture_object_pointer;
cudaTextureObject_t texture_object_array[30];
cudaArray* texture_cuda_array[30];

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
void rtx_cuda_malloc_pointer(void**& gpu_array, size_t size)
{
    printf("cudaMalloc] %p\n", &gpu_array);
    assert(size > 0);
    cudaError_t error = cudaMalloc(&gpu_array, size);
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
void rtx_cuda_malloc_texture(int unit_index, int width, int height)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

    cudaArray*& array = texture_cuda_array[unit_index];
    cudaError_t error = cudaMallocArray(&array, &desc, width, height);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMallocArray: %s\n", cudaGetErrorString(error));
    }
    cudaMalloc((void**)&texture_object_pointer, sizeof(cudaTextureObject_t*) * 30);
}
void rtx_cuda_memcpy_to_texture(int unit_index, int width_offset, int height_offset, void* data, size_t bytes)
{
    cudaArray* array = texture_cuda_array[unit_index];
    // cudaError_t error = cudaMemcpy2D(array, sizeof(float), data, sizeof(float), width_offset, height_offset, cudaMemcpyHostToDevice);
    cudaError_t error = cudaMemcpyToArray(array, 0, 0, data, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMemcpyToArray: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_bind_texture(int unit_index)
{
    cudaArray* array = texture_cuda_array[unit_index];

    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(cudaResourceDesc));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc desc;
    memset(&desc, 0, sizeof(cudaTextureDesc));
    desc.normalizedCoords = true;
    desc.readMode = cudaReadModeElementType;
    desc.filterMode = cudaFilterModePoint;
    desc.addressMode[0] = cudaAddressModeWrap;
    desc.addressMode[1] = cudaAddressModeWrap;
    cudaError_t error = cudaCreateTextureObject(&texture_object_array[unit_index], &resource, &desc, NULL);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaCreateTextureObject: %s\n", cudaGetErrorString(error));
    }
    printf("%p\n", texture_object_array);
    cudaMemcpy(texture_object_pointer, &texture_object_array[unit_index], sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
}
void rtx_cuda_free_texture(int unit_index)
{
    cudaArray* array = texture_cuda_array[unit_index];
    cudaFreeArray(array);
    array = NULL;
}

size_t rtx_cuda_get_available_shared_memory_bytes()
{
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    return dev.sharedMemPerBlock;
}