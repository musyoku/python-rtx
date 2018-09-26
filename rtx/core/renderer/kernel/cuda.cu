#include "../../header/enum.h"
#include "../header/bridge.h"
#include "../header/cuda_common.h"
#include "../header/cuda_texture.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

cudaTextureObject_t* g_gpu_serialized_mapping_texture_object_array;
cudaTextureObject_t g_cpu_serialized_mapping_texture_object_array[RTX_CUDA_MAX_TEXTURE_UNITS];
cudaArray* g_gpu_serialized_mapping_texture_cudaArray_ptr_array[RTX_CUDA_MAX_TEXTURE_UNITS];

void rtx_cuda_malloc(void** gpu_array, size_t size)
{
    assert(size > 0);
    cudaCheckError(cudaMalloc(gpu_array, size));
    cudaCheckError(cudaGetLastError());
}
void rtx_cuda_malloc_pointer(void**& gpu_array, size_t size)
{
    assert(size > 0);
    cudaCheckError(cudaMalloc(&gpu_array, size));
    cudaCheckError(cudaGetLastError());
}
void rtx_cuda_memcpy_host_to_device(void* gpu_array, void* cpu_array, size_t size)
{
    cudaCheckError(cudaMemcpy(gpu_array, cpu_array, size, cudaMemcpyHostToDevice));
}
void rtx_cuda_memcpy_device_to_host(void* cpu_array, void* gpu_array, size_t size)
{
    cudaCheckError(cudaMemcpy(cpu_array, gpu_array, size, cudaMemcpyDeviceToHost));
}
void rtx_cuda_free(void** array_ref)
{
    if (*array_ref != NULL) {
        cudaCheckError(cudaFree(*array_ref));
        *array_ref = NULL;
    }
}
void rtx_cuda_device_reset()
{
    cudaDeviceReset();
}
void rtx_cuda_malloc_texture(int unit_index, int width, int height)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaArray** array = &g_gpu_serialized_mapping_texture_cudaArray_ptr_array[unit_index];
    cudaCheckError(cudaMallocArray(array, &desc, width, height));
    cudaCheckError(cudaMalloc((void**)&g_gpu_serialized_mapping_texture_object_array, sizeof(cudaTextureObject_t) * RTX_CUDA_MAX_TEXTURE_UNITS));
}
void rtx_cuda_memcpy_to_texture(int unit_index, int width_offset, int height_offset, void* data, size_t bytes)
{
    cudaArray* array = g_gpu_serialized_mapping_texture_cudaArray_ptr_array[unit_index];
    cudaCheckError(cudaMemcpyToArray(array, 0, 0, data, bytes, cudaMemcpyHostToDevice));
}
void rtx_cuda_bind_texture(int unit_index)
{
    cudaArray* array = g_gpu_serialized_mapping_texture_cudaArray_ptr_array[unit_index];
    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(cudaResourceDesc));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));
    tex.normalizedCoords = true;
    tex.readMode = cudaReadModeElementType;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    cudaCheckError(cudaCreateTextureObject(&g_cpu_serialized_mapping_texture_object_array[unit_index], &resource, &tex, NULL));
    cudaCheckError(cudaMemcpy(g_gpu_serialized_mapping_texture_object_array, g_cpu_serialized_mapping_texture_object_array, sizeof(cudaTextureObject_t) * RTX_CUDA_MAX_TEXTURE_UNITS, cudaMemcpyHostToDevice));
}
void rtx_cuda_free_texture(int unit_index)
{
    cudaArray* array = g_gpu_serialized_mapping_texture_cudaArray_ptr_array[unit_index];
    cudaCheckError(cudaFreeArray(array));
    array = NULL;
}
size_t rtx_cuda_get_available_shared_memory_bytes()
{
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    return dev.sharedMemPerBlock;
}
size_t rtx_cuda_get_cudaTextureObject_t_bytes(){
    return sizeof(cudaTextureObject_t);
}