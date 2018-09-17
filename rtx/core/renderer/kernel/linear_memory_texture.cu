#include "../../header/enum.h"
#include "../header/bridge.h"
#include "../header/cuda_linear_memory_texture.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

cudaTextureObject_t* g_serial_ray_array_texture_object_cpu_ptr = NULL;
cudaTextureObject_t* g_serial_face_vertex_index_array_texture_object_cpu_ptr = NULL;
cudaTextureObject_t* g_serial_vertex_array_texture_object_cpu_ptr = NULL;
cudaTextureObject_t* g_serial_threaded_bvh_array_texture_object_cpu_ptr = NULL;
cudaTextureObject_t* g_serial_threaded_bvh_node_array_texture_object_cpu_ptr = NULL;

cudaTextureObject_t* g_serial_ray_array_texture_object_gpu_ptr = NULL;
cudaTextureObject_t* g_serial_face_vertex_index_array_texture_object_gpu_ptr = NULL;
cudaTextureObject_t* g_serial_vertex_array_texture_object_gpu_ptr = NULL;
cudaTextureObject_t* g_serial_threaded_bvh_array_texture_object_gpu_ptr = NULL;
cudaTextureObject_t* g_serial_threaded_bvh_node_array_texture_object_gpu_ptr = NULL;

void rtx_cuda_malloc_linear_memory_texture_object(
    cudaTextureObject_t** texture_object_cpu_ptr_ref,
    cudaTextureObject_t** texture_object_gpu_ptr_ref)
{
    *texture_object_cpu_ptr_ref = (cudaTextureObject_t*)malloc(sizeof(cudaTextureObject_t));
    cudaMalloc((void**)texture_object_gpu_ptr_ref, sizeof(cudaTextureObject_t));
}
void rtx_cuda_bind_linear_memory_texture_object(
    cudaTextureObject_t** texture_object_cpu_ptr_ref,
    cudaTextureObject_t** texture_object_gpu_ptr_ref,
    void* buffer,
    size_t bytes, cudaChannelFormatKind format)
{
    cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));

    tex.normalizedCoords = false;
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;
    tex.readMode = cudaReadModeElementType;
    tex.filterMode = cudaFilterModePoint;

    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(cudaResourceDesc));
    resource.resType = cudaResourceTypeLinear;
    resource.res.linear.devPtr = buffer;
    resource.res.linear.sizeInBytes = bytes;
    resource.res.linear.desc.f = format;
    resource.res.linear.desc.x = 32;
    resource.res.linear.desc.y = 32;
    resource.res.linear.desc.z = 32;
    resource.res.linear.desc.w = 32;
    cudaError_t status = cudaCreateTextureObject(*texture_object_cpu_ptr_ref, &resource, &tex, NULL);
    if (status != 0) {
        fprintf(stderr, "CUDA Error at cudaCreateTextureObject: %s\n", cudaGetErrorString(status));
    }
    status = cudaMemcpy(*texture_object_gpu_ptr_ref, *texture_object_cpu_ptr_ref, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    if (status != 0) {
        fprintf(stderr, "CUDA Error at cudaMemcpy: %s\n", cudaGetErrorString(status));
    }
}
void rtx_cuda_free_linear_memory_texture_object(
    cudaTextureObject_t* texture_object_cpu_ptr_ref,
    cudaTextureObject_t* texture_object_gpu_ptr_ref)
{
    if (texture_object_cpu_ptr_ref) {
        free(texture_object_cpu_ptr_ref);
    }
    if (texture_object_gpu_ptr_ref) {
        cudaFree(texture_object_gpu_ptr_ref);
    }
}
void rtx_cuda_allocate_linear_memory_texture_objects()
{
    rtx_cuda_malloc_linear_memory_texture_object(&g_serial_ray_array_texture_object_cpu_ptr, &g_serial_ray_array_texture_object_gpu_ptr);
    rtx_cuda_malloc_linear_memory_texture_object(&g_serial_face_vertex_index_array_texture_object_cpu_ptr, &g_serial_face_vertex_index_array_texture_object_gpu_ptr);
    rtx_cuda_malloc_linear_memory_texture_object(&g_serial_vertex_array_texture_object_cpu_ptr, &g_serial_vertex_array_texture_object_gpu_ptr);
    rtx_cuda_malloc_linear_memory_texture_object(&g_serial_threaded_bvh_array_texture_object_cpu_ptr, &g_serial_threaded_bvh_array_texture_object_gpu_ptr);
    rtx_cuda_malloc_linear_memory_texture_object(&g_serial_threaded_bvh_node_array_texture_object_cpu_ptr, &g_serial_threaded_bvh_node_array_texture_object_gpu_ptr);
}
void rtx_cuda_delete_linear_memory_texture_objects()
{
    rtx_cuda_free_linear_memory_texture_object(g_serial_ray_array_texture_object_cpu_ptr, g_serial_ray_array_texture_object_gpu_ptr);
    rtx_cuda_free_linear_memory_texture_object(g_serial_face_vertex_index_array_texture_object_cpu_ptr, g_serial_face_vertex_index_array_texture_object_gpu_ptr);
    rtx_cuda_free_linear_memory_texture_object(g_serial_vertex_array_texture_object_cpu_ptr, g_serial_vertex_array_texture_object_gpu_ptr);
    rtx_cuda_free_linear_memory_texture_object(g_serial_threaded_bvh_array_texture_object_cpu_ptr, g_serial_threaded_bvh_array_texture_object_gpu_ptr);
    rtx_cuda_free_linear_memory_texture_object(g_serial_threaded_bvh_node_array_texture_object_cpu_ptr, g_serial_threaded_bvh_node_array_texture_object_gpu_ptr);
}