#pragma once
#include <cuda_runtime.h>

extern cudaTextureObject_t* g_serial_ray_array_texture_object_cpu_ptr;
extern cudaTextureObject_t* g_serial_face_vertex_index_array_texture_object_cpu_ptr;
extern cudaTextureObject_t* g_serial_vertex_array_texture_object_cpu_ptr;
extern cudaTextureObject_t* g_serial_threaded_bvh_array_texture_object_cpu_ptr;
extern cudaTextureObject_t* g_serial_threaded_bvh_node_array_texture_object_cpu_ptr;

extern cudaTextureObject_t* g_serial_ray_array_texture_object_gpu_ptr;
extern cudaTextureObject_t* g_serial_face_vertex_index_array_texture_object_gpu_ptr;
extern cudaTextureObject_t* g_serial_vertex_array_texture_object_gpu_ptr;
extern cudaTextureObject_t* g_serial_threaded_bvh_array_texture_object_gpu_ptr;
extern cudaTextureObject_t* g_serial_threaded_bvh_node_array_texture_object_gpu_ptr;

void rtx_cuda_malloc_linear_memory_texture_object(
    cudaTextureObject_t** texture_object_cpu_ptr_ref,
    cudaTextureObject_t** texture_object_gpu_ptr_ref);
void rtx_cuda_bind_linear_memory_texture_object(
    cudaTextureObject_t** texture_object_cpu_ptr_ref,
    cudaTextureObject_t** texture_object_gpu_ptr_ref,
    void* buffer,
    size_t bytes, cudaChannelFormatKind format);
void rtx_cuda_free_linear_memory_texture_object(
    cudaTextureObject_t* texture_object_cpu_ptr_ref,
    cudaTextureObject_t* texture_object_gpu_ptr_ref);