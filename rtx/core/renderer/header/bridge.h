#pragma once
#include "../../header/struct.h"

#define THREADED_BVH_TERMINAL_NODE -1
#define THREADED_BVH_INNER_NODE -1
#define RTX_CUDA_MAX_TEXTURE_UNITS 50

void rtx_cuda_malloc(void** gpu_array, size_t size);
void rtx_cuda_malloc_pointer(void**& gpu_array, size_t size);
void rtx_cuda_memcpy_host_to_device(void* gpu_array, void* cpu_array, size_t size);
void rtx_cuda_memcpy_device_to_host(void* cpu_array, void* gpu_array, size_t size);
void rtx_cuda_free(void** array);
void rtx_cuda_device_reset();

void rtx_cuda_malloc_texture(int unit_index, int width, int height);
void rtx_cuda_free_texture(int unit_index);
void rtx_cuda_memcpy_to_texture(int unit_index, int width_offset, int height_offset, void* data, size_t bytes);
void rtx_cuda_bind_texture(int unit_index);

size_t rtx_cuda_get_available_shared_memory_bytes();
size_t rtx_cuda_get_cudaTextureObject_t_bytes();

// 引数が同じ関数を3作るのでプリプロセッサで行う
#define rtx_cuda_define_standard_launcher_function(name)                                                     \
void rtx_cuda_launch_standard_##name(                                                                    \
    rtxRay* gpu_ray_array, int ray_array_size,                                                           \
    rtxFaceVertexIndex* gpu_face_vertex_index_array, int face_vertex_index_array_size,                   \
    rtxVertex* gpu_vertex_array, int vertex_array_size,                                                  \
    rtxObject* gpu_object_array, int object_array_size,                                                  \
    rtxMaterialAttributeByte* gpu_material_attribute_byte_array, int material_attribute_byte_array_size, \
    rtxThreadedBVH* gpu_threaded_bvh_array, int threaded_bvh_array_size,                                 \
    rtxThreadedBVHNode* gpu_threaded_bvh_node_array, int threaded_bvh_node_array_size,                   \
    rtxRGBAColor* gpu_color_mapping_array, int color_mapping_array_size,                                 \
    rtxUVCoordinate* gpu_serialized_uv_coordinate_array, int uv_coordinate_array_size,                   \
    rtxRGBAPixel* gpu_render_array, int render_array_size,                                               \
    int num_active_texture_units,                                                                        \
    int num_threads,                                                                                     \
    int num_blocks,                                                                                      \
    int num_rays_per_thread,                                                                             \
    size_t shared_memory_bytes,                                                                          \
    int max_bounce,                                                                                      \
    int curand_seed);

rtx_cuda_define_standard_launcher_function(texture_memory_kernel)
rtx_cuda_define_standard_launcher_function(shared_memory_kernel)
rtx_cuda_define_standard_launcher_function(global_memory_kernel)