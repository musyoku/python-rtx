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

// 引数が同じ関数を作るのでプリプロセッサで行う
#define rtx_define_cuda_mcrt_kernel_launcher_function(memory_type)   \
    void rtx_cuda_launch_mcrt_##memory_type##_kernel(                \
        rtxFaceVertexIndex* gpu_face_vertex_index_array,             \
        rtxVertex* gpu_vertex_array,                                 \
        rtxObject* gpu_object_array,                                 \
        rtxMaterialAttributeByte* gpu_material_attribute_byte_array, \
        rtxThreadedBVH* gpu_threaded_bvh_array,                      \
        rtxThreadedBVHNode* gpu_threaded_bvh_node_array,             \
        rtxRGBAColor* gpu_color_mapping_array,                       \
        rtxUVCoordinate* gpu_serialized_uv_coordinate_array,         \
        rtxRGBAPixel* gpu_render_array,                              \
        rtxMCRTKernelArguments& args,                                \
        int num_threads, int num_blocks, size_t shared_memory_bytes);

rtx_define_cuda_mcrt_kernel_launcher_function(texture_memory)
rtx_define_cuda_mcrt_kernel_launcher_function(shared_memory)
rtx_define_cuda_mcrt_kernel_launcher_function(global_memory)

#define rtx_define_cuda_nee_kernel_launcher_function(memory_type)    \
    void rtx_cuda_launch_nee_##memory_type##_kernel(                 \
        rtxFaceVertexIndex* gpu_face_vertex_index_array,             \
        rtxVertex* gpu_vertex_array,                                 \
        rtxObject* gpu_object_array,                                 \
        rtxMaterialAttributeByte* gpu_material_attribute_byte_array, \
        rtxThreadedBVH* gpu_threaded_bvh_array,                      \
        rtxThreadedBVHNode* gpu_threaded_bvh_node_array,             \
        rtxRGBAColor* gpu_color_mapping_array,                       \
        rtxUVCoordinate* gpu_serialized_uv_coordinate_array,         \
        int* gpu_light_sampling_table,                               \
        rtxRGBAPixel* gpu_render_array,                              \
        rtxNEEKernelArguments& args,                                 \
        int num_threads, int num_blocks, size_t shared_memory_bytes);

rtx_define_cuda_nee_kernel_launcher_function(texture_memory)
rtx_define_cuda_nee_kernel_launcher_function(shared_memory)
rtx_define_cuda_nee_kernel_launcher_function(global_memory)