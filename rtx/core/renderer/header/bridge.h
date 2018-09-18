#pragma once
#include "../../header/struct.h"

#define THREADED_BVH_TERMINAL_NODE -1
#define THREADED_BVH_INNER_NODE -1

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

void rtx_cuda_allocate_linear_memory_texture_objects();
void rtx_cuda_delete_linear_memory_texture_objects();

size_t rtx_cuda_get_available_shared_memory_bytes();

void rtx_cuda_launch_standard_texture_memory_kernel(
    RTXRay* gpu_ray_array, int ray_array_size,
    RTXFace* gpu_face_vertex_index_array, int face_vertex_index_array_size,
    RTXVertex* gpu_vertex_array, int vertex_array_size,
    RTXObject* gpu_object_array, int object_array_size,
    RTXMaterialAttributeByte* gpu_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* gpu_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* gpu_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    RTXColor* gpu_color_mapping_array, int color_mapping_array_size,
    RTXPixel* gpu_render_array, int render_array_size,
    int num_threads,
    int num_blocks,
    int num_rays_per_thread,
    size_t shared_memory_bytes,
    int max_bounce,
    int curand_seed);

void rtx_cuda_launch_standard_global_memory_kernel(
    RTXRay* gpu_ray_array, int ray_array_size,
    RTXFace* gpu_face_vertex_index_array, int face_vertex_index_array_size,
    RTXVertex* gpu_vertex_array, int vertex_array_size,
    RTXObject* gpu_object_array, int object_array_size,
    RTXMaterialAttributeByte* gpu_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* gpu_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* gpu_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    RTXColor* gpu_color_mapping_array, int color_mapping_array_size,
    RTXPixel* gpu_render_array, int render_array_size,
    int num_threads,
    int num_blocks,
    int num_rays_per_thread,
    size_t shared_memory_bytes,
    int max_bounce,
    int curand_seed);

void rtx_cuda_launch_standard_shared_memory_kernel(
    RTXRay* gpu_ray_array, int ray_array_size,
    RTXFace* gpu_face_vertex_index_array, int face_vertex_index_array_size,
    RTXVertex* gpu_vertex_array, int vertex_array_size,
    RTXObject* gpu_object_array, int object_array_size,
    RTXMaterialAttributeByte* gpu_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* gpu_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* gpu_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    RTXColor* gpu_color_mapping_array, int color_mapping_array_size,
    RTXPixel* gpu_render_array, int render_array_size,
    int num_threads,
    int num_blocks,
    int num_rays_per_thread,
    size_t shared_memory_bytes,
    int max_bounce,
    int curand_seed);

void rtx_cuda_launch_next_event_estimation_kernel(
    RTXRay* gpu_ray_array, int ray_array_size,
    RTXFace* gpu_face_vertex_index_array, int face_vertex_index_array_size,
    RTXVertex* gpu_vertex_array, int vertex_array_size,
    RTXObject* gpu_object_array, int object_array_size,
    RTXMaterialAttributeByte* gpu_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* gpu_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* gpu_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    int* gpu_light_sampling_table, int light_sampling_table_size,
    RTXColor* gpu_color_mapping_array, int color_mapping_array_size,
    RTXPixel* gpu_render_array, int render_array_size,
    int num_threads,
    int num_blocks,
    int num_rays_per_pixel,
    int max_bounce,
    int curand_seed);