#pragma once

void rtx_cuda_malloc(void** gpu_array, size_t size);
void rtx_cuda_memcpy_host_to_device(void* gpu_array, void* cpu_array, size_t size);
void rtx_cuda_memcpy_device_to_host(void* cpu_array, void* gpu_array, size_t size);
void rtx_cuda_free(void** array);
void cuda_device_reset();

void rtx_cuda_ray_tracing_render(
    float*& gpu_ray_array, const int ray_array_size,
    int*& gpu_face_vertex_index_array, const int face_vertex_index_array_size,
    float*& gpu_vertex_array, const int vertex_array_size,
    int*& gpu_object_face_count_array, const int object_face_count_array_size,
    int*& gpu_object_face_offset_array, const int object_face_offset_array_size,
    int*& gpu_object_vertex_count_array, const int object_vertex_count_array_size,
    int*& gpu_object_vertex_offset_array, const int object_vertex_offset_array_size,
    int*& gpu_object_geometry_type_array, const int object_geometry_type_array_size,
    unsigned int*& gpu_scene_threaded_bvh_node_array, const int scene_threaded_bvh_node_array_size,
    float*& gpu_scene_threaded_bvh_aabb_array, const int scene_threaded_bvh_aabb_array_size,
    float*& gpu_render_array, const int render_array_size,
    const int num_rays,
    const int num_rays_per_pixel,
    const int max_bounce);