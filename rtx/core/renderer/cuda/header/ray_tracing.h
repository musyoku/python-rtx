#pragma once

void rtx_cuda_malloc(void** gpu_buffer, size_t size);
void rtx_cuda_memcpy_host_to_device(void** gpu_buffer, void** cpu_buffer, size_t size);
void rtx_cuda_memcpy_device_to_host(void* cpu_buffer, void* gpu_buffer, size_t size);
void rtx_cuda_free(void** buffer);
void cuda_device_reset();

void rtx_cuda_ray_tracing_render(
    float*& gpu_ray_buffer, const int ray_buffer_size,
    int*& gpu_face_vertex_index_buffer, const int face_vertex_index_buffer_size,
    int*& gpu_face_count_buffer, const int face_count_buffer_size,
    float*& gpu_vertex_buffer, const int vertex_buffer_size,
    int*& gpu_vertex_count_buffer, const int vertex_count_buffer,
    unsigned int*& gpu_scene_threaded_bvh, const int scene_threaded_bvh_size,
    float*& gpu_render_buffer, const int render_buffer_size,
    const int num_rays,
    const int num_rays_per_pixel,
    const int max_path_depth);