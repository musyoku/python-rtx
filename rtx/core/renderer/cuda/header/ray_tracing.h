#pragma once
enum RTX_CUDA_GEOMETRY_TYPE {
    RTX_CUDA_GEOMETRY_TYPE_STANDARD,
    RTX_CUDA_GEOMETRY_TYPE_SPHERE
};

void rtx_cuda_alloc(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    int*& gpu_object_types,
    float*& gpu_color_per_ray,
    const float* rays,
    const float* face_vertices,
    const int* object_types,
    const int num_rays,
    const int num_faces,
    const int faces_stride,
    const int num_pixels,
    const int num_rays_per_pixel);
void rtx_cuda_delete(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    float*& gpu_object_types,
    float*& gpu_color_per_ray);

void rtx_cuda_ray_tracing_render(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    int*& gpu_object_types,
    float*& gpu_color_per_ray,
    float*& color_per_ray,
    const int num_rays,
    const int num_faces,
    const int faces_stride,
    const int path_depth,
    const int num_pixels,
    const int num_rays_per_pixel);