#pragma once
enum RTX_CUDA_GEOMETRY_TYPE {
    RTX_CUDA_GEOMETRY_TYPE_STANDARD,
    RTX_CUDA_GEOMETRY_TYPE_SPHERE
};

void rtx_cuda_alloc(
    float*& gpu_face_vertices,
    const float* face_vertices,
    const int num_faces,
    const int faces_stride,
    float*& gpu_color_per_ray,
    const int num_pixels,
    const int num_rays_per_pixel);
void rtx_cuda_delete(float*& gpu_face_vertices, float*& gpu_color_per_ray);

void rtx_cuda_ray_tracing_render(
    const float* rays,
    const int num_rays,
    const float* face_vertices,
    const int* object_types,
    const int num_faces,
    const int faces_stride,
    const int path_depth,
    float* color_per_ray,
    const int num_pixels,
    const int num_rays_per_pixel,
    float*& gpu_face_vertices,
    float*& gpu_color_per_ray);