#pragma once

void rtx_cuda_alloc(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    float*& gpu_face_colors,
    int*& gpu_object_types,
    int*& gpu_material_types,
    float*& gpu_color_per_ray,
    float*& gpu_camera_inv_matrix,
    const float* rays,
    const float* face_vertices,
    const float* face_colors,
    const int* object_types,
    const int* material_types,
    const float* camera_matrix,
    const int num_rays,
    const int rays_stride,
    const int num_faces,
    const int faces_stride,
    const int color_stride,
    const int num_pixels,
    const int num_rays_per_pixel);

void rtx_cuda_delete(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    float*& gpu_face_colors,
    int*& gpu_object_types,
    int*& gpu_material_types,
    float*& gpu_color_per_ray,
    float*& gpu_camera_inv_matrix);

void rtx_cuda_copy(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    float*& gpu_camera_matrix,
    const float* rays,
    const float* face_vertices,
    const float* camera_matrix,
    const int num_rays,
    const int rays_stride,
    const int num_faces,
    const int faces_stride);
    
void rtx_cuda_ray_tracing_render(
    float*& gpu_rays,
    float*& gpu_face_vertices,
    float*& gpu_face_colors,
    int*& gpu_object_types,
    int*& gpu_material_types,
    float*& gpu_color_per_ray,
    float*& color_per_ray,
    float*& gpu_camera_matrix,
    const int num_rays,
    const int num_faces,
    const int faces_stride,
    const int color_stride,
    const int path_depth,
    const int num_pixels,
    const int num_rays_per_pixel);

void cuda_device_reset();