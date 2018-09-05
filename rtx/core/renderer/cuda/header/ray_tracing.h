#pragma once

typedef struct CUDAThreadedBVHNode {
    int hit_link;
    int miss_link;
    int start;
    int end;
} CUDAThreadedBVHNode;

typedef struct CUDAVector3f {
    float x;
    float y;
    float z;
} CUDAVector3f;

typedef CUDAVector3f CUDAGeometryVertex;

typedef struct CUDAGeometryFace {
    int a;
    int b;
    int c;
} CUDAGeometryFace;

typedef struct CUDARay {
    CUDAVector3f direction;
    CUDAVector3f origin;
} CUDARay;

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
    int*& gpu_object_geometry_attributes_array, const int object_geometry_attributes_array_size,
    int*& gpu_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    int*& gpu_threaded_bvh_num_nodes_array, const int threaded_bvh_num_nodes_array_size,
    int*& gpu_threaded_bvh_index_offset_array, const int threaded_bvh_index_offset_array_size,
    float*& gpu_threaded_bvh_aabb_array, const int threaded_bvh_aabb_array_size,
    float*& gpu_render_array, const int render_array_size,
    const int num_rays,
    const int num_rays_per_pixel,
    const int max_bounce);

void launch_test_linear_kernel(
    int*& gpu_node_array, const int num_nodes);
void launch_test_struct_kernel(
    CUDAThreadedBVHNode*& gpu_struct_array, const int num_nodes);