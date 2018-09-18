#pragma once
#include "../../header/enum.h"
#include "../../header/struct.h"
#include <cuda_runtime.h>

__global__ void standard_texture_memory_kernel(
    int ray_array_size,
    int face_vertex_index_array_size,
    int vertex_array_size,
    RTXObject* global_object_array, int object_array_size,
    RTXMaterialAttributeByte* global_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* global_threaded_bvh_array, int threaded_bvh_array_size,
    int threaded_bvh_node_array_size,
    RTXColor* global_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* texture_object_array, int texture_object_array_size,
    RTXPixel* global_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);

__global__ void standard_global_memory_kernel(
    RTXRay* global_ray_array, int ray_array_size,
    RTXFace* global_face_vertex_indices_array, int face_vertex_index_array_size,
    RTXVertex* global_vertex_array, int vertex_array_size,
    RTXObject* global_object_array, int object_array_size,
    RTXMaterialAttributeByte* global_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* global_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* global_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    RTXColor* global_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* texture_object_array, int texture_object_array_size,
    RTXPixel* global_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);

__global__ void standard_shared_memory_kernel(
    int ray_array_size,
    RTXFace* global_face_vertex_indices_array, int face_vertex_index_array_size,
    RTXVertex* global_vertex_array, int vertex_array_size,
    RTXObject* global_object_array, int object_array_size,
    RTXMaterialAttributeByte* global_material_attribute_byte_array, int material_attribute_byte_array_size,
    RTXThreadedBVH* global_threaded_bvh_array, int threaded_bvh_array_size,
    RTXThreadedBVHNode* global_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    RTXColor* global_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* texture_object_array, int texture_object_array_size,
    RTXPixel* global_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);