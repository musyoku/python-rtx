#pragma once
#include "../../header/enum.h"
#include "../../header/struct.h"
#include <cuda_runtime.h>

__global__ void standard_global_memory_kernel(
    const int ray_array_size,
    const int face_vertex_index_array_size,
    const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const int threaded_bvh_node_array_size,
    const RTXColor* global_color_mapping_array, const int color_mapping_array_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed);

__global__ void standard_shared_memory_kernel(
    const int ray_array_size,
    const RTXFace* global_face_vertex_indices_array, const int face_vertex_index_array_size,
    const RTXVertex* global_vertex_array, const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const RTXThreadedBVHNode* global_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    const RTXColor* global_color_mapping_array, const int color_mapping_array_size,
    const cudaTextureObject_t* texture_object_array, const int texture_object_array_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed);