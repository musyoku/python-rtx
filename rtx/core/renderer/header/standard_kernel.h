#pragma once
#include "../../header/enum.h"
#include "../../header/struct.h"
#include <cuda_runtime.h>

__global__ void standard_texture_memory_kernel(
    int ray_array_size,
    int face_vertex_index_array_size,
    int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);

__global__ void standard_global_memory_kernel(
    rtxRay* global_ray_array, int ray_array_size,
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array, int face_vertex_index_array_size,
    rtxVertex* global_serialized_vertex_array, int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);

__global__ void standard_shared_memory_kernel(
    int ray_array_size,
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array, int face_vertex_index_array_size,
    rtxVertex* global_serialized_vertex_array, int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed);