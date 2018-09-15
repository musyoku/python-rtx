#pragma once
#include "../../header/enum.h"
#include "../../header/struct.h"
#include <cuda_runtime.h>

__global__ void standard_kernel(
    const int ray_array_size,
    const int face_vertex_index_array_size,
    const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const int threaded_bvh_node_array_size,
    const int* global_light_sampling_table, const int light_sampling_table_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed);