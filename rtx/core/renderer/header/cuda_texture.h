#pragma once
#include <cuda_runtime.h>

extern cudaTextureObject_t* texture_object_pointer;
extern cudaTextureObject_t texture_object_array[30];
extern cudaArray* texture_cuda_array[30];

texture<float4, cudaTextureType1D, cudaReadModeElementType> g_serial_ray_array_texture_ref;
texture<int4, cudaTextureType1D, cudaReadModeElementType> g_serial_face_vertex_index_array_texture_ref;
texture<float4, cudaTextureType1D, cudaReadModeElementType> g_serial_vertex_array_texture_ref;
texture<float4, cudaTextureType1D, cudaReadModeElementType> g_serial_threaded_bvh_node_array_texture_ref;
texture<float2, cudaTextureType1D, cudaReadModeElementType> g_serial_uv_coordinate_array_texture_ref;