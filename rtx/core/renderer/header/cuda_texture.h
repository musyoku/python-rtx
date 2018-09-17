#pragma once
#include <cuda_runtime.h>

extern cudaTextureObject_t* texture_object_pointer;
extern cudaTextureObject_t texture_object_array[30];
extern cudaArray* texture_cuda_array[30];


texture<float4, cudaTextureType1D, cudaReadModeElementType> ray_texture;
texture<int4, cudaTextureType1D, cudaReadModeElementType> face_vertex_index_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> vertex_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> threaded_bvh_node_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> threaded_bvh_texture;