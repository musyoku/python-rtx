#pragma once
#include <cuda_runtime.h>

typedef struct CUDARay {
    float4 direction;
    float4 origin;
} CUDARay;

typedef struct CUDAThreadedBVHNode {
    int hit_node_index;
    int miss_node_index;
    int assigned_face_index_start;
    int assigned_face_index_end;
    float4 aabb_max;
    float4 aabb_min;
} CUDAThreadedBVHNode;

texture<float4, cudaTextureType1D, cudaReadModeElementType> ray_texture;
texture<int4, cudaTextureType1D, cudaReadModeElementType> face_vertex_index_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> vertex_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> threaded_bvh_node_texture;
texture<float4, cudaTextureType1D, cudaReadModeElementType> threaded_bvh_texture;