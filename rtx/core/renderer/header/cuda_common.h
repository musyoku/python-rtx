#pragma once
#include <cuda_runtime.h>

#define cudaCheckError(error)                                                                     \
    {                                                                                             \
        if (error != cudaSuccess) {                                                               \
            printf("CUDA Error at %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(0);                                                                              \
        }                                                                                         \
    }

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