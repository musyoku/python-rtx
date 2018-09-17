#pragma once
#include <cuda_runtime.h>

extern cudaTextureObject_t* texture_object_pointer;
extern cudaTextureObject_t texture_object_array[30];
extern cudaArray* texture_cuda_array[30];