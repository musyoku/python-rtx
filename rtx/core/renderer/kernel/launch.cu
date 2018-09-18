#include "../../header/enum.h"
#include "../header/bridge.h"
#include "../header/cuda_common.h"
#include "../header/cuda_linear_memory_texture.h"
#include "../header/cuda_texture.h"
#include "../header/next_event_estimation_kernel.h"
#include "../header/standard_kernel.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

void rtx_cuda_launch_standard_kernel(
    RTXRay*& gpu_ray_array, const int ray_array_size,
    RTXFace*& gpu_face_vertex_index_array, const int face_vertex_index_array_size,
    RTXVertex*& gpu_vertex_array, const int vertex_array_size,
    RTXObject*& gpu_object_array, const int object_array_size,
    RTXMaterialAttributeByte*& gpu_material_attribute_byte_array, const int material_attribute_byte_array_size,
    RTXThreadedBVH*& gpu_threaded_bvh_array, const int threaded_bvh_array_size,
    RTXThreadedBVHNode*& gpu_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    RTXColor*& gpu_color_mapping_array, const int color_mapping_array_size,
    RTXPixel*& gpu_render_array, const int render_array_size,
    const int num_threads,
    const int num_blocks,
    const int num_rays_per_pixel,
    const int max_bounce,
    const int curand_seed)
{
    // assert(gpu_ray_array != NULL);
    // assert(gpu_face_vertex_index_array != NULL);
    // assert(gpu_vertex_array != NULL);
    // assert(gpu_object_array != NULL);
    // assert(gpu_material_attribute_byte_array != NULL);
    // assert(gpu_threaded_bvh_array != NULL);
    // assert(gpu_threaded_bvh_node_array != NULL);
    // assert(gpu_render_array != NULL);
    // if (color_mapping_array_size > 0) {
    //     assert(gpu_color_mapping_array != NULL);
    // }

    // int num_rays = ray_array_size;

    // // int num_blocks = (num_rays - 1) / num_threads + 1;

    // int num_rays_per_thread = num_rays / (num_threads * num_blocks) + 1;

    // long required_shared_memory_bytes = sizeof(RTXFace) * face_vertex_index_array_size + sizeof(RTXVertex) * vertex_array_size + sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size + sizeof(RTXColor) * color_mapping_array_size;

    // // num_blocks = 1;
    // // num_rays_per_thread = 1;

    // cudaDeviceProp dev;
    // cudaGetDeviceProperties(&dev, 0);

    // printf("shared memory: %ld bytes\n", required_shared_memory_bytes);
    // printf("    face: %d * %d vertex: %d * %d object: %d * %d material: %d * %d color: %d * %d \n", sizeof(RTXFace), face_vertex_index_array_size, sizeof(RTXVertex), vertex_array_size, sizeof(RTXObject), object_array_size, sizeof(RTXMaterialAttributeByte), material_attribute_byte_array_size, sizeof(RTXColor), color_mapping_array_size);
    // printf("    bvh: %d * %d node: %d * %d\n", sizeof(RTXThreadedBVH), threaded_bvh_array_size, sizeof(RTXThreadedBVHNode), threaded_bvh_node_array_size);
    // printf("available: %d bytes\n", dev.sharedMemPerBlock);
    // printf("rays: %d\n", ray_array_size);

    // if (required_shared_memory_bytes > dev.sharedMemPerBlock) {
    //     // int required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size;
    //     long required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXColor) * color_mapping_array_size;
    //     printf("    shared memory: %ld bytes\n", required_shared_memory_bytes);
    //     printf("    available: %d bytes\n", dev.sharedMemPerBlock);
    //     printf("    num_blocks: %d num_threads: %d\n", num_blocks, num_threads);
    //     printf("using global memory kernel\n");

    //     assert(required_shared_memory_bytes <= dev.sharedMemPerBlock);

    //     standard_global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
    //         gpu_ray_array, ray_array_size,
    //         gpu_face_vertex_index_array, face_vertex_index_array_size,
    //         gpu_vertex_array, vertex_array_size,
    //         gpu_object_array, object_array_size,
    //         gpu_material_attribute_byte_array, material_attribute_byte_array_size,
    //         gpu_threaded_bvh_array, threaded_bvh_array_size,
    //         gpu_threaded_bvh_node_array, threaded_bvh_node_array_size,
    //         gpu_color_mapping_array, color_mapping_array_size,
    //         texture_object_pointer, 30,
    //         gpu_render_array,
    //         num_rays_per_thread,
    //         max_bounce,
    //         curand_seed);

    //     // rtx_cuda_bind_linear_memory_texture_object(
    //     //     &g_serial_ray_array_texture_object_cpu_ptr,
    //     //     &g_serial_ray_array_texture_object_gpu_ptr,
    //     //     gpu_ray_array,
    //     //     sizeof(RTXRay) * ray_array_size,
    //     //     cudaChannelFormatKindFloat);
    //     // rtx_cuda_bind_linear_memory_texture_object(
    //     //     &g_serial_face_vertex_index_array_texture_object_cpu_ptr,
    //     //     &g_serial_face_vertex_index_array_texture_object_gpu_ptr,
    //     //     gpu_face_vertex_index_array,
    //     //     sizeof(RTXFace) * face_vertex_index_array_size,
    //     //     cudaChannelFormatKindSigned);
    //     // rtx_cuda_bind_linear_memory_texture_object(
    //     //     &g_serial_vertex_array_texture_object_cpu_ptr,
    //     //     &g_serial_vertex_array_texture_object_gpu_ptr,
    //     //     gpu_vertex_array,
    //     //     sizeof(RTXVertex) * vertex_array_size,
    //     //     cudaChannelFormatKindFloat);
    //     // rtx_cuda_bind_linear_memory_texture_object(
    //     //     &g_serial_threaded_bvh_node_array_texture_object_cpu_ptr,
    //     //     &g_serial_threaded_bvh_node_array_texture_object_gpu_ptr,
    //     //     gpu_threaded_bvh_node_array,
    //     //     sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size,
    //     //     cudaChannelFormatKindFloat);
    //     // standard_global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
    //     //     g_serial_ray_array_texture_object_gpu_ptr, ray_array_size,
    //     //     g_serial_face_vertex_index_array_texture_object_gpu_ptr, face_vertex_index_array_size,
    //     //     g_serial_vertex_array_texture_object_gpu_ptr, vertex_array_size,
    //     //     gpu_object_array, object_array_size,
    //     //     gpu_material_attribute_byte_array, material_attribute_byte_array_size,
    //     //     gpu_threaded_bvh_array, threaded_bvh_array_size,
    //     //     g_serial_threaded_bvh_node_array_texture_object_gpu_ptr, threaded_bvh_node_array_size,
    //     //     gpu_color_mapping_array, color_mapping_array_size,
    //     //     texture_object_pointer, 30,
    //     //     gpu_render_array,
    //     //     num_rays_per_thread,
    //     //     max_bounce,
    //     //     curand_seed);
    // } else {
    //     printf("using shared memory kernel\n");
    //     cudaBindTexture(0, ray_texture, gpu_ray_array, cudaCreateChannelDesc<float4>(), sizeof(RTXRay) * ray_array_size);

    //     printf("%p\n", texture_object_array);
    //     standard_shared_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
    //         ray_array_size,
    //         gpu_face_vertex_index_array, face_vertex_index_array_size,
    //         gpu_vertex_array, vertex_array_size,
    //         gpu_object_array, object_array_size,
    //         gpu_material_attribute_byte_array, material_attribute_byte_array_size,
    //         gpu_threaded_bvh_array, threaded_bvh_array_size,
    //         gpu_threaded_bvh_node_array, threaded_bvh_node_array_size,
    //         gpu_color_mapping_array, color_mapping_array_size,
    //         texture_object_pointer, 30,
    //         gpu_render_array,
    //         num_rays_per_thread,
    //         max_bounce,
    //         curand_seed);

    //     cudaUnbindTexture(ray_texture);
    // }

    // // num_rays_per_thread = 1;

    // // printf("rays: %d, rays_per_kernel: %d, num_rays_per_thread: %d\n", num_rays, num_rays_per_kernel, num_rays_per_thread);
    // // printf("<<<%d, %d>>>\n", num_blocks, num_threads);
    // cudaError_t status = cudaGetLastError();
    // if (status != 0) {
    //     fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
    // }
    // cudaError_t error = cudaThreadSynchronize();
    // if (error != cudaSuccess) {
    //     fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
    // }

    // // cudaDeviceProp dev;
    // // cudaGetDeviceProperties(&dev, 0);

    // // printf(" device name : %s\n", dev.name);
    // // printf(" total global memory : %d (MB)\n", dev.totalGlobalMem/1024/1024);
    // // printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);
    // // printf(" register / block : %d\n", dev.regsPerBlock);
}
