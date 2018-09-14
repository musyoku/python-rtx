#include "../../header/enum.h"
#include "../header/ray_tracing.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

#define THREADED_BVH_TERMINAL_NODE -1
#define THREADED_BVH_INNER_NODE -1

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

__global__ void _global_memory_kernel(
    const int ray_array_size,
    const int face_vertex_index_array_size,
    const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const int threaded_bvh_node_array_size,
    const int* global_light_index_array, const int light_sampling_table_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed)
{
    extern __shared__ unsigned char shared_memory[];
    int thread_id = threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    RTXObject* shared_object_array = (RTXObject*)&shared_memory[offset];
    offset += sizeof(RTXObject) / sizeof(unsigned char) * object_array_size;

    RTXMaterialAttributeByte* shared_material_attribute_byte_array = (RTXMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(RTXMaterialAttributeByte) / sizeof(unsigned char) * material_attribute_byte_array_size;

    RTXThreadedBVH* shared_threaded_bvh_array = (RTXThreadedBVH*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVH) / sizeof(unsigned char) * threaded_bvh_array_size;

    int* shared_light_index_array = (int*)&shared_memory[offset];
    offset += sizeof(unsigned char) * light_sampling_table_size;

    // RTXThreadedBVHNode* shared_threaded_bvh_node_array = (RTXThreadedBVHNode*)&shared_memory[offset];
    // offset += sizeof(RTXThreadedBVHNode) / sizeof(int) * threaded_bvh_node_array_size;

    if (thread_id == 0) {
        for (int k = 0; k < object_array_size; k++) {
            shared_object_array[k] = global_object_array[k];
        }
        for (int k = 0; k < material_attribute_byte_array_size; k++) {
            shared_material_attribute_byte_array[k] = global_material_attribute_byte_array[k];
        }
        for (int k = 0; k < threaded_bvh_array_size; k++) {
            shared_threaded_bvh_array[k] = global_threaded_bvh_array[k];
        }
        for (int k = 0; k < light_sampling_table_size; k++) {
            shared_light_index_array[k] = global_light_index_array[k];
        }
        // for (int k = 0; k < threaded_bvh_node_array_size; k++) {
        //     shared_threaded_bvh_node_array[k] = global_threaded_bvh_node_array[k];
        // }
    }
    __syncthreads();

    // if (thread_id == 0) {
    //     // for (int i = 0; i < object_array_size; i++) {
    //     //     if (shared_object_array[i] != global_object_array[i]) {
    //     //         printf("Error: shared_object_array missmatch at %d\n", i);
    //     //         return;
    //     //     }
    //     // }
    //     // for (int i = 0; i < threaded_bvh_array_size; i++) {
    //     //     if (shared_threaded_bvh_array[i] != global_threaded_bvh_array[i]) {
    //     //         printf("Error: shared_threaded_bvh_array missmatch at %d\n", i);
    //     //         return;
    //     //     }
    //     // }
    //     for (int i = 0; i < light_sampling_table_size; i++) {
    //         if (shared_light_index_array[i] != global_light_index_array[i]) {
    //             printf("Error: shared_light_index_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    // }

    const float eps = 0.0000001;
    CUDARay ray;
    // RTXRay ray;
    float3 ray_direction_inv;
    float3 hit_point;
    float3 hit_face_normal;
    RTXPixel hit_color;
    RTXPixel reflection_decay;
    CUDAThreadedBVHNode node;

    int ray_loop = 0;
    int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + ray_loop;
    if (ray_index >= ray_array_size) {
        return;
    }

    // ray = global_ray_array[ray_index];
    ray.direction = tex1Dfetch(ray_texture, ray_index * 2 + 0);
    ray.origin = tex1Dfetch(ray_texture, ray_index * 2 + 1);

    ray_direction_inv.x = 1.0f / ray.direction.x;
    ray_direction_inv.y = 1.0f / ray.direction.y;
    ray_direction_inv.z = 1.0f / ray.direction.z;

    reflection_decay.r = 1.0f;
    reflection_decay.g = 1.0f;
    reflection_decay.b = 1.0f;

    bool did_hit_light = false;
    bool did_hit_object = false;

    int bounce = 0;
    int object_index = 0;
    int bvh_current_node_index = 0;
    int prev_bvh_current_node_index = -1;
    float min_distance = FLT_MAX;
    RTXObject object = shared_object_array[object_index];
    RTXThreadedBVH bvh = shared_threaded_bvh_array[object_index];
    for (;;) {
        do {
            if(prev_bvh_current_node_index != bvh_current_node_index){
                int bvh_node_position = bvh.node_index_offset + bvh_current_node_index;

                float4 attributes_float = tex1Dfetch(threaded_bvh_node_texture, bvh_node_position * 3 + 0);
                int4* attributes_integer_ptr = reinterpret_cast<int4*>(&attributes_float);
                node.hit_node_index = attributes_integer_ptr->x;
                node.miss_node_index = attributes_integer_ptr->y;
                node.assigned_face_index_start = attributes_integer_ptr->z;
                node.assigned_face_index_end = attributes_integer_ptr->w;
                node.aabb_max = tex1Dfetch(threaded_bvh_node_texture, bvh_node_position * 3 + 1);
                node.aabb_min = tex1Dfetch(threaded_bvh_node_texture, bvh_node_position * 3 + 2);

                prev_bvh_current_node_index = bvh_current_node_index;
            }

            bool is_inner_node = node.assigned_face_index_start == -1;

            if (is_inner_node) {
                // http://www.cs.utah.edu/~awilliam/box/box.pdf
                float tmin = ((ray_direction_inv.x < 0 ? node.aabb_max.x : node.aabb_min.x) - ray.origin.x) * ray_direction_inv.x;
                float tmax = ((ray_direction_inv.x < 0 ? node.aabb_min.x : node.aabb_max.x) - ray.origin.x) * ray_direction_inv.x;
                float tmp_tmin = ((ray_direction_inv.y < 0 ? node.aabb_max.y : node.aabb_min.y) - ray.origin.y) * ray_direction_inv.y;
                float tmp_tmax = ((ray_direction_inv.y < 0 ? node.aabb_min.y : node.aabb_max.y) - ray.origin.y) * ray_direction_inv.y;

                if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                    bvh_current_node_index = node.miss_node_index;
                    break;
                }
                if (tmp_tmin > tmin) {
                    tmin = tmp_tmin;
                }
                if (tmp_tmax < tmax) {
                    tmax = tmp_tmax;
                }
                tmp_tmin = ((ray_direction_inv.z < 0 ? node.aabb_max.z : node.aabb_min.z) - ray.origin.z) * ray_direction_inv.z;
                tmp_tmax = ((ray_direction_inv.z < 0 ? node.aabb_min.z : node.aabb_max.z) - ray.origin.z) * ray_direction_inv.z;
                if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                    bvh_current_node_index = node.miss_node_index;
                    break;
                }
                if (tmp_tmin > tmin) {
                    tmin = tmp_tmin;
                }
                if (tmp_tmax < tmax) {
                    tmax = tmp_tmax;
                }

                if (tmax < 0.001) {
                    bvh_current_node_index = node.miss_node_index;
                    break;
                }
            } else {
                int num_assigned_faces = node.assigned_face_index_end - node.assigned_face_index_start + 1;
                for (int m = 0; m < num_assigned_faces; m++) {
                    int index = node.assigned_face_index_start + m + object.face_index_offset;

                    int4 face = tex1Dfetch(face_vertex_index_texture, index);
                    // RTXFace face = global_face_vertex_index_array[index];

                    float4 va = tex1Dfetch(vertex_texture, face.x + object.vertex_index_offset);
                    float4 vb = tex1Dfetch(vertex_texture, face.y + object.vertex_index_offset);
                    float4 vc = tex1Dfetch(vertex_texture, face.z + object.vertex_index_offset);
                    // RTXVector4f va = global_vertex_array[face.a];
                    // RTXVector4f vb = global_vertex_array[face.b];
                    // RTXVector4f vc = global_vertex_array[face.c];

                    float3 edge_ba;
                    edge_ba.x = vb.x - va.x;
                    edge_ba.y = vb.y - va.y;
                    edge_ba.z = vb.z - va.z;

                    float3 edge_ca;
                    edge_ca.x = vc.x - va.x;
                    edge_ca.y = vc.y - va.y;
                    edge_ca.z = vc.z - va.z;

                    float3 h;
                    h.x = ray.direction.y * edge_ca.z - ray.direction.z * edge_ca.y;
                    h.y = ray.direction.z * edge_ca.x - ray.direction.x * edge_ca.z;
                    h.z = ray.direction.x * edge_ca.y - ray.direction.y * edge_ca.x;
                    float f = edge_ba.x * h.x + edge_ba.y * h.y + edge_ba.z * h.z;
                    if (f > -eps && f < eps) {
                        continue;
                    }

                    f = 1.0f / f;

                    float3 s;
                    s.x = ray.origin.x - va.x;
                    s.y = ray.origin.y - va.y;
                    s.z = ray.origin.z - va.z;
                    float dot = s.x * h.x + s.y * h.y + s.z * h.z;
                    float u = f * dot;
                    if (u < 0.0f || u > 1.0f) {
                        continue;
                    }

                    h.x = s.y * edge_ba.z - s.z * edge_ba.y;
                    h.y = s.z * edge_ba.x - s.x * edge_ba.z;
                    h.z = s.x * edge_ba.y - s.y * edge_ba.x;
                    dot = h.x * ray.direction.x + h.y * ray.direction.y + h.z * ray.direction.z;
                    float v = f * dot;
                    if (v < 0.0f || u + v > 1.0f) {
                        continue;
                    }
                    s.x = edge_ba.y * edge_ca.z - edge_ba.z * edge_ca.y;
                    s.y = edge_ba.z * edge_ca.x - edge_ba.x * edge_ca.z;
                    s.z = edge_ba.x * edge_ca.y - edge_ba.y * edge_ca.x;

                    float norm = sqrtf(s.x * s.x + s.y * s.y + s.z * s.z) + 1e-12;

                    s.x = s.x / norm;
                    s.y = s.y / norm;
                    s.z = s.z / norm;

                    dot = s.x * ray.direction.x + s.y * ray.direction.y + s.z * ray.direction.z;
                    if (dot > 0.0f) {
                        continue;
                    }

                    dot = edge_ca.x * h.x + edge_ca.y * h.y + edge_ca.z * h.z;
                    float t = f * dot;

                    if (t <= 0.001f) {
                        continue;
                    }
                    if (min_distance <= t) {
                        continue;
                    }

                    min_distance = t;
                    hit_point.x = ray.origin.x + t * ray.direction.x;
                    hit_point.y = ray.origin.y + t * ray.direction.y;
                    hit_point.z = ray.origin.z + t * ray.direction.z;

                    hit_face_normal.x = s.x;
                    hit_face_normal.y = s.y;
                    hit_face_normal.z = s.z;

                    int material_type = object.layerd_material_types.outside;
                    if (material_type == RTXMaterialTypeLambert) {
                        RTXLambertMaterialAttribute attr = ((RTXLambertMaterialAttribute*)&shared_material_attribute_byte_array[object.material_attribute_byte_array_offset])[0];
                        did_hit_light = false;
                        did_hit_object = true;
                    } else if (material_type == RTXMaterialTypeEmissive) {
                        RTXEmissiveMaterialAttribute attr = ((RTXEmissiveMaterialAttribute*)&shared_material_attribute_byte_array[object.material_attribute_byte_array_offset])[0];
                        did_hit_light = true;
                        did_hit_object = false;
                    }
                    hit_color.r = 0.9f;
                    hit_color.g = 0.9f;
                    hit_color.b = 0.9f;
                }
            }

            if (node.hit_node_index == THREADED_BVH_TERMINAL_NODE) {
                bvh_current_node_index = node.miss_node_index;
            } else {
                bvh_current_node_index = node.hit_node_index;
            }
        } while (false);

        bool hitting_light_determined = did_hit_light && (object_index == object_array_size - 1);
        bool hitting_object_determined = did_hit_object && (object_index == object_array_size - 1);

        if (hitting_light_determined) {
            reflection_decay.r *= hit_color.r;
            reflection_decay.g *= hit_color.g;
            reflection_decay.b *= hit_color.b;

            // global_render_array[ray_index] = reflection_decay;
        }

        if (hitting_object_determined) {
            ray.origin.x = hit_point.x;
            ray.origin.y = hit_point.y;
            ray.origin.z = hit_point.z;

            // diffuse reflection
            float diffuese_x = curand_normal(&state);
            float diffuese_y = curand_normal(&state);
            float diffuese_z = curand_normal(&state);
            float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
            diffuese_x /= norm;
            diffuese_y /= norm;
            diffuese_z /= norm;

            float dot = hit_face_normal.x * diffuese_x + hit_face_normal.y * diffuese_y + hit_face_normal.z * diffuese_z;
            if (dot < 0.0f) {
                diffuese_x = -diffuese_x;
                diffuese_y = -diffuese_y;
                diffuese_z = -diffuese_z;
            }
            ray.direction.x = diffuese_x;
            ray.direction.y = diffuese_y;
            ray.direction.z = diffuese_z;

            ray_direction_inv.x = 1.0f / ray.direction.x;
            ray_direction_inv.y = 1.0f / ray.direction.y;
            ray_direction_inv.z = 1.0f / ray.direction.z;

            reflection_decay.r *= hit_color.r;
            reflection_decay.g *= hit_color.g;
            reflection_decay.b *= hit_color.b;
        }

        // if (ray_index == 1000) {
        //     printf("ray: %d bounce: %d object: %d bvh: %d\n", ray_index, bounce, object_index, bvh_current_node_index);
        // }

        bool object_index_changed = false;
        bool should_go_to_next_object = bvh_current_node_index == THREADED_BVH_TERMINAL_NODE;
        if (should_go_to_next_object) {
            object_index++;
            object_index_changed = true;
        }

        bool should_go_to_next_bounce = object_index >= object_array_size;
        if (should_go_to_next_bounce) {
            bounce++;
            did_hit_light = false;
            did_hit_object = false;
            min_distance = FLT_MAX;
            bvh_current_node_index = 0;
        }

        bool should_go_to_first_object = should_go_to_next_bounce;
        if (should_go_to_first_object) {
            object_index = 0;
            object_index_changed = true;
        }
        if (object_index_changed) {
            object = shared_object_array[object_index];
            bvh = shared_threaded_bvh_array[object_index];
        }
        bool should_go_to_next_ray = bounce >= max_bounce;
        if (should_go_to_next_ray) {
            bounce = 0;
            ray_loop++;
            ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + ray_loop;
            if (ray_index >= ray_array_size) {
                return;
            }
            // ray = global_ray_array[ray_index];
            ray.direction = tex1Dfetch(ray_texture, ray_index * 2 + 0);
            ray.origin = tex1Dfetch(ray_texture, ray_index * 2 + 1);

            ray_direction_inv.x = 1.0f / ray.direction.x;
            ray_direction_inv.y = 1.0f / ray.direction.y;
            ray_direction_inv.z = 1.0f / ray.direction.z;

            reflection_decay.r = 1.0f;
            reflection_decay.g = 1.0f;
            reflection_decay.b = 1.0f;
        }
    }
}

__global__ void global_memory_kernel(
    const int ray_array_size,
    const int face_vertex_index_array_size,
    const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const int threaded_bvh_node_array_size,
    const int* global_light_index_array, const int light_sampling_table_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed)
{
    extern __shared__ unsigned char shared_memory[];
    int thread_id = threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    RTXObject* shared_object_array = (RTXObject*)&shared_memory[offset];
    offset += sizeof(RTXObject) / sizeof(unsigned char) * object_array_size;

    RTXMaterialAttributeByte* shared_material_attribute_byte_array = (RTXMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(RTXMaterialAttributeByte) / sizeof(unsigned char) * material_attribute_byte_array_size;

    RTXThreadedBVH* shared_threaded_bvh_array = (RTXThreadedBVH*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVH) / sizeof(unsigned char) * threaded_bvh_array_size;

    int* shared_light_index_array = (int*)&shared_memory[offset];
    offset += sizeof(unsigned char) * light_sampling_table_size;

    // RTXThreadedBVHNode* shared_threaded_bvh_node_array = (RTXThreadedBVHNode*)&shared_memory[offset];
    // offset += sizeof(RTXThreadedBVHNode) / sizeof(int) * threaded_bvh_node_array_size;

    if (thread_id == 0) {
        for (int k = 0; k < object_array_size; k++) {
            shared_object_array[k] = global_object_array[k];
        }
        for (int k = 0; k < material_attribute_byte_array_size; k++) {
            shared_material_attribute_byte_array[k] = global_material_attribute_byte_array[k];
        }
        for (int k = 0; k < threaded_bvh_array_size; k++) {
            shared_threaded_bvh_array[k] = global_threaded_bvh_array[k];
        }
        for (int k = 0; k < light_sampling_table_size; k++) {
            shared_light_index_array[k] = global_light_index_array[k];
        }
        // for (int k = 0; k < threaded_bvh_node_array_size; k++) {
        //     shared_threaded_bvh_node_array[k] = global_threaded_bvh_node_array[k];
        // }
    }
    __syncthreads();

    // if (thread_id == 0) {
    //     // for (int i = 0; i < object_array_size; i++) {
    //     //     if (shared_object_array[i] != global_object_array[i]) {
    //     //         printf("Error: shared_object_array missmatch at %d\n", i);
    //     //         return;
    //     //     }
    //     // }
    //     // for (int i = 0; i < threaded_bvh_array_size; i++) {
    //     //     if (shared_threaded_bvh_array[i] != global_threaded_bvh_array[i]) {
    //     //         printf("Error: shared_threaded_bvh_array missmatch at %d\n", i);
    //     //         return;
    //     //     }
    //     // }
    //     for (int i = 0; i < light_sampling_table_size; i++) {
    //         if (shared_light_index_array[i] != global_light_index_array[i]) {
    //             printf("Error: shared_light_index_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    // }

    const float eps = 0.0000001;
    CUDARay ray;
    // RTXRay ray;
    float3 ray_direction_inv;
    float3 hit_point;
    float3 hit_face_normal;
    RTXPixel hit_color;
    RTXPixel reflection_decay;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= ray_array_size) {
            return;
        }

        // ray = global_ray_array[ray_index];
        ray.direction = tex1Dfetch(ray_texture, ray_index * 2 + 0);
        ray.origin = tex1Dfetch(ray_texture, ray_index * 2 + 1);

        // RTXRay ray = global_ray_array[ray_index];
        // if(ray_index < 5){
        //     printf("ray: %d\n", ray_index);
        //     printf("%f == %f\n", ray.direction.x, direction.x);
        //     printf("%f == %f\n", ray.direction.y, direction.y);
        //     printf("%f == %f\n", ray.direction.z, direction.z);
        //     printf("%f == %f\n", ray.origin.x, origin.x);
        //     printf("%f == %f\n", ray.origin.y, origin.y);
        //     printf("%f == %f\n", ray.origin.z, origin.z);
        // }
        // assert(ray.direction.x == _ray.direction.x);
        // assert(ray.direction.y == _ray.direction.y);
        // assert(ray.direction.z == _ray.direction.z);
        // assert(ray.origin.x == _ray.origin.x);
        // assert(ray.origin.y == _ray.origin.y);
        // assert(ray.origin.z == _ray.origin.z);

        ray_direction_inv.x = 1.0f / ray.direction.x;
        ray_direction_inv.y = 1.0f / ray.direction.y;
        ray_direction_inv.z = 1.0f / ray.direction.z;

        reflection_decay.r = 1.0f;
        reflection_decay.g = 1.0f;
        reflection_decay.b = 1.0f;

        bool did_hit_light = false;

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;

            for (int object_index = 0; object_index < object_array_size; object_index++) {
                RTXObject object = shared_object_array[object_index];
                RTXThreadedBVH bvh = shared_threaded_bvh_array[object_index];

                int bvh_current_node_index = 0;
                for (int traversal = 0; traversal < bvh.num_nodes; traversal++) {
                    if (bvh_current_node_index == THREADED_BVH_TERMINAL_NODE) {
                        break;
                    }

                    int index = bvh.node_index_offset + bvh_current_node_index;

                    CUDAThreadedBVHNode node;
                    float4 attributes_float = tex1Dfetch(threaded_bvh_node_texture, index * 3 + 0);
                    int4* attributes_integer_ptr = reinterpret_cast<int4*>(&attributes_float);
                    node.hit_node_index = attributes_integer_ptr->x;
                    node.miss_node_index = attributes_integer_ptr->y;
                    node.assigned_face_index_start = attributes_integer_ptr->z;
                    node.assigned_face_index_end = attributes_integer_ptr->w;
                    node.aabb_max = tex1Dfetch(threaded_bvh_node_texture, index * 3 + 1);
                    node.aabb_min = tex1Dfetch(threaded_bvh_node_texture, index * 3 + 2);

                    bool is_inner_node = node.assigned_face_index_start == -1;
                    if (is_inner_node) {
                        // http://www.cs.utah.edu/~awilliam/box/box.pdf
                        float tmin = ((ray_direction_inv.x < 0 ? node.aabb_max.x : node.aabb_min.x) - ray.origin.x) * ray_direction_inv.x;
                        float tmax = ((ray_direction_inv.x < 0 ? node.aabb_min.x : node.aabb_max.x) - ray.origin.x) * ray_direction_inv.x;
                        float tmp_tmin = ((ray_direction_inv.y < 0 ? node.aabb_max.y : node.aabb_min.y) - ray.origin.y) * ray_direction_inv.y;
                        float tmp_tmax = ((ray_direction_inv.y < 0 ? node.aabb_min.y : node.aabb_max.y) - ray.origin.y) * ray_direction_inv.y;

                        if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                            bvh_current_node_index = node.miss_node_index;
                            continue;
                        }
                        if (tmp_tmin > tmin) {
                            tmin = tmp_tmin;
                        }
                        if (tmp_tmax < tmax) {
                            tmax = tmp_tmax;
                        }
                        tmp_tmin = ((ray_direction_inv.z < 0 ? node.aabb_max.z : node.aabb_min.z) - ray.origin.z) * ray_direction_inv.z;
                        tmp_tmax = ((ray_direction_inv.z < 0 ? node.aabb_min.z : node.aabb_max.z) - ray.origin.z) * ray_direction_inv.z;
                        if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                            bvh_current_node_index = node.miss_node_index;
                            continue;
                        }
                        if (tmp_tmin > tmin) {
                            tmin = tmp_tmin;
                        }
                        if (tmp_tmax < tmax) {
                            tmax = tmp_tmax;
                        }

                        if (tmax < 0.001) {
                            bvh_current_node_index = node.miss_node_index;
                            continue;
                        }
                    } else {
                        int num_assigned_faces = node.assigned_face_index_end - node.assigned_face_index_start + 1;
                        for (int m = 0; m < num_assigned_faces; m++) {
                            int index = node.assigned_face_index_start + m + object.face_index_offset;

                            int4 face = tex1Dfetch(face_vertex_index_texture, index);
                            // RTXFace face = global_face_vertex_index_array[index];

                            float4 va = tex1Dfetch(vertex_texture, face.x + object.vertex_index_offset);
                            float4 vb = tex1Dfetch(vertex_texture, face.y + object.vertex_index_offset);
                            float4 vc = tex1Dfetch(vertex_texture, face.z + object.vertex_index_offset);
                            // RTXVector4f va = global_vertex_array[face.a];
                            // RTXVector4f vb = global_vertex_array[face.b];
                            // RTXVector4f vc = global_vertex_array[face.c];

                            float3 edge_ba;
                            edge_ba.x = vb.x - va.x;
                            edge_ba.y = vb.y - va.y;
                            edge_ba.z = vb.z - va.z;

                            float3 edge_ca;
                            edge_ca.x = vc.x - va.x;
                            edge_ca.y = vc.y - va.y;
                            edge_ca.z = vc.z - va.z;

                            float3 h;
                            h.x = ray.direction.y * edge_ca.z - ray.direction.z * edge_ca.y;
                            h.y = ray.direction.z * edge_ca.x - ray.direction.x * edge_ca.z;
                            h.z = ray.direction.x * edge_ca.y - ray.direction.y * edge_ca.x;
                            float f = edge_ba.x * h.x + edge_ba.y * h.y + edge_ba.z * h.z;
                            if (f > -eps && f < eps) {
                                continue;
                            }

                            f = 1.0f / f;

                            float3 s;
                            s.x = ray.origin.x - va.x;
                            s.y = ray.origin.y - va.y;
                            s.z = ray.origin.z - va.z;
                            float dot = s.x * h.x + s.y * h.y + s.z * h.z;
                            float u = f * dot;
                            if (u < 0.0f || u > 1.0f) {
                                continue;
                            }

                            h.x = s.y * edge_ba.z - s.z * edge_ba.y;
                            h.y = s.z * edge_ba.x - s.x * edge_ba.z;
                            h.z = s.x * edge_ba.y - s.y * edge_ba.x;
                            dot = h.x * ray.direction.x + h.y * ray.direction.y + h.z * ray.direction.z;
                            float v = f * dot;
                            if (v < 0.0f || u + v > 1.0f) {
                                continue;
                            }
                            s.x = edge_ba.y * edge_ca.z - edge_ba.z * edge_ca.y;
                            s.y = edge_ba.z * edge_ca.x - edge_ba.x * edge_ca.z;
                            s.z = edge_ba.x * edge_ca.y - edge_ba.y * edge_ca.x;

                            float norm = sqrtf(s.x * s.x + s.y * s.y + s.z * s.z) + 1e-12;

                            s.x = s.x / norm;
                            s.y = s.y / norm;
                            s.z = s.z / norm;

                            dot = s.x * ray.direction.x + s.y * ray.direction.y + s.z * ray.direction.z;
                            if (dot > 0.0f) {
                                continue;
                            }

                            dot = edge_ca.x * h.x + edge_ca.y * h.y + edge_ca.z * h.z;
                            float t = f * dot;

                            if (t <= 0.001f) {
                                continue;
                            }
                            if (min_distance <= t) {
                                continue;
                            }

                            min_distance = t;
                            hit_point.x = ray.origin.x + t * ray.direction.x;
                            hit_point.y = ray.origin.y + t * ray.direction.y;
                            hit_point.z = ray.origin.z + t * ray.direction.z;

                            hit_face_normal.x = s.x;
                            hit_face_normal.y = s.y;
                            hit_face_normal.z = s.z;

                            int material_type = object.layerd_material_types.outside;
                            if (material_type == RTXMaterialTypeLambert) {
                                RTXLambertMaterialAttribute attr = ((RTXLambertMaterialAttribute*)&shared_material_attribute_byte_array[object.material_attribute_byte_array_offset])[0];
                                did_hit_light = false;
                            } else if (material_type == RTXMaterialTypeEmissive) {
                                RTXEmissiveMaterialAttribute attr = ((RTXEmissiveMaterialAttribute*)&shared_material_attribute_byte_array[object.material_attribute_byte_array_offset])[0];
                                did_hit_light = true;
                            }
                            hit_color.r = 0.9f;
                            hit_color.g = 0.9f;
                            hit_color.b = 0.9f;
                        }
                    }

                    if (node.hit_node_index == THREADED_BVH_TERMINAL_NODE) {
                        bvh_current_node_index = node.miss_node_index;
                    } else {
                        bvh_current_node_index = node.hit_node_index;
                    }
                }
            }

            if (did_hit_light) {
                reflection_decay.r *= hit_color.r;
                reflection_decay.g *= hit_color.g;
                reflection_decay.b *= hit_color.b;
                break;
            }

            ray.origin.x = hit_point.x;
            ray.origin.y = hit_point.y;
            ray.origin.z = hit_point.z;

            // diffuse reflection
            float diffuese_x = curand_normal(&state);
            float diffuese_y = curand_normal(&state);
            float diffuese_z = curand_normal(&state);
            float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
            diffuese_x /= norm;
            diffuese_y /= norm;
            diffuese_z /= norm;

            float dot = hit_face_normal.x * diffuese_x + hit_face_normal.y * diffuese_y + hit_face_normal.z * diffuese_z;
            if (dot < 0.0f) {
                diffuese_x = -diffuese_x;
                diffuese_y = -diffuese_y;
                diffuese_z = -diffuese_z;
            }
            ray.direction.x = diffuese_x;
            ray.direction.y = diffuese_y;
            ray.direction.z = diffuese_z;

            ray_direction_inv.x = 1.0f / ray.direction.x;
            ray_direction_inv.y = 1.0f / ray.direction.y;
            ray_direction_inv.z = 1.0f / ray.direction.z;

            reflection_decay.r *= hit_color.r;
            reflection_decay.g *= hit_color.g;
            reflection_decay.b *= hit_color.b;
        }

        if (did_hit_light == false) {
            reflection_decay.r = -1.0f;
            reflection_decay.g = -1.0f;
            reflection_decay.b = -1.0f;
        }

        global_render_array[ray_index] = reflection_decay;
    }
}

__global__ void shared_memory_kernel(
    const float* ray_array, const int ray_array_size,
    const int* face_vertex_index_array, const int face_vertex_index_array_size,
    const float* vertex_array, const int vertex_array_size,
    const int* object_face_count_array, const int object_face_count_array_size,
    const int* object_face_offset_array, const int object_face_offset_array_size,
    const int* object_vertex_count_array, const int object_vertex_count_array_size,
    const int* object_vertex_offset_array, const int object_vertex_offset_array_size,
    const int* object_geometry_type_array, const int object_geometry_attributes_array_size,
    const int* threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    const int* threaded_bvh_num_nodes_array, const int threaded_bvh_num_nodes_array_size,
    const int* threaded_bvh_index_offset_array, const int threaded_bvh_index_offset_array_size,
    const float* scene_threaded_bvh_aabb_array, const int threaded_bvh_aabb_array_size,
    float* render_array, const int render_array_size,
    const int num_rays,
    const int num_rays_per_thread,
    const int max_bounce)
{
}
// __global__ void _render(
//                 if (object_type == RTXObjectTypeSphereGeometry) {
//                     const float center_x = shared_face_vertices[index + 0];
//                     const float center_y = shared_face_vertices[index + 1];
//                     const float center_z = shared_face_vertices[index + 2];
//                     const float radius = shared_face_vertices[index + 4];

//                     const float oc_x = ray.origin.x - center_x;
//                     const float oc_y = ray.origin.y - center_y;
//                     const float oc_z = ray.origin.z - center_z;

//                     const float a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z;
//                     const float b = 2.0f * (ray.direction.x * oc_x + ray.direction.y * oc_y + ray.direction.z * oc_z);
//                     const float c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius;
//                     const float d = b * b - 4.0f * a * c;

//                     if (d <= 0) {
//                         continue;
//                     }
//                     const float root = sqrt(d);
//                     float t = (-b - root) / (2.0f * a);
//                     if (t <= 0.001f) {
//                         t = (-b + root) / (2.0f * a);
//                         if (t <= 0.001f) {
//                             continue;
//                         }
//                     }

//                     if (min_distance <= t) {
//                         continue;
//                     }
//                     min_distance = t;
//                     hit_point.x = ray.origin.x + t * ray.direction.x;
//                     hit_point.y = ray.origin.y + t * ray.direction.y;
//                     hit_point.z = ray.origin.z + t * ray.direction.z;

//                     float tmp_x = hit_point.x - center_x;
//                     float tmp_y = hit_point.y - center_y;
//                     float tmp_z = hit_point.z - center_z;
//                     float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

//                     hit_face_normal.x = tmp_x / norm;
//                     hit_face_normal.y = tmp_y / norm;
//                     hit_face_normal.z = tmp_z / norm;

//                     material_type = shared_material_types[face_index];

//                     hit_color_r = shared_face_colors[face_index * colors_stride + 0];
//                     hit_color_g = shared_face_colors[face_index * colors_stride + 1];
//                     hit_color_b = shared_face_colors[face_index * colors_stride + 2];

//                     did_hit_object = true;
//                     continue;
//                 }

void rtx_cuda_malloc(void** gpu_array, size_t size)
{
    assert(size > 0);
    cudaError_t error = cudaMalloc(gpu_array, size);
    // printf("malloc %p\n", *gpu_array);
    cudaError_t status = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMalloc: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_memcpy_host_to_device(void* gpu_array, void* cpu_array, size_t size)
{
    cudaError_t error = cudaMemcpy(gpu_array, cpu_array, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMemcpyHostToDevice: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_memcpy_device_to_host(void* cpu_array, void* gpu_array, size_t size)
{
    cudaError_t error = cudaMemcpy(cpu_array, gpu_array, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString(error));
    }
}
void rtx_cuda_free(void** array)
{
    if (*array != NULL) {
        // printf("free %p\n", *array);
        cudaError_t error = cudaFree(*array);
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error at cudaFree: %s\n", cudaGetErrorString(error));
        }
        *array = NULL;
    }
}
void cuda_device_reset()
{
    cudaDeviceReset();
}
void rtx_cuda_render(
    RTXRay*& gpu_ray_array, const int ray_array_size,
    RTXFace*& gpu_face_vertex_index_array, const int face_vertex_index_array_size,
    RTXVertex*& gpu_vertex_array, const int vertex_array_size,
    RTXObject*& gpu_object_array, const int object_array_size,
    RTXMaterialAttributeByte*& gpu_material_attribute_byte_array, const int material_attribute_byte_array_size,
    RTXThreadedBVH*& gpu_threaded_bvh_array, const int threaded_bvh_array_size,
    RTXThreadedBVHNode*& gpu_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    int*& gpu_light_sampling_table, const int light_sampling_table_size,
    RTXPixel*& gpu_render_array, const int render_array_size,
    const int num_threads,
    const int num_blocks,
    const int num_rays_per_pixel,
    const int max_bounce,
    const int curand_seed)
{
    assert(gpu_ray_array != NULL);
    assert(gpu_face_vertex_index_array != NULL);
    assert(gpu_vertex_array != NULL);
    assert(gpu_object_array != NULL);
    assert(gpu_material_attribute_byte_array != NULL);
    assert(gpu_threaded_bvh_array != NULL);
    assert(gpu_threaded_bvh_node_array != NULL);
    if (light_sampling_table_size > 0) {
        assert(gpu_light_sampling_table != NULL);
    }
    assert(gpu_render_array != NULL);

    int num_rays = ray_array_size;

    // int num_blocks = (num_rays - 1) / num_threads + 1;

    int num_rays_per_thread = num_rays / (num_threads * num_blocks) + 1;

    long required_shared_memory_bytes = sizeof(RTXFace) * face_vertex_index_array_size + sizeof(RTXVertex) * vertex_array_size + sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) + threaded_bvh_array_size * sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size + sizeof(int) * light_sampling_table_size;

    // num_blocks = 1;
    // num_rays_per_thread = 1;

    printf("bytes: %u\n", sizeof(curandStateXORWOW_t));

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);

    printf("shared memory: %ld bytes\n", required_shared_memory_bytes);
    printf("available: %d bytes\n", dev.sharedMemPerBlock);
    printf("rays: %d\n", ray_array_size);

    if (required_shared_memory_bytes > dev.sharedMemPerBlock) {
        // int required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size;
        long required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(int) * light_sampling_table_size;
        printf("    shared memory: %ld bytes\n", required_shared_memory_bytes);
        printf("    available: %d bytes\n", dev.sharedMemPerBlock);
        printf("    num_blocks: %d num_threads: %d\n", num_blocks, num_threads);

        if (required_shared_memory_bytes > dev.sharedMemPerBlock) {
            long required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(int) * light_sampling_table_size;
            printf("        shared memory: %ld bytes\n", required_shared_memory_bytes);
            printf("        available: %d bytes\n", dev.sharedMemPerBlock);
            printf("        num_blocks: %d num_threads: %d\n", num_blocks, num_threads);
            assert(required_shared_memory_bytes <= dev.sharedMemPerBlock);
        } else {
            cudaBindTexture(0, ray_texture, gpu_ray_array, cudaCreateChannelDesc<float4>(), sizeof(RTXRay) * ray_array_size);
            cudaBindTexture(0, face_vertex_index_texture, gpu_face_vertex_index_array, cudaCreateChannelDesc<int4>(), sizeof(RTXFace) * face_vertex_index_array_size);
            cudaBindTexture(0, vertex_texture, gpu_vertex_array, cudaCreateChannelDesc<float4>(), sizeof(RTXVertex) * vertex_array_size);
            cudaBindTexture(0, threaded_bvh_node_texture, gpu_threaded_bvh_node_array, cudaCreateChannelDesc<float4>(), sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size);

            global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
                ray_array_size,
                face_vertex_index_array_size,
                vertex_array_size,
                gpu_object_array, object_array_size,
                gpu_material_attribute_byte_array, material_attribute_byte_array_size,
                gpu_threaded_bvh_array, threaded_bvh_array_size,
                threaded_bvh_node_array_size,
                gpu_light_sampling_table, light_sampling_table_size,
                gpu_render_array,
                num_rays_per_thread,
                max_bounce,
                curand_seed);

            cudaUnbindTexture(ray_texture);
            cudaUnbindTexture(face_vertex_index_texture);
            cudaUnbindTexture(vertex_texture);
            cudaUnbindTexture(threaded_bvh_node_texture);
        }

    } else {
        cudaBindTexture(0, ray_texture, gpu_ray_array, cudaCreateChannelDesc<float4>(), sizeof(RTXRay) * ray_array_size);
        cudaBindTexture(0, face_vertex_index_texture, gpu_face_vertex_index_array, cudaCreateChannelDesc<int4>(), sizeof(RTXFace) * face_vertex_index_array_size);
        cudaBindTexture(0, vertex_texture, gpu_vertex_array, cudaCreateChannelDesc<float4>(), sizeof(RTXVertex) * vertex_array_size);
        cudaBindTexture(0, threaded_bvh_node_texture, gpu_threaded_bvh_node_array, cudaCreateChannelDesc<float4>(), sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size);

        global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
            ray_array_size,
            face_vertex_index_array_size,
            vertex_array_size,
            gpu_object_array, object_array_size,
            gpu_material_attribute_byte_array, material_attribute_byte_array_size,
            gpu_threaded_bvh_array, threaded_bvh_array_size,
            threaded_bvh_node_array_size,
            gpu_light_sampling_table, light_sampling_table_size,
            gpu_render_array,
            num_rays_per_thread,
            max_bounce,
            curand_seed);

        cudaUnbindTexture(ray_texture);
        cudaUnbindTexture(face_vertex_index_texture);
        cudaUnbindTexture(vertex_texture);
        cudaUnbindTexture(threaded_bvh_node_texture);
    }

    // num_rays_per_thread = 1;

    // printf("rays: %d, rays_per_kernel: %d, num_rays_per_thread: %d\n", num_rays, num_rays_per_kernel, num_rays_per_thread);
    // printf("<<<%d, %d>>>\n", num_blocks, num_threads);
    cudaError_t status = cudaGetLastError();
    if (status != 0) {
        fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
    }
    cudaError_t error = cudaThreadSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
    }

    // cudaDeviceProp dev;
    // cudaGetDeviceProperties(&dev, 0);

    // printf(" device name : %s\n", dev.name);
    // printf(" total global memory : %d (MB)\n", dev.totalGlobalMem/1024/1024);
    // printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);
    // printf(" register / block : %d\n", dev.regsPerBlock);
}

// __global__ void test_linear_kernel(
//     const int* node_array, const int num_nodes)
// {
//     int s = 0;
//     for (int j = 0; j < 1000; j++) {
//         for (int i = 0; i < num_nodes; i++) {
//             int hit = node_array[i * 4 + 0];
//             int miss = node_array[i * 4 + 1];
//             int start = node_array[i * 4 + 2];
//             int end = node_array[i * 4 + 3];
//             s += hit + miss + start + end;
//         }
//     }
// }

// __global__ void test_struct_kernel(
//     const RTXThreadedBVHNode* node_array, const int num_nodes)
// {
//     int s = 0;
//     for (int j = 0; j < 1000; j++) {
//         for (int i = 0; i < num_nodes; i++) {
//             RTXThreadedBVHNode node = node_array[i];
//             s += node.hit_node_index + node.miss_node_index + node.assigned_face_index_start + node.assigned_face_index_end;
//         }
//     }
// }

// void launch_test_linear_kernel(
//     int*& gpu_node_array, const int num_nodes)
// {
//     test_linear_kernel<<<256, 32>>>(gpu_node_array, num_nodes);
//     cudaError_t status = cudaGetLastError();
//     if (status != 0) {
//         fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
//     }
//     cudaError_t error = cudaThreadSynchronize();
//     if (error != cudaSuccess) {
//         fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
//     }
// }

// void launch_test_struct_kernel(
//     RTXThreadedBVHNode*& gpu_struct_array, const int num_nodes)
// {

//     test_struct_kernel<<<256, 32>>>(gpu_struct_array, num_nodes);
//     cudaError_t status = cudaGetLastError();
//     if (status != 0) {
//         fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
//     }
//     cudaError_t error = cudaThreadSynchronize();
//     if (error != cudaSuccess) {
//         fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
//     }
// }