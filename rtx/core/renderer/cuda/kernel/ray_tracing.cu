#include "../../../header/enum.h"
#include "../header/ray_tracing.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

#define THREADED_BVH_TERMINAL_NODE 8191
#define THREADED_BVH_INNER_NODE 63

__global__ void bvh_kernel(
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
    extern __shared__ int shared_memory[];
    int thread_id = threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(0, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    int* shared_face_vertex_index_array = &shared_memory[offset];
    offset += face_vertex_index_array_size;
    float* shared_vertex_array = (float*)&shared_memory[offset];
    offset += vertex_array_size;
    int* shared_object_face_count_array = &shared_memory[offset];
    offset += object_face_count_array_size;
    int* shared_object_face_offset_array = &shared_memory[offset];
    offset += object_face_offset_array_size;
    int* shared_object_vertex_count_array = &shared_memory[offset];
    offset += object_vertex_count_array_size;
    int* shared_object_vertex_offset_array = &shared_memory[offset];
    offset += object_vertex_offset_array_size;
    int* shared_object_geometry_attributes_array = &shared_memory[offset];
    offset += object_geometry_attributes_array_size;
    int* shared_threaded_bvh_node_array = (int*)&shared_memory[offset];
    offset += threaded_bvh_node_array_size;
    float* shared_threaded_bvh_aabb_array = (float*)&shared_memory[offset];
    offset += threaded_bvh_aabb_array_size;

    if (thread_id == 0) {
        for (int i = 0; i < face_vertex_index_array_size; i++) {
            shared_face_vertex_index_array[i] = face_vertex_index_array[i];
        }
        for (int i = 0; i < vertex_array_size; i++) {
            shared_vertex_array[i] = vertex_array[i];
        }
        for (int i = 0; i < object_face_count_array_size; i++) {
            shared_object_face_count_array[i] = object_face_count_array[i];
        }
        for (int i = 0; i < object_face_offset_array_size; i++) {
            shared_object_face_offset_array[i] = object_face_offset_array[i];
        }
        for (int i = 0; i < object_vertex_count_array_size; i++) {
            shared_object_vertex_count_array[i] = object_vertex_count_array[i];
        }
        for (int i = 0; i < object_vertex_offset_array_size; i++) {
            shared_object_vertex_offset_array[i] = object_vertex_offset_array[i];
        }
        for (int i = 0; i < object_geometry_attributes_array_size; i++) {
            shared_object_geometry_attributes_array[i] = object_geometry_type_array[i];
        }
        for (int i = 0; i < threaded_bvh_node_array_size; i++) {
            shared_threaded_bvh_node_array[i] = threaded_bvh_node_array[i];
        }
        for (int i = 0; i < threaded_bvh_aabb_array_size; i++) {
            shared_threaded_bvh_aabb_array[i] = scene_threaded_bvh_aabb_array[i];
        }
    }
    __syncthreads();

    // if (thread_id == 0) {
    //     for (int i = 0; i < face_vertex_index_array_size; i++) {
    //         if (shared_face_vertex_index_array[i] != face_vertex_index_array[i]) {
    //             printf("Error: shared_face_vertex_index_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < vertex_array_size; i++) {
    //         if (shared_vertex_array[i] != vertex_array[i]) {
    //             printf("Error: shared_vertex_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_face_count_array_size; i++) {
    //         if (shared_object_face_count_array[i] != object_face_count_array[i]) {
    //             printf("Error: shared_object_face_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_face_offset_array_size; i++) {
    //         if (shared_object_face_offset_array[i] != object_face_offset_array[i]) {
    //             printf("Error: shared_object_face_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_count_array_size; i++) {
    //         if (shared_object_vertex_count_array[i] != object_vertex_count_array[i]) {
    //             printf("Error: shared_object_vertex_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_offset_array_size; i++) {
    //         if (shared_object_vertex_offset_array[i] != object_vertex_offset_array[i]) {
    //             printf("Error: shared_object_vertex_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_node_array_size; i++) {
    //         if (shared_threaded_bvh_node_array[i] != threaded_bvh_node_array[i]) {
    //             printf("Error: shared_threaded_bvh_node_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_aabb_array_size; i++) {
    //         if (shared_threaded_bvh_aabb_array[i] != scene_threaded_bvh_aabb_array[i]) {
    //             printf("Error: shared_threaded_bvh_aabb_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    // }

    // bool did_hit_object;
    float ray_direction_x;
    float ray_direction_y;
    float ray_direction_z;
    float ray_origin_x;
    float ray_origin_y;
    float ray_origin_z;
    float ray_direction_inv_x;
    float ray_direction_inv_y;
    float ray_direction_inv_z;
    // int bvh_binary_node;
    // int bvh_object_index;
    // int bvh_current_node_id;
    // int bvh_hit_node_id;
    // int bvh_miss_node_id;
    // float aabb_max_x;
    // float aabb_max_y;
    // float aabb_max_z;
    // float aabb_min_x;
    // float aabb_min_y;
    // float aabb_min_z;
    // float tmin, tmax, tmp_tmin, tmp_tmax;
    // float min_distance;
    float hit_point_x;
    float hit_point_y;
    float hit_point_z;
    float hit_color_r;
    float hit_color_g;
    float hit_color_b;
    float hit_face_normal_x;
    float hit_face_normal_y;
    float hit_face_normal_z;
    float reflection_decay_r;
    float reflection_decay_g;
    float reflection_decay_b;
    // float va_x, va_y, va_z;
    // float vb_x, vb_y, vb_z;
    // float vc_x, vc_y, vc_z;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= num_rays) {
            return;
        }

        int array_index = ray_index * 8;
        ray_direction_x = ray_array[array_index + 0];
        ray_direction_y = ray_array[array_index + 1];
        ray_direction_z = ray_array[array_index + 2];
        ray_origin_x = ray_array[array_index + 4];
        ray_origin_y = ray_array[array_index + 5];
        ray_origin_z = ray_array[array_index + 6];
        ray_direction_inv_x = 1.0f / ray_direction_x;
        ray_direction_inv_y = 1.0f / ray_direction_y;
        ray_direction_inv_z = 1.0f / ray_direction_z;

        reflection_decay_r = 1.0f;
        reflection_decay_g = 1.0f;
        reflection_decay_b = 1.0f;

        bool did_hit_light = false;

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            // BVH traversal
            int bvh_current_node_id = 0;
            for (int traversal = 0; traversal < threaded_bvh_node_array_size; traversal++) {

                if (bvh_current_node_id == THREADED_BVH_TERMINAL_NODE) {
                    break;
                }
                assert(bvh_current_node_id < threaded_bvh_node_array_size);

                int bvh_binary_node = shared_threaded_bvh_node_array[bvh_current_node_id];
                int bvh_object_index = 0xFF & bvh_binary_node;
                int bvh_miss_node_id = 0xFF & (bvh_binary_node >> 8);
                int bvh_hit_node_id = 0xFF & (bvh_binary_node >> 16);

                if (bvh_object_index == THREADED_BVH_INNER_NODE) {
                    assert(bvh_current_node_id * 8 + 0 < threaded_bvh_aabb_array_size);
                    assert(bvh_current_node_id * 8 + 7 < threaded_bvh_aabb_array_size);
                    float aabb_max_x = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 0];
                    float aabb_max_y = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 1];
                    float aabb_max_z = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 2];
                    float aabb_min_x = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 4];
                    float aabb_min_y = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 5];
                    float aabb_min_z = shared_threaded_bvh_aabb_array[bvh_current_node_id * 8 + 6];

                    // http://www.cs.utah.edu/~awilliam/box/box.pdf
                    float tmin = ((ray_direction_inv_x < 0 ? aabb_max_x : aabb_min_x) - ray_origin_x) * ray_direction_inv_x;
                    float tmax = ((ray_direction_inv_x < 0 ? aabb_min_x : aabb_max_x) - ray_origin_x) * ray_direction_inv_x;
                    float tmp_tmin = ((ray_direction_inv_y < 0 ? aabb_max_y : aabb_min_y) - ray_origin_y) * ray_direction_inv_y;
                    float tmp_tmax = ((ray_direction_inv_y < 0 ? aabb_min_y : aabb_max_y) - ray_origin_y) * ray_direction_inv_y;

                    if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                        bvh_current_node_id = bvh_miss_node_id;
                        continue;
                    }
                    if (tmp_tmin > tmin) {
                        tmin = tmp_tmin;
                    }
                    if (tmp_tmax < tmax) {
                        tmax = tmp_tmax;
                    }
                    tmp_tmin = ((ray_direction_inv_z < 0 ? aabb_max_z : aabb_min_z) - ray_origin_z) * ray_direction_inv_z;
                    // tmp_tmax = ((ray_direction_inv_z < 0 ? aabb_min_z : aabb_max_z) - ray_origin_z) * ray_direction_inv_z;
                    // if (bvh_object_index == 3 && ray_index == 10 * 32 + 5) {
                    //     printf("+traversal: %d max: %f %f dir: %f inv: %f tmax: %f tmin: %f ttmax: %f ttmin: %f \n", traversal, aabb_max_z, aabb_min_z, ray_direction_z, ray_direction_inv_z, tmax, tmin, tmp_tmax, tmp_tmin);
                    //     printf("AABB(max): (%f, %f, %f) min: (%f, %f, %f)\n", aabb_max_x, aabb_max_y, aabb_max_z, aabb_min_x, aabb_min_y, aabb_min_z);
                    //     printf("ray: (%f, %f, %f)\n", ray_direction_x, ray_direction_y, ray_direction_z);
                    //     printf("object: %d\n", bvh_object_index);
                    //     reflection_decay_r = 0.0f;
                    //     reflection_decay_g = 0.0f;
                    //     reflection_decay_b = 0.0f;
                    // }
                    // if (ray_index == 20 * 32 + 5) {
                    //     printf("-traversal: %d max: %f %f dir: %f inv: %f tmax: %f tmin: %f ttmax: %f ttmin: %f \n", traversal, aabb_max_z, aabb_min_z, ray_direction_z, ray_direction_inv_z, tmax, tmin, tmp_tmax, tmp_tmin);
                    //     printf("AABB(max): (%f, %f, %f) min: (%f, %f, %f)\n", aabb_max_x, aabb_max_y, aabb_max_z, aabb_min_x, aabb_min_y, aabb_min_z);
                    //     printf("ray: (%f, %f, %f)\n", ray_direction_x, ray_direction_y, ray_direction_z);
                    //     reflection_decay_r = 0.0f;
                    //     reflection_decay_g = 0.0f;
                    //     reflection_decay_b = 0.0f;
                    // }
                    if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                        bvh_current_node_id = bvh_miss_node_id;
                        continue;
                    }
                    if (tmp_tmin > tmin) {
                        tmin = tmp_tmin;
                    }
                    if (tmp_tmax < tmax) {
                        tmax = tmp_tmax;
                    }

                    if (tmax < 0.001) {
                        bvh_current_node_id = bvh_miss_node_id;
                        continue;
                    }

                    // if (thread_id == 0) {
                    //     printf("index: %u hit: %u miss: %u object: %u binary: %u\n", bvh_current_node_id,
                    //         bvh_hit_node_id, bvh_miss_node_id, bvh_object_index, bvh_binary_node);
                    //     // printf("node: %u %u\n", bvh_current_node_id, bvh_binary_node);
                    //     // printf("object: %u\n", bvh_object_index);
                    //     // printf("miss: %u\n", bvh_miss_node_id);
                    //     // printf("hit: %u\n", bvh_hit_node_id);
                    //     // printf("AAAB(max): %f %f %f\n", aabb_max_x, aabb_max_y, aabb_max_z);
                    //     // printf("AAAB(min): %f %f %f\n", aabb_min_x, aabb_min_y, aabb_min_z);
                    //     // printf("t: %f %f\n", tmin, tmax);
                    //     // printf("next: %u\n", bvh_current_node_id);
                    // }

                    bvh_current_node_id = bvh_hit_node_id;
                } else {
                    int faces_count = shared_object_face_count_array[bvh_object_index];
                    int face_index_offset = shared_object_face_offset_array[bvh_object_index];
                    int vertices_count = shared_object_vertex_count_array[bvh_object_index];
                    int vertex_index_offset = shared_object_vertex_offset_array[bvh_object_index];
                    int geometry_type = shared_object_geometry_attributes_array[bvh_object_index];
                    // if (thread_id == 0) {
                    //     printf("object: %u count: %d offset: %d\n", bvh_object_index, vertices_count, vertex_index_offset);
                    // }

                    if (geometry_type == RTX_GEOMETRY_TYPE_STANDARD) {

                        for (int j = 0; j < faces_count; j++) {
                            int array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 0];
                            // if (thread_id == 0) {
                            //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                            // }
                            assert(array_index + 0 < vertex_array_size);
                            float va_x = shared_vertex_array[array_index + 0];
                            float va_y = shared_vertex_array[array_index + 1];
                            float va_z = shared_vertex_array[array_index + 2];

                            // if (thread_id == 0) {
                            //     printf("va: (%f, %f, %f)\n", va_x, va_y, va_z);
                            // }

                            array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 1];
                            // if (thread_id == 0) {
                            //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                            // }
                            assert(array_index + 0 < vertex_array_size);
                            float vb_x = shared_vertex_array[array_index + 0];
                            float vb_y = shared_vertex_array[array_index + 1];
                            float vb_z = shared_vertex_array[array_index + 2];

                            // if (thread_id == 0) {
                            //     printf("vb: (%f, %f, %f)\n", vb_x, vb_y, vb_z);
                            // }

                            array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 2];
                            // if (thread_id == 0) {
                            //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                            // }
                            assert(array_index + 0 < vertex_array_size);
                            float vc_x = shared_vertex_array[array_index + 0];
                            float vc_y = shared_vertex_array[array_index + 1];
                            float vc_z = shared_vertex_array[array_index + 2];

                            // if (thread_id == 0) {
                            //     printf("vc: (%f, %f, %f)\n", vc_x, vc_y, vc_z);
                            // }

                            float edge_ba_x = vb_x - va_x;
                            float edge_ba_y = vb_y - va_y;
                            float edge_ba_z = vb_z - va_z;

                            float edge_ca_x = vc_x - va_x;
                            float edge_ca_y = vc_y - va_y;
                            float edge_ca_z = vc_z - va_z;

                            float h_x = ray_direction_y * edge_ca_z - ray_direction_z * edge_ca_y;
                            float h_y = ray_direction_z * edge_ca_x - ray_direction_x * edge_ca_z;
                            float h_z = ray_direction_x * edge_ca_y - ray_direction_y * edge_ca_x;
                            float a = edge_ba_x * h_x + edge_ba_y * h_y + edge_ba_z * h_z;
                            if (a > -0.0000001 && a < 0.0000001) {
                                continue;
                            }
                            float f = 1.0f / a;

                            float s_x = ray_origin_x - va_x;
                            float s_y = ray_origin_y - va_y;
                            float s_z = ray_origin_z - va_z;
                            float dot = s_x * h_x + s_y * h_y + s_z * h_z;
                            float u = f * dot;
                            if (u < 0.0f || u > 1.0f) {
                                continue;
                            }
                            float q_x = s_y * edge_ba_z - s_z * edge_ba_y;
                            float q_y = s_z * edge_ba_x - s_x * edge_ba_z;
                            float q_z = s_x * edge_ba_y - s_y * edge_ba_x;
                            dot = q_x * ray_direction_x + q_y * ray_direction_y + q_z * ray_direction_z;
                            float v = f * dot;
                            if (v < 0.0f || u + v > 1.0f) {
                                continue;
                            }
                            float tmp_x = edge_ba_y * edge_ca_z - edge_ba_z * edge_ca_y;
                            float tmp_y = edge_ba_z * edge_ca_x - edge_ba_x * edge_ca_z;
                            float tmp_z = edge_ba_x * edge_ca_y - edge_ba_y * edge_ca_x;

                            float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                            tmp_x = tmp_x / norm;
                            tmp_y = tmp_y / norm;
                            tmp_z = tmp_z / norm;

                            dot = tmp_x * ray_direction_x + tmp_y * ray_direction_y + tmp_z * ray_direction_z;
                            if (dot > 0.0f) {
                                continue;
                            }

                            dot = edge_ca_x * q_x + edge_ca_y * q_y + edge_ca_z * q_z;
                            float t = f * dot;

                            if (t <= 0.001f) {
                                continue;
                            }
                            if (min_distance <= t) {
                                continue;
                            }

                            min_distance = t;
                            hit_point_x = ray_origin_x + t * ray_direction_x;
                            hit_point_y = ray_origin_y + t * ray_direction_y;
                            hit_point_z = ray_origin_z + t * ray_direction_z;

                            hit_face_normal_x = tmp_x;
                            hit_face_normal_y = tmp_y;
                            hit_face_normal_z = tmp_z;

                            hit_color_r = 0.8f;
                            hit_color_g = 0.8f;
                            hit_color_b = 0.8f;

                            if (bvh_object_index == 2) {
                                hit_color_r = 1.0f;
                                hit_color_g = 1.0f;
                                hit_color_b = 1.0f;
                                did_hit_light = true;
                                continue;
                            }

                            did_hit_object = true;
                            did_hit_light = false;
                        }
                    } else if (geometry_type == RTX_GEOMETRY_TYPE_SPHERE) {
                        assert(faces_count == 1);
                        for (int j = 0; j < faces_count; j++) {
                            int array_index = 4 * shared_face_vertex_index_array[face_index_offset * 4 + 0];
                            assert(array_index + 0 < vertex_array_size);

                            float center_x = shared_vertex_array[array_index + 0];
                            float center_y = shared_vertex_array[array_index + 1];
                            float center_z = shared_vertex_array[array_index + 2];
                            float radius = shared_vertex_array[array_index + 4];

                            float oc_x = ray_origin_x - center_x;
                            float oc_y = ray_origin_y - center_y;
                            float oc_z = ray_origin_z - center_z;

                            float a = ray_direction_x * ray_direction_x + ray_direction_y * ray_direction_y + ray_direction_z * ray_direction_z;
                            float b = 2.0f * (ray_direction_x * oc_x + ray_direction_y * oc_y + ray_direction_z * oc_z);
                            float c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius;
                            float d = b * b - 4.0f * a * c;

                            if (d <= 0) {
                                continue;
                            }
                            float root = sqrt(d);
                            float t = (-b - root) / (2.0f * a);
                            if (t <= 0.001f) {
                                t = (-b + root) / (2.0f * a);
                                if (t <= 0.001f) {
                                    continue;
                                }
                            }

                            if (min_distance <= t) {
                                continue;
                            }

                            min_distance = t;
                            hit_point_x = ray_origin_x + t * ray_direction_x;
                            hit_point_y = ray_origin_y + t * ray_direction_y;
                            hit_point_z = ray_origin_z + t * ray_direction_z;

                            float tmp_x = hit_point_x - center_x;
                            float tmp_y = hit_point_y - center_y;
                            float tmp_z = hit_point_z - center_z;
                            float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                            hit_face_normal_x = tmp_x / norm;
                            hit_face_normal_y = tmp_y / norm;
                            hit_face_normal_z = tmp_z / norm;

                            hit_color_r = 0.8f;
                            hit_color_g = 0.8f;
                            hit_color_b = 0.8f;

                            did_hit_object = true;
                            did_hit_light = false;
                        }
                    }

                    if (bvh_hit_node_id == THREADED_BVH_TERMINAL_NODE) {
                        bvh_current_node_id = bvh_miss_node_id;
                    } else {
                        bvh_current_node_id = bvh_hit_node_id;
                    }
                }
            }

            if (did_hit_light) {
                reflection_decay_r *= hit_color_r;
                reflection_decay_g *= hit_color_g;
                reflection_decay_b *= hit_color_b;
                break;
            }

            if (did_hit_object) {
                ray_origin_x = hit_point_x;
                ray_origin_y = hit_point_y;
                ray_origin_z = hit_point_z;

                // diffuse reflection
                float diffuese_x = curand_normal(&state);
                float diffuese_y = curand_normal(&state);
                float diffuese_z = curand_normal(&state);
                float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
                diffuese_x /= norm;
                diffuese_y /= norm;
                diffuese_z /= norm;

                float dot = hit_face_normal_x * diffuese_x + hit_face_normal_y * diffuese_y + hit_face_normal_z * diffuese_z;
                if (dot < 0.0f) {
                    diffuese_x = -diffuese_x;
                    diffuese_y = -diffuese_y;
                    diffuese_z = -diffuese_z;
                }
                ray_direction_x = diffuese_x;
                ray_direction_y = diffuese_y;
                ray_direction_z = diffuese_z;

                ray_direction_inv_x = 1.0f / ray_direction_x;
                ray_direction_inv_y = 1.0f / ray_direction_y;
                ray_direction_inv_z = 1.0f / ray_direction_z;

                reflection_decay_r *= hit_color_r;
                reflection_decay_g *= hit_color_g;
                reflection_decay_b *= hit_color_b;
            }
        }

        if (did_hit_light == false) {
            reflection_decay_r = 0.0f;
            reflection_decay_g = 0.0f;
            reflection_decay_b = 0.0f;
        }

        assert(ray_index * 4 + 2 < render_array_size);
        render_array[ray_index * 4 + 0] = reflection_decay_r;
        render_array[ray_index * 4 + 1] = reflection_decay_g;
        render_array[ray_index * 4 + 2] = reflection_decay_b;

        // render_array[ray_index * 3 + 0] = (ray_direction_x + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 1] = (ray_direction_y + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 2] = (ray_direction_z + 1.0f) / 2.0f;
    }
}
__global__ void global_memory_kernel(
    RTXRay*& global_ray_array, const int ray_array_size,
    RTXGeometryFace*& global_face_vertex_index_array, const int face_vertex_index_array_size,
    RTXGeometryVertex*& global_vertex_array, const int vertex_array_size,
    RTXObject*& global_object_array, const int object_array_size,
    RTXThreadedBVH*& global_threaded_bvh_array, const int threaded_bvh_array_size,
    RTXThreadedBVHNode*& global_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    RTXPixel*& global_render_array, const int render_array_size,
    const int num_rays_per_thread,
    const int max_bounce)
{
    extern __shared__ int shared_memory[];
    int thread_id = threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(0, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    RTXObject* shared_object_array = (RTXObject*)&shared_memory[offset];
    offset += sizeof(RTXObject) / sizeof(int) * object_array_size;

    RTXThreadedBVH* shared_threaded_bvh_array = (RTXThreadedBVH*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVH) / sizeof(int) * threaded_bvh_array_size;

    RTXThreadedBVHNode* shared_threaded_bvh_node_array = (RTXThreadedBVHNode*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVHNode) / sizeof(int) * threaded_bvh_node_array_size;

    if (thread_id == 0) {
        printf("%p\n", global_object_array);
    }
    return;

    if (thread_id == 0) {
        for (int k = 0; k < object_array_size; k++) {
            // RTXObject obj = global_object_array[k];
            // RTXObject _obj = shared_object_array[k];
            // if (thread_id == 0) {
            //     printf("global num_faces: %d face_index_offset: %d num_vertices:%d vertex_index_offset: %d bvh_enabled: %d bvh_index: %d \n", obj.num_faces, obj.face_index_offset, obj.num_vertices, obj.vertex_index_offset, obj.bvh_enabled, obj.bvh_index);
            //     printf("shared num_faces: %d face_index_offset: %d num_vertices:%d vertex_index_offset: %d bvh_enabled: %d bvh_index: %d \n", _obj.num_faces, _obj.face_index_offset, _obj.num_vertices, _obj.vertex_index_offset, _obj.bvh_enabled, _obj.bvh_index);
            // }
            shared_object_array[k] = global_object_array[k];
        }
        for (int k = 0; k < threaded_bvh_array_size; k++) {
            shared_threaded_bvh_array[k] = global_threaded_bvh_array[k];
        }
        for (int k = 0; k < threaded_bvh_node_array_size; k++) {
            shared_threaded_bvh_node_array[k] = global_threaded_bvh_node_array[k];
        }
    }
    __syncthreads();
    return;

    // if (thread_id == 0) {
    //     for (int i = 0; i < object_face_count_array_size; i++) {
    //         if (shared_object_face_count_array[i] != object_face_count_array[i]) {
    //             printf("Error: shared_object_face_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_face_offset_array_size; i++) {
    //         if (shared_object_face_offset_array[i] != object_face_offset_array[i]) {
    //             printf("Error: shared_object_face_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_count_array_size; i++) {
    //         if (shared_object_vertex_count_array[i] != object_vertex_count_array[i]) {
    //             printf("Error: shared_object_vertex_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_offset_array_size; i++) {
    //         if (shared_object_vertex_offset_array[i] != object_vertex_offset_array[i]) {
    //             printf("Error: shared_object_vertex_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_node_array_size; i++) {
    //         if (shared_threaded_bvh_node_array[i] != threaded_bvh_node_array[i]) {
    //             printf("Error: shared_threaded_bvh_node_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_num_nodes_array_size; i++) {
    //         if (shared_threaded_bvh_num_nodes_array[i] != threaded_bvh_num_nodes_array[i]) {
    //             printf("Error: shared_threaded_bvh_num_nodes_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_index_offset_array_size; i++) {
    //         if (shared_threaded_bvh_index_offset_array[i] != threaded_bvh_index_offset_array[i]) {
    //             printf("Error: shared_threaded_bvh_index_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_aabb_array_size; i++) {
    //         if (shared_threaded_bvh_aabb_array[i] != scene_threaded_bvh_aabb_array[i]) {
    //             printf("Error: shared_threaded_bvh_aabb_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    // }

    const float eps = 0.0000001;
    RTXRay ray;
    RTXVector3f ray_direction_inv;
    RTXVector3f hit_point;
    RTXVector3f hit_face_normal;
    RTXPixel hit_color;
    RTXPixel reflection_decay;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= ray_array_size) {
            return;
        }

        int array_index = ray_index * 8;
        ray = global_ray_array[array_index];
        ray_direction_inv.x = 1.0f / ray.direction.x;
        ray_direction_inv.y = 1.0f / ray.direction.y;
        ray_direction_inv.z = 1.0f / ray.direction.z;

        reflection_decay.r = 1.0f;
        reflection_decay.g = 1.0f;
        reflection_decay.b = 1.0f;

        bool did_hit_light = false;

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            for (int object_index = 0; object_index < object_array_size; object_index++) {
                RTXObject object = shared_object_array[object_index];

                // if (thread_id == 0) {
                //     printf("object: %d\n", object_index);
                //     printf("face: ");
                //     for (int i = face_index_offset; i < face_index_offset + faces_count; i++) {
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 0]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 1]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 2]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 3]);
                //     }
                //     printf("\n");
                //     printf("vertex: ");
                //     for (int i = 0; i < faces_count; i++) {
                //         int array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 0];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //         array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 1];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //         array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 2];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //     }
                //     printf("\n");
                // }
                if (object.bvh_enabled) {
                    RTXThreadedBVH bvh = shared_threaded_bvh_array[object.bvh_index];
                    // BVH traversal

                    for (int traversal = 0; traversal < bvh.num_nodes; traversal++) {
                        RTXThreadedBVHNode node = shared_threaded_bvh_node_array[bvh.node_index_offset + traversal];
                        if (thread_id == 0) {
                            printf("traversal: %d hit: %d miss: %d start: %d end: %d \n", traversal, node.hit_node_index, node.miss_node_index, node.assigned_face_index_start, node.assigned_face_index_end);
                        }
                    }

                    // int bvh_current_node_id = 0;
                    // for (int traversal = 0; traversal < num_nodes; traversal++) {
                    //     if (bvh_current_node_id == THREADED_BVH_TERMINAL_NODE) {
                    //         break;
                    //     }
                    //     int array_index = bvh_current_node_id + bvh_node_offset;
                    //     assert(array_index < threaded_bvh_node_array_size);

                    //     int bvh_binary_node = shared_threaded_bvh_node_array[array_index];
                    //     int bvh_object_index = 0x3F & bvh_binary_node;
                    //     int bvh_miss_node_id = 0x1FFF & (bvh_binary_node >> 6);
                    //     int bvh_hit_node_id = 0x1FFF & (bvh_binary_node >> 19);

                    //     if (bvh_object_index == THREADED_BVH_INNER_NODE) {
                    //         assert(array_index * 8 + 0 < threaded_bvh_aabb_array_size);
                    //         assert(array_index * 8 + 7 < threaded_bvh_aabb_array_size);
                    //         float aabb_max_x = shared_threaded_bvh_aabb_array[array_index * 8 + 0];
                    //         float aabb_max_y = shared_threaded_bvh_aabb_array[array_index * 8 + 1];
                    //         float aabb_max_z = shared_threaded_bvh_aabb_array[array_index * 8 + 2];
                    //         float aabb_min_x = shared_threaded_bvh_aabb_array[array_index * 8 + 4];
                    //         float aabb_min_y = shared_threaded_bvh_aabb_array[array_index * 8 + 5];
                    //         float aabb_min_z = shared_threaded_bvh_aabb_array[array_index * 8 + 6];

                    //         // http://www.cs.utah.edu/~awilliam/box/box.pdf
                    //         float tmin = ((ray_direction_inv_x < 0 ? aabb_max_x : aabb_min_x) - ray_origin_x) * ray_direction_inv_x;
                    //         float tmax = ((ray_direction_inv_x < 0 ? aabb_min_x : aabb_max_x) - ray_origin_x) * ray_direction_inv_x;
                    //         float tmp_tmin = ((ray_direction_inv_y < 0 ? aabb_max_y : aabb_min_y) - ray_origin_y) * ray_direction_inv_y;
                    //         float tmp_tmax = ((ray_direction_inv_y < 0 ? aabb_min_y : aabb_max_y) - ray_origin_y) * ray_direction_inv_y;

                    //         if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                    //             bvh_current_node_id = bvh_miss_node_id;
                    //             continue;
                    //         }
                    //         if (tmp_tmin > tmin) {
                    //             tmin = tmp_tmin;
                    //         }
                    //         if (tmp_tmax < tmax) {
                    //             tmax = tmp_tmax;
                    //         }
                    //         tmp_tmin = ((ray_direction_inv_z < 0 ? aabb_max_z : aabb_min_z) - ray_origin_z) * ray_direction_inv_z;
                    //         // tmp_tmax = ((ray_direction_inv_z < 0 ? aabb_min_z : aabb_max_z) - ray_origin_z) * ray_direction_inv_z;
                    //         // if (bvh_object_index == 3 && ray_index == 10 * 32 + 5) {
                    //         //     printf("+traversal: %d max: %f %f dir: %f inv: %f tmax: %f tmin: %f ttmax: %f ttmin: %f \n", traversal, aabb_max_z, aabb_min_z, ray_direction_z, ray_direction_inv_z, tmax, tmin, tmp_tmax, tmp_tmin);
                    //         //     printf("AABB(max): (%f, %f, %f) min: (%f, %f, %f)\n", aabb_max_x, aabb_max_y, aabb_max_z, aabb_min_x, aabb_min_y, aabb_min_z);
                    //         //     printf("ray: (%f, %f, %f)\n", ray_direction_x, ray_direction_y, ray_direction_z);
                    //         //     printf("object: %d\n", bvh_object_index);
                    //         //     reflection_decay_r = 0.0f;
                    //         //     reflection_decay_g = 0.0f;
                    //         //     reflection_decay_b = 0.0f;
                    //         // }
                    //         // if (ray_index == 20 * 32 + 5) {
                    //         //     printf("-traversal: %d max: %f %f dir: %f inv: %f tmax: %f tmin: %f ttmax: %f ttmin: %f \n", traversal, aabb_max_z, aabb_min_z, ray_direction_z, ray_direction_inv_z, tmax, tmin, tmp_tmax, tmp_tmin);
                    //         //     printf("AABB(max): (%f, %f, %f) min: (%f, %f, %f)\n", aabb_max_x, aabb_max_y, aabb_max_z, aabb_min_x, aabb_min_y, aabb_min_z);
                    //         //     printf("ray: (%f, %f, %f)\n", ray_direction_x, ray_direction_y, ray_direction_z);
                    //         //     reflection_decay_r = 0.0f;
                    //         //     reflection_decay_g = 0.0f;
                    //         //     reflection_decay_b = 0.0f;
                    //         // }
                    //         if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {
                    //             bvh_current_node_id = bvh_miss_node_id;
                    //             continue;
                    //         }
                    //         if (tmp_tmin > tmin) {
                    //             tmin = tmp_tmin;
                    //         }
                    //         if (tmp_tmax < tmax) {
                    //             tmax = tmp_tmax;
                    //         }

                    //         if (tmax < 0.001) {
                    //             bvh_current_node_id = bvh_miss_node_id;
                    //             continue;
                    //         }

                    //         // if (thread_id == 0) {
                    //         //     printf("index: %u hit: %u miss: %u object: %u binary: %u\n", bvh_current_node_id,
                    //         //         bvh_hit_node_id, bvh_miss_node_id, bvh_object_index, bvh_binary_node);
                    //         //     // printf("node: %u %u\n", bvh_current_node_id, bvh_binary_node);
                    //         //     // printf("object: %u\n", bvh_object_index);
                    //         //     // printf("miss: %u\n", bvh_miss_node_id);
                    //         //     // printf("hit: %u\n", bvh_hit_node_id);
                    //         //     // printf("AAAB(max): %f %f %f\n", aabb_max_x, aabb_max_y, aabb_max_z);
                    //         //     // printf("AAAB(min): %f %f %f\n", aabb_min_x, aabb_min_y, aabb_min_z);
                    //         //     // printf("t: %f %f\n", tmin, tmax);
                    //         //     // printf("next: %u\n", bvh_current_node_id);
                    //         // }
                    //         reflection_decay_r = 1.0f;
                    //         reflection_decay_g = 0.0f;
                    //         reflection_decay_b = 0.0f;

                    //         bvh_current_node_id = bvh_hit_node_id;
                    //     } else {
                    //         if (bvh_hit_node_id == THREADED_BVH_TERMINAL_NODE) {
                    //             bvh_current_node_id = bvh_miss_node_id;
                    //         } else {
                    //             bvh_current_node_id = bvh_hit_node_id;
                    //         }
                    //     }
                    // }
                } else {
                    // if (geometry_type == RTX_GEOMETRY_TYPE_STANDARD) {
                    //     for (int j = 0; j < faces_count; j++) {
                    //         array_index = 4 * face_vertex_index_array[(face_index_offset + j) * 4 + 0];
                    //         if (array_index < 0) {
                    //             continue;
                    //         }
                    //         // if (thread_id == 0) {
                    //         //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                    //         // }
                    //         assert(array_index + 0 < vertex_array_size);
                    //         float va_x = vertex_array[array_index + 0];
                    //         float va_y = vertex_array[array_index + 1];
                    //         float va_z = vertex_array[array_index + 2];

                    //         // if (thread_id == 0) {
                    //         //     printf("va: (%f, %f, %f)\n", va_x, va_y, va_z);
                    //         // }

                    //         array_index = 4 * face_vertex_index_array[(face_index_offset + j) * 4 + 1];
                    //         if (array_index < 0) {
                    //             continue;
                    //         }
                    //         // if (thread_id == 0) {
                    //         //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                    //         // }
                    //         assert(array_index + 0 < vertex_array_size);
                    //         float vb_x = vertex_array[array_index + 0];
                    //         float vb_y = vertex_array[array_index + 1];
                    //         float vb_z = vertex_array[array_index + 2];

                    //         // if (thread_id == 0) {
                    //         //     printf("vb: (%f, %f, %f)\n", vb_x, vb_y, vb_z);
                    //         // }

                    //         array_index = 4 * face_vertex_index_array[(face_index_offset + j) * 4 + 2];
                    //         if (array_index < 0) {
                    //             continue;
                    //         }
                    //         // if (thread_id == 0) {
                    //         //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                    //         // }
                    //         assert(array_index + 0 < vertex_array_size);
                    //         float vc_x = vertex_array[array_index + 0];
                    //         float vc_y = vertex_array[array_index + 1];
                    //         float vc_z = vertex_array[array_index + 2];

                    //         // if (thread_id == 0) {
                    //         //     printf("vc: (%f, %f, %f)\n", vc_x, vc_y, vc_z);
                    //         // }

                    //         float edge_ba_x = vb_x - va_x;
                    //         float edge_ba_y = vb_y - va_y;
                    //         float edge_ba_z = vb_z - va_z;

                    //         float edge_ca_x = vc_x - va_x;
                    //         float edge_ca_y = vc_y - va_y;
                    //         float edge_ca_z = vc_z - va_z;

                    //         float h_x = ray_direction_y * edge_ca_z - ray_direction_z * edge_ca_y;
                    //         float h_y = ray_direction_z * edge_ca_x - ray_direction_x * edge_ca_z;
                    //         float h_z = ray_direction_x * edge_ca_y - ray_direction_y * edge_ca_x;
                    //         float a = edge_ba_x * h_x + edge_ba_y * h_y + edge_ba_z * h_z;
                    //         if (a > -eps && a < eps) {
                    //             continue;
                    //         }
                    //         float f = 1.0f / a;

                    //         float s_x = ray_origin_x - va_x;
                    //         float s_y = ray_origin_y - va_y;
                    //         float s_z = ray_origin_z - va_z;
                    //         float dot = s_x * h_x + s_y * h_y + s_z * h_z;
                    //         float u = f * dot;
                    //         if (u < 0.0f || u > 1.0f) {
                    //             continue;
                    //         }
                    //         float q_x = s_y * edge_ba_z - s_z * edge_ba_y;
                    //         float q_y = s_z * edge_ba_x - s_x * edge_ba_z;
                    //         float q_z = s_x * edge_ba_y - s_y * edge_ba_x;
                    //         dot = q_x * ray_direction_x + q_y * ray_direction_y + q_z * ray_direction_z;
                    //         float v = f * dot;
                    //         if (v < 0.0f || u + v > 1.0f) {
                    //             continue;
                    //         }
                    //         float tmp_x = edge_ba_y * edge_ca_z - edge_ba_z * edge_ca_y;
                    //         float tmp_y = edge_ba_z * edge_ca_x - edge_ba_x * edge_ca_z;
                    //         float tmp_z = edge_ba_x * edge_ca_y - edge_ba_y * edge_ca_x;

                    //         float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                    //         tmp_x = tmp_x / norm;
                    //         tmp_y = tmp_y / norm;
                    //         tmp_z = tmp_z / norm;

                    //         dot = tmp_x * ray_direction_x + tmp_y * ray_direction_y + tmp_z * ray_direction_z;
                    //         if (dot > 0.0f) {
                    //             continue;
                    //         }

                    //         dot = edge_ca_x * q_x + edge_ca_y * q_y + edge_ca_z * q_z;
                    //         float t = f * dot;

                    //         if (t <= 0.001f) {
                    //             continue;
                    //         }
                    //         if (min_distance <= t) {
                    //             continue;
                    //         }

                    //         min_distance = t;
                    //         hit_point_x = ray_origin_x + t * ray_direction_x;
                    //         hit_point_y = ray_origin_y + t * ray_direction_y;
                    //         hit_point_z = ray_origin_z + t * ray_direction_z;

                    //         hit_face_normal_x = tmp_x;
                    //         hit_face_normal_y = tmp_y;
                    //         hit_face_normal_z = tmp_z;

                    //         hit_color_r = 0.8f;
                    //         hit_color_g = 0.8f;
                    //         hit_color_b = 0.8f;

                    //         if (object_index == 0) {
                    //             hit_color_r = 1.0f;
                    //             hit_color_g = 1.0f;
                    //             hit_color_b = 1.0f;
                    //             did_hit_light = true;
                    //             continue;
                    //         }

                    //         did_hit_object = true;
                    //         did_hit_light = false;
                    //     }
                    // } else if (geometry_type == RTX_GEOMETRY_TYPE_SPHERE) {
                    //     array_index = 4 * face_vertex_index_array[face_index_offset * 4 + 0];
                    //     assert(array_index + 0 < vertex_array_size);

                    //     float center_x = vertex_array[array_index + 0];
                    //     float center_y = vertex_array[array_index + 1];
                    //     float center_z = vertex_array[array_index + 2];
                    //     float radius = vertex_array[array_index + 4];

                    //     float oc_x = ray_origin_x - center_x;
                    //     float oc_y = ray_origin_y - center_y;
                    //     float oc_z = ray_origin_z - center_z;

                    //     float a = ray_direction_x * ray_direction_x + ray_direction_y * ray_direction_y + ray_direction_z * ray_direction_z;
                    //     float b = 2.0f * (ray_direction_x * oc_x + ray_direction_y * oc_y + ray_direction_z * oc_z);
                    //     float c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius;
                    //     float d = b * b - 4.0f * a * c;

                    //     if (d <= 0) {
                    //         continue;
                    //     }
                    //     float root = sqrt(d);
                    //     float t = (-b - root) / (2.0f * a);
                    //     if (t <= 0.001f) {
                    //         t = (-b + root) / (2.0f * a);
                    //         if (t <= 0.001f) {
                    //             continue;
                    //         }
                    //     }

                    //     if (min_distance <= t) {
                    //         continue;
                    //     }

                    //     min_distance = t;
                    //     hit_point_x = ray_origin_x + t * ray_direction_x;
                    //     hit_point_y = ray_origin_y + t * ray_direction_y;
                    //     hit_point_z = ray_origin_z + t * ray_direction_z;

                    //     float tmp_x = hit_point_x - center_x;
                    //     float tmp_y = hit_point_y - center_y;
                    //     float tmp_z = hit_point_z - center_z;
                    //     float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                    //     hit_face_normal_x = tmp_x / norm;
                    //     hit_face_normal_y = tmp_y / norm;
                    //     hit_face_normal_z = tmp_z / norm;

                    //     hit_color_r = 0.8f;
                    //     hit_color_g = 0.8f;
                    //     hit_color_b = 0.8f;

                    //     did_hit_object = true;
                    //     did_hit_light = false;
                    //     continue;
                    // }
                }
            }

            // if (did_hit_light) {
            //     reflection_decay_r *= hit_color_r;
            //     reflection_decay_g *= hit_color_g;
            //     reflection_decay_b *= hit_color_b;
            //     break;
            // }

            // if (did_hit_object) {
            //     ray_origin_x = hit_point_x;
            //     ray_origin_y = hit_point_y;
            //     ray_origin_z = hit_point_z;

            //     // diffuse reflection
            //     float diffuese_x = curand_normal(&state);
            //     float diffuese_y = curand_normal(&state);
            //     float diffuese_z = curand_normal(&state);
            //     float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
            //     diffuese_x /= norm;
            //     diffuese_y /= norm;
            //     diffuese_z /= norm;

            //     float dot = hit_face_normal_x * diffuese_x + hit_face_normal_y * diffuese_y + hit_face_normal_z * diffuese_z;
            //     if (dot < 0.0f) {
            //         diffuese_x = -diffuese_x;
            //         diffuese_y = -diffuese_y;
            //         diffuese_z = -diffuese_z;
            //     }
            //     ray_direction_x = diffuese_x;
            //     ray_direction_y = diffuese_y;
            //     ray_direction_z = diffuese_z;

            //     ray_direction_inv_x = 1.0f / ray_direction_x;
            //     ray_direction_inv_y = 1.0f / ray_direction_y;
            //     ray_direction_inv_z = 1.0f / ray_direction_z;

            //     reflection_decay_r *= hit_color_r;
            //     reflection_decay_g *= hit_color_g;
            //     reflection_decay_b *= hit_color_b;
            // }
        }

        // if (did_hit_light == false) {
        //     reflection_decay_r = 0.0f;
        //     reflection_decay_g = 0.0f;
        //     reflection_decay_b = 0.0f;
        // }

        // assert(ray_index * 4 + 2 < render_array_size);
        // render_array[ray_index * 4 + 0] = reflection_decay_r;
        // render_array[ray_index * 4 + 1] = reflection_decay_g;
        // render_array[ray_index * 4 + 2] = reflection_decay_b;

        // render_array[ray_index * 3 + 0] = (ray_direction_x + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 1] = (ray_direction_y + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 2] = (ray_direction_z + 1.0f) / 2.0f;
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
    extern __shared__ int shared_memory[];
    int thread_id = threadIdx.x;
    curandStateXORWOW_t state;
    curand_init(0, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    int* shared_face_vertex_index_array = &shared_memory[offset];
    offset += face_vertex_index_array_size;
    float* shared_vertex_array = (float*)&shared_memory[offset];
    offset += vertex_array_size;
    int* shared_object_face_count_array = &shared_memory[offset];
    offset += object_face_count_array_size;
    int* shared_object_face_offset_array = &shared_memory[offset];
    offset += object_face_offset_array_size;
    int* shared_object_vertex_count_array = &shared_memory[offset];
    offset += object_vertex_count_array_size;
    int* shared_object_vertex_offset_array = &shared_memory[offset];
    offset += object_vertex_offset_array_size;
    int* shared_object_geometry_attributes_array = &shared_memory[offset];
    offset += object_geometry_attributes_array_size;
    int* shared_threaded_bvh_node_array = (int*)&shared_memory[offset];
    offset += threaded_bvh_node_array_size;
    float* shared_threaded_bvh_aabb_array = (float*)&shared_memory[offset];
    offset += threaded_bvh_aabb_array_size;

    if (thread_id == 0) {
        for (int i = 0; i < face_vertex_index_array_size; i++) {
            shared_face_vertex_index_array[i] = face_vertex_index_array[i];
        }
        for (int i = 0; i < vertex_array_size; i++) {
            shared_vertex_array[i] = vertex_array[i];
        }
        for (int i = 0; i < object_face_count_array_size; i++) {
            shared_object_face_count_array[i] = object_face_count_array[i];
        }
        for (int i = 0; i < object_face_offset_array_size; i++) {
            shared_object_face_offset_array[i] = object_face_offset_array[i];
        }
        for (int i = 0; i < object_vertex_count_array_size; i++) {
            shared_object_vertex_count_array[i] = object_vertex_count_array[i];
        }
        for (int i = 0; i < object_vertex_offset_array_size; i++) {
            shared_object_vertex_offset_array[i] = object_vertex_offset_array[i];
        }
        for (int i = 0; i < object_geometry_attributes_array_size; i++) {
            shared_object_geometry_attributes_array[i] = object_geometry_type_array[i];
        }
        for (int i = 0; i < threaded_bvh_node_array_size; i++) {
            shared_threaded_bvh_node_array[i] = threaded_bvh_node_array[i];
        }
        for (int i = 0; i < threaded_bvh_aabb_array_size; i++) {
            shared_threaded_bvh_aabb_array[i] = scene_threaded_bvh_aabb_array[i];
        }
    }
    __syncthreads();
    // if (thread_id == 0) {
    //     for (int i = 0; i < face_vertex_index_array_size; i++) {
    //         if (shared_face_vertex_index_array[i] != face_vertex_index_array[i]) {
    //             printf("Error: shared_face_vertex_index_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < vertex_array_size; i++) {
    //         if (shared_vertex_array[i] != vertex_array[i]) {
    //             printf("Error: shared_vertex_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_face_count_array_size; i++) {
    //         if (shared_object_face_count_array[i] != object_face_count_array[i]) {
    //             printf("Error: shared_object_face_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_face_offset_array_size; i++) {
    //         if (shared_object_face_offset_array[i] != object_face_offset_array[i]) {
    //             printf("Error: shared_object_face_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_count_array_size; i++) {
    //         if (shared_object_vertex_count_array[i] != object_vertex_count_array[i]) {
    //             printf("Error: shared_object_vertex_count_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < object_vertex_offset_array_size; i++) {
    //         if (shared_object_vertex_offset_array[i] != object_vertex_offset_array[i]) {
    //             printf("Error: shared_object_vertex_offset_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_node_array_size; i++) {
    //         if (shared_threaded_bvh_node_array[i] != threaded_bvh_node_array[i]) {
    //             printf("Error: shared_threaded_bvh_node_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    //     for (int i = 0; i < threaded_bvh_aabb_array_size; i++) {
    //         if (shared_threaded_bvh_aabb_array[i] != scene_threaded_bvh_aabb_array[i]) {
    //             printf("Error: shared_threaded_bvh_aabb_array missmatch at %d\n", i);
    //             return;
    //         }
    //     }
    // }

    const float eps = 0.0000001;
    float ray_direction_x;
    float ray_direction_y;
    float ray_direction_z;
    float ray_origin_x;
    float ray_origin_y;
    float ray_origin_z;
    float ray_direction_inv_x;
    float ray_direction_inv_y;
    float ray_direction_inv_z;
    float hit_point_x;
    float hit_point_y;
    float hit_point_z;
    float hit_color_r;
    float hit_color_g;
    float hit_color_b;
    float hit_face_normal_x;
    float hit_face_normal_y;
    float hit_face_normal_z;
    float reflection_decay_r;
    float reflection_decay_g;
    float reflection_decay_b;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= num_rays) {
            return;
        }

        int array_index = ray_index * 8;
        ray_direction_x = ray_array[array_index + 0];
        ray_direction_y = ray_array[array_index + 1];
        ray_direction_z = ray_array[array_index + 2];
        ray_origin_x = ray_array[array_index + 4];
        ray_origin_y = ray_array[array_index + 5];
        ray_origin_z = ray_array[array_index + 6];
        ray_direction_inv_x = 1.0f / ray_direction_x;
        ray_direction_inv_y = 1.0f / ray_direction_y;
        ray_direction_inv_z = 1.0f / ray_direction_z;

        reflection_decay_r = 1.0f;
        reflection_decay_g = 1.0f;
        reflection_decay_b = 1.0f;

        bool did_hit_light = false;

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            for (int object_index = 0; object_index < object_face_count_array_size; object_index++) {
                int faces_count = shared_object_face_count_array[object_index];
                int face_index_offset = shared_object_face_offset_array[object_index];
                int vertices_count = shared_object_vertex_count_array[object_index];
                int vertex_index_offset = shared_object_vertex_offset_array[object_index];
                int geometry_type = shared_object_geometry_attributes_array[object_index];
                // if (thread_id == 0) {
                //     printf("object: %d\n", object_index);
                //     printf("face: ");
                //     for (int i = face_index_offset; i < face_index_offset + faces_count; i++) {
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 0]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 1]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 2]);
                //         printf("%d ", shared_face_vertex_index_array[i * 4 + 3]);
                //     }
                //     printf("\n");
                //     printf("vertex: ");
                //     for (int i = 0; i < faces_count; i++) {
                //         int array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 0];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //         array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 1];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //         array_index = shared_face_vertex_index_array[(face_index_offset + i) * 4 + 2];
                //         printf("(%f, %f, %f) ", shared_vertex_array[array_index * 4 + 0], shared_vertex_array[array_index * 4 + 1], shared_vertex_array[array_index * 4 + 2]);
                //     }
                //     printf("\n");
                // }
                if (geometry_type == RTX_GEOMETRY_TYPE_STANDARD) {
                    for (int j = 0; j < faces_count; j++) {
                        array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 0];
                        if (array_index < 0) {
                            continue;
                        }
                        // if (thread_id == 0) {
                        //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                        // }
                        assert(array_index + 0 < vertex_array_size);
                        float va_x = shared_vertex_array[array_index + 0];
                        float va_y = shared_vertex_array[array_index + 1];
                        float va_z = shared_vertex_array[array_index + 2];

                        // if (thread_id == 0) {
                        //     printf("va: (%f, %f, %f)\n", va_x, va_y, va_z);
                        // }

                        array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 1];
                        if (array_index < 0) {
                            continue;
                        }
                        // if (thread_id == 0) {
                        //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                        // }
                        assert(array_index + 0 < vertex_array_size);
                        float vb_x = shared_vertex_array[array_index + 0];
                        float vb_y = shared_vertex_array[array_index + 1];
                        float vb_z = shared_vertex_array[array_index + 2];

                        // if (thread_id == 0) {
                        //     printf("vb: (%f, %f, %f)\n", vb_x, vb_y, vb_z);
                        // }

                        array_index = 4 * shared_face_vertex_index_array[(face_index_offset + j) * 4 + 2];
                        if (array_index < 0) {
                            continue;
                        }
                        // if (thread_id == 0) {
                        //     printf("object: %u face: %d vertex: %d\n", bvh_object_index, j, array_index);
                        // }
                        assert(array_index + 0 < vertex_array_size);
                        float vc_x = shared_vertex_array[array_index + 0];
                        float vc_y = shared_vertex_array[array_index + 1];
                        float vc_z = shared_vertex_array[array_index + 2];

                        // if (thread_id == 0) {
                        //     printf("vc: (%f, %f, %f)\n", vc_x, vc_y, vc_z);
                        // }

                        float edge_ba_x = vb_x - va_x;
                        float edge_ba_y = vb_y - va_y;
                        float edge_ba_z = vb_z - va_z;

                        float edge_ca_x = vc_x - va_x;
                        float edge_ca_y = vc_y - va_y;
                        float edge_ca_z = vc_z - va_z;

                        float h_x = ray_direction_y * edge_ca_z - ray_direction_z * edge_ca_y;
                        float h_y = ray_direction_z * edge_ca_x - ray_direction_x * edge_ca_z;
                        float h_z = ray_direction_x * edge_ca_y - ray_direction_y * edge_ca_x;
                        float a = edge_ba_x * h_x + edge_ba_y * h_y + edge_ba_z * h_z;
                        if (a > -eps && a < eps) {
                            continue;
                        }
                        float f = 1.0f / a;

                        float s_x = ray_origin_x - va_x;
                        float s_y = ray_origin_y - va_y;
                        float s_z = ray_origin_z - va_z;
                        float dot = s_x * h_x + s_y * h_y + s_z * h_z;
                        float u = f * dot;
                        if (u < 0.0f || u > 1.0f) {
                            continue;
                        }
                        float q_x = s_y * edge_ba_z - s_z * edge_ba_y;
                        float q_y = s_z * edge_ba_x - s_x * edge_ba_z;
                        float q_z = s_x * edge_ba_y - s_y * edge_ba_x;
                        dot = q_x * ray_direction_x + q_y * ray_direction_y + q_z * ray_direction_z;
                        float v = f * dot;
                        if (v < 0.0f || u + v > 1.0f) {
                            continue;
                        }
                        float tmp_x = edge_ba_y * edge_ca_z - edge_ba_z * edge_ca_y;
                        float tmp_y = edge_ba_z * edge_ca_x - edge_ba_x * edge_ca_z;
                        float tmp_z = edge_ba_x * edge_ca_y - edge_ba_y * edge_ca_x;

                        float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                        tmp_x = tmp_x / norm;
                        tmp_y = tmp_y / norm;
                        tmp_z = tmp_z / norm;

                        dot = tmp_x * ray_direction_x + tmp_y * ray_direction_y + tmp_z * ray_direction_z;
                        if (dot > 0.0f) {
                            continue;
                        }

                        dot = edge_ca_x * q_x + edge_ca_y * q_y + edge_ca_z * q_z;
                        float t = f * dot;

                        if (t <= 0.001f) {
                            continue;
                        }
                        if (min_distance <= t) {
                            continue;
                        }

                        min_distance = t;
                        hit_point_x = ray_origin_x + t * ray_direction_x;
                        hit_point_y = ray_origin_y + t * ray_direction_y;
                        hit_point_z = ray_origin_z + t * ray_direction_z;

                        hit_face_normal_x = tmp_x;
                        hit_face_normal_y = tmp_y;
                        hit_face_normal_z = tmp_z;

                        hit_color_r = 0.8f;
                        hit_color_g = 0.8f;
                        hit_color_b = 0.8f;

                        if (object_index == 2) {
                            hit_color_r = 1.0f;
                            hit_color_g = 1.0f;
                            hit_color_b = 1.0f;
                            did_hit_light = true;
                            continue;
                        }

                        did_hit_object = true;
                        did_hit_light = false;
                    }
                } else if (geometry_type == RTX_GEOMETRY_TYPE_SPHERE) {
                    array_index = 4 * shared_face_vertex_index_array[face_index_offset * 4 + 0];
                    assert(array_index + 0 < vertex_array_size);

                    float center_x = shared_vertex_array[array_index + 0];
                    float center_y = shared_vertex_array[array_index + 1];
                    float center_z = shared_vertex_array[array_index + 2];
                    float radius = shared_vertex_array[array_index + 4];

                    float oc_x = ray_origin_x - center_x;
                    float oc_y = ray_origin_y - center_y;
                    float oc_z = ray_origin_z - center_z;

                    float a = ray_direction_x * ray_direction_x + ray_direction_y * ray_direction_y + ray_direction_z * ray_direction_z;
                    float b = 2.0f * (ray_direction_x * oc_x + ray_direction_y * oc_y + ray_direction_z * oc_z);
                    float c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius;
                    float d = b * b - 4.0f * a * c;

                    if (d <= 0) {
                        continue;
                    }
                    float root = sqrt(d);
                    float t = (-b - root) / (2.0f * a);
                    if (t <= 0.001f) {
                        t = (-b + root) / (2.0f * a);
                        if (t <= 0.001f) {
                            continue;
                        }
                    }

                    if (min_distance <= t) {
                        continue;
                    }

                    min_distance = t;
                    hit_point_x = ray_origin_x + t * ray_direction_x;
                    hit_point_y = ray_origin_y + t * ray_direction_y;
                    hit_point_z = ray_origin_z + t * ray_direction_z;

                    float tmp_x = hit_point_x - center_x;
                    float tmp_y = hit_point_y - center_y;
                    float tmp_z = hit_point_z - center_z;
                    float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

                    hit_face_normal_x = tmp_x / norm;
                    hit_face_normal_y = tmp_y / norm;
                    hit_face_normal_z = tmp_z / norm;

                    hit_color_r = 0.8f;
                    hit_color_g = 0.8f;
                    hit_color_b = 0.8f;

                    did_hit_object = true;
                    did_hit_light = false;
                    continue;
                }
            }

            if (did_hit_light) {
                reflection_decay_r *= hit_color_r;
                reflection_decay_g *= hit_color_g;
                reflection_decay_b *= hit_color_b;
                break;
            }

            if (did_hit_object) {
                ray_origin_x = hit_point_x;
                ray_origin_y = hit_point_y;
                ray_origin_z = hit_point_z;

                // diffuse reflection
                float diffuese_x = curand_normal(&state);
                float diffuese_y = curand_normal(&state);
                float diffuese_z = curand_normal(&state);
                float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
                diffuese_x /= norm;
                diffuese_y /= norm;
                diffuese_z /= norm;

                float dot = hit_face_normal_x * diffuese_x + hit_face_normal_y * diffuese_y + hit_face_normal_z * diffuese_z;
                if (dot < 0.0f) {
                    diffuese_x = -diffuese_x;
                    diffuese_y = -diffuese_y;
                    diffuese_z = -diffuese_z;
                }
                ray_direction_x = diffuese_x;
                ray_direction_y = diffuese_y;
                ray_direction_z = diffuese_z;

                ray_direction_inv_x = 1.0f / ray_direction_x;
                ray_direction_inv_y = 1.0f / ray_direction_y;
                ray_direction_inv_z = 1.0f / ray_direction_z;

                reflection_decay_r *= hit_color_r;
                reflection_decay_g *= hit_color_g;
                reflection_decay_b *= hit_color_b;
            }
        }

        if (did_hit_light == false) {
            reflection_decay_r = 0.0f;
            reflection_decay_g = 0.0f;
            reflection_decay_b = 0.0f;
        }

        assert(ray_index * 4 + 2 < render_array_size);
        render_array[ray_index * 4 + 0] = reflection_decay_r;
        render_array[ray_index * 4 + 1] = reflection_decay_g;
        render_array[ray_index * 4 + 2] = reflection_decay_b;

        // render_array[ray_index * 3 + 0] = (ray_direction_x + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 1] = (ray_direction_y + 1.0f) / 2.0f;
        // render_array[ray_index * 3 + 2] = (ray_direction_z + 1.0f) / 2.0f;
    }
}
// __global__ void _render(
//     const float* rays,
//     const float* face_vertices,
//     const float* face_colors,
//     const int* object_types,
//     const int* material_types,
//     float* color_per_ray,
//     const float* camera_inv_matrix,
//     const int num_rays_per_thread,
//     const int thread_offset,
//     const int num_rays,
//     const int num_faces,
//     const int faces_stride,
//     const int colors_stride,
//     const int max_bounce)
// {
//     int tid = threadIdx.x;
//     curandStateXORWOW_t state;
//     curand_init(0, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

//     __shared__ float shared_face_vertices[41 * 12];
//     __shared__ float shared_face_colors[41 * 3];
//     __shared__ int shared_object_types[41];
//     __shared__ int shared_material_types[41];
//     __shared__ float shared_camera_inv_matrix[4][4];

//     if (threadIdx.x == 0) {
//         for (int n = 0; n < num_faces; n++) {
//             for (int s = 0; s < faces_stride; s++) {
//                 shared_face_vertices[n * faces_stride + s] = face_vertices[n * faces_stride + s];
//             }
//             for (int s = 0; s < colors_stride; s++) {
//                 shared_face_colors[n * colors_stride + s] = face_colors[n * colors_stride + s];
//             }
//             shared_object_types[n] = object_types[n];
//             shared_material_types[n] = material_types[n];
//         }
//         shared_camera_inv_matrix[0][0] = camera_inv_matrix[0];
//         shared_camera_inv_matrix[0][1] = camera_inv_matrix[1];
//         shared_camera_inv_matrix[0][2] = camera_inv_matrix[2];
//         shared_camera_inv_matrix[0][3] = camera_inv_matrix[3];
//         shared_camera_inv_matrix[1][0] = camera_inv_matrix[4];
//         shared_camera_inv_matrix[1][1] = camera_inv_matrix[5];
//         shared_camera_inv_matrix[1][2] = camera_inv_matrix[6];
//         shared_camera_inv_matrix[1][3] = camera_inv_matrix[7];
//         shared_camera_inv_matrix[2][0] = camera_inv_matrix[8];
//         shared_camera_inv_matrix[2][1] = camera_inv_matrix[9];
//         shared_camera_inv_matrix[2][2] = camera_inv_matrix[10];
//         shared_camera_inv_matrix[2][3] = camera_inv_matrix[11];
//         shared_camera_inv_matrix[3][0] = camera_inv_matrix[12];
//         shared_camera_inv_matrix[3][1] = camera_inv_matrix[13];
//         shared_camera_inv_matrix[3][2] = camera_inv_matrix[14];
//         shared_camera_inv_matrix[3][3] = camera_inv_matrix[15];
//     }
//     __syncthreads();

//     for (int n = 0; n < num_rays_per_thread; n++) {
//         int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n + thread_offset;
//         if (ray_index >= num_rays) {
//             return;
//         }

//         const int p = ray_index * 7;
//         float ray_direction_x = rays[p + 0];
//         float ray_direction_y = rays[p + 1];
//         float ray_direction_z = rays[p + 2];
//         float ray_origin_x = rays[p + 3];
//         float ray_origin_y = rays[p + 4];
//         float ray_origin_z = rays[p + 5];
//         float ray_direction_inv_x = 1.0f / ray_direction_x;
//         float ray_direction_inv_y = 1.0f / ray_direction_y;
//         float ray_direction_inv_z = 1.0f / ray_direction_z;

//         float color_r = 0.0;
//         float color_g = 0.0;
//         float color_b = 0.0;

//         int object_type = 0;
//         int material_type = 0;
//         float hit_point_x = 0.0f;
//         float hit_point_y = 0.0f;
//         float hit_point_z = 0.0f;
//         float hit_color_r = 0.0f;
//         float hit_color_g = 0.0f;
//         float hit_color_b = 0.0f;
//         float hit_face_normal_x = 0.0f;
//         float hit_face_normal_y = 0.0f;
//         float hit_face_normal_z = 0.0f;

//         color_r = 1.0f;
//         color_g = 1.0f;
//         color_b = 1.0f;

//         const float eps = 0.0000001;
//         float reflection_decay_r = 1.0f;
//         float reflection_decay_g = 1.0f;
//         float reflection_decay_b = 1.0f;
//         bool did_hit_light = false;

//         for (int depth = 0; depth < max_bounce; depth++) {
//             float min_distance = FLT_MAX;
//             bool did_hit_object = false;

//             for (int face_index = 0; face_index < num_faces; face_index++) {
//                 object_type = shared_object_types[face_index];
//                 const int index = face_index * faces_stride;

//                 if (object_type == RTX_GEOMETRY_TYPE_STANDARD) {
//                     const float va_x = shared_face_vertices[index + 0];
//                     const float va_y = shared_face_vertices[index + 1];
//                     const float va_z = shared_face_vertices[index + 2];

//                     const float vb_x = shared_face_vertices[index + 4];
//                     const float vb_y = shared_face_vertices[index + 5];
//                     const float vb_z = shared_face_vertices[index + 6];

//                     const float vc_x = shared_face_vertices[index + 8];
//                     const float vc_y = shared_face_vertices[index + 9];
//                     const float vc_z = shared_face_vertices[index + 10];

//                     const float edge_ba_x = vb_x - va_x;
//                     const float edge_ba_y = vb_y - va_y;
//                     const float edge_ba_z = vb_z - va_z;

//                     const float edge_ca_x = vc_x - va_x;
//                     const float edge_ca_y = vc_y - va_y;
//                     const float edge_ca_z = vc_z - va_z;

//                     const float h_x = ray_direction_y * edge_ca_z - ray_direction_z * edge_ca_y;
//                     const float h_y = ray_direction_z * edge_ca_x - ray_direction_x * edge_ca_z;
//                     const float h_z = ray_direction_x * edge_ca_y - ray_direction_y * edge_ca_x;
//                     const float a = edge_ba_x * h_x + edge_ba_y * h_y + edge_ba_z * h_z;
//                     if (a > -eps && a < eps) {
//                         continue;
//                     }
//                     const float f = 1.0f / a;

//                     const float s_x = ray_origin_x - va_x;
//                     const float s_y = ray_origin_y - va_y;
//                     const float s_z = ray_origin_z - va_z;
//                     float dot = s_x * h_x + s_y * h_y + s_z * h_z;
//                     const float u = f * dot;
//                     if (u < 0.0f || u > 1.0f) {
//                         continue;
//                     }
//                     const float q_x = s_y * edge_ba_z - s_z * edge_ba_y;
//                     const float q_y = s_z * edge_ba_x - s_x * edge_ba_z;
//                     const float q_z = s_x * edge_ba_y - s_y * edge_ba_x;
//                     dot = q_x * ray_direction_x + q_y * ray_direction_y + q_z * ray_direction_z;
//                     const float v = f * dot;
//                     if (v < 0.0f || u + v > 1.0f) {
//                         continue;
//                     }
//                     float tmp_x = edge_ba_y * edge_ca_z - edge_ba_z * edge_ca_y;
//                     float tmp_y = edge_ba_z * edge_ca_x - edge_ba_x * edge_ca_z;
//                     float tmp_z = edge_ba_x * edge_ca_y - edge_ba_y * edge_ca_x;

//                     float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

//                     tmp_x = tmp_x / norm;
//                     tmp_y = tmp_y / norm;
//                     tmp_z = tmp_z / norm;

//                     dot = tmp_x * ray_direction_x + tmp_y * ray_direction_y + tmp_z * ray_direction_z;
//                     if (dot > 0.0f) {
//                         continue;
//                     }

//                     dot = edge_ca_x * q_x + edge_ca_y * q_y + edge_ca_z * q_z;
//                     const float t = f * dot;

//                     if (t <= 0.001f) {
//                         continue;
//                     }
//                     if (min_distance <= t) {
//                         continue;
//                     }

//                     min_distance = t;
//                     hit_point_x = ray_origin_x + t * ray_direction_x;
//                     hit_point_y = ray_origin_y + t * ray_direction_y;
//                     hit_point_z = ray_origin_z + t * ray_direction_z;

//                     hit_face_normal_x = tmp_x;
//                     hit_face_normal_y = tmp_y;
//                     hit_face_normal_z = tmp_z;

//                     material_type = shared_material_types[face_index];

//                     hit_color_r = shared_face_colors[face_index * colors_stride + 0];
//                     hit_color_g = shared_face_colors[face_index * colors_stride + 1];
//                     hit_color_b = shared_face_colors[face_index * colors_stride + 2];

//                     did_hit_object = true;
//                     continue;
//                 }
//                 if (object_type == RTX_GEOMETRY_TYPE_SPHERE) {
//                     const float center_x = shared_face_vertices[index + 0];
//                     const float center_y = shared_face_vertices[index + 1];
//                     const float center_z = shared_face_vertices[index + 2];
//                     const float radius = shared_face_vertices[index + 4];

//                     const float oc_x = ray_origin_x - center_x;
//                     const float oc_y = ray_origin_y - center_y;
//                     const float oc_z = ray_origin_z - center_z;

//                     const float a = ray_direction_x * ray_direction_x + ray_direction_y * ray_direction_y + ray_direction_z * ray_direction_z;
//                     const float b = 2.0f * (ray_direction_x * oc_x + ray_direction_y * oc_y + ray_direction_z * oc_z);
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
//                     hit_point_x = ray_origin_x + t * ray_direction_x;
//                     hit_point_y = ray_origin_y + t * ray_direction_y;
//                     hit_point_z = ray_origin_z + t * ray_direction_z;

//                     float tmp_x = hit_point_x - center_x;
//                     float tmp_y = hit_point_y - center_y;
//                     float tmp_z = hit_point_z - center_z;
//                     float norm = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z) + 1e-12;

//                     hit_face_normal_x = tmp_x / norm;
//                     hit_face_normal_y = tmp_y / norm;
//                     hit_face_normal_z = tmp_z / norm;

//                     material_type = shared_material_types[face_index];

//                     hit_color_r = shared_face_colors[face_index * colors_stride + 0];
//                     hit_color_g = shared_face_colors[face_index * colors_stride + 1];
//                     hit_color_b = shared_face_colors[face_index * colors_stride + 2];

//                     did_hit_object = true;
//                     continue;
//                 }
//                 // http://www.cs.utah.edu/~awilliam/box/box.pdf
//                 if (object_type == 333) {
//                     float _min_x = shared_face_vertices[index + 0];
//                     float _min_y = shared_face_vertices[index + 1];
//                     float _min_z = shared_face_vertices[index + 2];
//                     float _max_x = shared_face_vertices[index + 4];
//                     float _max_y = shared_face_vertices[index + 5];
//                     float _max_z = shared_face_vertices[index + 6];

//                     float min_x = shared_camera_inv_matrix[0][0] * _min_x + shared_camera_inv_matrix[1][0] * _min_y + shared_camera_inv_matrix[2][0] * _min_z + shared_camera_inv_matrix[3][0];
//                     float min_y = shared_camera_inv_matrix[0][1] * _min_x + shared_camera_inv_matrix[1][1] * _min_y + shared_camera_inv_matrix[2][1] * _min_z + shared_camera_inv_matrix[3][1];
//                     float min_z = shared_camera_inv_matrix[0][2] * _min_x + shared_camera_inv_matrix[1][2] * _min_y + shared_camera_inv_matrix[2][2] * _min_z + shared_camera_inv_matrix[3][2];

//                     float max_x = shared_camera_inv_matrix[0][0] * _max_x + shared_camera_inv_matrix[1][0] * _max_y + shared_camera_inv_matrix[2][0] * _max_z + shared_camera_inv_matrix[3][0];
//                     float max_y = shared_camera_inv_matrix[0][1] * _max_x + shared_camera_inv_matrix[1][1] * _max_y + shared_camera_inv_matrix[2][1] * _max_z + shared_camera_inv_matrix[3][1];
//                     float max_z = shared_camera_inv_matrix[0][2] * _max_x + shared_camera_inv_matrix[1][2] * _max_y + shared_camera_inv_matrix[2][2] * _max_z + shared_camera_inv_matrix[3][2];

//                     const bool sign_x = ray_direction_inv_x < 0;
//                     const bool sign_y = ray_direction_inv_y < 0;
//                     const bool sign_z = ray_direction_inv_z < 0;
//                     float tmin, tmax, tymin, tymax, tzmin, tzmax;
//                     tmin = ((sign_x ? max_x : min_x) - ray_origin_x) * ray_direction_inv_x;
//                     tmax = ((sign_x ? min_x : max_x) - ray_origin_x) * ray_direction_inv_x;
//                     tymin = ((sign_y ? max_y : min_y) - ray_origin_y) * ray_direction_inv_y;
//                     tymax = ((sign_y ? min_y : max_y) - ray_origin_y) * ray_direction_inv_y;
//                     if ((tmin > tymax) || (tymin > tmax)) {
//                         continue;
//                     }
//                     if (tymin > tmin) {
//                         tmin = tymin;
//                     }
//                     if (tymax < tmax) {
//                         tmax = tymax;
//                     }
//                     tzmin = ((sign_z ? max_z : min_z) - ray_origin_z) * ray_direction_inv_z;
//                     tzmax = ((sign_z ? min_z : max_z) - ray_origin_z) * ray_direction_inv_z;
//                     if ((tmin > tzmax) || (tzmin > tmax)) {
//                         continue;
//                     }
//                     if (tzmin > tmin) {
//                         tmin = tzmin;
//                     }
//                     if (tzmax < tmax) {
//                         tmax = tzmax;
//                     }
//                     material_type = RTX_MATERIAL_TYPE_EMISSIVE;

//                     hit_color_r = 1.0f;
//                     hit_color_g = 1.0f;
//                     hit_color_b = 1.0f;

//                     did_hit_object = true;
//                     continue;
//                 }
//             }

//             if (did_hit_object) {
//                 ray_origin_x = hit_point_x;
//                 ray_origin_y = hit_point_y;
//                 ray_origin_z = hit_point_z;

//                 if (material_type == RTX_MATERIAL_TYPE_EMISSIVE) {
//                     color_r = reflection_decay_r * hit_color_r;
//                     color_g = reflection_decay_g * hit_color_g;
//                     color_b = reflection_decay_b * hit_color_b;
//                     did_hit_light = true;
//                     break;
//                 }

//                 // detect backface
//                 // float dot = hit_face_normal_x * ray_direction_x + hit_face_normal_y * ray_direction_y + hit_face_normal_z * ray_direction_z;
//                 // if (dot > 0.0f) {
//                 //     hit_face_normal_x *= -1.0f;
//                 //     hit_face_normal_y *= -1.0f;
//                 //     hit_face_normal_z *= -1.0f;
//                 // }

//                 // diffuse reflection
//                 float diffuese_x = curand_normal(&state);
//                 float diffuese_y = curand_normal(&state);
//                 float diffuese_z = curand_normal(&state);
//                 const float norm = sqrt(diffuese_x * diffuese_x + diffuese_y * diffuese_y + diffuese_z * diffuese_z);
//                 diffuese_x /= norm;
//                 diffuese_y /= norm;
//                 diffuese_z /= norm;

//                 float dot = hit_face_normal_x * diffuese_x + hit_face_normal_y * diffuese_y + hit_face_normal_z * diffuese_z;
//                 if (dot < 0.0f) {
//                     diffuese_x = -diffuese_x;
//                     diffuese_y = -diffuese_y;
//                     diffuese_z = -diffuese_z;
//                 }
//                 ray_direction_x = diffuese_x;
//                 ray_direction_y = diffuese_y;
//                 ray_direction_z = diffuese_z;

//                 ray_direction_inv_x = 1.0f / ray_direction_x;
//                 ray_direction_inv_y = 1.0f / ray_direction_y;
//                 ray_direction_inv_z = 1.0f / ray_direction_z;

//                 reflection_decay_r *= hit_color_r;
//                 reflection_decay_g *= hit_color_g;
//                 reflection_decay_b *= hit_color_b;
//             }
//         }

//         if (did_hit_light == false) {
//             color_r = 0.0f;
//             color_g = 0.0f;
//             color_b = 0.0f;
//         }
//         color_per_ray[ray_index * 3 + 0] = color_r;
//         color_per_ray[ray_index * 3 + 1] = color_g;
//         color_per_ray[ray_index * 3 + 2] = color_b;
//     }
// }
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
void rtx_cuda_ray_tracing_render(
    RTXRay*& gpu_ray_array, const int ray_array_size,
    RTXGeometryFace*& gpu_face_vertex_index_array, const int face_vertex_index_array_size,
    RTXGeometryVertex*& gpu_vertex_array, const int vertex_array_size,
    RTXObject*& gpu_object_array, const int object_array_size,
    RTXThreadedBVH*& gpu_threaded_bvh_array, const int threaded_bvh_array_size,
    RTXThreadedBVHNode*& gpu_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    RTXPixel*& gpu_render_array, const int render_array_size,
    const int num_rays_per_pixel,
    const int max_bounce)
{
    assert(gpu_ray_array != NULL);
    assert(gpu_face_vertex_index_array != NULL);
    assert(gpu_vertex_array != NULL);
    assert(gpu_object_array != NULL);
    assert(gpu_threaded_bvh_array != NULL);
    assert(gpu_threaded_bvh_node_array != NULL);
    assert(gpu_render_array != NULL);

    int num_rays = ray_array_size;

    int num_threads = 32;
    // int num_blocks = (num_rays - 1) / num_threads + 1;
    int num_blocks = 256;

    int num_rays_per_thread = num_rays / (num_threads * num_blocks) + 1;

    int required_shared_memory_bytes = sizeof(RTXGeometryFace) * face_vertex_index_array_size + sizeof(RTXGeometryVertex) * vertex_array_size + sizeof(RTXObject) * object_array_size + sizeof(RTXThreadedBVH) + threaded_bvh_array_size * sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size;

    num_blocks = 1;
    num_rays_per_thread = 1;

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);

    printf("shared memory: %d bytes\n", required_shared_memory_bytes);
    printf("available: %d bytes\n", dev.sharedMemPerBlock);

    if (required_shared_memory_bytes > dev.sharedMemPerBlock) {
        int required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size;
        printf("shared memory: %d bytes\n", required_shared_memory_bytes);
        printf("available: %d bytes\n", dev.sharedMemPerBlock);
        printf("RTXObject: %d * %d = %d\n", sizeof(RTXObject), object_array_size, sizeof(RTXObject) * object_array_size);
        printf("RTXThreadedBVH: %d * %d = %d\n", sizeof(RTXThreadedBVH), threaded_bvh_array_size, sizeof(RTXThreadedBVH) * threaded_bvh_array_size);
        printf("RTXThreadedBVHNode: %d * %d = %d\n", sizeof(RTXThreadedBVHNode), threaded_bvh_node_array_size, sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size);
        assert(required_shared_memory_bytes <= dev.sharedMemPerBlock);

        global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
            gpu_ray_array, ray_array_size,
            gpu_face_vertex_index_array, face_vertex_index_array_size,
            gpu_vertex_array, vertex_array_size,
            gpu_object_array, object_array_size,
            gpu_threaded_bvh_array, threaded_bvh_array_size,
            gpu_threaded_bvh_node_array, threaded_bvh_node_array_size,
            gpu_render_array, render_array_size,
            num_rays_per_pixel,
            max_bounce);
    } else {
        // shared_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
        //     gpu_ray_array, ray_array_size,
        //     gpu_face_vertex_index_array, face_vertex_index_array_size,
        //     gpu_vertex_array, vertex_array_size,
        //     gpu_object_face_count_array, object_face_count_array_size,
        //     gpu_object_face_offset_array, object_face_offset_array_size,
        //     gpu_object_vertex_count_array, object_vertex_count_array_size,
        //     gpu_object_vertex_offset_array, object_vertex_offset_array_size,
        //     gpu_object_geometry_attributes_array, object_geometry_attributes_array_size,
        //     gpu_threaded_bvh_node_array, threaded_bvh_node_array_size,
        //     gpu_threaded_bvh_num_nodes_array, threaded_bvh_num_nodes_array_size,
        //     gpu_threaded_bvh_index_offset_array, threaded_bvh_index_offset_array_size,
        //     gpu_threaded_bvh_aabb_array, threaded_bvh_aabb_array_size,
        //     gpu_render_array, render_array_size,
        //     num_rays,
        //     num_rays_per_thread,
        //     max_bounce);
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

__global__ void test_linear_kernel(
    const int* node_array, const int num_nodes)
{
    int s = 0;
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < num_nodes; i++) {
            int hit = node_array[i * 4 + 0];
            int miss = node_array[i * 4 + 1];
            int start = node_array[i * 4 + 2];
            int end = node_array[i * 4 + 3];
            s += hit + miss + start + end;
        }
    }
}

__global__ void test_struct_kernel(
    const RTXThreadedBVHNode* node_array, const int num_nodes)
{
    int s = 0;
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < num_nodes; i++) {
            RTXThreadedBVHNode node = node_array[i];
            s += node.hit_node_index + node.miss_node_index + node.assigned_face_index_start + node.assigned_face_index_end;
        }
    }
}

void launch_test_linear_kernel(
    int*& gpu_node_array, const int num_nodes)
{
    test_linear_kernel<<<256, 32>>>(gpu_node_array, num_nodes);
    cudaError_t status = cudaGetLastError();
    if (status != 0) {
        fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
    }
    cudaError_t error = cudaThreadSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
    }
}

void launch_test_struct_kernel(
    RTXThreadedBVHNode*& gpu_struct_array, const int num_nodes)
{

    test_struct_kernel<<<256, 32>>>(gpu_struct_array, num_nodes);
    cudaError_t status = cudaGetLastError();
    if (status != 0) {
        fprintf(stderr, "CUDA Error at kernel: %s\n", cudaGetErrorString(status));
    }
    cudaError_t error = cudaThreadSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error at cudaThreadSynchronize: %s\n", cudaGetErrorString(error));
    }
}