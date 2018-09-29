#include "../../../header/enum.h"
#include "../../../header/struct.h"
#include "../../header/bridge.h"
#include "../../header/cuda_common.h"
#include "../../header/cuda_functions.h"
#include "../../header/cuda_texture.h"
#include "../../header/standard_kernel.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

__global__ void standard_global_memory_kernel(
    rtxRay* global_serialized_ray_array, int ray_array_size,
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array, int face_vertex_index_array_size,
    rtxVertex* global_serialized_vertex_array, int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    rtxUVCoordinate* global_serialized_uv_coordinate_array, int uv_coordinate_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    int num_active_texture_units,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed)
{
    extern __shared__ char shared_memory[];
    curandStatePhilox4_32_10_t curand_state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &curand_state);

    // グローバルメモリの直列データを共有メモリにコピーする
    int offset = 0;
    rtxObject* shared_serialized_object_array = (rtxObject*)&shared_memory[offset];
    offset += sizeof(rtxObject) * object_array_size;

    rtxMaterialAttributeByte* shared_serialized_material_attribute_byte_array = (rtxMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(rtxMaterialAttributeByte) * material_attribute_byte_array_size;

    rtxThreadedBVH* shared_serialized_threaded_bvh_array = (rtxThreadedBVH*)&shared_memory[offset];
    offset += sizeof(rtxThreadedBVH) * threaded_bvh_array_size;

    rtxRGBAColor* shared_serialized_color_mapping_array = (rtxRGBAColor*)&shared_memory[offset];
    offset += sizeof(rtxRGBAColor) * color_mapping_array_size;

    cudaTextureObject_t* shared_serialized_texture_object_array = (cudaTextureObject_t*)&shared_memory[offset];
    offset += sizeof(cudaTextureObject_t) * num_active_texture_units;

    // ブロック内のどれか1スレッドが代表して共有メモリに内容をコピー
    if (threadIdx.x == 0) {
        for (int m = 0; m < object_array_size; m++) {
            shared_serialized_object_array[m] = global_serialized_object_array[m];
        }
        for (int m = 0; m < material_attribute_byte_array_size; m++) {
            shared_serialized_material_attribute_byte_array[m] = global_serialized_material_attribute_byte_array[m];
        }
        for (int m = 0; m < threaded_bvh_array_size; m++) {
            shared_serialized_threaded_bvh_array[m] = global_serialized_threaded_bvh_array[m];
        }
        for (int m = 0; m < color_mapping_array_size; m++) {
            shared_serialized_color_mapping_array[m] = global_serialized_color_mapping_array[m];
        }
        for (int m = 0; m < num_active_texture_units; m++) {
            shared_serialized_texture_object_array[m] = global_serialized_mapping_texture_object_array[m];
        }
    }
    __syncthreads();

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= ray_array_size) {
            return;
        }
        rtxRay ray = global_serialized_ray_array[ray_index];
        float3 hit_point;
        float3 hit_face_normal;
        rtxVertex hit_va;
        rtxVertex hit_vb;
        rtxVertex hit_vc;
        rtxFaceVertexIndex hit_face;
        rtxObject hit_object;

        float3 ray_direction_inv = {
            1.0f / ray.direction.x,
            1.0f / ray.direction.y,
            1.0f / ray.direction.z,
        };

        // 出力する画素
        rtxRGBAPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };

        // 光輸送経路のウェイト
        rtxRGBAColor path_weight = { 1.0f, 1.0f, 1.0f };

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            // シーン上の全オブジェクトについて
            for (int object_index = 0; object_index < object_array_size; object_index++) {
                rtxObject object = shared_serialized_object_array[object_index];

                // 各ジオメトリのThreaded BVH
                rtxThreadedBVH bvh = shared_serialized_threaded_bvh_array[object_index];

                // BVHの各ノードを遷移していく
                int bvh_current_node_index = 0;
                for (int traversal = 0; traversal < bvh.num_nodes; traversal++) {
                    if (bvh_current_node_index == THREADED_BVH_TERMINAL_NODE) {
                        // 終端ノードならこのオブジェクトにはヒットしていない
                        break;
                    }
                    int serialized_node_index = bvh.serial_node_index_offset + bvh_current_node_index;
                    rtxThreadedBVHNode node = global_serialized_threaded_bvh_node_array[serialized_node_index];

                    bool is_inner_node = node.assigned_face_index_start == -1;
                    if (is_inner_node) {
                        // 中間ノードの場合AABBとの衝突判定を行う
                        // 詳細は以下参照
                        // An Efficient and Robust Ray–Box Intersection Algorithm
                        // http://www.cs.utah.edu/~awilliam/box/box.pdf
                        rtx_cuda_kernel_bvh_traversal_one_step_or_continue(node, ray_direction_inv, bvh_current_node_index);
                    } else {
                        // 葉ノード
                        // 割り当てられたジオメトリの各面との衝突判定を行う
                        // アルゴリズムの詳細は以下
                        // Fast Minimum Storage Ray/Triangle Intersectio
                        // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
                        int num_assigned_faces = node.assigned_face_index_end - node.assigned_face_index_start + 1;
                        if (object.geometry_type == RTXGeometryTypeStandard) {

                            for (int m = 0; m < num_assigned_faces; m++) {
                                int serialized_face_index = node.assigned_face_index_start + m + object.serialized_face_index_offset;
                                const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_face_index];

                                const rtxVertex va = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                                const rtxVertex vb = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                                const rtxVertex vc = global_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                                float3 face_normal;
                                float distance;
                                rtx_cuda_kernel_intersect_triangle_or_continue(va, vb, vc, face_normal, distance, min_distance);

                                min_distance = distance;
                                hit_point.x = ray.origin.x + distance * ray.direction.x;
                                hit_point.y = ray.origin.y + distance * ray.direction.y;
                                hit_point.z = ray.origin.z + distance * ray.direction.z;

                                hit_face_normal.x = face_normal.x;
                                hit_face_normal.y = face_normal.y;
                                hit_face_normal.z = face_normal.z;

                                hit_va = va;
                                hit_vb = vb;
                                hit_vc = vc;
                                hit_face = face;

                                did_hit_object = true;
                                hit_object = object;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int serialized_array_index = node.assigned_face_index_start + object.serialized_face_index_offset;
                            const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_array_index];

                            const rtxVertex center = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex radius = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];

                            float distance;
                            rtx_cuda_kernel_intersect_sphere_or_continue(center, radius, distance, min_distance);

                            min_distance = distance;
                            hit_point.x = ray.origin.x + distance * ray.direction.x;
                            hit_point.y = ray.origin.y + distance * ray.direction.y;
                            hit_point.z = ray.origin.z + distance * ray.direction.z;

                            const float3 normal = {
                                hit_point.x - center.x,
                                hit_point.y - center.y,
                                hit_point.z - center.z,
                            };
                            const float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) + 1e-12;

                            hit_face_normal.x = normal.x / norm;
                            hit_face_normal.y = normal.y / norm;
                            hit_face_normal.z = normal.z / norm;

                            did_hit_object = true;
                            hit_object = object;
                        }
                    }

                    if (node.hit_node_index == THREADED_BVH_TERMINAL_NODE) {
                        bvh_current_node_index = node.miss_node_index;
                    } else {
                        bvh_current_node_index = node.hit_node_index;
                    }
                }
            }

            // 反射方向のサンプリング
            float3 unit_next_ray_direction;
            float cosine_term;
            rtx_cuda_kernel_sample_ray_direction(unit_next_ray_direction, cosine_term, curand_state);


            //  衝突点の色を検出
            rtxRGBAColor hit_color;
            bool did_hit_light = false;
            if (did_hit_object) {
                rtx_cuda_kernel_fetch_hit_color_in_linear_memory(
                    hit_point,
                    hit_face_normal,
                    hit_object,
                    hit_face,
                    hit_color,
                    ray.direction,
                    unit_next_ray_direction,
                    shared_serialized_material_attribute_byte_array,
                    shared_serialized_color_mapping_array,
                    shared_serialized_texture_object_array,
                    global_serialized_uv_coordinate_array,
                    did_hit_light);
            }

            // 光源に当たった場合トレースを打ち切り
            if (did_hit_light) {
                pixel.r += hit_color.r * path_weight.r;
                pixel.g += hit_color.g * path_weight.g;
                pixel.b += hit_color.b * path_weight.b;
                break;
            }

            if (did_hit_object) {
                rtx_cuda_kernel_update_ray(ray,
                    hit_point,
                    unit_next_ray_direction,
                    cosine_term,
                    path_weight);
            }
        }

        global_serialized_render_array[ray_index] = pixel;
    }
}
void rtx_cuda_launch_standard_global_memory_kernel(
    rtxRay* gpu_serialized_ray_array, int ray_array_size,
    rtxFaceVertexIndex* gpu_serialized_face_vertex_index_array, int face_vertex_index_array_size,
    rtxVertex* gpu_serialized_vertex_array, int vertex_array_size,
    rtxObject* gpu_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* gpu_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* gpu_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* gpu_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* gpu_serialized_color_mapping_array, int color_mapping_array_size,
    rtxUVCoordinate* gpu_serialized_uv_coordinate_array, int uv_coordinate_array_size,
    rtxRGBAPixel* gpu_serialized_render_array, int render_array_size,
    int num_active_texture_units,
    int num_threads,
    int num_blocks,
    int num_rays_per_thread,
    size_t shared_memory_bytes,
    int max_bounce,
    int curand_seed)
{
    rtx_cuda_check_kernel_arguments();
    standard_global_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        gpu_serialized_ray_array, ray_array_size,
        gpu_serialized_face_vertex_index_array, face_vertex_index_array_size,
        gpu_serialized_vertex_array, vertex_array_size,
        gpu_serialized_object_array, object_array_size,
        gpu_serialized_material_attribute_byte_array, material_attribute_byte_array_size,
        gpu_serialized_threaded_bvh_array, threaded_bvh_array_size,
        gpu_serialized_threaded_bvh_node_array, threaded_bvh_node_array_size,
        gpu_serialized_color_mapping_array, color_mapping_array_size,
        gpu_serialized_uv_coordinate_array, uv_coordinate_array_size,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_serialized_render_array,
        num_active_texture_units,
        num_rays_per_thread,
        max_bounce,
        curand_seed);
    cudaCheckError(cudaThreadSynchronize());
}