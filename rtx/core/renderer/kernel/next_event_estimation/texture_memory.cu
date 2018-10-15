#include "../../../header/enum.h"
#include "../../../header/struct.h"
#include "../../header/bridge.h"
#include "../../header/cuda_common.h"
#include "../../header/cuda_functions.h"
#include "../../header/cuda_texture.h"
#include "../../header/next_event_estimation_kernel.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

__global__ void nee_texture_memory_kernel(
    int ray_array_size,
    int face_vertex_index_array_size,
    int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    int* global_light_sampling_table, int light_sampling_table_size,
    float total_light_face_area,
    rtxRGBAPixel* global_serialized_render_array,
    int num_active_texture_units,
    int num_rays_per_thread,
    int max_bounce,
    int curand_seed)
{
    extern __shared__ char shared_memory[];
    curandStatePhilox4_32_10_t curand_state_philox4;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &curand_state_philox4);

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

    int* shared_light_sampling_table = (int*)&shared_memory[offset];
    offset += sizeof(int) * light_sampling_table_size;

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
        for (int m = 0; m < light_sampling_table_size; m++) {
            shared_light_sampling_table[m] = global_light_sampling_table[m];
        }
    }
    __syncthreads();

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= ray_array_size) {
            return;
        }
        rtxCUDARay* ray;
        rtxCUDARay shadow_ray;
        rtxCUDARay primary_ray;
        float3 hit_point;
        float3 hit_face_normal;
        float4 hit_va;
        float4 hit_vb;
        float4 hit_vc;
        rtxFaceVertexIndex hit_face;
        rtxObject hit_object;
        int hit_object_index;
        float g_term;
        float brdf;

        primary_ray.direction = tex1Dfetch(g_serialized_ray_array_texture_ref, ray_index * 2 + 0);
        primary_ray.origin = tex1Dfetch(g_serialized_ray_array_texture_ref, ray_index * 2 + 1);

        float3 ray_direction_inv;
        ray = &primary_ray;

        // 出力する画素
        rtxRGBAPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };

        // 光輸送経路のウェイト
        rtxRGBAColor path_weight = { 1.0f, 1.0f, 1.0f };
        rtxRGBAColor next_path_weight;

        // レイが当たるたびにシャドウレイを飛ばすので2倍ループが必要
        int total_rays = max_bounce * 2;

        for (int iter = 0; iter < total_rays; iter++) {
            bool is_shadow_ray = (iter & 1) == 1; // iter % 2

            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            ray_direction_inv.x = 1.0f / ray->direction.x;
            ray_direction_inv.y = 1.0f / ray->direction.y;
            ray_direction_inv.z = 1.0f / ray->direction.z;

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

                    // BVHノードの読み込み
                    // rtxCUDAThreadedBVHNodeは48バイトなのでfloat4（16バイト）x3に分割
                    // さらにint型の要素が4つあるためreinterpret_castでint4として解釈する
                    int serialized_node_index = bvh.serial_node_index_offset + bvh_current_node_index;
                    rtxCUDAThreadedBVHNode node;
                    float4 attributes_as_float4 = tex1Dfetch(g_serialized_threaded_bvh_node_array_texture_ref, serialized_node_index * 3 + 0);
                    int4* attributes_as_int4_ptr = reinterpret_cast<int4*>(&attributes_as_float4);
                    node.hit_node_index = attributes_as_int4_ptr->x;
                    node.miss_node_index = attributes_as_int4_ptr->y;
                    node.assigned_face_index_start = attributes_as_int4_ptr->z;
                    node.assigned_face_index_end = attributes_as_int4_ptr->w;
                    node.aabb_max = tex1Dfetch(g_serialized_threaded_bvh_node_array_texture_ref, serialized_node_index * 3 + 1);
                    node.aabb_min = tex1Dfetch(g_serialized_threaded_bvh_node_array_texture_ref, serialized_node_index * 3 + 2);

                    bool is_inner_node = node.assigned_face_index_start == -1;
                    if (is_inner_node) {
                        // 中間ノードの場合AABBとの衝突判定を行う
                        // 詳細は以下参照
                        // An Efficient and Robust Ray–Box Intersection Algorithm
                        // http://www.cs.utah.edu/~awilliam/box/box.pdf
                        rtx_cuda_kernel_bvh_traversal_one_step_or_continue((*ray), node, ray_direction_inv, bvh_current_node_index);
                    } else {
                        // 葉ノード
                        // 割り当てられたジオメトリの各面との衝突判定を行う
                        // アルゴリズムの詳細は以下
                        // Fast Minimum Storage Ray/Triangle Intersectio
                        // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
                        int num_assigned_faces = node.assigned_face_index_end - node.assigned_face_index_start + 1;
                        if (object.geometry_type == RTXGeometryTypeStandard) {

                            for (int m = 0; m < num_assigned_faces; m++) {
                                const int serialized_face_index = node.assigned_face_index_start + m + object.serialized_face_index_offset;
                                const int4 face = tex1Dfetch(g_serialized_face_vertex_index_array_texture_ref, serialized_face_index);

                                const float4 va = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.x + object.serialized_vertex_index_offset);
                                const float4 vb = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.y + object.serialized_vertex_index_offset);
                                const float4 vc = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.z + object.serialized_vertex_index_offset);

                                float3 face_normal;
                                float distance;
                                rtx_cuda_kernel_intersect_triangle_or_continue((*ray), va, vb, vc, face_normal, distance, min_distance);

                                min_distance = distance;
                                hit_point.x = ray->origin.x + distance * ray->direction.x;
                                hit_point.y = ray->origin.y + distance * ray->direction.y;
                                hit_point.z = ray->origin.z + distance * ray->direction.z;

                                hit_face_normal.x = face_normal.x;
                                hit_face_normal.y = face_normal.y;
                                hit_face_normal.z = face_normal.z;

                                hit_va = va;
                                hit_vb = vb;
                                hit_vc = vc;

                                hit_face.a = face.x;
                                hit_face.b = face.y;
                                hit_face.c = face.z;

                                did_hit_object = true;
                                hit_object = object;
                                hit_object_index = object_index;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int serialized_array_index = node.assigned_face_index_start + object.serialized_face_index_offset;
                            const int4 face = tex1Dfetch(g_serialized_face_vertex_index_array_texture_ref, serialized_array_index);

                            const float4 center = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.x + object.serialized_vertex_index_offset);
                            const float4 radius = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.y + object.serialized_vertex_index_offset);

                            float distance;
                            rtx_cuda_kernel_intersect_sphere_or_continue((*ray), center, radius, distance, min_distance);

                            min_distance = distance;
                            hit_point.x = ray->origin.x + distance * ray->direction.x;
                            hit_point.y = ray->origin.y + distance * ray->direction.y;
                            hit_point.z = ray->origin.z + distance * ray->direction.z;

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
                            hit_object_index = object_index;
                        }
                    }

                    if (node.hit_node_index == THREADED_BVH_TERMINAL_NODE) {
                        bvh_current_node_index = node.miss_node_index;
                    } else {
                        bvh_current_node_index = node.hit_node_index;
                    }
                }
            }

            if (did_hit_object == false) {
                pixel.r = 1.0f;
                pixel.g = 0.0f;
                pixel.b = 0.0f;
                break;
            }

            //  衝突点の色を検出
            rtxRGBAColor hit_color;

            if (is_shadow_ray) {
                // 光源に当たった場合寄与を加算
                int material_type = hit_object.layerd_material_types.outside;
                if (material_type == RTXMaterialTypeEmissive) {
                    rtx_cuda_kernel_fetch_light_color_in_texture_memory(
                        hit_point,
                        hit_object,
                        hit_face,
                        hit_color,
                        shared_serialized_material_attribute_byte_array,
                        shared_serialized_color_mapping_array,
                        shared_serialized_texture_object_array,
                        g_serialized_uv_coordinate_array_texture_ref);
                    rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    pixel.r += min(brdf * hit_color.r * path_weight.r * total_light_face_area * g_term, attr.brightness);
                    pixel.g += min(brdf * hit_color.g * path_weight.g * total_light_face_area * g_term, attr.brightness);
                    pixel.b += min(brdf * hit_color.b * path_weight.b * total_light_face_area * g_term, attr.brightness);
                }

                path_weight.r = next_path_weight.r;
                path_weight.g = next_path_weight.g;
                path_weight.b = next_path_weight.b;

                ray = &primary_ray;
            } else {
                // 反射方向のサンプリング
                float3 unit_next_ray_direction;
                float cosine_term;
                rtx_cuda_kernel_sample_ray_direction(unit_next_ray_direction,
                    cosine_term,
                    curand_state_philox4);

                bool did_hit_light = false;
                rtx_cuda_kernel_fetch_hit_color_in_texture_memory(
                    hit_point,
                    hit_face_normal,
                    hit_object,
                    hit_face,
                    hit_color,
                    ray->direction,
                    unit_next_ray_direction,
                    shared_serialized_material_attribute_byte_array,
                    shared_serialized_color_mapping_array,
                    shared_serialized_texture_object_array,
                    g_serialized_uv_coordinate_array_texture_ref,
                    did_hit_light);

                // 光源に当たった場合トレースを打ち切り
                if (did_hit_light) {
                    if (iter > 0) {
                        break;
                    }
                    pixel.r += hit_color.r * path_weight.r;
                    pixel.g += hit_color.g * path_weight.g;
                    pixel.b += hit_color.b * path_weight.b;
                    break;
                }

                // 入射方向のサンプリング
                ray->origin.x = hit_point.x;
                ray->origin.y = hit_point.y;
                ray->origin.z = hit_point.z;
                ray->direction.x = unit_next_ray_direction.x;
                ray->direction.y = unit_next_ray_direction.y;
                ray->direction.z = unit_next_ray_direction.z;

                int material_type = hit_object.layerd_material_types.outside;
                if (material_type == RTXMaterialTypeLambert) {
                    rtxLambertMaterialAttribute attr = ((rtxLambertMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    brdf = attr.albedo / M_PI;
                } else if (material_type == RTXMaterialTypeOrenNayar) {
                    rtxOrenNayarMaterialAttribute attr = ((rtxOrenNayarMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    brdf = attr.albedo / M_PI;
                }

                next_path_weight.r = path_weight.r * M_PI * hit_color.r * cosine_term;
                next_path_weight.g = path_weight.g * M_PI * hit_color.g * cosine_term;
                next_path_weight.b = path_weight.b * M_PI * hit_color.b * cosine_term;

                // 光源のサンプリング
                float4 uniform4 = curand_uniform4(&curand_state_philox4);
                const int table_index = floorf(uniform4.x * float(light_sampling_table_size));
                const int object_index = shared_light_sampling_table[table_index];
                rtxObject object = shared_serialized_object_array[object_index];
                const int face_index = floorf(uniform4.y * float(object.num_faces));
                const int serialized_face_index = face_index + object.serialized_face_index_offset;
                const int4 face = tex1Dfetch(g_serialized_face_vertex_index_array_texture_ref, serialized_face_index);
                const float4 va = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.x + object.serialized_vertex_index_offset);
                const float4 vb = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.y + object.serialized_vertex_index_offset);
                const float4 vc = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.z + object.serialized_vertex_index_offset);
                float4 weight = curand_uniform4(&curand_state_philox4);
                const float sum = weight.x + weight.y + weight.z;
                weight.x /= sum;
                weight.y /= sum;
                weight.z /= sum;
                const float3 random_point = {
                    weight.x * va.x + weight.y * vb.x + weight.z * vc.x,
                    weight.x * va.y + weight.y * vb.y + weight.z * vc.y,
                    weight.x * va.z + weight.y * vb.z + weight.z * vc.z,
                };
                shadow_ray.origin.x = hit_point.x;
                shadow_ray.origin.y = hit_point.y;
                shadow_ray.origin.z = hit_point.z;

                shadow_ray.direction.x = random_point.x - hit_point.x;
                shadow_ray.direction.y = random_point.y - hit_point.y;
                shadow_ray.direction.z = random_point.z - hit_point.z;
                const float light_distance = sqrtf(shadow_ray.direction.x * shadow_ray.direction.x + shadow_ray.direction.y * shadow_ray.direction.y + shadow_ray.direction.z * shadow_ray.direction.z);
                shadow_ray.direction.x /= light_distance;
                shadow_ray.direction.y /= light_distance;
                shadow_ray.direction.z /= light_distance;

                float dot1 = shadow_ray.direction.x * hit_face_normal.x
                    + shadow_ray.direction.y * hit_face_normal.y
                    + shadow_ray.direction.z * hit_face_normal.z;
                if (dot1 <= 0.0f) {
                    ray = &primary_ray;
                    iter += 1;
                    continue;
                }

                float3 edge_ba = {
                    vb.x - va.x,
                    vb.y - va.y,
                    vb.z - va.z,
                };
                float3 edge_ca = {
                    vc.x - va.x,
                    vc.y - va.y,
                    vc.z - va.z,
                };
                float3 light_normal;
                light_normal.x = edge_ba.y * edge_ca.z - edge_ba.z * edge_ca.y;
                light_normal.y = edge_ba.z * edge_ca.x - edge_ba.x * edge_ca.z;
                light_normal.z = edge_ba.x * edge_ca.y - edge_ba.y * edge_ca.x;
                const float norm = sqrtf(light_normal.x * light_normal.x + light_normal.y * light_normal.y + light_normal.z * light_normal.z) + 1e-12;
                light_normal.x = light_normal.x / norm;
                light_normal.y = light_normal.y / norm;
                light_normal.z = light_normal.z / norm;
                float dot2 = -(shadow_ray.direction.x * light_normal.x + shadow_ray.direction.y * light_normal.y + shadow_ray.direction.z * light_normal.z);
                if (dot2 < 0.0f) {
                    dot2 *= -1.0f;
                }
                g_term = dot1 * dot2 / (light_distance * light_distance);
                g_term = dot1 / (light_distance * light_distance);

                if (g_term > 1 && threadIdx.x == 78 && 289024 < ray_index && ray_index <= 289152) {
                    printf("hit_point: %f, %f, %f\n", hit_point.x, hit_point.y, hit_point.z);
                    printf("random_point: %f, %f, %f\n", random_point.x, random_point.y, random_point.z);
                    printf("shadow_ray.direction: %f, %f, %f\n", shadow_ray.direction.x, shadow_ray.direction.y, shadow_ray.direction.z);

                    printf("G: %f = %f * %f / %f iter: %d thread: %d\n", g_term, dot1, dot2, (light_distance * light_distance), iter, threadIdx.x);
                }
                ray = &shadow_ray;
            }
        }

        global_serialized_render_array[ray_index] = pixel;
    }
}

void rtx_cuda_launch_nee_texture_memory_kernel(
    rtxRay* gpu_serialized_ray_array, int ray_array_size,
    rtxFaceVertexIndex* gpu_serialized_face_vertex_index_array, int face_vertex_index_array_size,
    rtxVertex* gpu_serialized_vertex_array, int vertex_array_size,
    rtxObject* gpu_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* gpu_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* gpu_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* gpu_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* gpu_serialized_color_mapping_array, int color_mapping_array_size,
    rtxUVCoordinate* gpu_serialized_uv_coordinate_array, int uv_coordinate_array_size,
    int* gpu_light_sampling_table, int light_sampling_table_size,
    float total_light_face_area,
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

    cudaBindTexture(0, g_serialized_ray_array_texture_ref, gpu_serialized_ray_array, cudaCreateChannelDesc<float4>(), sizeof(rtxRay) * ray_array_size);
    cudaBindTexture(0, g_serialized_face_vertex_index_array_texture_ref, gpu_serialized_face_vertex_index_array, cudaCreateChannelDesc<int4>(), sizeof(rtxFaceVertexIndex) * face_vertex_index_array_size);
    cudaBindTexture(0, g_serialized_vertex_array_texture_ref, gpu_serialized_vertex_array, cudaCreateChannelDesc<float4>(), sizeof(rtxVertex) * vertex_array_size);
    cudaBindTexture(0, g_serialized_threaded_bvh_node_array_texture_ref, gpu_serialized_threaded_bvh_node_array, cudaCreateChannelDesc<float4>(), sizeof(rtxThreadedBVHNode) * threaded_bvh_node_array_size);
    cudaBindTexture(0, g_serialized_uv_coordinate_array_texture_ref, gpu_serialized_uv_coordinate_array, cudaCreateChannelDesc<float2>(), sizeof(rtxUVCoordinate) * uv_coordinate_array_size);

    nee_texture_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        ray_array_size,
        face_vertex_index_array_size,
        vertex_array_size,
        gpu_serialized_object_array, object_array_size,
        gpu_serialized_material_attribute_byte_array, material_attribute_byte_array_size,
        gpu_serialized_threaded_bvh_array, threaded_bvh_array_size,
        threaded_bvh_node_array_size,
        gpu_serialized_color_mapping_array, color_mapping_array_size,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_light_sampling_table, light_sampling_table_size,
        total_light_face_area,
        gpu_serialized_render_array,
        num_active_texture_units,
        num_rays_per_thread,
        max_bounce,
        curand_seed);

    cudaCheckError(cudaThreadSynchronize());

    cudaUnbindTexture(g_serialized_ray_array_texture_ref);
    cudaUnbindTexture(g_serialized_face_vertex_index_array_texture_ref);
    cudaUnbindTexture(g_serialized_vertex_array_texture_ref);
    cudaUnbindTexture(g_serialized_threaded_bvh_node_array_texture_ref);
    cudaUnbindTexture(g_serialized_uv_coordinate_array_texture_ref);
}