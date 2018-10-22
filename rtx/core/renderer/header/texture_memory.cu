#include "../../../header/enum.h"
#include "../../../header/struct.h"
#include "../../header/bridge.h"
#include "../../header/cuda_common.h"
#include "../../header/cuda_functions.h"
#include "../../header/cuda_texture.h"
#include "../../header/mcrt_kernel.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

__global__ void mcrt_texture_memory_kernel(
    int num_rays,
    int face_vertex_index_array_size,
    int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    int num_active_texture_units,
    int num_rays_per_thread,
    int num_rays_per_pixel,
    int max_bounce,
    RTXCameraType camera_type,
    float ray_origin_z,
    int screen_width, int screen_height,
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

    int ray_index_offset = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread;
    int num_generated_rays_per_pixel = num_rays_per_thread * int(ceilf(float(num_rays_per_pixel) / float(num_rays_per_thread)));

    int target_pixel_index = ray_index_offset / num_generated_rays_per_pixel;
    if (target_pixel_index >= screen_width * screen_height) {
        return;
    }
    int target_pixel_x = target_pixel_index % screen_width;
    int target_pixel_y = target_pixel_index / screen_width;
    float aspect_ratio = float(screen_width) / float(screen_height);
    int render_buffer_index = ray_index_offset / num_rays_per_thread;

    // 出力する画素
    rtxRGBAPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };

    // float4 supersampling_noise;
    // supersampling_noise = curand_uniform4(&curand_state);

    // uint32_t w = curand_seed;
    // uint32_t x = w << 13;
    // uint32_t y = (w >> 9) ^ (x << 6);
    // uint32_t z = y >> 7;

    unsigned long xors_x = curand_seed;
    unsigned long xors_y = 362436069;
    unsigned long xors_z = 521288629;
    unsigned long xors_w = 88675123;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = ray_index_offset + n;
        int ray_index_in_pixel = ray_index % num_generated_rays_per_pixel;
        if (ray_index_in_pixel >= num_rays_per_pixel) {
            return;
        }

        // レイの生成
        rtxCUDARay ray;
        // スーパーサンプリング
        float2 noise;

        unsigned long t = xors_x ^ (xors_x << 11);
        xors_x = xors_y;
        xors_y = xors_z;
        xors_z = xors_w;
        xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
        noise.x = float(xors_w & 0xFFFF) / 65535.0;

        t = xors_x ^ (xors_x << 11);
        xors_x = xors_y;
        xors_y = xors_z;
        xors_z = xors_w;
        xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
        noise.y = float(xors_w & 0xFFFF) / 65535.0;

        // 方向
        ray.direction.x = 2.0f * float(target_pixel_x + noise.x) / float(screen_width) - 1.0f;
        ray.direction.y = -(2.0f * float(target_pixel_y + noise.y) / float(screen_height) - 1.0f) / aspect_ratio;
        ray.direction.z = -ray_origin_z;
        // 正規化
        const float norm = sqrtf(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z);
        ray.direction.x /= norm;
        ray.direction.y /= norm;
        ray.direction.z /= norm;
        // 始点
        if (camera_type == RTXCameraTypePerspective) {
            ray.origin = { 0.0f, 0.0f, ray_origin_z };
        } else {
            ray.origin = { ray.direction.x, ray.direction.y, ray_origin_z };
        }

        float3 hit_point;
        float3 unit_hit_face_normal;
        float4 hit_va;
        float4 hit_vb;
        float4 hit_vc;
        rtxFaceVertexIndex hit_face;
        rtxObject hit_object;

        float3 ray_direction_inv = {
            1.0f / ray.direction.x,
            1.0f / ray.direction.y,
            1.0f / ray.direction.z,
        };

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
                        __rtx_bvh_traversal_one_step_or_continue(ray, node, ray_direction_inv, bvh_current_node_index);
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
                                const int4 face = tex1Dfetch(g_serialized_face_vertex_index_array_texture_ref, serialized_face_index);

                                const float4 va = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.x + object.serialized_vertex_index_offset);
                                const float4 vb = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.y + object.serialized_vertex_index_offset);
                                const float4 vc = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.z + object.serialized_vertex_index_offset);

                                float3 face_normal;
                                float distance;
                                __rtx_intersect_triangle_or_continue(ray, va, vb, vc, face_normal, distance, min_distance);

                                min_distance = distance;
                                hit_point.x = ray.origin.x + distance * ray.direction.x;
                                hit_point.y = ray.origin.y + distance * ray.direction.y;
                                hit_point.z = ray.origin.z + distance * ray.direction.z;

                                unit_hit_face_normal.x = face_normal.x;
                                unit_hit_face_normal.y = face_normal.y;
                                unit_hit_face_normal.z = face_normal.z;

                                hit_va = va;
                                hit_vb = vb;
                                hit_vc = vc;

                                hit_face.a = face.x;
                                hit_face.b = face.y;
                                hit_face.c = face.z;

                                did_hit_object = true;
                                hit_object = object;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int serialized_array_index = node.assigned_face_index_start + object.serialized_face_index_offset;
                            const int4 face = tex1Dfetch(g_serialized_face_vertex_index_array_texture_ref, serialized_array_index);

                            const float4 center = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.x + object.serialized_vertex_index_offset);
                            const float4 radius = tex1Dfetch(g_serialized_vertex_array_texture_ref, face.y + object.serialized_vertex_index_offset);

                            float distance;
                            __rtx_intersect_sphere_or_continue(ray, center, radius, distance, min_distance);

                            min_distance = distance;
                            hit_point.x = ray.origin.x + distance * ray.direction.x;
                            hit_point.y = ray.origin.y + distance * ray.direction.y;
                            hit_point.z = ray.origin.z + distance * ray.direction.z;

                            const float3 normal = {
                                hit_point.x - center.x,
                                hit_point.y - center.y,
                                hit_point.z - center.z,
                            };
                            const float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

                            unit_hit_face_normal.x = normal.x / norm;
                            unit_hit_face_normal.y = normal.y / norm;
                            unit_hit_face_normal.z = normal.z / norm;

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

            if (did_hit_object == false) {
                break;
            }

            // 反射方向のサンプリング
            float3 unit_next_path_direction;
            float cosine_term;
            __rtx_sample_ray_direction(
                unit_hit_face_normal,
                unit_next_path_direction,
                cosine_term,
                curand_state);

            //  衝突点の色を検出
            rtxRGBAColor hit_color;
            bool did_hit_light = false;
            float brdf = 0.0f;
            __rtx_fetch_color_in_texture_memory(
                hit_point,
                unit_hit_face_normal,
                hit_object,
                hit_face,
                hit_color,
                ray.direction,
                unit_next_path_direction,
                shared_serialized_material_attribute_byte_array,
                shared_serialized_color_mapping_array,
                shared_serialized_texture_object_array,
                g_serialized_uv_coordinate_array_texture_ref,
                brdf,
                did_hit_light);

            // 光源に当たった場合トレースを打ち切り
            if (did_hit_light) {
                float intensity = brdf;
                pixel.r += hit_color.r * path_weight.r * intensity;
                pixel.g += hit_color.g * path_weight.g * intensity;
                pixel.b += hit_color.b * path_weight.b * intensity;
                break;
            }

            __rtx_update_ray(ray, hit_point, unit_next_path_direction);

            // 経路のウェイトを更新
            float inv_pdf = 2.0f * M_PI;
            path_weight.r *= hit_color.r * brdf * cosine_term * inv_pdf;
            path_weight.g *= hit_color.g * brdf * cosine_term * inv_pdf;
            path_weight.b *= hit_color.b * brdf * cosine_term * inv_pdf;
        }
    }
    global_serialized_render_array[render_buffer_index] = pixel;
}

void rtx_cuda_launch_mcrt_texture_memory_kernel(
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
    int num_rays_per_pixel,
    size_t shared_memory_bytes,
    int max_bounce,
    RTXCameraType camera_type,
    float ray_origin_z,
    int screen_width, int screen_height,
    int curand_seed)
{
    __check_kernel_arguments();

    // cudaBindTexture(0, g_serialized_ray_array_texture_ref, gpu_serialized_ray_array, cudaCreateChannelDesc<float4>(), sizeof(rtxRay) * ray_array_size);
    cudaBindTexture(0, g_serialized_face_vertex_index_array_texture_ref, gpu_serialized_face_vertex_index_array, cudaCreateChannelDesc<int4>(), sizeof(rtxFaceVertexIndex) * face_vertex_index_array_size);
    cudaBindTexture(0, g_serialized_vertex_array_texture_ref, gpu_serialized_vertex_array, cudaCreateChannelDesc<float4>(), sizeof(rtxVertex) * vertex_array_size);
    cudaBindTexture(0, g_serialized_threaded_bvh_node_array_texture_ref, gpu_serialized_threaded_bvh_node_array, cudaCreateChannelDesc<float4>(), sizeof(rtxThreadedBVHNode) * threaded_bvh_node_array_size);
    cudaBindTexture(0, g_serialized_uv_coordinate_array_texture_ref, gpu_serialized_uv_coordinate_array, cudaCreateChannelDesc<float2>(), sizeof(rtxUVCoordinate) * uv_coordinate_array_size);

    mcrt_texture_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        ray_array_size,
        face_vertex_index_array_size,
        vertex_array_size,
        gpu_serialized_object_array, object_array_size,
        gpu_serialized_material_attribute_byte_array, material_attribute_byte_array_size,
        gpu_serialized_threaded_bvh_array, threaded_bvh_array_size,
        threaded_bvh_node_array_size,
        gpu_serialized_color_mapping_array, color_mapping_array_size,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_serialized_render_array,
        num_active_texture_units,
        num_rays_per_thread,
        num_rays_per_pixel,
        max_bounce,
        camera_type,
        ray_origin_z,
        screen_width, screen_height,
        curand_seed);

    cudaCheckError(cudaThreadSynchronize());

    // cudaUnbindTexture(g_serialized_ray_array_texture_ref);
    cudaUnbindTexture(g_serialized_face_vertex_index_array_texture_ref);
    cudaUnbindTexture(g_serialized_vertex_array_texture_ref);
    cudaUnbindTexture(g_serialized_threaded_bvh_node_array_texture_ref);
    cudaUnbindTexture(g_serialized_uv_coordinate_array_texture_ref);
}