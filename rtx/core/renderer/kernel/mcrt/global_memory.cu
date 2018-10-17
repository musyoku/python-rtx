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

__global__ void mcrt_global_memory_kernel(
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
    int num_rays_per_pixel,
    int max_bounce,
    RTXCameraType camera_type,
    float ray_origin_z,
    int screen_width, int screen_height,
    rtxRGBAColor ambient_color, 
    int curand_seed)
{
    extern __shared__ char shared_memory[];
    curandStatePhilox4_32_10_t curand_state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &curand_state);
    __xorshift_init(curand_seed);

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
        __xorshift_uniform(noise.x, xors_x, xors_y, xors_z, xors_w);
        __xorshift_uniform(noise.y, xors_x, xors_y, xors_z, xors_w);
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
        // BVHのAABBとの衝突判定で使う
        float3 ray_direction_inv = {
            1.0f / ray.direction.x,
            1.0f / ray.direction.y,
            1.0f / ray.direction.z,
        };

        float3 hit_point;
        float3 unit_hit_face_normal;
        rtxVertex hit_va;
        rtxVertex hit_vb;
        rtxVertex hit_vc;
        rtxFaceVertexIndex hit_face;
        rtxObject hit_object;

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
                                const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_face_index];

                                const rtxVertex va = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                                const rtxVertex vb = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                                const rtxVertex vc = global_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

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
                            const float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) + 1e-12;

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
                pixel.r += ambient_color.r;
                pixel.g += ambient_color.g;
                pixel.b += ambient_color.b;
                break;
            }

            // 反射方向のサンプリング
            float3 unit_next_ray_direction;
            float cosine_term;
            __rtx_sample_ray_direction(
                unit_hit_face_normal,
                unit_next_ray_direction,
                cosine_term,
                curand_state);

            //  衝突点の色を検出
            rtxRGBAColor hit_color;
            bool did_hit_light = false;
            float brdf = 0.0f;
            rtx_cuda_fetch_hit_color_in_linear_memory(
                hit_point,
                unit_hit_face_normal,
                hit_object,
                hit_face,
                hit_color,
                ray.direction,
                unit_next_ray_direction,
                shared_serialized_material_attribute_byte_array,
                shared_serialized_color_mapping_array,
                shared_serialized_texture_object_array,
                global_serialized_uv_coordinate_array,
                brdf,
                did_hit_light);

            // 光源に当たった場合トレースを打ち切り
            if (did_hit_light) {
                rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                if (bounce == 0 && attr.visible == false) {
                    pixel.r += ambient_color.r;
                    pixel.g += ambient_color.g;
                    pixel.b += ambient_color.b;
                } else {
                    pixel.r += hit_color.r * path_weight.r * attr.brightness;
                    pixel.g += hit_color.g * path_weight.g * attr.brightness;
                    pixel.b += hit_color.b * path_weight.b * attr.brightness;
                }
                break;
            }

            __rtx_update_ray(ray, ray_direction_inv, hit_point, unit_next_ray_direction);

            // 経路のウェイトを更新
            float inv_pdf = 2.0f * M_PI;
            path_weight.r *= hit_color.r * brdf * cosine_term * inv_pdf;
            path_weight.g *= hit_color.g * brdf * cosine_term * inv_pdf;
            path_weight.b *= hit_color.b * brdf * cosine_term * inv_pdf;
        }
    }
    global_serialized_render_array[render_buffer_index] = pixel;
}
void rtx_cuda_launch_mcrt_global_memory_kernel(
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
    rtxRGBAColor ambient_color, 
    int curand_seed)
{
    __check_kernel_arguments();
    mcrt_global_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
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
        num_rays_per_pixel,
        max_bounce,
        camera_type,
        ray_origin_z,
        screen_width, screen_height,
        ambient_color,
        curand_seed);
    cudaCheckError(cudaThreadSynchronize());
}