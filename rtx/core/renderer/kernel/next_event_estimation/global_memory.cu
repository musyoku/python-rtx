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

__global__ void nee_global_memory_kernel(
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array, int face_vertex_index_array_size,
    rtxVertex* global_serialized_vertex_array, int vertex_array_size,
    rtxObject* global_serialized_object_array, int object_array_size,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array, int material_attribute_byte_array_size,
    rtxThreadedBVH* global_serialized_threaded_bvh_array, int threaded_bvh_array_size,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array, int threaded_bvh_node_array_size,
    rtxRGBAColor* global_serialized_color_mapping_array, int color_mapping_array_size,
    rtxUVCoordinate* global_serialized_uv_coordinate_array, int uv_coordinate_array_size,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    int* global_light_sampling_table, int light_sampling_table_size,
    float total_light_face_area,
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

        rtxCUDARay* ray;
        rtxCUDARay shadow_ray;
        rtxCUDARay primary_ray;
        float3 hit_point;
        float3 unit_hit_face_normal;
        rtxVertex hit_va;
        rtxVertex hit_vb;
        rtxVertex hit_vc;
        rtxFaceVertexIndex hit_face;
        rtxObject hit_object;
        float g_term;
        float brdf;

        // スーパーサンプリング
        float2 noise;
        __xorshift_uniform(noise.x, xors_x, xors_y, xors_z, xors_w);
        __xorshift_uniform(noise.y, xors_x, xors_y, xors_z, xors_w);
        // 方向
        primary_ray.direction.x = 2.0f * float(target_pixel_x + noise.x) / float(screen_width) - 1.0f;
        primary_ray.direction.y = -(2.0f * float(target_pixel_y + noise.y) / float(screen_height) - 1.0f) / aspect_ratio;
        primary_ray.direction.z = -ray_origin_z;
        // 正規化
        const float norm = sqrtf(primary_ray.direction.x * primary_ray.direction.x + primary_ray.direction.y * primary_ray.direction.y + primary_ray.direction.z * primary_ray.direction.z);
        primary_ray.direction.x /= norm;
        primary_ray.direction.y /= norm;
        primary_ray.direction.z /= norm;
        // 始点
        if (camera_type == RTXCameraTypePerspective) {
            primary_ray.origin = { 0.0f, 0.0f, ray_origin_z };
        } else {
            primary_ray.origin = { primary_ray.direction.x, primary_ray.direction.y, ray_origin_z };
        }

        float3 ray_direction_inv;
        ray = &primary_ray;

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
                    rtxThreadedBVHNode node = global_serialized_threaded_bvh_node_array[serialized_node_index];

                    bool is_inner_node = node.assigned_face_index_start == -1;
                    if (is_inner_node) {
                        // 中間ノードの場合AABBとの衝突判定を行う
                        // 詳細は以下参照
                        // An Efficient and Robust Ray–Box Intersection Algorithm
                        // http://www.cs.utah.edu/~awilliam/box/box.pdf
                        __rtx_bvh_traversal_one_step_or_continue((*ray), node, ray_direction_inv, bvh_current_node_index);
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
                                const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_face_index];

                                const rtxVertex va = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                                const rtxVertex vb = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                                const rtxVertex vc = global_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                                float3 face_normal;
                                float distance;
                                __rtx_intersect_triangle_or_continue((*ray), va, vb, vc, face_normal, distance, min_distance);

                                min_distance = distance;
                                hit_point.x = ray->origin.x + distance * ray->direction.x;
                                hit_point.y = ray->origin.y + distance * ray->direction.y;
                                hit_point.z = ray->origin.z + distance * ray->direction.z;

                                unit_hit_face_normal.x = face_normal.x;
                                unit_hit_face_normal.y = face_normal.y;
                                unit_hit_face_normal.z = face_normal.z;

                                hit_va = va;
                                hit_vb = vb;
                                hit_vc = vc;

                                hit_face.a = face.a;
                                hit_face.b = face.b;
                                hit_face.c = face.c;

                                did_hit_object = true;
                                hit_object = object;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int serialized_array_index = node.assigned_face_index_start + object.serialized_face_index_offset;
                            const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_array_index];

                            const rtxVertex center = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex radius = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_sphere_or_continue((*ray), center, radius, distance, min_distance);

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

            //  衝突点の色を検出
            rtxRGBAColor hit_color;

            if (is_shadow_ray) {
                // 光源に当たった場合寄与を加算
                int material_type = hit_object.layerd_material_types.outside;
                if (material_type == RTXMaterialTypeEmissive) {
                    __rtx_fetch_light_color_in_texture_memory(
                        hit_point,
                        hit_object,
                        hit_face,
                        hit_color,
                        shared_serialized_material_attribute_byte_array,
                        shared_serialized_color_mapping_array,
                        shared_serialized_texture_object_array,
                        g_serialized_uv_coordinate_array_texture_ref);
                    rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    float emission = attr.brightness;
                    float inv_pdf = total_light_face_area;
                    // pixel.r += min(path_weight.r * emission * brdf * hit_color.r * inv_pdf * g_term, attr.brightness);
                    // pixel.g += min(path_weight.g * emission * brdf * hit_color.g * inv_pdf * g_term, attr.brightness);
                    // pixel.b += min(path_weight.b * emission * brdf * hit_color.b * inv_pdf * g_term, attr.brightness);
                    pixel.r += path_weight.r * emission * brdf * hit_color.r * inv_pdf * g_term;
                    pixel.g += path_weight.g * emission * brdf * hit_color.g * inv_pdf * g_term;
                    pixel.b += path_weight.b * emission * brdf * hit_color.b * inv_pdf * g_term;
                }

                path_weight.r = next_path_weight.r;
                path_weight.g = next_path_weight.g;
                path_weight.b = next_path_weight.b;

                ray = &primary_ray;
            } else {
                // 反射方向のサンプリング
                float3 unit_next_ray_direction;
                float cosine_term;
                __rtx_sample_ray_direction(
                    unit_hit_face_normal,
                    unit_next_ray_direction,
                    cosine_term,
                    curand_state);

                bool did_hit_light = false;
                brdf = 0.0f;
                __rtx_fetch_hit_color_in_texture_memory(
                    hit_point,
                    unit_hit_face_normal,
                    hit_object,
                    hit_face,
                    hit_color,
                    ray->direction,
                    unit_next_ray_direction,
                    shared_serialized_material_attribute_byte_array,
                    shared_serialized_color_mapping_array,
                    shared_serialized_texture_object_array,
                    g_serialized_uv_coordinate_array_texture_ref,
                    brdf,
                    did_hit_light);

                // 光源に当たった場合トレースを打ち切り
                if (did_hit_light) {
                    if (iter > 0) {
                        break;
                    }
                    // 最初のパスで光源に当たった場合のみ寄与を加算
                    rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    if (attr.visible) {
                        pixel.r += hit_color.r * path_weight.r * attr.brightness;
                        pixel.g += hit_color.g * path_weight.g * attr.brightness;
                        pixel.b += hit_color.b * path_weight.b * attr.brightness;
                    } else {
                        pixel.r += ambient_color.r;
                        pixel.g += ambient_color.g;
                        pixel.b += ambient_color.b;
                    }
                    break;
                }

                // 入射方向のサンプリング
                ray->origin.x = hit_point.x;
                ray->origin.y = hit_point.y;
                ray->origin.z = hit_point.z;
                ray->direction.x = unit_next_ray_direction.x;
                ray->direction.y = unit_next_ray_direction.y;
                ray->direction.z = unit_next_ray_direction.z;

                float inv_pdf = 2.0f * M_PI;
                next_path_weight.r = path_weight.r * brdf * hit_color.r * cosine_term * inv_pdf;
                next_path_weight.g = path_weight.g * brdf * hit_color.g * cosine_term * inv_pdf;
                next_path_weight.b = path_weight.b * brdf * hit_color.b * cosine_term * inv_pdf;

                // 光源のサンプリング
                float2 uniform2;
                __xorshift_uniform(uniform2.x, xors_x, xors_y, xors_z, xors_w);
                __xorshift_uniform(uniform2.y, xors_x, xors_y, xors_z, xors_w);
                const int table_index = floorf(uniform2.x * float(light_sampling_table_size));
                const int object_index = shared_light_sampling_table[table_index];
                rtxObject object = shared_serialized_object_array[object_index];
                const int face_index = floorf(uniform2.y * float(object.num_faces));
                const int serialized_face_index = face_index + object.serialized_face_index_offset;
                const rtxFaceVertexIndex face = global_serialized_face_vertex_indices_array[serialized_face_index];
                const rtxVertex va = global_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                const rtxVertex vb = global_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                const rtxVertex vc = global_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                // 面上の一点の一様なサンプリング
                // http://www.cs.princeton.edu/~funk/tog02.pdf
                float r1, r2;
                __xorshift_uniform(r1, xors_x, xors_y, xors_z, xors_w);
                __xorshift_uniform(r2, xors_x, xors_y, xors_z, xors_w);
                r1 = sqrtf(r1);
                const float3 random_point = {
                    (1.0f - r1) * va.x + (r1 * (1.0f - r2)) * vb.x + (r1 * r2) * vc.x,
                    (1.0f - r1) * va.y + (r1 * (1.0f - r2)) * vb.y + (r1 * r2) * vc.y,
                    (1.0f - r1) * va.z + (r1 * (1.0f - r2)) * vb.z + (r1 * r2) * vc.z,
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

                const float dot1 = shadow_ray.direction.x * unit_hit_face_normal.x
                    + shadow_ray.direction.y * unit_hit_face_normal.y
                    + shadow_ray.direction.z * unit_hit_face_normal.z;
                if (dot1 <= 0.0f) {
                    ray = &primary_ray;
                    iter += 1;
                    continue;
                }

                const float3 edge_ba = {
                    vb.x - va.x,
                    vb.y - va.y,
                    vb.z - va.z,
                };
                const float3 edge_ca = {
                    vc.x - va.x,
                    vc.y - va.y,
                    vc.z - va.z,
                };
                float3 light_normal;
                light_normal.x = edge_ba.y * edge_ca.z - edge_ba.z * edge_ca.y;
                light_normal.y = edge_ba.z * edge_ca.x - edge_ba.x * edge_ca.z;
                light_normal.z = edge_ba.x * edge_ca.y - edge_ba.y * edge_ca.x;
                const float norm = sqrtf(light_normal.x * light_normal.x + light_normal.y * light_normal.y + light_normal.z * light_normal.z) + 1e-12;
                light_normal.x /= norm;
                light_normal.y /= norm;
                light_normal.z /= norm;
                const float dot2 = fabsf(shadow_ray.direction.x * light_normal.x + shadow_ray.direction.y * light_normal.y + shadow_ray.direction.z * light_normal.z);
                
                // ハック
                const float r = max(light_distance, 0.5f);
                g_term = dot1 * dot2 / (r * r);
                ray = &shadow_ray;
            }
        }
    }
    global_serialized_render_array[render_buffer_index] = pixel;
}
void rtx_cuda_launch_nee_global_memory_kernel(
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
    nee_global_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        gpu_serialized_face_vertex_index_array, face_vertex_index_array_size,
        gpu_serialized_vertex_array, vertex_array_size,
        gpu_serialized_object_array, object_array_size,
        gpu_serialized_material_attribute_byte_array, material_attribute_byte_array_size,
        gpu_serialized_threaded_bvh_array, threaded_bvh_array_size,
        gpu_serialized_threaded_bvh_node_array, threaded_bvh_node_array_size,
        gpu_serialized_color_mapping_array, color_mapping_array_size,
        gpu_serialized_uv_coordinate_array, uv_coordinate_array_size,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_light_sampling_table, light_sampling_table_size,
        total_light_face_area,
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