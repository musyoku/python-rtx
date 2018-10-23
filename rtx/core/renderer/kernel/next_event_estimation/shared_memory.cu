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

__global__ void nee_shared_memory_kernel(
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array,
    rtxVertex* global_serialized_vertex_array,
    rtxObject* global_serialized_object_array,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array,
    rtxThreadedBVH* global_serialized_threaded_bvh_array,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array,
    rtxRGBAColor* global_serialized_color_mapping_array,
    rtxUVCoordinate* global_serialized_uv_coordinate_array,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    int* global_light_sampling_table,
    rtxRGBAPixel* global_serialized_render_array,
    rtxNEEKernelArguments args)
{
    extern __shared__ char shared_memory[];
    curandStatePhilox4_32_10_t curand_state;
    curand_init(args.curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &curand_state);
    __xorshift_init(args.curand_seed);

    // グローバルメモリの直列データを共有メモリにコピーする
    int offset = 0;
    rtxFaceVertexIndex* shared_serialized_face_vertex_indices_array = (rtxFaceVertexIndex*)&shared_memory[offset];
    offset += sizeof(rtxFaceVertexIndex) * args.face_vertex_index_array_size;

    rtxVertex* shared_serialized_vertex_array = (rtxVertex*)&shared_memory[offset];
    offset += sizeof(rtxVertex) * args.vertex_array_size;

    rtxObject* shared_serialized_object_array = (rtxObject*)&shared_memory[offset];
    offset += sizeof(rtxObject) * args.object_array_size;

    rtxMaterialAttributeByte* shared_serialized_material_attribute_byte_array = (rtxMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(rtxMaterialAttributeByte) * args.material_attribute_byte_array_size;

    rtxThreadedBVH* shared_serialized_threaded_bvh_array = (rtxThreadedBVH*)&shared_memory[offset];
    offset += sizeof(rtxThreadedBVH) * args.threaded_bvh_array_size;

    rtxThreadedBVHNode* shared_serialized_threaded_bvh_node_array = (rtxThreadedBVHNode*)&shared_memory[offset];
    offset += sizeof(rtxThreadedBVHNode) * args.threaded_bvh_node_array_size;

    rtxRGBAColor* shared_serialized_color_mapping_array = (rtxRGBAColor*)&shared_memory[offset];
    offset += sizeof(rtxRGBAColor) * args.color_mapping_array_size;

    rtxUVCoordinate* shared_serialized_uv_coordinate_array = (rtxUVCoordinate*)&shared_memory[offset];
    offset += sizeof(rtxUVCoordinate) * args.uv_coordinate_array_size;

    // cudaTextureObject_tは8バイトなのでアライメントに気をつける
    offset += sizeof(cudaTextureObject_t) - offset % sizeof(cudaTextureObject_t);
    cudaTextureObject_t* shared_serialized_texture_object_array = (cudaTextureObject_t*)&shared_memory[offset];
    offset += sizeof(cudaTextureObject_t) * args.num_active_texture_units;

    int* shared_light_sampling_table = (int*)&shared_memory[offset];
    offset += sizeof(int) * args.light_sampling_table_size;

    if (threadIdx.x == 0) {
        for (int m = 0; m < args.face_vertex_index_array_size; m++) {
            shared_serialized_face_vertex_indices_array[m] = global_serialized_face_vertex_indices_array[m];
        }
        for (int m = 0; m < args.vertex_array_size; m++) {
            shared_serialized_vertex_array[m] = global_serialized_vertex_array[m];
        }
        for (int m = 0; m < args.object_array_size; m++) {
            shared_serialized_object_array[m] = global_serialized_object_array[m];
        }
        for (int m = 0; m < args.material_attribute_byte_array_size; m++) {
            shared_serialized_material_attribute_byte_array[m] = global_serialized_material_attribute_byte_array[m];
        }
        for (int m = 0; m < args.threaded_bvh_array_size; m++) {
            shared_serialized_threaded_bvh_array[m] = global_serialized_threaded_bvh_array[m];
        }
        for (int m = 0; m < args.threaded_bvh_node_array_size; m++) {
            shared_serialized_threaded_bvh_node_array[m] = global_serialized_threaded_bvh_node_array[m];
        }
        for (int m = 0; m < args.color_mapping_array_size; m++) {
            shared_serialized_color_mapping_array[m] = global_serialized_color_mapping_array[m];
        }
        for (int m = 0; m < args.uv_coordinate_array_size; m++) {
            shared_serialized_uv_coordinate_array[m] = global_serialized_uv_coordinate_array[m];
        }
        for (int m = 0; m < args.num_active_texture_units; m++) {
            shared_serialized_texture_object_array[m] = global_serialized_mapping_texture_object_array[m];
        }
        for (int m = 0; m < args.light_sampling_table_size; m++) {
            shared_light_sampling_table[m] = global_light_sampling_table[m];
        }
    }
    __syncthreads();

    int ray_index_offset = (blockIdx.x * blockDim.x + threadIdx.x) * args.num_rays_per_thread;
    int num_generated_rays_per_pixel = args.num_rays_per_thread * int(ceilf(float(args.num_rays_per_pixel) / float(args.num_rays_per_thread)));

    int target_pixel_index = ray_index_offset / num_generated_rays_per_pixel;
    if (target_pixel_index >= args.screen_width * args.screen_height) {
        return;
    }
    int target_pixel_x = target_pixel_index % args.screen_width;
    int target_pixel_y = target_pixel_index / args.screen_width;
    float aspect_ratio = float(args.screen_width) / float(args.screen_height);
    int render_buffer_index = ray_index_offset / args.num_rays_per_thread;

    // 出力する画素
    rtxRGBAPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int n = 0; n < args.num_rays_per_thread; n++) {
        int ray_index = ray_index_offset + n;
        int ray_index_in_pixel = ray_index % num_generated_rays_per_pixel;
        if (ray_index_in_pixel >= args.num_rays_per_pixel) {
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
        rtxRGBAColor hit_object_color;
        float g_term;
        float shadow_ray_brdf;

        // レイの生成
        __rtx_generate_ray(primary_ray, args, aspect_ratio);

        float3 ray_direction_inv;
        ray = &primary_ray;

        // 光輸送経路のウェイト
        rtxRGBAColor path_weight = { 1.0f, 1.0f, 1.0f };
        rtxRGBAColor next_path_weight;

        // レイが当たるたびにシャドウレイを飛ばすので2倍ループが必要
        int total_rays = args.max_bounce * 2;

        for (int iter = 0; iter < total_rays; iter++) {
            bool is_shadow_ray = (iter & 1) == 1; // iter % 2

            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            ray_direction_inv.x = 1.0f / ray->direction.x;
            ray_direction_inv.y = 1.0f / ray->direction.y;
            ray_direction_inv.z = 1.0f / ray->direction.z;

            // シーン上の全オブジェクトについて
            for (int object_index = 0; object_index < args.object_array_size; object_index++) {
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
                    rtxThreadedBVHNode node = shared_serialized_threaded_bvh_node_array[serialized_node_index];

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
                                const rtxFaceVertexIndex face = shared_serialized_face_vertex_indices_array[serialized_face_index];

                                const rtxVertex va = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                                const rtxVertex vb = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                                const rtxVertex vc = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

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
                            const rtxFaceVertexIndex face = shared_serialized_face_vertex_indices_array[serialized_array_index];

                            const rtxVertex center = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex radius = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_sphere_or_continue((*ray), center, radius, distance, min_distance);

                            min_distance = distance;
                            hit_point.x = ray->origin.x + distance * ray->direction.x;
                            hit_point.y = ray->origin.y + distance * ray->direction.y;
                            hit_point.z = ray->origin.z + distance * ray->direction.z;

                            unit_hit_face_normal.x = hit_point.x - center.x;
                            unit_hit_face_normal.y = hit_point.y - center.y;
                            unit_hit_face_normal.z = hit_point.z - center.z;
                            __rtx_normalize_vector(unit_hit_face_normal);

                            did_hit_object = true;
                            hit_object = object;
                        } else if (object.geometry_type == RTXGeometryTypeCylinder) {
                            rtxFaceVertexIndex face;
                            int offset = node.assigned_face_index_start + object.serialized_face_index_offset;

                            // Load cylinder parameters
                            face = shared_serialized_face_vertex_indices_array[offset + 0];
                            const rtxVertex params = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const float radius = params.x;
                            const float y_max = params.y;
                            const float y_min = params.z;

                            // Load transformation matrix
                            face = shared_serialized_face_vertex_indices_array[offset + 1];
                            const rtxVertex trans_a = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex trans_b = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex trans_c = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                            // Load inverse transformation matrix
                            face = shared_serialized_face_vertex_indices_array[offset + 2];
                            const rtxVertex inv_trans_a = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_b = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_c = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_cylinder_or_continue(
                                (*ray),
                                trans_a, trans_b, trans_c,
                                inv_trans_a, inv_trans_b, inv_trans_c,
                                unit_hit_face_normal,
                                distance,
                                min_distance);
                            min_distance = distance;

                            // hit point in view space
                            hit_point.x = ray->origin.x + distance * ray->direction.x;
                            hit_point.y = ray->origin.y + distance * ray->direction.y;
                            hit_point.z = ray->origin.z + distance * ray->direction.z;

                            did_hit_object = true;
                            hit_object = object;
                        } else if (object.geometry_type == RTXGeometryTypeCone) {
                            rtxFaceVertexIndex face;
                            int offset = node.assigned_face_index_start + object.serialized_face_index_offset;

                            // Load cylinder parameters
                            face = shared_serialized_face_vertex_indices_array[offset + 0];
                            const rtxVertex params = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const float radius = params.x;
                            const float height = params.y;

                            // Load transformation matrix
                            face = shared_serialized_face_vertex_indices_array[offset + 1];
                            const rtxVertex trans_a = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex trans_b = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex trans_c = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                            // Load inverse transformation matrix
                            face = shared_serialized_face_vertex_indices_array[offset + 2];
                            const rtxVertex inv_trans_a = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_b = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_c = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_cone_or_continue(
                                (*ray),
                                trans_a, trans_b, trans_c,
                                inv_trans_a, inv_trans_b, inv_trans_c,
                                unit_hit_face_normal,
                                distance,
                                min_distance);
                            min_distance = distance;

                            // hit point in view space
                            hit_point.x = ray->origin.x + distance * ray->direction.x;
                            hit_point.y = ray->origin.y + distance * ray->direction.y;
                            hit_point.z = ray->origin.z + distance * ray->direction.z;

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
                if (iter == 0) {
                    pixel.r += args.ambient_color.r;
                    pixel.g += args.ambient_color.g;
                    pixel.b += args.ambient_color.b;
                }
                break;
            }

            if (is_shadow_ray) {

                // 光源に当たった場合寄与を加算
                rtxRGBAColor hit_light_color;
                int material_type = hit_object.layerd_material_types.outside;
                if (material_type == RTXMaterialTypeEmissive) {
                    __rtx_fetch_color_in_linear_memory(
                        hit_point,
                        hit_object,
                        hit_face,
                        hit_light_color,
                        shared_serialized_material_attribute_byte_array,
                        shared_serialized_color_mapping_array,
                        shared_serialized_texture_object_array,
                        shared_serialized_uv_coordinate_array);

                    rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    float emission = attr.intensity;
                    float inv_pdf = args.total_light_face_area;
                    pixel.r += path_weight.r * emission * shadow_ray_brdf * hit_light_color.r * hit_object_color.r * inv_pdf * g_term;
                    pixel.g += path_weight.g * emission * shadow_ray_brdf * hit_light_color.g * hit_object_color.g * inv_pdf * g_term;
                    pixel.b += path_weight.b * emission * shadow_ray_brdf * hit_light_color.b * hit_object_color.b * inv_pdf * g_term;
                }

                path_weight.r = next_path_weight.r;
                path_weight.g = next_path_weight.g;
                path_weight.b = next_path_weight.b;

                ray = &primary_ray;
            } else {
                int material_type = hit_object.layerd_material_types.outside;
                bool did_hit_light = material_type == RTXMaterialTypeEmissive;

                __rtx_fetch_color_in_linear_memory(
                    hit_point,
                    hit_object,
                    hit_face,
                    hit_object_color,
                    shared_serialized_material_attribute_byte_array,
                    shared_serialized_color_mapping_array,
                    shared_serialized_texture_object_array,
                    shared_serialized_uv_coordinate_array);

                // 光源に当たった場合トレースを打ち切り
                if (did_hit_light) {
                    if (iter > 0) {
                        break;
                    }
                    // 最初のパスで光源に当たった場合のみ寄与を加算
                    rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    if (attr.visible) {
                        pixel.r += hit_object_color.r * path_weight.r * attr.intensity;
                        pixel.g += hit_object_color.g * path_weight.g * attr.intensity;
                        pixel.b += hit_object_color.b * path_weight.b * attr.intensity;
                    } else {
                        pixel.r += args.ambient_color.r;
                        pixel.g += args.ambient_color.g;
                        pixel.b += args.ambient_color.b;
                    }
                    break;
                }

                // 入射方向のサンプリング
                float3 unit_next_path_direction;
                float cosine_term;
                __rtx_sample_ray_direction(
                    unit_hit_face_normal,
                    unit_next_path_direction,
                    cosine_term,
                    curand_state);

                float input_ray_brdf = 0.0f;
                __rtx_compute_brdf(
                    unit_hit_face_normal,
                    hit_object,
                    hit_face,
                    primary_ray.direction,
                    unit_next_path_direction,
                    shared_serialized_material_attribute_byte_array,
                    input_ray_brdf);

                float inv_pdf = 2.0f * M_PI;
                next_path_weight.r = path_weight.r * input_ray_brdf * hit_object_color.r * cosine_term * inv_pdf;
                next_path_weight.g = path_weight.g * input_ray_brdf * hit_object_color.g * cosine_term * inv_pdf;
                next_path_weight.b = path_weight.b * input_ray_brdf * hit_object_color.b * cosine_term * inv_pdf;

                float4 random_uniform4 = curand_uniform4(&curand_state);

                // 光源のサンプリング
                const int table_index = min(int(floorf(random_uniform4.x * float(args.light_sampling_table_size))), args.light_sampling_table_size - 1);
                const int object_index = shared_light_sampling_table[table_index];
                rtxObject object = shared_serialized_object_array[object_index];

                float light_distance;
                float3 unit_light_normal;
                if (object.geometry_type == RTXGeometryTypeStandard) {
                    const int face_index = min(int(floorf(random_uniform4.y * float(object.num_faces))), object.num_faces - 1);
                    const int serialized_face_index = face_index + object.serialized_face_index_offset;
                    const rtxFaceVertexIndex face = shared_serialized_face_vertex_indices_array[serialized_face_index];
                    const rtxVertex va = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                    const rtxVertex vb = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                    const rtxVertex vc = shared_serialized_vertex_array[face.c + object.serialized_vertex_index_offset];
                    __rtx_nee_sample_point_in_triangle(random_uniform4, va, vb, vc, shadow_ray, light_distance, unit_light_normal);
                } else if (object.geometry_type == RTXGeometryTypeSphere) {
                    const int serialized_array_index = object.serialized_face_index_offset;
                    const rtxFaceVertexIndex face = shared_serialized_face_vertex_indices_array[serialized_array_index];
                    const rtxVertex center = shared_serialized_vertex_array[face.a + object.serialized_vertex_index_offset];
                    const rtxVertex radius = shared_serialized_vertex_array[face.b + object.serialized_vertex_index_offset];
                    __rtx_nee_sample_point_in_sphere(curand_state, unit_light_normal, shadow_ray, light_distance);
                }

                const float dot_ray_face = shadow_ray.direction.x * unit_hit_face_normal.x
                    + shadow_ray.direction.y * unit_hit_face_normal.y
                    + shadow_ray.direction.z * unit_hit_face_normal.z;

                if (dot_ray_face <= 0.0f) {
                    ray = &primary_ray;
                    iter += 1;
                    path_weight.r = next_path_weight.r;
                    path_weight.g = next_path_weight.g;
                    path_weight.b = next_path_weight.b;
                    continue;
                }

                shadow_ray.origin.x = hit_point.x;
                shadow_ray.origin.y = hit_point.y;
                shadow_ray.origin.z = hit_point.z;

                shadow_ray_brdf = 0.0f;
                __rtx_compute_brdf(
                    unit_hit_face_normal,
                    hit_object,
                    hit_face,
                    primary_ray.direction,
                    shadow_ray.direction,
                    shared_serialized_material_attribute_byte_array,
                    shadow_ray_brdf);

                // 次のパス
                primary_ray.origin.x = hit_point.x;
                primary_ray.origin.y = hit_point.y;
                primary_ray.origin.z = hit_point.z;
                primary_ray.direction.x = unit_next_path_direction.x;
                primary_ray.direction.y = unit_next_path_direction.y;
                primary_ray.direction.z = unit_next_path_direction.z;

                const float dot_ray_light = fabsf(shadow_ray.direction.x * unit_light_normal.x + shadow_ray.direction.y * unit_light_normal.y + shadow_ray.direction.z * unit_light_normal.z);

                // ハック
                const float r = max(light_distance, 0.5f);
                g_term = dot_ray_face * dot_ray_light / (r * r);

                ray = &shadow_ray;
            }
        }
    }
    global_serialized_render_array[render_buffer_index] = pixel;
}

void rtx_cuda_launch_nee_shared_memory_kernel(
    rtxFaceVertexIndex* gpu_serialized_face_vertex_index_array,
    rtxVertex* gpu_serialized_vertex_array,
    rtxObject* gpu_serialized_object_array,
    rtxMaterialAttributeByte* gpu_serialized_material_attribute_byte_array,
    rtxThreadedBVH* gpu_serialized_threaded_bvh_array,
    rtxThreadedBVHNode* gpu_serialized_threaded_bvh_node_array,
    rtxRGBAColor* gpu_serialized_color_mapping_array,
    rtxUVCoordinate* gpu_serialized_uv_coordinate_array,
    int* gpu_light_sampling_table,
    rtxRGBAPixel* gpu_serialized_render_array,
    rtxNEEKernelArguments& args,
    int num_threads,
    int num_blocks,
    size_t shared_memory_bytes)
{
    __check_kernel_arguments();
    nee_shared_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        gpu_serialized_face_vertex_index_array,
        gpu_serialized_vertex_array,
        gpu_serialized_object_array,
        gpu_serialized_material_attribute_byte_array,
        gpu_serialized_threaded_bvh_array,
        gpu_serialized_threaded_bvh_node_array,
        gpu_serialized_color_mapping_array,
        gpu_serialized_uv_coordinate_array,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_light_sampling_table,
        gpu_serialized_render_array,
        args);
    cudaCheckError(cudaThreadSynchronize());
}
