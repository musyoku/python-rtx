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

__global__ void mcrt_shared_memory_kernel(
    rtxFaceVertexIndex* global_serialized_face_vertex_indices_array,
    rtxVertex* global_serialized_vertex_array,
    rtxObject* global_serialized_object_array,
    rtxMaterialAttributeByte* global_serialized_material_attribute_byte_array,
    rtxThreadedBVH* global_serialized_threaded_bvh_array,
    rtxThreadedBVHNode* global_serialized_threaded_bvh_node_array,
    rtxRGBAColor* global_serialized_color_mapping_array,
    rtxUVCoordinate* global_serialized_uv_coordinate_array,
    cudaTextureObject_t* global_serialized_mapping_texture_object_array,
    rtxRGBAPixel* global_serialized_render_array,
    rtxMCRTKernelArguments args)
{
    extern __shared__ char shared_memory[];
    curandStatePhilox4_32_10_t curand_state;
    curand_init(args.curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &curand_state);
    __xorshift_init(args.curand_seed);

    // グローバルメモリの直列データを共有メモリにコピーする
    int offset = 0;
    rtxFaceVertexIndex* shared_face_vertex_indices_array = (rtxFaceVertexIndex*)&shared_memory[offset];
    offset += sizeof(rtxFaceVertexIndex) * args.face_vertex_index_array_size;

    rtxVertex* shared_vertex_array = (rtxVertex*)&shared_memory[offset];
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

    cudaTextureObject_t* shared_serialized_texture_object_array = (cudaTextureObject_t*)&shared_memory[offset];
    offset += sizeof(cudaTextureObject_t) * args.num_active_texture_units;

    if (threadIdx.x == 0) {
        for (int m = 0; m < args.face_vertex_index_array_size; m++) {
            shared_face_vertex_indices_array[m] = global_serialized_face_vertex_indices_array[m];
        }
        for (int m = 0; m < args.vertex_array_size; m++) {
            shared_vertex_array[m] = global_serialized_vertex_array[m];
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

        // レイの生成
        rtxCUDARay ray;
        __rtx_generate_ray(ray, args, aspect_ratio);

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

        for (int bounce = 0; bounce < args.max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

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

                    rtxThreadedBVHNode node = shared_serialized_threaded_bvh_node_array[bvh.serial_node_index_offset + bvh_current_node_index];

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
                                int index = node.assigned_face_index_start + m + object.serialized_face_index_offset;
                                const rtxFaceVertexIndex face = shared_face_vertex_indices_array[index];

                                const rtxVertex va = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                                const rtxVertex vb = shared_vertex_array[face.b + object.serialized_vertex_index_offset];
                                const rtxVertex vc = shared_vertex_array[face.c + object.serialized_vertex_index_offset];

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
                            int index = node.assigned_face_index_start + object.serialized_face_index_offset;
                            const rtxFaceVertexIndex face = shared_face_vertex_indices_array[index];

                            const rtxVertex center = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex radius = shared_vertex_array[face.b + object.serialized_vertex_index_offset];

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
                        } else if (object.geometry_type == RTXGeometryTypeCylinder) {
                            rtxFaceVertexIndex face;
                            int offset = node.assigned_face_index_start + object.serialized_face_index_offset;

                            // Load cylinder parameters
                            face = shared_face_vertex_indices_array[offset + 0];
                            const rtxVertex params = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const float radius = params.x;
                            const float y_max = params.y;
                            const float y_min = params.z;

                            // Load transformation matrix
                            face = shared_face_vertex_indices_array[offset + 1];
                            const rtxVertex trans_a = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex trans_b = shared_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex trans_c = shared_vertex_array[face.c + object.serialized_vertex_index_offset];

                            // Load inverse transformation matrix
                            face = shared_face_vertex_indices_array[offset + 2];
                            const rtxVertex inv_trans_a = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_b = shared_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_c = shared_vertex_array[face.c + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_cylinder_or_continue(
                                ray,
                                trans_a, trans_b, trans_c,
                                inv_trans_a, inv_trans_b, inv_trans_c,
                                unit_hit_face_normal,
                                distance,
                                min_distance);
                            min_distance = distance;

                            // hit point in view space
                            hit_point.x = ray.origin.x + distance * ray.direction.x;
                            hit_point.y = ray.origin.y + distance * ray.direction.y;
                            hit_point.z = ray.origin.z + distance * ray.direction.z;

                            did_hit_object = true;
                            hit_object = object;
                        } else if (object.geometry_type == RTXGeometryTypeCone) {
                            rtxFaceVertexIndex face;
                            int offset = node.assigned_face_index_start + object.serialized_face_index_offset;

                            // Load cylinder parameters
                            face = shared_face_vertex_indices_array[offset + 0];
                            const rtxVertex params = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const float radius = params.x;
                            const float height = params.y;

                            // Load transformation matrix
                            face = shared_face_vertex_indices_array[offset + 1];
                            const rtxVertex trans_a = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex trans_b = shared_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex trans_c = shared_vertex_array[face.c + object.serialized_vertex_index_offset];

                            // Load inverse transformation matrix
                            face = shared_face_vertex_indices_array[offset + 2];
                            const rtxVertex inv_trans_a = shared_vertex_array[face.a + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_b = shared_vertex_array[face.b + object.serialized_vertex_index_offset];
                            const rtxVertex inv_trans_c = shared_vertex_array[face.c + object.serialized_vertex_index_offset];

                            float distance;
                            __rtx_intersect_cone_or_continue(
                                ray,
                                trans_a, trans_b, trans_c,
                                inv_trans_a, inv_trans_b, inv_trans_c,
                                unit_hit_face_normal,
                                distance,
                                min_distance);
                            min_distance = distance;

                            // hit point in view space
                            hit_point.x = ray.origin.x + distance * ray.direction.x;
                            hit_point.y = ray.origin.y + distance * ray.direction.y;
                            hit_point.z = ray.origin.z + distance * ray.direction.z;

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
                if (bounce == 0){
                    pixel.r += args.ambient_color.r;
                    pixel.g += args.ambient_color.g;
                    pixel.b += args.ambient_color.b;
                }
                break;
            }

            //  衝突点の色を検出
            rtxRGBAColor hit_color;
            __rtx_fetch_color_in_linear_memory(
                hit_point,
                hit_object,
                hit_face,
                hit_color,
                shared_serialized_material_attribute_byte_array,
                shared_serialized_color_mapping_array,
                shared_serialized_texture_object_array,
                shared_serialized_uv_coordinate_array);

            int material_type = hit_object.layerd_material_types.outside;
            bool did_hit_light = material_type == RTXMaterialTypeEmissive;

            // 光源に当たった場合トレースを打ち切り
            if (did_hit_light) {
                rtxEmissiveMaterialAttribute attr = ((rtxEmissiveMaterialAttribute*)&shared_serialized_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                if (bounce == 0 && attr.visible == false) {
                    pixel.r += args.ambient_color.r;
                    pixel.g += args.ambient_color.g;
                    pixel.b += args.ambient_color.b;
                } else {
                    pixel.r += hit_color.r * path_weight.r * attr.intensity;
                    pixel.g += hit_color.g * path_weight.g * attr.intensity;
                    pixel.b += hit_color.b * path_weight.b * attr.intensity;
                }
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

            float brdf = 0.0f;
            __rtx_compute_brdf(
                unit_hit_face_normal,
                hit_object,
                hit_face,
                ray.direction,
                unit_next_ray_direction,
                shared_serialized_material_attribute_byte_array,
                brdf);

            __rtx_update_ray(
                ray,
                ray_direction_inv,
                hit_point,
                unit_next_ray_direction);

            // 経路のウェイトを更新
            float inv_pdf = 2.0f * M_PI;
            path_weight.r *= hit_color.r * brdf * cosine_term * inv_pdf;
            path_weight.g *= hit_color.g * brdf * cosine_term * inv_pdf;
            path_weight.b *= hit_color.b * brdf * cosine_term * inv_pdf;
        }
    }
    global_serialized_render_array[render_buffer_index] = pixel;
}

void rtx_cuda_launch_mcrt_shared_memory_kernel(
    rtxFaceVertexIndex* gpu_serialized_face_vertex_index_array,
    rtxVertex* gpu_serialized_vertex_array,
    rtxObject* gpu_serialized_object_array,
    rtxMaterialAttributeByte* gpu_serialized_material_attribute_byte_array,
    rtxThreadedBVH* gpu_serialized_threaded_bvh_array,
    rtxThreadedBVHNode* gpu_serialized_threaded_bvh_node_array,
    rtxRGBAColor* gpu_serialized_color_mapping_array,
    rtxUVCoordinate* gpu_serialized_uv_coordinate_array,
    rtxRGBAPixel* gpu_serialized_render_array,
    rtxMCRTKernelArguments& args,
    int num_threads, int num_blocks, size_t shared_memory_bytes)
{
    __check_kernel_arguments();
    mcrt_shared_memory_kernel<<<num_blocks, num_threads, shared_memory_bytes>>>(
        gpu_serialized_face_vertex_index_array,
        gpu_serialized_vertex_array,
        gpu_serialized_object_array,
        gpu_serialized_material_attribute_byte_array,
        gpu_serialized_threaded_bvh_array,
        gpu_serialized_threaded_bvh_node_array,
        gpu_serialized_color_mapping_array,
        gpu_serialized_uv_coordinate_array,
        g_gpu_serialized_mapping_texture_object_array,
        gpu_serialized_render_array,
        args);
    cudaCheckError(cudaThreadSynchronize());
}
