#include "../../header/enum.h"
#include "../header/cuda.h"
#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

__global__ void standard_global_memory_kernel(
    const int ray_array_size,
    const int face_vertex_index_array_size,
    const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const int threaded_bvh_node_array_size,
    const RTXColor* global_color_mapping_array, const int color_mapping_array_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed)
{
    extern __shared__ unsigned char shared_memory[];
    int thread_id = threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    RTXObject* shared_object_array = (RTXObject*)&shared_memory[offset];
    offset += sizeof(RTXObject) / sizeof(unsigned char) * object_array_size;

    RTXMaterialAttributeByte* shared_material_attribute_byte_array = (RTXMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(RTXMaterialAttributeByte) / sizeof(unsigned char) * material_attribute_byte_array_size;

    RTXThreadedBVH* shared_threaded_bvh_array = (RTXThreadedBVH*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVH) / sizeof(unsigned char) * threaded_bvh_array_size;

    RTXColor* shared_color_mapping_array = (RTXColor*)&shared_memory[offset];
    offset += sizeof(RTXColor) / sizeof(unsigned char) * color_mapping_array_size;

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
        for (int k = 0; k < color_mapping_array_size; k++) {
            shared_color_mapping_array[k] = global_color_mapping_array[k];
        }
    }
    __syncthreads();

    const float eps = 0.0000001;
    CUDARay ray;
    // RTXRay ray;
    float3 ray_direction_inv;
    float3 hit_point;
    float3 hit_face_normal;
    RTXObject hit_object;

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

        RTXPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };
        RTXPixel path_weight = { 1.0f, 1.0f, 1.0f };

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

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
                        if (object.geometry_type == RTXGeometryTypeStandard) {

                            for (int m = 0; m < num_assigned_faces; m++) {
                                int index = node.assigned_face_index_start + m + object.face_index_offset;

                                const int4 face = tex1Dfetch(face_vertex_index_texture, index);
                                // RTXFace face = global_face_vertex_index_array[index];

                                const float4 va = tex1Dfetch(vertex_texture, face.x + object.vertex_index_offset);
                                const float4 vb = tex1Dfetch(vertex_texture, face.y + object.vertex_index_offset);
                                const float4 vc = tex1Dfetch(vertex_texture, face.z + object.vertex_index_offset);
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

                                did_hit_object = true;
                                hit_object = object;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int index = node.assigned_face_index_start + object.face_index_offset;

                            const int4 face = tex1Dfetch(face_vertex_index_texture, index);

                            const float4 center = tex1Dfetch(vertex_texture, face.x + object.vertex_index_offset);
                            const float4 radius = tex1Dfetch(vertex_texture, face.y + object.vertex_index_offset);

                            float4 oc;
                            oc.x = ray.origin.x - center.x;
                            oc.y = ray.origin.y - center.y;
                            oc.z = ray.origin.z - center.z;

                            const float a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z;
                            const float b = 2.0f * (ray.direction.x * oc.x + ray.direction.y * oc.y + ray.direction.z * oc.z);
                            const float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - radius.x * radius.x;
                            const float d = b * b - 4.0f * a * c;

                            if (d <= 0) {
                                continue;
                            }
                            const float root = sqrt(d);
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
                            hit_point.x = ray.origin.x + t * ray.direction.x;
                            hit_point.y = ray.origin.y + t * ray.direction.y;
                            hit_point.z = ray.origin.z + t * ray.direction.z;

                            float4 tmp;
                            tmp.x = hit_point.x - center.x;
                            tmp.y = hit_point.y - center.y;
                            tmp.z = hit_point.z - center.z;
                            const float norm = sqrtf(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z) + 1e-12;

                            hit_face_normal.x = tmp.x / norm;
                            hit_face_normal.y = tmp.y / norm;
                            hit_face_normal.z = tmp.z / norm;

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

            RTXPixel hit_color;
            bool did_hit_light = false;
            if (did_hit_object) {
                int material_type = hit_object.layerd_material_types.outside;
                int mapping_type = hit_object.mapping_type;

                if (mapping_type == RTXMappingTypeSolidColor) {
                    RTXColor color = shared_color_mapping_array[hit_object.mapping_index];
                    hit_color.r = color.r;
                    hit_color.g = color.g;
                    hit_color.b = color.b;
                }

                if (material_type == RTXMaterialTypeLambert) {
                    RTXLambertMaterialAttribute attr = ((RTXLambertMaterialAttribute*)&shared_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                } else if (material_type == RTXMaterialTypeEmissive) {
                    RTXEmissiveMaterialAttribute attr = ((RTXEmissiveMaterialAttribute*)&shared_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    did_hit_light = true;
                    hit_color.r *= attr.brightness;
                    hit_color.g *= attr.brightness;
                    hit_color.b *= attr.brightness;
                }
            }

            if (did_hit_light) {
                pixel.r += hit_color.r * path_weight.r;
                pixel.g += hit_color.g * path_weight.g;
                pixel.b += hit_color.b * path_weight.b;
            }

            if (did_hit_object) {
                float4 path;
                path.x = hit_point.x - ray.origin.x;
                path.y = hit_point.y - ray.origin.y;
                path.z = hit_point.z - ray.origin.z;
                float distance = sqrt(path.x * path.x + path.y * path.y + path.z * path.z);

                ray.origin.x = hit_point.x;
                ray.origin.y = hit_point.y;
                ray.origin.z = hit_point.z;

                // diffuse reflection
                float unit_diffuese_x = curand_normal(&state);
                float unit_diffuese_y = curand_normal(&state);
                float unit_diffuese_z = curand_normal(&state);
                float norm = sqrt(unit_diffuese_x * unit_diffuese_x + unit_diffuese_y * unit_diffuese_y + unit_diffuese_z * unit_diffuese_z);
                unit_diffuese_x /= norm;
                unit_diffuese_y /= norm;
                unit_diffuese_z /= norm;

                float dot = hit_face_normal.x * unit_diffuese_x + hit_face_normal.y * unit_diffuese_y + hit_face_normal.z * unit_diffuese_z;
                if (dot < 0.0f) {
                    unit_diffuese_x = -unit_diffuese_x;
                    unit_diffuese_y = -unit_diffuese_y;
                    unit_diffuese_z = -unit_diffuese_z;
                }
                ray.direction.x = unit_diffuese_x;
                ray.direction.y = unit_diffuese_y;
                ray.direction.z = unit_diffuese_z;

                float cosine_term = hit_face_normal.x * unit_diffuese_x + hit_face_normal.y * unit_diffuese_y + hit_face_normal.y * unit_diffuese_y;

                ray_direction_inv.x = 1.0f / ray.direction.x;
                ray_direction_inv.y = 1.0f / ray.direction.y;
                ray_direction_inv.z = 1.0f / ray.direction.z;

                path_weight.r *= hit_color.r * cosine_term;
                path_weight.g *= hit_color.g * cosine_term;
                path_weight.b *= hit_color.b * cosine_term;
            }
        }

        global_render_array[ray_index] = pixel;
    }
}

__global__ void standard_shared_memory_kernel(
    const int ray_array_size,
    const RTXFace* global_face_vertex_indices_array, const int face_vertex_index_array_size,
    const RTXVertex* global_vertex_array, const int vertex_array_size,
    const RTXObject* global_object_array, const int object_array_size,
    const RTXMaterialAttributeByte* global_material_attribute_byte_array, const int material_attribute_byte_array_size,
    const RTXThreadedBVH* global_threaded_bvh_array, const int threaded_bvh_array_size,
    const RTXThreadedBVHNode* global_threaded_bvh_node_array, const int threaded_bvh_node_array_size,
    const RTXColor* global_color_mapping_array, const int color_mapping_array_size,
    RTXPixel* global_render_array,
    const int num_rays_per_thread,
    const int max_bounce,
    const int curand_seed)
{
    extern __shared__ unsigned char shared_memory[];
    int thread_id = threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(curand_seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int offset = 0;
    RTXFace* shared_face_vertex_indices_array = (RTXFace*)&shared_memory[offset];
    offset += sizeof(RTXFace) / sizeof(unsigned char) * face_vertex_index_array_size;

    RTXVertex* shared_vertex_array = (RTXVertex*)&shared_memory[offset];
    offset += sizeof(RTXVertex) / sizeof(unsigned char) * vertex_array_size;

    RTXObject* shared_object_array = (RTXObject*)&shared_memory[offset];
    offset += sizeof(RTXObject) / sizeof(unsigned char) * object_array_size;

    RTXMaterialAttributeByte* shared_material_attribute_byte_array = (RTXMaterialAttributeByte*)&shared_memory[offset];
    offset += sizeof(RTXMaterialAttributeByte) / sizeof(unsigned char) * material_attribute_byte_array_size;

    RTXThreadedBVH* shared_threaded_bvh_array = (RTXThreadedBVH*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVH) / sizeof(unsigned char) * threaded_bvh_array_size;

    RTXThreadedBVHNode* shared_threaded_bvh_node_array = (RTXThreadedBVHNode*)&shared_memory[offset];
    offset += sizeof(RTXThreadedBVHNode) / sizeof(unsigned char) * threaded_bvh_node_array_size;

    RTXColor* shared_color_mapping_array = (RTXColor*)&shared_memory[offset];
    offset += sizeof(RTXColor) / sizeof(unsigned char) * color_mapping_array_size;

    if (thread_id == 0) {
        for (int k = 0; k < face_vertex_index_array_size; k++) {
            shared_face_vertex_indices_array[k] = global_face_vertex_indices_array[k];
        }
        for (int k = 0; k < vertex_array_size; k++) {
            shared_vertex_array[k] = global_vertex_array[k];
        }
        for (int k = 0; k < object_array_size; k++) {
            shared_object_array[k] = global_object_array[k];
        }
        for (int k = 0; k < material_attribute_byte_array_size; k++) {
            shared_material_attribute_byte_array[k] = global_material_attribute_byte_array[k];
        }
        for (int k = 0; k < threaded_bvh_array_size; k++) {
            shared_threaded_bvh_array[k] = global_threaded_bvh_array[k];
        }
        for (int k = 0; k < threaded_bvh_node_array_size; k++) {
            shared_threaded_bvh_node_array[k] = global_threaded_bvh_node_array[k];
        }
        for (int k = 0; k < color_mapping_array_size; k++) {
            shared_color_mapping_array[k] = global_color_mapping_array[k];
        }
    }
    __syncthreads();

    const float eps = 0.0000001;
    CUDARay ray;
    // RTXRay ray;
    float3 ray_direction_inv;
    float3 hit_point;
    float3 hit_face_normal;
    RTXObject hit_object;

    for (int n = 0; n < num_rays_per_thread; n++) {
        int ray_index = (blockIdx.x * blockDim.x + threadIdx.x) * num_rays_per_thread + n;
        if (ray_index >= ray_array_size) {
            return;
        }

        ray.direction = tex1Dfetch(ray_texture, ray_index * 2 + 0);
        ray.origin = tex1Dfetch(ray_texture, ray_index * 2 + 1);

        ray_direction_inv.x = 1.0f / ray.direction.x;
        ray_direction_inv.y = 1.0f / ray.direction.y;
        ray_direction_inv.z = 1.0f / ray.direction.z;

        RTXPixel pixel = { 0.0f, 0.0f, 0.0f, 0.0f };
        RTXPixel path_weight = { 1.0f, 1.0f, 1.0f };

        for (int bounce = 0; bounce < max_bounce; bounce++) {
            float min_distance = FLT_MAX;
            bool did_hit_object = false;

            for (int object_index = 0; object_index < object_array_size; object_index++) {
                RTXObject object = shared_object_array[object_index];
                RTXThreadedBVH bvh = shared_threaded_bvh_array[object_index];

                int bvh_current_node_index = 0;
                for (int traversal = 0; traversal < bvh.num_nodes; traversal++) {
                    if (bvh_current_node_index == THREADED_BVH_TERMINAL_NODE) {
                        break;
                    }

                    RTXThreadedBVHNode node = shared_threaded_bvh_node_array[bvh.node_index_offset + bvh_current_node_index];

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
                        if (object.geometry_type == RTXGeometryTypeStandard) {

                            for (int m = 0; m < num_assigned_faces; m++) {
                                int index = node.assigned_face_index_start + m + object.face_index_offset;

                                const RTXFace face = shared_face_vertex_indices_array[index];

                                const RTXVertex va = shared_vertex_array[face.a + object.vertex_index_offset];
                                const RTXVertex vb = shared_vertex_array[face.b + object.vertex_index_offset];
                                const RTXVertex vc = shared_vertex_array[face.c + object.vertex_index_offset];

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

                                did_hit_object = true;
                                hit_object = object;
                            }
                        } else if (object.geometry_type == RTXGeometryTypeSphere) {
                            int index = node.assigned_face_index_start + object.face_index_offset;

                            const RTXFace face = shared_face_vertex_indices_array[index];

                            const RTXVertex center = shared_vertex_array[face.a + object.vertex_index_offset];
                            const RTXVertex radius = shared_vertex_array[face.b + object.vertex_index_offset];

                            float4 oc;
                            oc.x = ray.origin.x - center.x;
                            oc.y = ray.origin.y - center.y;
                            oc.z = ray.origin.z - center.z;

                            const float a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z;
                            const float b = 2.0f * (ray.direction.x * oc.x + ray.direction.y * oc.y + ray.direction.z * oc.z);
                            const float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - radius.x * radius.x;
                            const float d = b * b - 4.0f * a * c;

                            if (d <= 0) {
                                continue;
                            }
                            const float root = sqrt(d);
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
                            hit_point.x = ray.origin.x + t * ray.direction.x;
                            hit_point.y = ray.origin.y + t * ray.direction.y;
                            hit_point.z = ray.origin.z + t * ray.direction.z;

                            float4 tmp;
                            tmp.x = hit_point.x - center.x;
                            tmp.y = hit_point.y - center.y;
                            tmp.z = hit_point.z - center.z;
                            const float norm = sqrtf(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z) + 1e-12;

                            hit_face_normal.x = tmp.x / norm;
                            hit_face_normal.y = tmp.y / norm;
                            hit_face_normal.z = tmp.z / norm;

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

            RTXPixel hit_color;
            bool did_hit_light = false;
            if (did_hit_object) {
                int material_type = hit_object.layerd_material_types.outside;
                int mapping_type = hit_object.mapping_type;

                if (mapping_type == RTXMappingTypeSolidColor) {
                    RTXColor color = shared_color_mapping_array[hit_object.mapping_index];
                    hit_color.r = color.r;
                    hit_color.g = color.g;
                    hit_color.b = color.b;
                }

                if (material_type == RTXMaterialTypeLambert) {
                    RTXLambertMaterialAttribute attr = ((RTXLambertMaterialAttribute*)&shared_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                } else if (material_type == RTXMaterialTypeEmissive) {
                    RTXEmissiveMaterialAttribute attr = ((RTXEmissiveMaterialAttribute*)&shared_material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];
                    did_hit_light = true;
                    hit_color.r *= attr.brightness;
                    hit_color.g *= attr.brightness;
                    hit_color.b *= attr.brightness;
                }
            }

            if (did_hit_light) {
                pixel.r += hit_color.r * path_weight.r;
                pixel.g += hit_color.g * path_weight.g;
                pixel.b += hit_color.b * path_weight.b;
            }

            if (did_hit_object) {
                float4 path;
                path.x = hit_point.x - ray.origin.x;
                path.y = hit_point.y - ray.origin.y;
                path.z = hit_point.z - ray.origin.z;
                float distance = sqrt(path.x * path.x + path.y * path.y + path.z * path.z);

                ray.origin.x = hit_point.x;
                ray.origin.y = hit_point.y;
                ray.origin.z = hit_point.z;

                // diffuse reflection
                float unit_diffuese_x = curand_normal(&state);
                float unit_diffuese_y = curand_normal(&state);
                float unit_diffuese_z = curand_normal(&state);
                float norm = sqrt(unit_diffuese_x * unit_diffuese_x + unit_diffuese_y * unit_diffuese_y + unit_diffuese_z * unit_diffuese_z);
                unit_diffuese_x /= norm;
                unit_diffuese_y /= norm;
                unit_diffuese_z /= norm;

                float cosine_term = hit_face_normal.x * unit_diffuese_x + hit_face_normal.y * unit_diffuese_y + hit_face_normal.z * unit_diffuese_z;
                if (cosine_term < 0.0f) {
                    unit_diffuese_x *= -1;
                    unit_diffuese_y *= -1;
                    unit_diffuese_z *= -1;
                    cosine_term *= -1;
                }
                ray.direction.x = unit_diffuese_x;
                ray.direction.y = unit_diffuese_y;
                ray.direction.z = unit_diffuese_z;

                ray_direction_inv.x = 1.0f / ray.direction.x;
                ray_direction_inv.y = 1.0f / ray.direction.y;
                ray_direction_inv.z = 1.0f / ray.direction.z;

                path_weight.r *= hit_color.r * cosine_term;
                path_weight.g *= hit_color.g * cosine_term;
                path_weight.b *= hit_color.b * cosine_term;
            }
        }

        global_render_array[ray_index] = pixel;
    }
}

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
    assert(gpu_ray_array != NULL);
    assert(gpu_face_vertex_index_array != NULL);
    assert(gpu_vertex_array != NULL);
    assert(gpu_object_array != NULL);
    assert(gpu_material_attribute_byte_array != NULL);
    assert(gpu_threaded_bvh_array != NULL);
    assert(gpu_threaded_bvh_node_array != NULL);
    assert(gpu_render_array != NULL);
    if (color_mapping_array_size > 0) {
        assert(gpu_color_mapping_array != NULL);
    }

    int num_rays = ray_array_size;

    // int num_blocks = (num_rays - 1) / num_threads + 1;

    int num_rays_per_thread = num_rays / (num_threads * num_blocks) + 1;

    long required_shared_memory_bytes = sizeof(RTXFace) * face_vertex_index_array_size + sizeof(RTXVertex) * vertex_array_size + sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size + sizeof(RTXColor) * color_mapping_array_size;

    // num_blocks = 1;
    // num_rays_per_thread = 1;

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);

    printf("shared memory: %ld bytes\n", required_shared_memory_bytes);
    printf("    face: %d * %d vertex: %d * %d object: %d * %d material: %d * %d color: %d * %d \n", sizeof(RTXFace), face_vertex_index_array_size, sizeof(RTXVertex), vertex_array_size, sizeof(RTXObject), object_array_size, sizeof(RTXMaterialAttributeByte), material_attribute_byte_array_size, sizeof(RTXColor), color_mapping_array_size);
    printf("    bvh: %d * %d node: %d * %d\n", sizeof(RTXThreadedBVH), threaded_bvh_array_size, sizeof(RTXThreadedBVHNode), threaded_bvh_node_array_size);
    printf("available: %d bytes\n", dev.sharedMemPerBlock);
    printf("rays: %d\n", ray_array_size);

    if (required_shared_memory_bytes > dev.sharedMemPerBlock) {
        // int required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size;
        long required_shared_memory_bytes = sizeof(RTXObject) * object_array_size + sizeof(RTXMaterialAttributeByte) * material_attribute_byte_array_size + sizeof(RTXThreadedBVH) * threaded_bvh_array_size + sizeof(RTXColor) * color_mapping_array_size;
        printf("    shared memory: %ld bytes\n", required_shared_memory_bytes);
        printf("    available: %d bytes\n", dev.sharedMemPerBlock);
        printf("    num_blocks: %d num_threads: %d\n", num_blocks, num_threads);
        printf("using global memory kernel\n");

        assert(required_shared_memory_bytes <= dev.sharedMemPerBlock);

        cudaBindTexture(0, ray_texture, gpu_ray_array, cudaCreateChannelDesc<float4>(), sizeof(RTXRay) * ray_array_size);
        cudaBindTexture(0, face_vertex_index_texture, gpu_face_vertex_index_array, cudaCreateChannelDesc<int4>(), sizeof(RTXFace) * face_vertex_index_array_size);
        cudaBindTexture(0, vertex_texture, gpu_vertex_array, cudaCreateChannelDesc<float4>(), sizeof(RTXVertex) * vertex_array_size);
        cudaBindTexture(0, threaded_bvh_node_texture, gpu_threaded_bvh_node_array, cudaCreateChannelDesc<float4>(), sizeof(RTXThreadedBVHNode) * threaded_bvh_node_array_size);

        standard_global_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
            ray_array_size,
            face_vertex_index_array_size,
            vertex_array_size,
            gpu_object_array, object_array_size,
            gpu_material_attribute_byte_array, material_attribute_byte_array_size,
            gpu_threaded_bvh_array, threaded_bvh_array_size,
            threaded_bvh_node_array_size,
            gpu_color_mapping_array, color_mapping_array_size,
            gpu_render_array,
            num_rays_per_thread,
            max_bounce,
            curand_seed);

        cudaUnbindTexture(ray_texture);
        cudaUnbindTexture(face_vertex_index_texture);
        cudaUnbindTexture(vertex_texture);
        cudaUnbindTexture(threaded_bvh_node_texture);

    } else {
        printf("using shared memory kernel\n");
        cudaBindTexture(0, ray_texture, gpu_ray_array, cudaCreateChannelDesc<float4>(), sizeof(RTXRay) * ray_array_size);

        standard_shared_memory_kernel<<<num_blocks, num_threads, required_shared_memory_bytes>>>(
            ray_array_size,
            gpu_face_vertex_index_array, face_vertex_index_array_size,
            gpu_vertex_array, vertex_array_size,
            gpu_object_array, object_array_size,
            gpu_material_attribute_byte_array, material_attribute_byte_array_size,
            gpu_threaded_bvh_array, threaded_bvh_array_size,
            gpu_threaded_bvh_node_array, threaded_bvh_node_array_size,
            gpu_color_mapping_array, color_mapping_array_size,
            gpu_render_array,
            num_rays_per_thread,
            max_bounce,
            curand_seed);

        cudaUnbindTexture(ray_texture);
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
