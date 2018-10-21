#pragma once

// インライン関数でも速度が落ちるのですべてプリプロセッサで埋め込む

#define __rtx_intersect_triangle_or_continue(ray, va, vb, vc, s, t, min_distance)    \
    {                                                                                \
        const float eps = 0.000001;                                                  \
        float3 edge_ba = {                                                           \
            vb.x - va.x,                                                             \
            vb.y - va.y,                                                             \
            vb.z - va.z,                                                             \
        };                                                                           \
        float3 edge_ca = {                                                           \
            vc.x - va.x,                                                             \
            vc.y - va.y,                                                             \
            vc.z - va.z,                                                             \
        };                                                                           \
        float3 h = {                                                                 \
            ray.direction.y * edge_ca.z - ray.direction.z * edge_ca.y,               \
            ray.direction.z * edge_ca.x - ray.direction.x * edge_ca.z,               \
            ray.direction.x * edge_ca.y - ray.direction.y * edge_ca.x,               \
        };                                                                           \
        float f = edge_ba.x * h.x + edge_ba.y * h.y + edge_ba.z * h.z;               \
        if (f > -eps && f < eps) {                                                   \
            continue;                                                                \
        }                                                                            \
        f = 1.0f / f;                                                                \
        s.x = ray.origin.x - va.x;                                                   \
        s.y = ray.origin.y - va.y;                                                   \
        s.z = ray.origin.z - va.z;                                                   \
        float dot = s.x * h.x + s.y * h.y + s.z * h.z;                               \
        float u = f * dot;                                                           \
        if (u < 0.0f || u > 1.0f) {                                                  \
            continue;                                                                \
        }                                                                            \
        h.x = s.y * edge_ba.z - s.z * edge_ba.y;                                     \
        h.y = s.z * edge_ba.x - s.x * edge_ba.z;                                     \
        h.z = s.x * edge_ba.y - s.y * edge_ba.x;                                     \
        dot = h.x * ray.direction.x + h.y * ray.direction.y + h.z * ray.direction.z; \
        float v = f * dot;                                                           \
        if (v < 0.0f || u + v > 1.0f) {                                              \
            continue;                                                                \
        }                                                                            \
        s.x = edge_ba.y * edge_ca.z - edge_ba.z * edge_ca.y;                         \
        s.y = edge_ba.z * edge_ca.x - edge_ba.x * edge_ca.z;                         \
        s.z = edge_ba.x * edge_ca.y - edge_ba.y * edge_ca.x;                         \
        float norm = sqrtf(s.x * s.x + s.y * s.y + s.z * s.z);                       \
        s.x = s.x / norm;                                                            \
        s.y = s.y / norm;                                                            \
        s.z = s.z / norm;                                                            \
        dot = s.x * ray.direction.x + s.y * ray.direction.y + s.z * ray.direction.z; \
        if (dot > 0.0f) {                                                            \
            continue;                                                                \
        }                                                                            \
        dot = edge_ca.x * h.x + edge_ca.y * h.y + edge_ca.z * h.z;                   \
        t = f * dot;                                                                 \
        if (t <= 0.001f) {                                                           \
            continue;                                                                \
        }                                                                            \
        if (min_distance <= t) {                                                     \
            continue;                                                                \
        }                                                                            \
    }

// http://www.pbr-book.org/3ed-2018/Utilities/Mathematical_Routines.html#SolvingQuadraticEquations
#define __rtx_intersect_sphere_or_continue(ray, center, radius, t, min_distance)                                                   \
    {                                                                                                                              \
        float3 oc = {                                                                                                              \
            ray.origin.x - center.x,                                                                                               \
            ray.origin.y - center.y,                                                                                               \
            ray.origin.z - center.z,                                                                                               \
        };                                                                                                                         \
        const float a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z; \
        const float b = 2.0f * (ray.direction.x * oc.x + ray.direction.y * oc.y + ray.direction.z * oc.z);                         \
        const float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - radius.x * radius.x;                                           \
        const float discrim = b * b - 4.0f * a * c;                                                                                \
        if (discrim <= 0) {                                                                                                        \
            continue;                                                                                                              \
        }                                                                                                                          \
        const float root = sqrtf(discrim);                                                                                         \
        t = (-b - root) / (2.0f * a);                                                                                              \
        if (t <= 0.001f) {                                                                                                         \
            t = (-b + root) / (2.0f * a);                                                                                          \
            if (t <= 0.001f) {                                                                                                     \
                continue;                                                                                                          \
            }                                                                                                                      \
        }                                                                                                                          \
        if (min_distance <= t) {                                                                                                   \
            continue;                                                                                                              \
        }                                                                                                                          \
    }

// http://woo4.me/wootracer/cylinder-intersection/
#define __rtx_intersect_cylinder_or_continue(ray, trans_a, trans_b, trans_c, inv_trans_a, inv_trans_b, inv_trans_c, unit_hit_face_normal, t, min_distance) \
    {                                                                                                                                                      \
        /* 方向ベクトルの変換では平行移動の成分を無視する */                                                                        \
        float3 d = {                                                                                                                                       \
            ray.direction.x * inv_trans_a.x + ray.direction.y * inv_trans_a.y + ray.direction.z * inv_trans_a.z,                                           \
            ray.direction.x * inv_trans_b.x + ray.direction.y * inv_trans_b.y + ray.direction.z * inv_trans_b.z,                                           \
            ray.direction.x * inv_trans_c.x + ray.direction.y * inv_trans_c.y + ray.direction.z * inv_trans_c.z,                                           \
        };                                                                                                                                                 \
        float3 o = {                                                                                                                                       \
            ray.origin.x * inv_trans_a.x + ray.origin.y * inv_trans_a.y + ray.origin.z * inv_trans_a.z + inv_trans_a.w,                                    \
            ray.origin.x * inv_trans_b.x + ray.origin.y * inv_trans_b.y + ray.origin.z * inv_trans_b.z + inv_trans_b.w,                                    \
            ray.origin.x * inv_trans_c.x + ray.origin.y * inv_trans_c.y + ray.origin.z * inv_trans_c.z + inv_trans_c.w,                                    \
        };                                                                                                                                                 \
        const float a = d.x * d.x + d.z * d.z;                                                                                                             \
        const float b = 2.0f * (d.x * o.x + d.z * o.z);                                                                                                    \
        const float c = (o.x * o.x + o.z * o.z) - radius * radius;                                                                                         \
        const float discrim = b * b - 4.0f * a * c;                                                                                                        \
        if (discrim <= 0) {                                                                                                                                \
            continue;                                                                                                                                      \
        }                                                                                                                                                  \
        const float root = sqrtf(discrim);                                                                                                                 \
        float t0 = (-b + root) / (2.0f * a);                                                                                                               \
        float t1 = (-b - root) / (2.0f * a);                                                                                                               \
        if (t0 > t1) {                                                                                                                                     \
            __swapf(t0, t1);                                                                                                                               \
        }                                                                                                                                                  \
        if (min_distance <= t0) {                                                                                                                          \
            continue;                                                                                                                                      \
        }                                                                                                                                                  \
        float y0 = o.y + t0 * d.y;                                                                                                                         \
        float y1 = o.y + t1 * d.y;                                                                                                                         \
        float3 p_hit;                                                                                                                                      \
        /* face normal in model space */                                                                                                                   \
        float3 normal;                                                                                                                                     \
        if (y0 < y_min) {                                                                                                                                  \
            if (y1 < y_min) {                                                                                                                              \
                continue;                                                                                                                                  \
            }                                                                                                                                              \
            float th = t0 + (t1 - t0) * (y0 - y_min) / (y0 - y1);                                                                                          \
            if (th <= 0.0f) {                                                                                                                              \
                continue;                                                                                                                                  \
            }                                                                                                                                              \
            __rtx_make_ray(p_hit, o, th, d);                                                                                                               \
            normal.x = 0.0f;                                                                                                                               \
            normal.y = -1.0f;                                                                                                                              \
            normal.z = 0.0f;                                                                                                                               \
            t = th;                                                                                                                                        \
        } else if (y0 >= y_min && y0 <= y_max) {                                                                                                           \
            if (t0 <= 0.001f) {                                                                                                                            \
                continue;                                                                                                                                  \
            }                                                                                                                                              \
            __rtx_make_ray(p_hit, o, t0, d);                                                                                                               \
            normal.x = p_hit.x;                                                                                                                            \
            normal.y = 0.0f;                                                                                                                               \
            normal.z = p_hit.z;                                                                                                                            \
            const float norm = sqrtf(normal.x * normal.x + normal.z * normal.z);                                                                           \
            normal.x /= norm;                                                                                                                              \
            normal.z /= norm;                                                                                                                              \
            t = t0;                                                                                                                                        \
        } else if (y0 > y_max) {                                                                                                                           \
            if (y1 > y_max) {                                                                                                                              \
                continue;                                                                                                                                  \
            }                                                                                                                                              \
            float th = t0 + (t1 - t0) * (y0 - y_max) / (y0 - y1);                                                                                          \
            if (th <= 0) {                                                                                                                                 \
                continue;                                                                                                                                  \
            }                                                                                                                                              \
            __rtx_make_ray(p_hit, o, th, d);                                                                                                               \
            normal.x = 0.0f;                                                                                                                               \
            normal.y = 1.0f;                                                                                                                               \
            normal.z = 0.0f;                                                                                                                               \
            t = th;                                                                                                                                        \
        } else {                                                                                                                                           \
            continue;                                                                                                                                      \
        }                                                                                                                                                  \
        /* face normal in view space*/                                                                                                                     \
        unit_hit_face_normal.x = normal.x * trans_a.x + normal.y * trans_a.y + normal.z * trans_a.z;                                                       \
        unit_hit_face_normal.y = normal.x * trans_b.x + normal.y * trans_b.y + normal.z * trans_b.z;                                                       \
        unit_hit_face_normal.z = normal.x * trans_c.x + normal.y * trans_c.y + normal.z * trans_c.z;                                                       \
    }

// http://www.pbr-book.org/3ed-2018/Shapes/Other_Quadrics.html
#define __rtx_intersect_cone_or_continue(ray, trans_a, trans_b, trans_c, inv_trans_a, inv_trans_b, inv_trans_c, unit_hit_face_normal, t, min_distance) \
    {                                                                                                                                                  \
        /* 方向ベクトルの変換では平行移動の成分を無視する */                                                                    \
        float3 d = {                                                                                                                                   \
            ray.direction.x * inv_trans_a.x + ray.direction.y * inv_trans_a.y + ray.direction.z * inv_trans_a.z,                                       \
            ray.direction.x * inv_trans_b.x + ray.direction.y * inv_trans_b.y + ray.direction.z * inv_trans_b.z,                                       \
            ray.direction.x * inv_trans_c.x + ray.direction.y * inv_trans_c.y + ray.direction.z * inv_trans_c.z,                                       \
        };                                                                                                                                             \
        float3 o = {                                                                                                                                   \
            ray.origin.x * inv_trans_a.x + ray.origin.y * inv_trans_a.y + ray.origin.z * inv_trans_a.z + inv_trans_a.w,                                \
            ray.origin.x * inv_trans_b.x + ray.origin.y * inv_trans_b.y + ray.origin.z * inv_trans_b.z + inv_trans_b.w,                                \
            ray.origin.x * inv_trans_c.x + ray.origin.y * inv_trans_c.y + ray.origin.z * inv_trans_c.z + inv_trans_c.w,                                \
        };                                                                                                                                             \
        const float coeff = (height * height) / (radius * radius);                                                                                     \
        const float a = coeff * (d.x * d.x + d.z * d.z) - (d.y * d.y);                                                                                 \
        const float b = 2.0f * coeff * (d.x * o.x + d.z * o.z) + 2.0f * (d.y * height - d.y * o.y);                                                    \
        const float c = coeff * (o.x * o.x + o.z * o.z) + (2.0f * o.y * height) - (o.y * o.y) - (height * height);                                     \
        const float discrim = b * b - 4.0f * a * c;                                                                                                    \
        if (discrim <= 0) {                                                                                                                            \
            continue;                                                                                                                                  \
        }                                                                                                                                              \
        float t0, t1;                                                                                                                                  \
        if (fabs(a) <= 1e-6) { /* 誤差対策 */                                                                                                      \
            t0 = -c / b;                                                                                                                               \
            t1 = 9999.0f;                                                                                                                              \
        } else {                                                                                                                                       \
            const float root = sqrtf(discrim);                                                                                                         \
            t0 = (-b + root) / (2.0f * a);                                                                                                             \
            t1 = (-b - root) / (2.0f * a);                                                                                                             \
        }                                                                                                                                              \
        float y0 = o.y + t0 * d.y;                                                                                                                     \
        float3 p_hit;                                                                                                                                  \
        bool did_hit_bottom = false;                                                                                                                   \
        if (y0 > height) {                                                                                                                             \
            __swapf(t0, t1);                                                                                                                           \
            y0 = o.y + t0 * d.y;                                                                                                                       \
            if (y0 < 0.0f || height < y0) {                                                                                                            \
                continue;                                                                                                                              \
            }                                                                                                                                          \
            __rtx_make_ray(p_hit, o, t0, d);                                                                                                           \
        } else {                                                                                                                                       \
            if (t0 > t1) {                                                                                                                             \
                __swapf(t0, t1);                                                                                                                       \
            }                                                                                                                                          \
            y0 = o.y + t0 * d.y;                                                                                                                       \
            float y1 = o.y + t1 * d.y;                                                                                                                 \
            if (y0 < 0.0f) {                                                                                                                           \
                if (y1 < 0.0f) {                                                                                                                       \
                    continue;                                                                                                                          \
                }                                                                                                                                      \
                if (height < y1) {                                                                                                                     \
                    continue;                                                                                                                          \
                }                                                                                                                                      \
                did_hit_bottom = true;                                                                                                                 \
            } else if (0.0f <= y0 && y0 <= height) {                                                                                                   \
                if (y1 > height) {                                                                                                                     \
                    did_hit_bottom = true;                                                                                                             \
                }                                                                                                                                      \
            } else {                                                                                                                                   \
                continue;                                                                                                                              \
            }                                                                                                                                          \
            if (did_hit_bottom) {                                                                                                                      \
                t0 = -o.y / d.y;                                                                                                                       \
            }                                                                                                                                          \
            __rtx_make_ray(p_hit, o, t0, d);                                                                                                           \
        }                                                                                                                                              \
        if (t0 <= 0.001) {                                                                                                                             \
            continue;                                                                                                                                  \
        }                                                                                                                                              \
        if (min_distance <= t0) {                                                                                                                      \
            continue;                                                                                                                                  \
        }                                                                                                                                              \
        /* face normal in model space */                                                                                                               \
        float3 normal;                                                                                                                                 \
        if (did_hit_bottom) {                                                                                                                          \
            normal.x = 0.0f;                                                                                                                           \
            normal.y = -1.0f;                                                                                                                          \
            normal.z = 0.0f;                                                                                                                           \
        } else {                                                                                                                                       \
            float norm;                                                                                                                                \
            normal.x = p_hit.x;                                                                                                                        \
            normal.y = height / radius;                                                                                                                \
            normal.z = p_hit.z;                                                                                                                        \
            norm = sqrtf(normal.x * normal.x + normal.z * normal.z);                                                                                   \
            normal.x /= norm;                                                                                                                          \
            normal.y = height / radius;                                                                                                                \
            normal.z /= norm;                                                                                                                          \
            norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);                                                             \
            normal.x /= norm;                                                                                                                          \
            normal.y /= norm;                                                                                                                          \
            normal.z /= norm;                                                                                                                          \
        }                                                                                                                                              \
        /* face normal in view space*/                                                                                                                 \
        unit_hit_face_normal.x = normal.x * trans_a.x + normal.y * trans_a.y + normal.z * trans_a.z;                                                   \
        unit_hit_face_normal.y = normal.x * trans_b.x + normal.y * trans_b.y + normal.z * trans_b.z;                                                   \
        unit_hit_face_normal.z = normal.x * trans_c.x + normal.y * trans_c.y + normal.z * trans_c.z;                                                   \
                                                                                                                                                       \
        t = t0;                                                                                                                                        \
    }

#define __rtx_bvh_traversal_one_step_or_continue(ray, node, ray_direction_inv, bvh_current_node_index)                         \
    {                                                                                                                          \
        float tmin = ((ray_direction_inv.x < 0 ? node.aabb_max.x : node.aabb_min.x) - ray.origin.x) * ray_direction_inv.x;     \
        float tmax = ((ray_direction_inv.x < 0 ? node.aabb_min.x : node.aabb_max.x) - ray.origin.x) * ray_direction_inv.x;     \
        float tmp_tmin = ((ray_direction_inv.y < 0 ? node.aabb_max.y : node.aabb_min.y) - ray.origin.y) * ray_direction_inv.y; \
        float tmp_tmax = ((ray_direction_inv.y < 0 ? node.aabb_min.y : node.aabb_max.y) - ray.origin.y) * ray_direction_inv.y; \
        if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {                                                                          \
            bvh_current_node_index = node.miss_node_index;                                                                     \
            continue;                                                                                                          \
        }                                                                                                                      \
        if (tmp_tmin > tmin) {                                                                                                 \
            tmin = tmp_tmin;                                                                                                   \
        }                                                                                                                      \
        if (tmp_tmax < tmax) {                                                                                                 \
            tmax = tmp_tmax;                                                                                                   \
        }                                                                                                                      \
        tmp_tmin = ((ray_direction_inv.z < 0 ? node.aabb_max.z : node.aabb_min.z) - ray.origin.z) * ray_direction_inv.z;       \
        tmp_tmax = ((ray_direction_inv.z < 0 ? node.aabb_min.z : node.aabb_max.z) - ray.origin.z) * ray_direction_inv.z;       \
        if ((tmin > tmp_tmax) || (tmp_tmin > tmax)) {                                                                          \
            bvh_current_node_index = node.miss_node_index;                                                                     \
            continue;                                                                                                          \
        }                                                                                                                      \
        if (tmp_tmin > tmin) {                                                                                                 \
            tmin = tmp_tmin;                                                                                                   \
        }                                                                                                                      \
        if (tmp_tmax < tmax) {                                                                                                 \
            tmax = tmp_tmax;                                                                                                   \
        }                                                                                                                      \
        /* 計算誤差を防ぐ */                                                                                            \
        if (tmax < 0.001) {                                                                                                    \
            bvh_current_node_index = node.miss_node_index;                                                                     \
            continue;                                                                                                          \
        }                                                                                                                      \
    }

#define rtx_cuda_fetch_uv_coordinate_in_linear_memory(                                                              \
    x,                                                                                                              \
    y,                                                                                                              \
    uv_coordinate_array,                                                                                            \
    hit_face,                                                                                                       \
    hit_object)                                                                                                     \
    {                                                                                                               \
        const rtxUVCoordinate uv_a = uv_coordinate_array[hit_face.a + hit_object.serialized_uv_coordinates_offset]; \
        const rtxUVCoordinate uv_b = uv_coordinate_array[hit_face.b + hit_object.serialized_uv_coordinates_offset]; \
        const rtxUVCoordinate uv_c = uv_coordinate_array[hit_face.c + hit_object.serialized_uv_coordinates_offset]; \
        x = max(0.0f, lambda.x * uv_a.u + lambda.y * uv_b.u + lambda.z * uv_c.u);                                   \
        y = max(0.0f, lambda.x * uv_a.v + lambda.y * uv_b.v + lambda.z * uv_c.v);                                   \
    }
#define rtx_cuda_fetch_uv_coordinate_in_texture_memory(                                                                \
    x,                                                                                                                 \
    y,                                                                                                                 \
    uv_coordinate_array,                                                                                               \
    hit_face,                                                                                                          \
    hit_object)                                                                                                        \
    {                                                                                                                  \
        const float2 uv_a = tex1Dfetch(uv_coordinate_array, hit_face.a + hit_object.serialized_uv_coordinates_offset); \
        const float2 uv_b = tex1Dfetch(uv_coordinate_array, hit_face.b + hit_object.serialized_uv_coordinates_offset); \
        const float2 uv_c = tex1Dfetch(uv_coordinate_array, hit_face.c + hit_object.serialized_uv_coordinates_offset); \
        x = max(0.0f, lambda.x * uv_a.x + lambda.y * uv_b.x + lambda.z * uv_c.x);                                      \
        y = max(0.0f, lambda.x * uv_a.y + lambda.y * uv_b.y + lambda.z * uv_c.y);                                      \
    }
#define __rtx_fetch_color_in_linear_memory(                              \
    hit_point,                                                           \
    hit_object,                                                          \
    hit_face,                                                            \
    hit_color,                                                           \
    material_attribute_byte_array,                                       \
    color_mapping_array,                                                 \
    texture_object_array,                                                \
    uv_coordinate_array)                                                 \
    {                                                                    \
        __rtx_fetch_color(rtx_cuda_fetch_uv_coordinate_in_linear_memory, \
            hit_point,                                                   \
            hit_object,                                                  \
            hit_face,                                                    \
            hit_color,                                                   \
            material_attribute_byte_array,                               \
            color_mapping_array,                                         \
            texture_object_array,                                        \
            uv_coordinate_array);                                        \
    }
#define __rtx_fetch_color_in_texture_memory(                              \
    hit_point,                                                            \
    hit_object,                                                           \
    hit_face,                                                             \
    hit_color,                                                            \
    material_attribute_byte_array,                                        \
    color_mapping_array,                                                  \
    texture_object_array,                                                 \
    uv_coordinate_array)                                                  \
    {                                                                     \
        __rtx_fetch_color(rtx_cuda_fetch_uv_coordinate_in_texture_memory, \
            hit_point,                                                    \
            hit_object,                                                   \
            hit_face,                                                     \
            hit_color,                                                    \
            material_attribute_byte_array,                                \
            color_mapping_array,                                          \
            texture_object_array,                                         \
            uv_coordinate_array);                                         \
    }

#define __rtx_fetch_color(                                                                                                                                                                                 \
    fetch_uv_coordinates,                                                                                                                                                                                  \
    hit_point,                                                                                                                                                                                             \
    hit_object,                                                                                                                                                                                            \
    hit_face,                                                                                                                                                                                              \
    hit_color,                                                                                                                                                                                             \
    material_attribute_byte_array,                                                                                                                                                                         \
    color_mapping_array,                                                                                                                                                                                   \
    texture_object_array,                                                                                                                                                                                  \
    uv_coordinate_array)                                                                                                                                                                                   \
    {                                                                                                                                                                                                      \
        int mapping_type = hit_object.mapping_type;                                                                                                                                                        \
        int geometry_type = hit_object.geometry_type;                                                                                                                                                      \
        if (mapping_type == RTXMappingTypeSolidColor) {                                                                                                                                                    \
            rtxRGBAColor color = color_mapping_array[hit_object.mapping_index];                                                                                                                            \
            hit_color.r = color.r;                                                                                                                                                                         \
            hit_color.g = color.g;                                                                                                                                                                         \
            hit_color.b = color.b;                                                                                                                                                                         \
        } else if (mapping_type == RTXMappingTypeTexture) {                                                                                                                                                \
            if (geometry_type == RTXGeometryTypeStandard) {                                                                                                                                                \
                /* 衝突位置を重心座標系で表す */                                                                                                                                              \
                /* https://shikihuiku.wordpress.com/2017/05/23/barycentric-coordinates%E3%81%AE%E8%A8%88%E7%AE%97%E3%81%A8perspective-correction-partial-derivative%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/*/ \
                float3 d1 = { hit_va.x - hit_vc.x, hit_va.y - hit_vc.y, hit_va.z - hit_vc.z };                                                                                                             \
                float3 d2 = { hit_vb.x - hit_vc.x, hit_vb.y - hit_vc.y, hit_vb.z - hit_vc.z };                                                                                                             \
                float3 d = { hit_point.x - hit_vc.x, hit_point.y - hit_vc.y, hit_point.z - hit_vc.z };                                                                                                     \
                const float d1x = d1.x * d1.x + d1.y * d1.y + d1.z * d1.z;                                                                                                                                 \
                const float d1y = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;                                                                                                                                 \
                const float d2x = d1y;                                                                                                                                                                     \
                const float d2y = d2.x * d2.x + d2.y * d2.y + d2.z * d2.z;                                                                                                                                 \
                const float dx = d.x * d1.x + d.y * d1.y + d.z * d1.z;                                                                                                                                     \
                const float dy = d.x * d2.x + d.y * d2.y + d.z * d2.z;                                                                                                                                     \
                const float det = d1x * d2y - d1y * d2x;                                                                                                                                                   \
                float3 lambda = { (dx * d2y - dy * d2x) / det, (d1x * dy - d1y * dx) / det, 0.0f };                                                                                                        \
                lambda.z = 1.0f - lambda.x - lambda.y;                                                                                                                                                     \
                float x, y;                                                                                                                                                                                \
                fetch_uv_coordinates(x, y, uv_coordinate_array, hit_face, hit_object);                                                                                                                     \
                float4 color = tex2D<float4>(texture_object_array[hit_object.mapping_index], x, y);                                                                                                        \
                hit_color.r = color.x;                                                                                                                                                                     \
                hit_color.g = color.y;                                                                                                                                                                     \
                hit_color.b = color.z;                                                                                                                                                                     \
            } else {                                                                                                                                                                                       \
                hit_color.r = 0.0f;                                                                                                                                                                        \
                hit_color.g = 0.0f;                                                                                                                                                                        \
                hit_color.b = 0.0f;                                                                                                                                                                        \
            }                                                                                                                                                                                              \
        }                                                                                                                                                                                                  \
    }

#define __rtx_compute_brdf(                                                                                                                                                                   \
    hit_face_normal,                                                                                                                                                                          \
    hit_object,                                                                                                                                                                               \
    hit_face,                                                                                                                                                                                 \
    unit_current_ray_direction,                                                                                                                                                               \
    unit_next_ray_direction,                                                                                                                                                                  \
    material_attribute_byte_array,                                                                                                                                                            \
    brdf)                                                                                                                                                                                     \
    {                                                                                                                                                                                         \
        int material_type = hit_object.layerd_material_types.outside;                                                                                                                         \
        if (material_type == RTXMaterialTypeLambert) {                                                                                                                                        \
            rtxLambertMaterialAttribute attr = ((rtxLambertMaterialAttribute*)&material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];                            \
            float cos_ref = hit_face_normal.x * unit_next_ray_direction.x + hit_face_normal.y * unit_next_ray_direction.y + hit_face_normal.z * unit_next_ray_direction.z;                    \
            brdf = attr.albedo * cos_ref / M_PI;                                                                                                                                              \
        } else if (material_type == RTXMaterialTypeOrenNayar) {                                                                                                                               \
            /* https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model */                                                                                                          \
            rtxOrenNayarMaterialAttribute attr = ((rtxOrenNayarMaterialAttribute*)&material_attribute_byte_array[hit_object.material_attribute_byte_array_offset])[0];                        \
            const float squared_roughness = attr.roughness * attr.roughness;                                                                                                                  \
            const float a = 1.0f - 0.5f * ((squared_roughness) / (squared_roughness + 0.33));                                                                                                 \
            const float b = 0.45f * ((squared_roughness) / (squared_roughness + 0.09));                                                                                                       \
            const float cos_view = -(hit_face_normal.x * unit_current_ray_direction.x + hit_face_normal.y * unit_current_ray_direction.y + hit_face_normal.z * unit_current_ray_direction.z); \
            const float cos_ref = hit_face_normal.x * unit_next_ray_direction.x + hit_face_normal.y * unit_next_ray_direction.y + hit_face_normal.z * unit_next_ray_direction.z;              \
            const float theta_view = acos(cos_view);                                                                                                                                          \
            const float theta_ref = acos(cos_ref);                                                                                                                                            \
            const float sin_alpha = sin(max(theta_view, theta_ref));                                                                                                                          \
            const float tan_beta = tan(min(theta_view, theta_ref));                                                                                                                           \
            float3 cross_view = {                                                                                                                                                             \
                -(unit_current_ray_direction.y * hit_face_normal.z - unit_current_ray_direction.z * hit_face_normal.y),                                                                       \
                -(unit_current_ray_direction.z * hit_face_normal.x - unit_current_ray_direction.x * hit_face_normal.z),                                                                       \
                -(unit_current_ray_direction.x * hit_face_normal.y - unit_current_ray_direction.y * hit_face_normal.x),                                                                       \
            };                                                                                                                                                                                \
            float norm = sqrtf(cross_view.x * cross_view.x + cross_view.y * cross_view.y + cross_view.z * cross_view.z);                                                                      \
            cross_view.x /= norm;                                                                                                                                                             \
            cross_view.y /= norm;                                                                                                                                                             \
            cross_view.z /= norm;                                                                                                                                                             \
            float3 cross_ref = {                                                                                                                                                              \
                unit_next_ray_direction.y * hit_face_normal.z - unit_next_ray_direction.z * hit_face_normal.y,                                                                                \
                unit_next_ray_direction.z * hit_face_normal.x - unit_next_ray_direction.x * hit_face_normal.z,                                                                                \
                unit_next_ray_direction.x * hit_face_normal.y - unit_next_ray_direction.y * hit_face_normal.x,                                                                                \
            };                                                                                                                                                                                \
            norm = sqrtf(cross_ref.x * cross_ref.x + cross_ref.y * cross_ref.y + cross_ref.z * cross_ref.z);                                                                                  \
            cross_ref.x /= norm;                                                                                                                                                              \
            cross_ref.y /= norm;                                                                                                                                                              \
            cross_ref.z /= norm;                                                                                                                                                              \
            const float cos_phi = cross_view.x * cross_ref.x + cross_view.y * cross_ref.y + cross_view.z * cross_ref.z;                                                                       \
            const float coeff = attr.albedo * cos_ref * (a + (b * max(0.0f, cos_phi) * sin_alpha * tan_beta));                                                                                \
            brdf = coeff / M_PI;                                                                                                                                                              \
        }                                                                                                                                                                                     \
    }

#define __rtx_fetch_bvh_node_in_texture_memory(node, texture_ref, node_index)          \
    {                                                                                  \
        float4 attributes_as_float4 = tex1Dfetch(texture_ref, node_index * 3 + 0);     \
        int4* attributes_as_int4_ptr = reinterpret_cast<int4*>(&attributes_as_float4); \
        node.hit_node_index = attributes_as_int4_ptr->x;                               \
        node.miss_node_index = attributes_as_int4_ptr->y;                              \
        node.assigned_face_index_start = attributes_as_int4_ptr->z;                    \
        node.assigned_face_index_end = attributes_as_int4_ptr->w;                      \
        node.aabb_max = tex1Dfetch(texture_ref, node_index * 3 + 1);                   \
        node.aabb_min = tex1Dfetch(texture_ref, node_index * 3 + 2);                   \
    }

#define __rtx_sample_ray_direction(                                                                                                                \
    unit_hit_face_normal,                                                                                                                          \
    direction,                                                                                                                                     \
    cosine_term,                                                                                                                                   \
    curand_state)                                                                                                                                  \
    {                                                                                                                                              \
        float4 unit_diffuse = curand_normal4(&curand_state);                                                                                       \
        float norm = sqrtf(unit_diffuse.x * unit_diffuse.x + unit_diffuse.y * unit_diffuse.y + unit_diffuse.z * unit_diffuse.z);                   \
        unit_diffuse.x /= norm;                                                                                                                    \
        unit_diffuse.y /= norm;                                                                                                                    \
        unit_diffuse.z /= norm;                                                                                                                    \
        cosine_term = unit_hit_face_normal.x * unit_diffuse.x + unit_hit_face_normal.y * unit_diffuse.y + unit_hit_face_normal.z * unit_diffuse.z; \
        if (cosine_term < 0.0f) {                                                                                                                  \
            unit_diffuse.x *= -1;                                                                                                                  \
            unit_diffuse.y *= -1;                                                                                                                  \
            unit_diffuse.z *= -1;                                                                                                                  \
            cosine_term *= -1;                                                                                                                     \
        }                                                                                                                                          \
        direction.x = unit_diffuse.x;                                                                                                              \
        direction.y = unit_diffuse.y;                                                                                                              \
        direction.z = unit_diffuse.z;                                                                                                              \
    }

#define __rtx_update_ray(                             \
    ray,                                              \
    ray_direction_inv,                                \
    hit_point,                                        \
    unit_next_ray_direction)                          \
    {                                                 \
        ray.origin.x = hit_point.x;                   \
        ray.origin.y = hit_point.y;                   \
        ray.origin.z = hit_point.z;                   \
        ray.direction.x = unit_next_ray_direction.x;  \
        ray.direction.y = unit_next_ray_direction.y;  \
        ray.direction.z = unit_next_ray_direction.z;  \
        ray_direction_inv.x = 1.0f / ray.direction.x; \
        ray_direction_inv.y = 1.0f / ray.direction.y; \
        ray_direction_inv.z = 1.0f / ray.direction.z; \
    }

#define __check_kernel_arguments()                                    \
    {                                                                 \
        assert(gpu_serialized_face_vertex_index_array != NULL);       \
        assert(gpu_serialized_vertex_array != NULL);                  \
        assert(gpu_serialized_object_array != NULL);                  \
        assert(gpu_serialized_material_attribute_byte_array != NULL); \
        assert(gpu_serialized_threaded_bvh_array != NULL);            \
        assert(gpu_serialized_threaded_bvh_node_array != NULL);       \
        assert(gpu_serialized_render_array != NULL);                  \
        if (args.color_mapping_array_size > 0) {                      \
            assert(gpu_serialized_color_mapping_array != NULL);       \
        }                                                             \
        if (args.uv_coordinate_array_size > 0) {                      \
            assert(gpu_serialized_uv_coordinate_array != NULL);       \
        }                                                             \
    }

#define __xorshift_uniform(ret, x, y, z, w) \
    {                                       \
        unsigned long t = x ^ (x << 11);    \
        t = (t ^ (t >> 8));                 \
        x = y;                              \
        y = z;                              \
        z = w;                              \
        w = (w ^ (w >> 19)) ^ t;            \
        ret = float(w & 0xFFFF) / 65535.0;  \
    }

#define __xorshift_init(curand_seed)    \
    unsigned long xors_x = curand_seed; \
    unsigned long xors_y = 362436069;   \
    unsigned long xors_z = 521288629;   \
    unsigned long xors_w = 88675123;

#define __swapf(a, b)        \
    {                        \
        const float tmp = a; \
        a = b;               \
        b = tmp;             \
    }

#define __rtx_make_ray(ret, o, t, d) \
    ret.x = o.x + t * d.x;           \
    ret.y = o.y + t * d.y;           \
    ret.z = o.z + t * d.z;

#define __rtx_generate_ray(ray, args, aspect_ratio)                                                                                          \
    /* スーパーサンプリング */                                                                                                     \
    float2 noise = { 0.0f, 0.0f };                                                                                                           \
    if (args.supersampling_enabled) {                                                                                                        \
        __xorshift_uniform(noise.x, xors_x, xors_y, xors_z, xors_w);                                                                         \
        __xorshift_uniform(noise.y, xors_x, xors_y, xors_z, xors_w);                                                                         \
    }                                                                                                                                        \
    /* 方向 */                                                                                                                             \
    ray.direction.x = 2.0f * float(target_pixel_x + noise.x) / float(args.screen_width) - 1.0f;                                              \
    ray.direction.y = -(2.0f * float(target_pixel_y + noise.y) / float(args.screen_height) - 1.0f) / aspect_ratio;                           \
    ray.direction.z = -args.ray_origin_z;                                                                                                    \
    /* 始点 */                                                                                                                             \
    if (args.camera_type == RTXCameraTypePerspective) {                                                                                      \
        ray.origin.x = 0.0f;                                                                                                                 \
        ray.origin.y = 0.0f;                                                                                                                 \
        ray.origin.z = args.ray_origin_z;                                                                                                    \
        ray.origin.w = 0.0f;                                                                                                                 \
        /* 正規化 */                                                                                                                      \
        const float norm = sqrtf(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z); \
        ray.direction.x /= norm;                                                                                                             \
        ray.direction.y /= norm;                                                                                                             \
        ray.direction.z /= norm;                                                                                                             \
    } else {                                                                                                                                 \
        ray.origin.x = ray.direction.x * args.ray_origin_z;                                                                                  \
        ray.origin.y = ray.direction.y * args.ray_origin_z;                                                                                  \
        ray.origin.z = args.ray_origin_z;                                                                                                    \
        ray.origin.w = 0.0f;                                                                                                                 \
        ray.direction.x = 0.0f;                                                                                                              \
        ray.direction.y = 0.0f;                                                                                                              \
        ray.direction.z = -1.0f;                                                                                                             \
    }
