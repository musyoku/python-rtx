#include "renderer.h"
#include "../../../geometry/sphere.h"
#include "../../../geometry/standard.h"
#include "../intersect.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

namespace rtx {

using namespace cpu;
namespace py = pybind11;

bool hit_test_sphere(
    glm::vec3& center,
    float radius,
    std::unique_ptr<Ray>& ray,
    float& min_distance,
    glm::vec3& hit_point,
    glm::vec3& face_normal)
{
    float t = intersect_sphere(center, radius, ray);
    if (t <= 0.001f) {
        return false;
    }
    if (min_distance <= t) {
        return false;
    }
    min_distance = t;
    hit_point = ray->point(t);
    face_normal = glm::normalize(hit_point - center);
    return true;
}

bool hit_test_triangle(
    glm::vec3& va,
    glm::vec3& vb,
    glm::vec3& vc,
    glm::vec3& face_normal,
    std::unique_ptr<Ray>& ray,
    float& min_distance,
    glm::vec3& hit_point)
{
    float t = intersect_triangle(va, vb, vc, ray);
    if (t <= 0.001f) {
        return false;
    }
    if (min_distance <= t) {
        return false;
    }
    min_distance = t;
    hit_point = ray->point(t);
    return true;
}

bool hit_test(std::vector<std::shared_ptr<Mesh>>& mesh_array,
    std::unique_ptr<Ray>& ray,
    glm::vec3& new_origin,
    glm::vec3& hit_face_normal,
    std::shared_ptr<Mesh>& hit_mesh)
{
    bool did_hit = false;
    glm::vec3 hit_point = glm::vec3(0.0f);
    float min_distance = FLT_MAX;
    for (auto& mesh : mesh_array) {
        auto& geometry = mesh->_geometry;

        if (geometry->type() == GeometryTypeSphere) {
            SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
            if (hit_test_sphere(sphere->_center, sphere->_radius, ray, min_distance, hit_point, hit_face_normal)) {
                new_origin = hit_point;
                did_hit = true;
                hit_mesh = mesh;
            }
        }

        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            for (unsigned int n = 0; n < standard_geometry->_face_vertex_indices_array.size(); n++) {
                glm::vec<3, int>& face = standard_geometry->_face_vertex_indices_array[n];
                glm::vec3& va = standard_geometry->_vertex_array[face[0]];
                glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
                glm::vec3& vc = standard_geometry->_vertex_array[face[2]];
                glm::vec3& triangle_face_normal = standard_geometry->_face_normal_array[n];
                if (hit_test_triangle(va, vb, vc, triangle_face_normal, ray, min_distance, hit_point)) {
                    new_origin = hit_point;
                    did_hit = true;
                    hit_mesh = mesh;
                    hit_face_normal = triangle_face_normal;
                }
            }
        }
    }
    return did_hit;
}

RayTracingCPURenderer::RayTracingCPURenderer()
{
    std::random_device seed_gen;
    _normal_engine = std::default_random_engine(seed_gen());
    _normal_distribution = std::normal_distribution<float>(0.0, 1.0);
}

glm::vec3 RayTracingCPURenderer::compute_color(std::vector<std::shared_ptr<Mesh>>& mesh_array,
    std::unique_ptr<Ray>& ray,
    int current_reflection,
    int max_reflextions)
{
    if (current_reflection == max_reflextions) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }
    glm::vec3 new_origin = glm::vec3(0.0f);
    glm::vec3 hit_face_normal = glm::vec3(0.0f);
    std::shared_ptr<Mesh> hit_mesh;
    if (hit_test(mesh_array, ray, new_origin, hit_face_normal, hit_mesh)) {
        ray->_origin = new_origin;
        glm::vec3 face_normal = hit_face_normal;

        const auto& material = hit_mesh->_material;
        if (material->type() == MaterialTypeEmissive) {
            return material->emit_color();
        }

        // detect backface
        const float t = glm::dot(hit_face_normal, ray->_direction);
        if (t > 0.0f) {
            face_normal *= -1.0f;
        }

        // diffuse
        glm::vec3 diffuse_vec = glm::vec3(_normal_distribution(_normal_engine), _normal_distribution(_normal_engine), _normal_distribution(_normal_engine));
        glm::vec3 unit_diffuse_vec = diffuse_vec / glm::length(diffuse_vec);
        const float dot = glm::dot(face_normal, unit_diffuse_vec);
        if (dot < 0.0f) {
            unit_diffuse_vec *= -1.0f;
        }

        // specular
        glm::vec3 unit_specular_vec = ray->_direction - 2.0f * glm::dot(ray->_direction, face_normal) * face_normal;

        ray->_direction = material->reflect_ray(unit_diffuse_vec, unit_specular_vec);

        glm::vec3 input_color = compute_color(mesh_array, ray, current_reflection + 1, max_reflextions);

        // return (unit_diffuse_vec + 1.0f) * 0.5f;
        return material->reflect_color(input_color);
    }
    return glm::vec3(0.0f);
}

void RayTracingCPURenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<float, py::array::c_style> buffer)
{
    py::gil_scoped_release release;

    _height = buffer.shape(0);
    _width = buffer.shape(1);
    int channels = buffer.shape(2);
    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }
    auto pixel = buffer.mutable_unchecked<3>();
    std::vector<std::unique_ptr<Ray>> ray_array;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);

    std::vector<std::shared_ptr<Mesh>> mesh_array;
    for (auto& mesh : scene->_mesh_array) {
        auto& geometry = mesh->_geometry;
        if (geometry->type() == GeometryTypeSphere) {
            SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
            std::shared_ptr<SphereGeometry> geometry_in_view_space = std::make_shared<SphereGeometry>(sphere->_radius);

            glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;

            glm::vec4 homogeneous_center = glm::vec4(sphere->_center, 1.0f);
            glm::vec4 homogeneous_center_in_view_space = mv_matrix * homogeneous_center;
            geometry_in_view_space->_center = glm::vec3(homogeneous_center_in_view_space.x, homogeneous_center_in_view_space.y, homogeneous_center_in_view_space.z);

            std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(geometry_in_view_space, mesh->_material);
            mesh_array.emplace_back(mesh_in_view_space);
        }
        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            std::shared_ptr<StandardGeometry> standard_geometry_in_view_space = std::make_shared<StandardGeometry>();
            standard_geometry_in_view_space->_face_vertex_indices_array = standard_geometry->_face_vertex_indices_array;

            glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;

            for (auto& vertex : standard_geometry->_vertex_array) {
                glm::vec4 homogeneous_vertex = glm::vec4(vertex, 1.0f);
                glm::vec4 homogeneous_vertex_in_view_space = mv_matrix * homogeneous_vertex;
                standard_geometry_in_view_space->_vertex_array.emplace_back(glm::vec3(homogeneous_vertex_in_view_space.x, homogeneous_vertex_in_view_space.y, homogeneous_vertex_in_view_space.z));
            }

            for (auto& face_normal : standard_geometry->_face_normal_array) {
                glm::vec4 homogeneous_face_normal = glm::vec4(face_normal, 1.0f);
                glm::vec4 homogeneous_face_normal_in_view_space = mv_matrix * homogeneous_face_normal;
                glm::vec3 face_normal_in_view_space = glm::normalize(glm::vec3(homogeneous_face_normal_in_view_space.x, homogeneous_face_normal_in_view_space.y, homogeneous_face_normal_in_view_space.z));
                standard_geometry_in_view_space->_face_normal_array.emplace_back(face_normal_in_view_space);
            }

            std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(standard_geometry_in_view_space, mesh->_material);
            mesh_array.emplace_back(mesh_in_view_space);
        }
    }

    int ns = options->num_rays_per_pixel();

    // #pragma omp parallel for
    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 pixel_color = glm::vec3(0, 0, 0);

            for (int m = 0; m < ns; m++) {
                float ray_target_x = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                float ray_target_y = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
                glm::vec3 direction = glm::normalize(glm::vec3(ray_target_x, ray_target_y, -1.0f));
                std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);

                glm::vec3 color = compute_color(mesh_array, ray, 0, options->path_depth());
                pixel_color.r += color.r;
                pixel_color.g += color.g;
                pixel_color.b += color.b;
            }

            pixel(y, x, 0) = glm::clamp(pixel_color.r / float(ns), 0.0f, 1.0f);
            pixel(y, x, 1) = glm::clamp(pixel_color.g / float(ns), 0.0f, 1.0f);
            pixel(y, x, 2) = glm::clamp(pixel_color.b / float(ns), 0.0f, 1.0f);
        }
    }
}

void RayTracingCPURenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    unsigned char* buffer,
    int height,
    int width,
    int channels)
{
    _height = height;
    _width = width;
    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }
    std::vector<std::unique_ptr<Ray>> ray_array;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);

    int ns = options->num_rays_per_pixel();

    std::vector<std::shared_ptr<Mesh>> mesh_array;
    for (auto mesh : scene->_mesh_array) {
        auto geometry = mesh->_geometry;
        if (geometry->type() == GeometryTypeSphere) {
            SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
            std::shared_ptr<SphereGeometry> geometry_in_view_space = std::make_shared<SphereGeometry>(sphere->_radius);

            glm::vec4 homogeneous_center = glm::vec4(sphere->_center, 1.0f);
            glm::vec4 homogeneous_center_in_view_space = camera->_view_matrix * mesh->_model_matrix * homogeneous_center;
            geometry_in_view_space->_center = glm::vec3(homogeneous_center_in_view_space.x, homogeneous_center_in_view_space.y, homogeneous_center_in_view_space.z);

            std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(geometry_in_view_space, mesh->_material);
            mesh_array.emplace_back(mesh_in_view_space);
        }
    }

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 pixel_color = glm::vec3(0, 0, 0);

            for (int m = 0; m < ns; m++) {
                float ray_target_x = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                float ray_target_y = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
                glm::vec3 direction = glm::normalize(glm::vec3(ray_target_x, ray_target_y, -1.0f));
                std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);

                glm::vec3 color = compute_color(mesh_array, ray, 0, options->path_depth());
                pixel_color.r += color.r;
                pixel_color.g += color.g;
                pixel_color.b += color.b;
            }

            int index = y * width * channels + x * channels;
            buffer[index + 0] = glm::clamp((int)(pixel_color.r / float(ns) * 255.0f), 0, 255);
            buffer[index + 1] = glm::clamp((int)(pixel_color.g / float(ns) * 255.0f), 0, 255);
            buffer[index + 2] = glm::clamp((int)(pixel_color.b / float(ns) * 255.0f), 0, 255);
        }
    }
}
}