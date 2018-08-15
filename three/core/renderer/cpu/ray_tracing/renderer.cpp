#include "renderer.h"
#include "../../../geometry/sphere.h"
#include "../hit_test.h"
#include <iostream>
#include <memory>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
namespace three {

using namespace cpu;
namespace py = pybind11;

bool hit_test_sphere(std::shared_ptr<Mesh>& mesh,
    std::shared_ptr<Camera>& camera,
    std::unique_ptr<Ray>& ray,
    float& min_distance,
    glm::vec3& hit_point,
    glm::vec3& reflection_normal)
{
    auto geometry = mesh->_geometry;
    SphereGeometry* sphere = (SphereGeometry*)geometry.get();
    glm::vec4 homogeneous_position = camera->_view_matrix * mesh->_model_matrix * sphere->_center;
    glm::vec3 position = glm::vec3(homogeneous_position.x, homogeneous_position.y, homogeneous_position.z);
    float t = hit_sphere(position, sphere->_radius, ray);
    if (t <= 0.001f) {
        return false;
    }
    if (min_distance <= t) {
        return false;
    }
    min_distance = t;
    hit_point = ray->point(t);
    reflection_normal = glm::normalize(hit_point - position);
    return true;
}

bool hit_test(std::shared_ptr<Scene>& scene,
    std::shared_ptr<Camera>& camera,
    std::unique_ptr<Ray>& ray,
    glm::vec3& new_origin,
    glm::vec3& reflection_normal,
    std::shared_ptr<Mesh>& hit_mesh)
{
    bool did_hit = false;
    glm::vec3 hit_point = glm::vec3(0.0f);
    float min_distance = FLT_MAX;
    for (auto mesh : scene->_mesh_array) {
        auto geometry = mesh->_geometry;

        if (geometry->type() == GeometryTypeSphere) {
            if (hit_test_sphere(mesh, camera, ray, min_distance, hit_point, reflection_normal)) {
                new_origin = hit_point;
                did_hit = true;
                hit_mesh = mesh;
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

glm::vec3 RayTracingCPURenderer::compute_color(std::shared_ptr<Scene>& scene,
    std::shared_ptr<Camera>& camera,
    std::unique_ptr<Ray>& ray,
    int current_reflection,
    int max_reflextions)
{
    if (current_reflection == max_reflextions) {
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }
    glm::vec3 new_origin = glm::vec3(0.0f);
    glm::vec3 reflection_normal = glm::vec3(0.0f);
    std::shared_ptr<Mesh> hit_mesh;
    if (hit_test(scene, camera, ray, new_origin, reflection_normal, hit_mesh)) {
        ray->_origin = new_origin;

        glm::vec3& direction = ray->_direction;
        auto material = hit_mesh->_material;

        // diffuse
        glm::vec3 diffuse_vec = glm::vec3(_normal_distribution(_normal_engine), _normal_distribution(_normal_engine), _normal_distribution(_normal_engine));
        glm::vec3 unit_diffuse_vec = diffuse_vec / glm::length(diffuse_vec);
        float dot = glm::dot(reflection_normal, unit_diffuse_vec);
        if (dot < 0.0f) {
            unit_diffuse_vec *= -1.0f;
        }

        // specular
        glm::vec3 unit_specular_vec = direction - 2.0f * glm::dot(direction, reflection_normal) * reflection_normal;

        ray->_direction = material->reflect_ray(unit_diffuse_vec, unit_specular_vec);

        glm::vec3 input_color = compute_color(scene, camera, ray, current_reflection + 1, max_reflextions);
        return material->reflect_color(input_color);
    }
    return glm::vec3(1.0f);
}

void RayTracingCPURenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<float, py::array::c_style> buffer)
{
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

    int ns = options->num_rays_per_pixel();

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 pixel_color = glm::vec3(0, 0, 0);

            for (int m = 0; m < ns; m++) {
                float ray_target_x = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                float ray_target_y = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
                glm::vec3 direction = glm::normalize(glm::vec3(ray_target_x, ray_target_y, -1.0f));
                std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);

                glm::vec3 color = compute_color(scene, camera, ray, 0, options->path_depth());
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
}