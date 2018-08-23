#include "renderer.h"
#include "../../../geometry/sphere.h"
#include "../../../geometry/standard.h"
#include "../header/ray_tracing.h"
#include <iostream>
#include <memory>
#include <vector>

namespace rtx {

namespace py = pybind11;

RayTracingCUDARenderer::RayTracingCUDARenderer()
{
    _face_vertices = nullptr;
    _object_types = nullptr;
    _rays = nullptr;
    _color_per_ray = nullptr;
    _gpu_face_vertices = nullptr;
    _gpu_color_per_ray = nullptr;
    _initialized = false;
}
void RayTracingCUDARenderer::render(
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

    int num_faces = 0;

    for (auto& mesh : scene->_mesh_array) {
        auto& geometry = mesh->_geometry;
        if (geometry->type() == GeometryTypeSphere) {
            num_faces += 1;
            continue;
        }
        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            num_faces += standard_geometry->_face_vertex_indices_array.size();
            continue;
        }
    }

    int faces_stride = 4 * 3;
    if (_initialized == false) {
        _face_vertices = new float[num_faces * faces_stride];
        _object_types = new int[num_faces];

        int face_index = 0;
        for (auto& mesh : scene->_mesh_array) {
            int index = face_index * faces_stride;
            auto& geometry = mesh->_geometry;
            if (geometry->type() == GeometryTypeSphere) {
                SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
                _face_vertices[index + 0] = sphere->_center.x;
                _face_vertices[index + 1] = sphere->_center.y;
                _face_vertices[index + 2] = sphere->_center.z;
                _face_vertices[index + 3] = 1.0f;
                _face_vertices[index + 4] = sphere->_radius;

                _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_SPHERE;
            }
            if (geometry->type() == GeometryTypeStandard) {
                StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
                for (auto& face : standard_geometry->_face_vertex_indices_array) {
                    glm::vec3& va = standard_geometry->_vertex_array[face[0]];
                    glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
                    glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

                    _face_vertices[index + 0] = va.x;
                    _face_vertices[index + 1] = va.y;
                    _face_vertices[index + 2] = va.z;
                    _face_vertices[index + 3] = 1.0f;

                    _face_vertices[index + 4] = vb.x;
                    _face_vertices[index + 5] = vb.y;
                    _face_vertices[index + 6] = vb.z;
                    _face_vertices[index + 7] = 1.0f;

                    _face_vertices[index + 8] = vc.x;
                    _face_vertices[index + 9] = vc.y;
                    _face_vertices[index + 10] = vc.z;
                    _face_vertices[index + 11] = 1.0f;

                    _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_STANDARD;
                }
            }
            face_index += 1;
        }
    }

    int num_rays_per_pixel = options->num_rays_per_pixel();
    int num_rays = _height * _width * num_rays_per_pixel;
    int rays_stride = 7;

    if (_initialized == false) {
        _rays = new float[num_rays * rays_stride];

        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width; x++) {
                glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);

                for (int m = 0; m < num_rays_per_pixel; m++) {
                    int index = y * _width * num_rays_per_pixel * rays_stride + x * num_rays_per_pixel * rays_stride + m * rays_stride;

                    // direction
                    _rays[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                    _rays[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
                    _rays[index + 2] = -1.0f;

                    // origin
                    _rays[index + 3] = 0.0f;
                    _rays[index + 4] = 0.0f;
                    _rays[index + 5] = 1.0f;

                    // length
                    _rays[index + 6] = 1.0f;
                }
            }
        }
    }
    int num_pixels = _height * _width;
    if (_initialized == false) {
        _color_per_ray = new float[num_pixels * 3 * num_rays_per_pixel];
    }

    if (_initialized == false) {
        rtx_cuda_alloc(_gpu_face_vertices,
            _face_vertices,
            num_faces,
            faces_stride,
            _gpu_color_per_ray,
            num_pixels,
            num_rays_per_pixel);
    }

    rtx_cuda_ray_tracing_render(
        _rays,
        num_rays,
        _face_vertices,
        _object_types,
        num_faces,
        faces_stride,
        options->path_depth(),
        _color_per_ray,
        num_pixels,
        num_rays_per_pixel,
        _gpu_face_vertices,
        _gpu_color_per_ray);

    _initialized = true;

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * _width * num_rays_per_pixel * 3 + x * num_rays_per_pixel * 3 + m * 3;
                sum_r += _color_per_ray[index + 0];
                sum_g += _color_per_ray[index + 1];
                sum_b += _color_per_ray[index + 2];
            }
            pixel(y, x, 0) = sum_r / float(num_rays_per_pixel);
            pixel(y, x, 1) = sum_g / float(num_rays_per_pixel);
            pixel(y, x, 2) = sum_b / float(num_rays_per_pixel);
        }
    }

    // rtx_cuda_delete(_gpu_face_vertices, _gpu_color_per_ray);
    // delete[] _face_vertices;
    // delete[] _object_types;
    // delete[] _rays;
    // delete[] _color_per_ray;
}
}