#include "renderer.h"
#include "../../../class/enum.h"
#include "../../../camera/perspective.h"
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
    _face_colors = nullptr;
    _object_types = nullptr;
    _material_types = nullptr;
    _rays = nullptr;
    _color_per_ray = nullptr;
    _gpu_rays = nullptr;
    _gpu_face_vertices = nullptr;
    _gpu_face_colors = nullptr;
    _gpu_object_types = nullptr;
    _gpu_material_types = nullptr;
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
    int color_stride = 3;
    if (_initialized == false) {
        std::cout << num_faces << " * " << faces_stride << std::endl;
        _face_vertices = new float[num_faces * faces_stride];
        _face_colors = new float[num_faces * color_stride];
        _object_types = new int[num_faces];
        _material_types = new int[num_faces];
    }
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

    int face_index = 0;
    for (auto& mesh : mesh_array) {
        auto& geometry = mesh->_geometry;
        auto& material = mesh->_material;
        if (geometry->type() == GeometryTypeSphere) {
            SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
            int index = face_index * faces_stride;
            _face_vertices[index + 0] = sphere->_center.x;
            _face_vertices[index + 1] = sphere->_center.y;
            _face_vertices[index + 2] = sphere->_center.z;
            _face_vertices[index + 3] = 1.0f;
            _face_vertices[index + 4] = sphere->_radius;

            index = face_index * color_stride;
            glm::vec3 color = material->color();
            _face_colors[index + 0] = color.r;
            _face_colors[index + 1] = color.g;
            _face_colors[index + 2] = color.b;

            _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_SPHERE;
            _material_types[face_index] = material->type();
            face_index += 1;
        }
        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            for (auto& face : standard_geometry->_face_vertex_indices_array) {
                glm::vec3& va = standard_geometry->_vertex_array[face[0]];
                glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
                glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

                int index = face_index * faces_stride;
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

                index = face_index * color_stride;
                glm::vec3 color = material->color();
                _face_colors[index + 0] = color.r;
                _face_colors[index + 1] = color.g;
                _face_colors[index + 2] = color.b;

                _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_STANDARD;
                _material_types[face_index] = material->type();
                face_index += 1;
            }
        }
    }

    int num_rays_per_pixel = options->num_rays_per_pixel();
    int num_rays = _height * _width * num_rays_per_pixel;
    int rays_stride = 7;

    if (_initialized == false) {
        _rays = new float[num_rays * rays_stride];
    }
    glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
    if (camera->type() == CameraTypePerspective) {
        PerspectiveCamera* perspective = static_cast<PerspectiveCamera*>(camera.get());
        origin.z = 1.0f / tanf(perspective->_fov_rad / 2.0f);
    }

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {

            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * _width * num_rays_per_pixel * rays_stride + x * num_rays_per_pixel * rays_stride + m * rays_stride;

                // direction
                _rays[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                _rays[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
                _rays[index + 2] = -origin.z;

                // origin
                _rays[index + 3] = origin.x;
                _rays[index + 4] = origin.y;
                _rays[index + 5] = origin.z;

                // length
                _rays[index + 6] = 1.0f;
            }
        }
    }

    int num_pixels = _height * _width;
    if (_initialized == false) {
        _color_per_ray = new float[num_pixels * 3 * num_rays_per_pixel];
    }

    if (_initialized == false) {
        rtx_cuda_alloc(
            _gpu_rays,
            _gpu_face_vertices,
            _gpu_face_colors,
            _gpu_object_types,
            _gpu_material_types,
            _gpu_color_per_ray,
            _rays,
            _face_vertices,
            _face_colors,
            _object_types,
            _material_types,
            num_rays,
            rays_stride,
            num_faces,
            faces_stride,
            color_stride,
            num_pixels,
            num_rays_per_pixel);
    }

    rtx_cuda_copy(_gpu_rays, _gpu_face_vertices, _rays, _face_vertices, num_rays, rays_stride, num_faces, faces_stride);

    rtx_cuda_ray_tracing_render(
        _gpu_rays,
        _gpu_face_vertices,
        _gpu_face_colors,
        _gpu_object_types,
        _gpu_material_types,
        _gpu_color_per_ray,
        _color_per_ray,
        num_rays,
        num_faces,
        faces_stride,
        color_stride,
        options->path_depth(),
        num_pixels,
        num_rays_per_pixel);

    // _initialized = true;

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

    _initialized = true;
    // rtx_cuda_delete(_gpu_rays,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _face_vertices;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _rays;
    // delete[] _color_per_ray;
}
void RayTracingCUDARenderer::render(
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
    int color_stride = 3;
    if (_initialized == false) {
        _face_vertices = new float[num_faces * faces_stride];
        _face_colors = new float[num_faces * color_stride];
        _object_types = new int[num_faces];
        _material_types = new int[num_faces];

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

        int face_index = 0;
        for (auto& mesh : mesh_array) {
            auto& geometry = mesh->_geometry;
            auto& material = mesh->_material;
            if (geometry->type() == GeometryTypeSphere) {
                SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
                int index = face_index * faces_stride;
                _face_vertices[index + 0] = sphere->_center.x;
                _face_vertices[index + 1] = sphere->_center.y;
                _face_vertices[index + 2] = sphere->_center.z;
                _face_vertices[index + 3] = 1.0f;
                _face_vertices[index + 4] = sphere->_radius;

                index = face_index * color_stride;
                glm::vec3 color = material->color();
                _face_colors[index + 0] = color.r;
                _face_colors[index + 1] = color.g;
                _face_colors[index + 2] = color.b;

                _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_SPHERE;
                _material_types[face_index] = material->type();
                face_index += 1;
            }
            if (geometry->type() == GeometryTypeStandard) {
                StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
                for (auto& face : standard_geometry->_face_vertex_indices_array) {
                    glm::vec3& va = standard_geometry->_vertex_array[face[0]];
                    glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
                    glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

                    int index = face_index * faces_stride;
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

                    index = face_index * color_stride;
                    glm::vec3 color = material->color();
                    _face_colors[index + 0] = color.r;
                    _face_colors[index + 1] = color.g;
                    _face_colors[index + 2] = color.b;

                    _object_types[face_index] = RTX_CUDA_GEOMETRY_TYPE_STANDARD;
                    _material_types[face_index] = material->type();
                    face_index += 1;
                }
            }
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
        rtx_cuda_alloc(
            _gpu_rays,
            _gpu_face_vertices,
            _gpu_face_colors,
            _gpu_object_types,
            _gpu_material_types,
            _gpu_color_per_ray,
            _rays,
            _face_vertices,
            _face_colors,
            _object_types,
            _material_types,
            num_rays,
            rays_stride,
            num_faces,
            faces_stride,
            color_stride,
            num_pixels,
            num_rays_per_pixel);
    }

    rtx_cuda_copy(_gpu_rays, _gpu_face_vertices, _rays, _face_vertices, num_rays, rays_stride, num_faces, faces_stride);

    rtx_cuda_ray_tracing_render(
        _gpu_rays,
        _gpu_face_vertices,
        _gpu_face_colors,
        _gpu_object_types,
        _gpu_material_types,
        _gpu_color_per_ray,
        _color_per_ray,
        num_rays,
        num_faces,
        faces_stride,
        color_stride,
        options->path_depth(),
        num_pixels,
        num_rays_per_pixel);

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
            int index = y * width * channels + x * channels;
            buffer[index + 0] = std::min(std::max((int)(sum_r / float(num_rays_per_pixel) * 255.0f), 0), 255);
            buffer[index + 1] = std::min(std::max((int)(sum_g / float(num_rays_per_pixel) * 255.0f), 0), 255);
            buffer[index + 2] = std::min(std::max((int)(sum_b / float(num_rays_per_pixel) * 255.0f), 0), 255);
        }
    }

    _initialized = true;
    // rtx_cuda_delete(_gpu_rays,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _face_vertices;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _rays;
    // delete[] _color_per_ray;
}
}