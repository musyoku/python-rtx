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

    int total_faces = 0;

    for (auto& mesh : scene->_mesh_array) {
        auto& geometry = mesh->_geometry;
        if (geometry->type() == GeometryTypeSphere) {
            total_faces += 1;
            continue;
        }
        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            total_faces += standard_geometry->_face_vertex_indices_array.size();
            continue;
        }
    }
    std::cout << total_faces << std::endl;
    int stride = 4 * 3;
    float* face_vertices = new float[total_faces * stride];

    int face_index = 0;
    for (auto& mesh : scene->_mesh_array) {
        int index = face_index * stride;
        auto& geometry = mesh->_geometry;
        if (geometry->type() == GeometryTypeSphere) {
            SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
            face_vertices[index + 0] = sphere->_center.x;
            face_vertices[index + 1] = sphere->_center.y;
            face_vertices[index + 2] = sphere->_center.z;
            face_vertices[index + 3] = 1.0f;
            face_vertices[index + 4] = sphere->_radius;
        }
        if (geometry->type() == GeometryTypeStandard) {
            StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
            for (auto& face : standard_geometry->_face_vertex_indices_array) {
                glm::vec3& va = standard_geometry->_vertex_array[face[0]];
                glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
                glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

                face_vertices[index + 0] = va.x;
                face_vertices[index + 1] = va.y;
                face_vertices[index + 2] = va.z;
                face_vertices[index + 3] = 1.0f;

                face_vertices[index + 4] = vb.x;
                face_vertices[index + 5] = vb.y;
                face_vertices[index + 6] = vb.z;
                face_vertices[index + 7] = 1.0f;

                face_vertices[index + 8] = vc.x;
                face_vertices[index + 9] = vc.y;
                face_vertices[index + 10] = vc.z;
                face_vertices[index + 11] = 1.0f;
            }
        }
        face_index += 1;
    }
    rtx_cuda_ray_tracing_render(face_vertices, total_faces, stride);
    delete[] face_vertices;
}
}