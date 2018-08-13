#include "../core/camera/perspective.h"
#include "../core/class/camera.h"
#include "../core/class/geometry.h"
#include "../core/class/material.h"
#include "../core/class/mesh.h"
#include "../core/class/renderer.h"
#include "../core/class/scene.h"
#include "../core/geometry/sphere.h"
#include "../core/material/mesh/standard.h"
#include "../core/renderer/cpu/ray_tracing/renderer.h"
#include "../core/renderer/options/ray_tracing.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(three, module)
{
    py::class_<Geometry, std::shared_ptr<Geometry>>(module, "Geometry");
    py::class_<Material, std::shared_ptr<Material>>(module, "Material");
    py::class_<Renderer, std::shared_ptr<Renderer>>(module, "Renderer");
    py::class_<Camera, std::shared_ptr<Camera>>(module, "Camera");

    py::class_<Mesh, std::shared_ptr<Mesh>>(module, "Mesh")
        .def(py::init<std::shared_ptr<Geometry>, std::shared_ptr<Material>>(), py::arg("geometry"), py::arg("material"))
        .def("set_scale", &Mesh::set_scale)
        .def("set_position", &Mesh::set_position)
        .def("set_rotation", &Mesh::set_rotation);

    py::class_<Scene, std::shared_ptr<Scene>>(module, "Scene")
        .def(py::init<>())
        .def("add", (void (Scene::*)(std::shared_ptr<Mesh>)) & Scene::add);

    py::class_<SphereGeometry, Geometry, std::shared_ptr<SphereGeometry>>(module, "SphereGeometry")
        .def(py::init<float>(), py::arg("radius"));

    py::class_<MeshStandardMaterial, Material, std::shared_ptr<MeshStandardMaterial>>(module, "MeshStandardMaterial")
        .def(py::init<>());

    py::class_<RayTracingCPURenderer, Renderer, std::shared_ptr<RayTracingCPURenderer>>(module, "RayTracingCPURenderer")
        .def(py::init<>())
        .def("render", (void (RayTracingCPURenderer::*)(std::shared_ptr<Scene>, std::shared_ptr<Camera>, std::shared_ptr<RayTracingOptions>, py::array_t<int, py::array::c_style>)) & RayTracingCPURenderer::render);

    py::class_<RayTracingOptions, std::shared_ptr<RayTracingOptions>>(module, "RayTracingOptions")
        .def(py::init<>())
        .def_property("num_rays_per_pixel", &RayTracingOptions::get_num_rays_per_pixel, &RayTracingOptions::set_num_rays_per_pixel);

    py::class_<PerspectiveCamera, Camera, std::shared_ptr<PerspectiveCamera>>(module, "PerspectiveCamera")
        .def(py::init<py::tuple, py::tuple, py::tuple, float, float, float, float>(),
            py::arg("eye"), py::arg("center"), py::arg("up"), py::arg("fov_rad"), py::arg("aspect_ratio"), py::arg("z_near"), py::arg("z_far"))
        .def("look_at", &PerspectiveCamera::look_at, py::arg("eye"), py::arg("center"), py::arg("up"));
}