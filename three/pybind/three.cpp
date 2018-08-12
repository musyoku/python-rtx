#include "../core/classes/geometry.h"
#include "../core/classes/material.h"
#include "../core/classes/mesh.h"
#include "../core/classes/renderer.h"
#include "../core/classes/scene.h"
#include "../core/geometries/sphere.h"
#include "../core/materials/mesh/standard.h"
#include "../core/renderers/cpu/ray_tracing.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(three, module)
{
    py::class_<Geometry, std::shared_ptr<Geometry>>(module, "Geometry");
    py::class_<Material, std::shared_ptr<Material>>(module, "Material");
    py::class_<Renderer, std::shared_ptr<Renderer>>(module, "Renderer");

    py::class_<Mesh, std::shared_ptr<Mesh>>(module, "Mesh")
        .def(py::init<std::shared_ptr<Geometry>, std::shared_ptr<Material>>(), py::arg("geometry"), py::arg("material"));

    py::class_<Scene, std::shared_ptr<Scene>>(module, "Scene")
        .def(py::init<>())
        .def("add", (void (Scene::*)(std::shared_ptr<Mesh>)) & Scene::add);

    py::class_<SphereGeometry, Geometry, std::shared_ptr<SphereGeometry>>(module, "SphereGeometry")
        .def(py::init<float>(), py::arg("radius"));

    py::class_<MeshStandardMaterial, Material, std::shared_ptr<MeshStandardMaterial>>(module, "MeshStandardMaterial")
        .def(py::init<>());

    py::class_<RayTracingCPURenderer, Renderer, std::shared_ptr<RayTracingCPURenderer>>(module, "RayTracingCPURenderer")
        .def(py::init<>());
}