#include "../core/classes/geometry.h"
#include "../core/classes/material.h"
#include "../core/classes/mesh.h"
#include "../core/geometries/sphere.h"
#include "../core/materials/mesh/standard.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(three, module)
{
    py::class_<Mesh, std::shared_ptr<Mesh>>(module, "Mesh")
        .def(py::init<Geometry*, Material*>(), py::arg("geometry"), py::arg("material"));

    py::class_<SphereGeometry, std::shared_ptr<SphereGeometry>>(module, "SphereGeometry")
        .def(py::init<float>(), py::arg("radius"));

    py::class_<MeshStandardMaterial, std::shared_ptr<MeshStandardMaterial>>(module, "MeshStandardMaterial")
        .def(py::init<>());
}