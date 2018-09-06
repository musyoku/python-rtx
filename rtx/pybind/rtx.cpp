#include "../core/camera/perspective.h"
#include "../core/class/camera.h"
#include "../core/class/geometry.h"
#include "../core/class/material.h"
#include "../core/class/mesh.h"
#include "../core/class/scene.h"
#include "../core/geometry/box.h"
#include "../core/geometry/plain.h"
#include "../core/geometry/sphere.h"
#include "../core/geometry/standard.h"
#include "../core/material/mesh/emissive.h"
#include "../core/material/mesh/lambert.h"
#include "../core/material/mesh/metal.h"
#include "../core/renderer/arguments/cuda_kernel.h"
#include "../core/renderer/arguments/ray_tracing.h"
#include "../core/renderer/renderer.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace rtx;

PYBIND11_MODULE(rtx, module)
{
    py::class_<Geometry, std::shared_ptr<Geometry>>(module, "Geometry");
    py::class_<Material, std::shared_ptr<Material>>(module, "Material");
    py::class_<Camera, std::shared_ptr<Camera>>(module, "Camera");

    py::class_<Mesh, std::shared_ptr<Mesh>>(module, "Mesh")
        .def(py::init<std::shared_ptr<Geometry>, std::shared_ptr<Material>>(), py::arg("geometry"), py::arg("material"))
        .def("set_scale", (void (Mesh::*)(py::tuple)) & Mesh::set_scale)
        .def("set_position", (void (Mesh::*)(py::tuple)) & Mesh::set_position)
        .def("set_rotation", (void (Mesh::*)(py::tuple)) & Mesh::set_rotation);

    py::class_<Scene, std::shared_ptr<Scene>>(module, "Scene")
        .def(py::init<>())
        .def("add", (void (Scene::*)(std::shared_ptr<Mesh>)) & Scene::add);

    py::class_<SphereGeometry, Geometry, std::shared_ptr<SphereGeometry>>(module, "SphereGeometry")
        .def(py::init<float>(), py::arg("radius"));
    py::class_<StandardGeometry, Geometry, std::shared_ptr<StandardGeometry>>(module, "StandardGeometry")
        .def(py::init<py::array_t<int, py::array::c_style>, py::array_t<float, py::array::c_style>>(), py::arg("face_vertex_indeces"), py::arg("vertices"))
        .def(py::init<py::array_t<int, py::array::c_style>, py::array_t<float, py::array::c_style>, int>(), py::arg("face_vertex_indeces"), py::arg("vertices"), py::arg("bvh_max_triangles_per_node"));
    py::class_<PlainGeometry, Geometry, std::shared_ptr<PlainGeometry>>(module, "PlainGeometry")
        .def(py::init<float, float>(), py::arg("width"), py::arg("height"));
    py::class_<BoxGeometry, Geometry, std::shared_ptr<BoxGeometry>>(module, "BoxGeometry")
        .def(py::init<float, float, float>(), py::arg("width"), py::arg("height"), py::arg("depth"));

    py::class_<MeshLambertMaterial, Material, std::shared_ptr<MeshLambertMaterial>>(module, "MeshLambertMaterial")
        .def(py::init<py::tuple, float>(), py::arg("color"), py::arg("diffuse_reflectance"));
    py::class_<MeshMetalMaterial, Material, std::shared_ptr<MeshMetalMaterial>>(module, "MeshMetalMaterial")
        .def(py::init<float, float>(), py::arg("roughness"), py::arg("specular_reflectance"));
    py::class_<MeshEmissiveMaterial, Material, std::shared_ptr<MeshEmissiveMaterial>>(module, "MeshEmissiveMaterial")
        .def(py::init<py::tuple>(), py::arg("color"));

    py::class_<Renderer, std::shared_ptr<Renderer>>(module, "Renderer")
        .def(py::init<>())
        .def("render", (void (Renderer::*)(std::shared_ptr<Scene>, std::shared_ptr<Camera>, std::shared_ptr<RayTracingArguments>, std::shared_ptr<CUDAKernelLaunchArguments>, py::array_t<float, py::array::c_style>)) & Renderer::render, py::arg("scene"), py::arg("camera"), py::arg("rt_args"), py::arg("cuda_args"), py::arg("render_buffer"));

    py::class_<RayTracingArguments, std::shared_ptr<RayTracingArguments>>(module, "RayTracingArguments")
        .def(py::init<>())
        .def_property("num_rays_per_pixel", &RayTracingArguments::num_rays_per_pixel, &RayTracingArguments::set_num_rays_per_pixel)
        .def_property("max_bounce", &RayTracingArguments::max_bounce, &RayTracingArguments::set_max_bounce);
    py::class_<CUDAKernelLaunchArguments, std::shared_ptr<CUDAKernelLaunchArguments>>(module, "CUDAKernelLaunchArguments")
        .def(py::init<>())
        .def_property("num_threads", &CUDAKernelLaunchArguments::num_threads, &CUDAKernelLaunchArguments::set_num_threads)
        .def_property("num_blocks", &CUDAKernelLaunchArguments::num_blocks, &CUDAKernelLaunchArguments::set_num_blocks);

    py::class_<PerspectiveCamera, Camera, std::shared_ptr<PerspectiveCamera>>(module, "PerspectiveCamera")
        .def(py::init<py::tuple, py::tuple, py::tuple, float, float, float, float>(),
            py::arg("eye"), py::arg("center"), py::arg("up"), py::arg("fov_rad"), py::arg("aspect_ratio"), py::arg("z_near"), py::arg("z_far"))
        .def_property("fov_rad", &PerspectiveCamera::fov_rad, &PerspectiveCamera::set_fov_rad)
        .def("look_at", (void (PerspectiveCamera::*)(py::tuple, py::tuple, py::tuple)) & PerspectiveCamera::look_at, py::arg("eye"), py::arg("center"), py::arg("up"));
}