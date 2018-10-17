#include "../core/camera/orthographic.h"
#include "../core/camera/perspective.h"
#include "../core/class/camera.h"
#include "../core/class/geometry.h"
#include "../core/class/mapping.h"
#include "../core/class/material.h"
#include "../core/class/object.h"
#include "../core/class/scene.h"
#include "../core/geometry/box.h"
#include "../core/geometry/plain.h"
#include "../core/geometry/sphere.h"
#include "../core/geometry/standard.h"
#include "../core/mapping/solid_color.h"
#include "../core/mapping/texture.h"
#include "../core/material/emissive.h"
#include "../core/material/lambert.h"
#include "../core/material/oren_nayar.h"
#include "../core/renderer/arguments/cuda_kernel.h"
#include "../core/renderer/arguments/ray_tracing.h"
#include "../core/renderer/renderer.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace rtx;

PYBIND11_MODULE(rtx, module)
{
    // Base classes
    py::class_<Geometry, std::shared_ptr<Geometry>>(module, "Geometry")
        .def("set_scale", (void (Geometry::*)(py::tuple)) & Geometry::set_scale)
        .def("set_position", (void (Geometry::*)(py::tuple)) & Geometry::set_position)
        .def("set_rotation", (void (Geometry::*)(py::tuple)) & Geometry::set_rotation);
    py::class_<Material, std::shared_ptr<Material>>(module, "Material");
    py::class_<Mapping, std::shared_ptr<Mapping>>(module, "Mapping");
    py::class_<Camera, std::shared_ptr<Camera>>(module, "Camera");
    py::class_<Object, std::shared_ptr<Object>>(module, "Object")
        .def(py::init<std::shared_ptr<Geometry>, std::shared_ptr<Material>, std::shared_ptr<Mapping>>(), py::arg("geometry"), py::arg("material"), py::arg("mapping"))
        .def(py::init<std::shared_ptr<Geometry>, std::shared_ptr<LayeredMaterial>, std::shared_ptr<Mapping>>(), py::arg("geometry"), py::arg("material"), py::arg("mapping"));

    // Scene
    py::class_<Scene, std::shared_ptr<Scene>>(module, "Scene")
        .def(py::init<py::tuple>(), py::arg("ambient_color"))
        .def("add", &Scene::add)
        .def("num_triangles", &Scene::num_triangles);

    // Geometries
    py::class_<SphereGeometry, Geometry, std::shared_ptr<SphereGeometry>>(module, "SphereGeometry")
        .def(py::init<float>(), py::arg("radius"));
    py::class_<StandardGeometry, Geometry, std::shared_ptr<StandardGeometry>>(module, "StandardGeometry")
        .def(py::init<py::array_t<int, py::array::c_style>, py::array_t<float, py::array::c_style>>(), py::arg("face_vertex_indeces"), py::arg("vertices"))
        .def(py::init<py::array_t<int, py::array::c_style>, py::array_t<float, py::array::c_style>, int>(), py::arg("face_vertex_indeces"), py::arg("vertices"), py::arg("bvh_max_triangles_per_node"));
    py::class_<PlainGeometry, Geometry, std::shared_ptr<PlainGeometry>>(module, "PlainGeometry")
        .def(py::init<float, float>(), py::arg("width"), py::arg("height"));
    py::class_<BoxGeometry, Geometry, std::shared_ptr<BoxGeometry>>(module, "BoxGeometry")
        .def(py::init<float, float, float>(), py::arg("width"), py::arg("height"), py::arg("depth"));

    // Materials
    py::class_<LambertMaterial, Material, std::shared_ptr<LambertMaterial>>(module, "LambertMaterial")
        .def(py::init<float>(), py::arg("albedo"));
    py::class_<OrenNayarMaterial, Material, std::shared_ptr<OrenNayarMaterial>>(module, "OrenNayarMaterial")
        .def(py::init<float, float>(), py::arg("albedo"), py::arg("roughness"));
    py::class_<EmissiveMaterial, Material, std::shared_ptr<EmissiveMaterial>>(module, "EmissiveMaterial")
        .def(py::init<float>(), py::arg("brightness"))
        .def(py::init<float, float>(), py::arg("brightness"), py::arg("visible"));
    py::class_<LayeredMaterial, std::shared_ptr<LayeredMaterial>>(module, "LayeredMaterial")
        .def(py::init<std::shared_ptr<Material>>())
        .def(py::init<std::shared_ptr<Material>, std::shared_ptr<Material>>())
        .def(py::init<std::shared_ptr<Material>, std::shared_ptr<Material>, std::shared_ptr<Material>>());

    // Mappings
    py::class_<SolidColorMapping, Mapping, std::shared_ptr<SolidColorMapping>>(module, "SolidColorMapping")
        .def(py::init<py::tuple>(), py::arg("color"));
    py::class_<TextureMapping, Mapping, std::shared_ptr<TextureMapping>>(module, "TextureMapping")
        .def(py::init<py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>>(), py::arg("texture"), py::arg("uv_coordinates"));

    // Arguments
    py::class_<RayTracingArguments, std::shared_ptr<RayTracingArguments>>(module, "RayTracingArguments")
        .def(py::init<>())
        .def_property("num_rays_per_pixel", &RayTracingArguments::num_rays_per_pixel, &RayTracingArguments::set_num_rays_per_pixel)
        .def_property("next_event_estimation_enabled", &RayTracingArguments::next_event_estimation_enabled, &RayTracingArguments::set_next_event_estimation_enabled)
        .def_property("max_bounce", &RayTracingArguments::max_bounce, &RayTracingArguments::set_max_bounce);
    py::class_<CUDAKernelLaunchArguments, std::shared_ptr<CUDAKernelLaunchArguments>>(module, "CUDAKernelLaunchArguments")
        .def(py::init<>())
        .def_property("num_threads", &CUDAKernelLaunchArguments::num_threads, &CUDAKernelLaunchArguments::set_num_threads)
        .def_property("num_rays_per_thread", &CUDAKernelLaunchArguments::num_rays_per_thread, &CUDAKernelLaunchArguments::set_num_rays_per_thread);

    // Cameras
    py::class_<PerspectiveCamera, Camera, std::shared_ptr<PerspectiveCamera>>(module, "PerspectiveCamera")
        .def(py::init<py::tuple, py::tuple, py::tuple, float, float, float, float>(),
            py::arg("eye"), py::arg("center"), py::arg("up"), py::arg("fov_rad"), py::arg("aspect_ratio"), py::arg("z_near"), py::arg("z_far"))
        .def_property("fov_rad", &PerspectiveCamera::fov_rad, &PerspectiveCamera::set_fov_rad)
        .def("look_at", (void (PerspectiveCamera::*)(py::tuple, py::tuple, py::tuple)) & PerspectiveCamera::look_at, py::arg("eye"), py::arg("center"), py::arg("up"));
    py::class_<OrthographicCamera, Camera, std::shared_ptr<OrthographicCamera>>(module, "OrthographicCamera")
        .def(py::init<py::tuple, py::tuple, py::tuple>(), py::arg("eye"), py::arg("center"), py::arg("up"))
        .def("look_at", (void (OrthographicCamera::*)(py::tuple, py::tuple, py::tuple)) & OrthographicCamera::look_at, py::arg("eye"), py::arg("center"), py::arg("up"));

    py::class_<Renderer, std::shared_ptr<Renderer>>(module, "Renderer")
        .def(py::init<>())
        .def("render", (void (Renderer::*)(std::shared_ptr<Scene>, std::shared_ptr<Camera>, std::shared_ptr<RayTracingArguments>, std::shared_ptr<CUDAKernelLaunchArguments>, py::array_t<float, py::array::c_style>)) & Renderer::render, py::arg("scene"), py::arg("camera"), py::arg("rt_args"), py::arg("cuda_args"), py::arg("render_buffer"));
}