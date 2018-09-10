#pragma once
#include "../class/camera.h"
#include "../class/ray.h"
#include "../class/scene.h"
#include "../header/array.h"
#include "../header/glm.h"
#include "../header/struct.h"
#include "arguments/cuda_kernel.h"
#include "arguments/ray_tracing.h"
#include "bvh/bvh.h"
#include "header/ray_tracing.h"
#include <array>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class Renderer {
private:
    // host
    rtx::array<RTXRay> _cpu_ray_array;
    rtx::array<RTXFace> _cpu_face_vertex_indices_array;
    rtx::array<RTXVertex> _cpu_vertex_array;
    rtx::array<RTXObject> _cpu_object_array;
    rtx::array<RTXThreadedBVH> _cpu_threaded_bvh_array;
    rtx::array<RTXThreadedBVHNode> _cpu_threaded_bvh_node_array;
    rtx::array<RTXPixel> _cpu_render_array;
    std::vector<int> _cpu_light_index_array;

    // device
    RTXRay* _gpu_ray_array;
    RTXFace* _gpu_face_vertex_indices_array;
    RTXVertex* _gpu_vertex_array;
    RTXObject* _gpu_object_array;
    RTXThreadedBVH* _gpu_threaded_bvh_array;
    RTXThreadedBVHNode* _gpu_threaded_bvh_node_array;
    RTXPixel* _gpu_render_array;
    int* _gpu_light_index_array;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingArguments> _rt_args;
    std::shared_ptr<CUDAKernelLaunchArguments> _cuda_args;
    std::vector<std::shared_ptr<Object>> _transformed_geometry_array;
    std::vector<std::shared_ptr<Object>> _transformed_light_array;
    std::vector<std::shared_ptr<BVH>> _geometry_bvh_array;
    std::unordered_map<int, int> _map_object_bvh;

    int _prev_height;
    int _prev_width;

    void construct_bvh();
    void transform_objects_to_view_space();
    void transform_geometries_to_view_space();
    void transform_lights_to_view_space();
    void serialize_objects();
    void serialize_rays(int height, int width);
    void render_objects(int height, int width);

public:
    Renderer();
    ~Renderer();
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingArguments> rt_args,
        std::shared_ptr<CUDAKernelLaunchArguments> cuda_args,
        pybind11::array_t<float, pybind11::array::c_style> array);
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingArguments> rt_args,
        std::shared_ptr<CUDAKernelLaunchArguments> cuda_args,
        unsigned char* array,
        int height,
        int width,
        int channels,
        int num_blocks,
        int num_threads);
};
}