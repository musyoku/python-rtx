#pragma once
#include "../../../class/ray.h"
#include "../../../class/renderer.h"
#include "../../../header/array.h"
#include "../../../header/glm.h"
#include "../../../header/struct.h"
#include "../../options/ray_tracing.h"
#include "../bvh/bvh.h"
#include "../header/ray_tracing.h"
#include <array>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class RayTracingCUDARenderer : public Renderer {
private:
    // host
    rtx::array<RTXRay> _cpu_ray_array;
    rtx::array<RTXGeometryFace> _cpu_face_vertex_indices_array;
    rtx::array<RTXGeometryVertex> _cpu_vertex_array;
    rtx::array<RTXObject> _cpu_object_array;
    rtx::array<RTXThreadedBVH> _cpu_threaded_bvh_array;
    rtx::array<RTXThreadedBVHNode> _cpu_threaded_bvh_node_array;
    rtx::array<RTXPixel> _cpu_render_array;

    // device
    RTXRay* _gpu_ray_array;
    RTXGeometryFace* _gpu_face_vertex_indices_array;
    RTXGeometryVertex* _gpu_vertex_array;
    RTXObject* _gpu_object_array;
    RTXThreadedBVH* _gpu_threaded_bvh_array;
    RTXThreadedBVHNode* _gpu_threaded_bvh_node_array;
    RTXPixel* _gpu_render_array;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingOptions> _options;
    std::vector<std::shared_ptr<Geometry>> _transformed_geometry_array;
    std::vector<std::shared_ptr<BVH>> _bvh_array;
    std::unordered_map<int, int> _map_object_bvh;

    int _prev_height;
    int _prev_width;

    void construct_bvh();
    void transform_geometries_to_view_space();
    void serialize_objects();
    void serialize_rays(int height, int width);
    void render_objects(int height, int width);

public:
    RayTracingCUDARenderer();
    ~RayTracingCUDARenderer();
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        pybind11::array_t<float, pybind11::array::c_style> array);
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        unsigned char* array,
        int height,
        int width,
        int channels,
        int num_blocks, 
        int num_threads);
};
}