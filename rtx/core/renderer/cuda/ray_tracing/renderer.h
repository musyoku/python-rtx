#pragma once
#include "../../../bvh/scene.h"
#include "../../../class/ray.h"
#include "../../../class/renderer.h"
#include "../../../header/array.h"
#include "../../../header/glm.h"
#include "../../options/ray_tracing.h"
#include <array>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class RayTracingCUDARenderer : public Renderer {
private:
    // host
    rtx::array<float> _ray_array;
    rtx::array<int> _face_vertex_index_array;
    rtx::array<float> _vertex_array;
    rtx::array<int> _object_face_offset_array;
    rtx::array<int> _object_face_count_array;
    rtx::array<int> _object_vertex_offset_array;
    rtx::array<int> _object_vertex_count_array;
    rtx::array<int> _object_geometry_type_array;
    rtx::array<unsigned int> _scene_threaded_bvh_node_array;
    rtx::array<float> _scene_threaded_bvh_aabb_array;
    rtx::array<float> _render_array;
    // device
    float* _gpu_ray_array;
    int* _gpu_face_vertex_index_array;
    float* _gpu_vertex_array;
    int* _gpu_object_face_offset_array;
    int* _gpu_object_face_count_array;
    int* _gpu_object_vertex_offset_array;
    int* _gpu_object_vertex_count_array;
    int* _gpu_object_geometry_type_array;
    unsigned int* _gpu_scene_threaded_bvh_node_array;
    float* _gpu_scene_threaded_bvh_aabb_array;
    float* _gpu_render_array;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingOptions> _options;
    std::unique_ptr<bvh::scene::SceneBVH> _bvh;
    std::vector<std::shared_ptr<Geometry>> _transformed_geometry_array;

    int _prev_height;
    int _prev_width;

    void construct_bvh();
    void transform_geometries_to_view_space();
    void serialize_geometries();
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
        int channels);
};
}