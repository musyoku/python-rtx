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
    rtx::array<float> _rays;
    rtx::array<int> _faces;
    rtx::array<float> _vertices;
    rtx::array<int> _face_serialization_offsets;
    rtx::array<int> _vertex_serialization_offsets;
    rtx::array<float> _object_colors;
    rtx::array<int> _geometry_types;
    rtx::array<int> _material_types;
    rtx::array<float> _color_buffer;
    rtx::array<unsigned int> _scene_bvh_nodes;
    // device
    float* _gpu_rays;
    int* _gpu_faces;
    float* _gpu_vertices;
    float* _gpu_object_colors;
    int* _gpu_geometry_types;
    int* _gpu_material_types;
    float* _gpu_color_buffer;
    unsigned int* _gpu_scene_bvh_nodes;

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
        pybind11::array_t<float, pybind11::array::c_style> buffer);
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        unsigned char* buffer,
        int height,
        int width,
        int channels);
};
}