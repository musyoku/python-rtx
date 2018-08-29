#pragma once
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
    rtx::array<float> _bvh_hit_path;
    rtx::array<float> _bvh_miss_path;
    rtx::array<bool> _bvh_is_leaf;
    rtx::array<float> _bvh_geometry_type;
    rtx::array<float> _bvh_face_start_index;
    rtx::array<float> _bvh_face_end_index;
    // device
    float* _gpu_rays;
    int* _gpu_faces;
    float* _gpu_vertices;
    float* _gpu_object_colors;
    int* _gpu_geometry_types;
    int* _gpu_material_types;
    float* _gpu_color_buffer;
    float* _gpu_bvh_hit_path;
    float* _gpu_bvh_miss_path;
    bool* _gpu_bvh_is_leaf;
    float* _gpu_bvh_object_index;
    float* _gpu_bvh_face_start_index;
    float* _gpu_bvh_face_end_index;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingOptions> _options;

    void construct_bvh();
    void serialize_objects();
    void serialize_mesh_buffer();

    void render_objects();

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