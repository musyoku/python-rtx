#pragma once
#include "../../../class/ray.h"
#include "../../../class/bvh.h"
#include "../../../class/renderer.h"
#include "../../../header/array.h"
#include "../../../header/glm.h"
#include "../../options/ray_tracing.h"
#include <array>
#include <memory>
#include <map>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class RayTracingCUDARenderer : public Renderer {
private:
    // host

    // [ray_index * 4 + 0] -> ray.x
    // [ray_index * 4 + 1] -> ray.y
    // [ray_index * 4 + 2] -> ray.z
    rtx::array<float> _ray_array;

    // [face_index * 4 + 0] -> vertex_index_a
    // [face_index * 4 + 1] -> vertex_index_b
    // [face_index * 4 + 2] -> vertex_index_c
    rtx::array<int> _face_vertex_indices_array;

    // [vertex_index * 4 + 0] -> vertex.x
    // [vertex_index * 4 + 1] -> vertex.y
    // [vertex_index * 4 + 2] -> vertex.z
    rtx::array<float> _vertex_array;

    // [object_index] -> offset in _face_vertex_indices_array
    rtx::array<int> _object_face_offset_array;

    // [object_index] -> #faces of the object
    rtx::array<int> _object_face_count_array;

    // [object_index] -> offset in _vertex_array
    rtx::array<int> _object_vertex_offset_array;

    // [object_index] -> #vertices of the object
    rtx::array<int> _object_vertex_count_array;

    // [object_index] -> attribute
    // an integer value obtained by concatenating 8 attribute values represented by 4 bit integer
    rtx::array<int> _object_geometry_attributes_array;

    // [bvh_index] -> node connections
    // an integer value obtained by concatenating 4 node indices represented by 8 bit integer
    rtx::array<int> _threaded_bvh_node_array;

    // [bvh_index] -> #nodes of the BVH tree.
    rtx::array<int> _threaded_bvh_num_nodes_array;

    // [bvh_index] -> offset in _threaded_bvh_node_array
    rtx::array<int> _threaded_bvh_index_offset_array;

    // [bvh_iondex * 8 + 0] -> aabb_max.x 
    // [bvh_iondex * 8 + 1] -> aabb_max.y 
    // [bvh_iondex * 8 + 2] -> aabb_max.z 
    // [bvh_iondex * 8 + 3] -> -1
    // [bvh_iondex * 8 + 4] -> aabb_min.x 
    // [bvh_iondex * 8 + 5] -> aabb_min.y 
    // [bvh_iondex * 8 + 6] -> aabb_min.z 
    // [bvh_iondex * 8 + 7] -> -1 
    rtx::array<float> _threaded_bvh_aabb_array;

    // [ray_index * 4 + 0] -> pixel.r
    // [ray_index * 4 + 1] -> pixel.g
    // [ray_index * 4 + 2] -> pixel.b
    rtx::array<float> _render_array;

    // device
    float* _gpu_ray_array;
    int* _gpu_face_vertex_indices_array;
    float* _gpu_vertex_array;
    int* _gpu_object_face_offset_array;
    int* _gpu_object_face_count_array;
    int* _gpu_object_vertex_offset_array;
    int* _gpu_object_vertex_count_array;
    int* _gpu_object_geometry_attributes_array;
    int* _gpu_threaded_bvh_node_array;
    float* _gpu_threaded_bvh_aabb_array;
    float* _gpu_render_array;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingOptions> _options;

    // [object_index] -> geometry
    std::vector<std::shared_ptr<Geometry>> _transformed_geometry_array;

    // [object_index] -> bvh
    std::vector<std::shared_ptr<BVH>> _bvh_array;

    // (object_index, bvh_index)
    std::unordered_map<int, int> _map_object_bvh;

    int _prev_height;
    int _prev_width;

    void construct_bvh();
    void transform_geometries_to_view_space();
    void serialize_geometries();
    void serialize_geometry_attributes();
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