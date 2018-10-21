#pragma once
#include "../class/camera.h"
#include "../class/scene.h"
#include "../header/array.h"
#include "../header/glm.h"
#include "../header/struct.h"
#include "../mapping/texture.h"
#include "arguments/cuda_kernel.h"
#include "arguments/ray_tracing.h"
#include "bvh/bvh.h"
#include <array>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class Renderer {
private:
    // Host
    rtx::array<rtxFaceVertexIndex> _cpu_face_vertex_indices_array;
    rtx::array<rtxVertex> _cpu_vertex_array;
    rtx::array<rtxObject> _cpu_object_array;
    rtx::array<rtxMaterialAttributeByte> _cpu_material_attribute_byte_array;
    rtx::array<rtxThreadedBVH> _cpu_threaded_bvh_array;
    rtx::array<rtxThreadedBVHNode> _cpu_threaded_bvh_node_array;
    rtx::array<rtxRGBAPixel> _cpu_render_array;
    rtx::array<rtxRGBAPixel> _cpu_render_buffer_array;
    rtx::array<int> _cpu_light_sampling_table;
    rtx::array<rtxRGBAColor> _cpu_color_mapping_array;
    rtx::array<rtxUVCoordinate> _cpu_serialized_uv_coordinate_array;

    // Device
    rtxFaceVertexIndex* _gpu_face_vertex_indices_array;
    rtxVertex* _gpu_vertex_array;
    rtxObject* _gpu_object_array;
    rtxMaterialAttributeByte* _gpu_material_attribute_byte_array;
    rtxThreadedBVH* _gpu_threaded_bvh_array;
    rtxThreadedBVHNode* _gpu_threaded_bvh_node_array;
    rtxRGBAPixel* _gpu_render_array;
    int* _gpu_light_sampling_table;
    rtxRGBAColor* _gpu_color_mapping_array;
    rtxUVCoordinate* _gpu_serialized_uv_coordinate_array;

    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracingArguments> _rt_args;
    std::shared_ptr<CUDAKernelLaunchArguments> _cuda_args;
    std::vector<std::shared_ptr<Object>> _transformed_object_array;
    std::vector<std::shared_ptr<BVH>> _geometry_bvh_array;
    std::vector<TextureMapping*> _texture_mapping_ptr_array;

    float _total_light_face_area;
    int _screen_height;
    int _screen_width;
    int _total_frames;

    void check_arguments();
    void construct_bvh();
    void transform_objects_to_view_space();
    void transform_objects_to_view_space_parallel();
    void transform_geometries_to_view_space();
    void transform_lights_to_view_space();
    void serialize_geometries();
    void serialize_textures();
    void serialize_materials();
    void serialize_color_mappings();
    void serialize_light_sampling_table();
    void serialize_objects();
    void serialize_rays(int height, int width);
    void compute_face_area_of_lights();
    void render_objects(int height, int width);
    void launch_mcrt_kernel();
    void launch_nee_kernel();

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