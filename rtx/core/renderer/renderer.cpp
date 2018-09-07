#include "renderer.h"
#include "../camera/perspective.h"
#include "../geometry/box.h"
#include "../geometry/sphere.h"
#include "../geometry/standard.h"
#include "../header/enum.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

namespace rtx {

namespace py = pybind11;

Renderer::Renderer()
{
    _gpu_ray_array = NULL;
    _gpu_face_vertex_indices_array = NULL;
    _gpu_vertex_array = NULL;
    _gpu_object_array = NULL;
    _gpu_threaded_bvh_array = NULL;
    _gpu_threaded_bvh_node_array = NULL;
    _gpu_render_array = NULL;
}
Renderer::~Renderer()
{
    rtx_cuda_free((void**)&_gpu_ray_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_indices_array);
    rtx_cuda_free((void**)&_gpu_vertex_array);
    rtx_cuda_free((void**)&_gpu_object_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
    rtx_cuda_free((void**)&_gpu_render_array);
}
void Renderer::transform_geometries_to_view_space()
{
    int num_objects = _scene->_mesh_array.size() + _scene->_light_array.size();
    std::vector<std::shared_ptr<Object>> transformed_object_array;
    transformed_object_array.reserve(num_objects);
    for (auto& mesh : _scene->_mesh_array) {
        auto& geometry = mesh->_geometry;
        glm::mat4 transformation_matrix = _camera->_view_matrix * mesh->_model_matrix;
        // std::cout << "transform: " << std::endl;
        // std::cout << transformation_matrix[0][0] << ", " << transformation_matrix[0][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[0][3] << std::endl;
        // std::cout << transformation_matrix[1][0] << ", " << transformation_matrix[1][1] << ", " << transformation_matrix[1][2] << ", " << transformation_matrix[1][3] << std::endl;
        // std::cout << transformation_matrix[2][0] << ", " << transformation_matrix[2][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[2][3] << std::endl;
        // std::cout << transformation_matrix[3][0] << ", " << transformation_matrix[3][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[3][3] << std::endl;
        // Transform vertices from model space to view space
        auto transformed_geometry = geometry->transoform(transformation_matrix);
        transformed_object_array.push_back(transformed_geometry);
    }
    for (auto& light : _scene->_light_array) {
        glm::mat4 transformation_matrix = _camera->_view_matrix * light->_model_matrix;
        // std::cout << "transform: " << std::endl;
        // std::cout << transformation_matrix[0][0] << ", " << transformation_matrix[0][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[0][3] << std::endl;
        // std::cout << transformation_matrix[1][0] << ", " << transformation_matrix[1][1] << ", " << transformation_matrix[1][2] << ", " << transformation_matrix[1][3] << std::endl;
        // std::cout << transformation_matrix[2][0] << ", " << transformation_matrix[2][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[2][3] << std::endl;
        // std::cout << transformation_matrix[3][0] << ", " << transformation_matrix[3][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[3][3] << std::endl;
        // Transform vertices from model space to view space
        auto transformed_geometry = light->transoform(transformation_matrix);
        transformed_object_array.push_back(transformed_geometry);
    }
    _transformed_object_array = transformed_object_array;
}
void Renderer::serialize_objects()
{
    assert(_transformed_object_array.size() > 0);
    int total_faces = 0;
    int total_vertices = 0;
    for (auto& geometry : _transformed_object_array) {
        total_faces += geometry->num_faces();
        total_vertices += geometry->num_vertices();
    }
    int num_objects = _transformed_object_array.size();
    _cpu_face_vertex_indices_array = rtx::array<RTXFace>(total_faces);
    _cpu_vertex_array = rtx::array<RTXVertex>(total_vertices);
    _cpu_object_array = rtx::array<RTXObject>(num_objects);

    int serialized_array_offset = 0;
    int vertex_offset = 0;
    std::vector<int> vertex_index_offset_array;
    std::vector<int> vertex_count_array;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_object_array.at(object_index);
        geometry->serialize_vertices(_cpu_vertex_array, serialized_array_offset);
        vertex_index_offset_array.push_back(serialized_array_offset);
        vertex_count_array.push_back(geometry->num_vertices());
        serialized_array_offset += geometry->num_vertices();
    }

    serialized_array_offset = 0;
    std::vector<int> face_index_offset_array;
    std::vector<int> face_count_array;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        // std::cout << "object: " << object_index << std::endl;
        auto& geometry = _transformed_object_array.at(object_index);
        int vertex_offset = vertex_index_offset_array[object_index];
        if (geometry->bvh_enabled()) {
            int bvh_index = _map_object_bvh.at(object_index);
            auto& bvh = _bvh_array.at(bvh_index);
            std::vector<std::shared_ptr<bvh::Node>> nodes;
            bvh->_root->collect_leaves(nodes);
            std::shared_ptr<StandardGeometry> standard = std::static_pointer_cast<StandardGeometry>(geometry);
            auto& face_vertex_indices_array = standard->_face_vertex_indices_array;
            // printf("============================================================\n");
            for (int face_index = 0; face_index < face_vertex_indices_array.size(); face_index++) {
                auto& face = face_vertex_indices_array[face_index];
                // printf("[%d] (%d, %d, %d)\n", face_index, face[0], face[1], face[2]);
            }
            // printf("============================================================\n");
            for (auto& node : nodes) {
                int serialized_array_index = node->_assigned_face_index_start + serialized_array_offset;
                assert(serialized_array_index < total_faces);
                for (int face_index : node->_assigned_face_indices) {
                    glm::vec3i face = face_vertex_indices_array[face_index];
                    _cpu_face_vertex_indices_array[serialized_array_index] = { face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset };
                    serialized_array_index++;
                    // printf("[%d] (%d, %d, %d)\n", face_index, face[0], face[1], face[2]);
                }
            }
        } else {
            geometry->serialize_faces(_cpu_face_vertex_indices_array, serialized_array_offset, vertex_offset);
        }
        // std::cout << "face: ";serialized_array_offset
        // for(int i = array_index;i < next_array_index;i++){
        //     std::cout << _cpu_face_vertex_indices_array[i] << " ";
        // }
        // std::cout << std::endl;
        face_index_offset_array.push_back(serialized_array_offset);
        face_count_array.push_back(geometry->num_faces());
        serialized_array_offset += geometry->num_faces();
    }

    assert(vertex_index_offset_array.size() == vertex_count_array.size());
    assert(face_index_offset_array.size() == face_count_array.size());
    assert(vertex_index_offset_array.size() == face_index_offset_array.size());

    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_object_array.at(object_index);
        RTXObject object;
        object.bvh_enabled = geometry->bvh_enabled();
        object.bvh_index = -1;
        if (object.bvh_enabled) {
            assert(_map_object_bvh.find(object_index) != _map_object_bvh.end());
            object.bvh_index = _map_object_bvh[object_index];
        }
        object.num_faces = face_count_array[object_index];
        object.face_index_offset = face_index_offset_array[object_index];
        object.num_vertices = vertex_count_array[object_index];
        object.vertex_index_offset = vertex_index_offset_array[object_index];
        _cpu_object_array[object_index] = object;
    }
    // for (int object_index = 0; object_index < num_objects; object_index++) {
    //     std::cout << "vertex_offset: " << _object_vertex_offset_array[object_index] << " face_offset: " << _face_offset_array[object_index] << std::endl;
    // }
}
void Renderer::construct_bvh()
{
    assert(_transformed_object_array.size() > 0);
    _map_object_bvh = std::unordered_map<int, int>();

    int bvh_index = 0;
    int num_bvh_enabled_objects = 0;
    for (int object_index = 0; object_index < (int)_transformed_object_array.size(); object_index++) {
        auto& geometry = _transformed_object_array[object_index];
        if (geometry->bvh_enabled() == false) {
            continue;
        }
        num_bvh_enabled_objects += 1;
        _map_object_bvh[object_index] = bvh_index;
        bvh_index++;
    }
    _bvh_array = std::vector<std::shared_ptr<BVH>>(num_bvh_enabled_objects);

    int total_nodes = 0;

#pragma omp parallel for
    for (int object_index = 0; object_index < (int)_transformed_object_array.size(); object_index++) {
        auto& geometry = _transformed_object_array[object_index];
        if (geometry->bvh_enabled() == false) {
            continue;
        }
        assert(geometry->bvh_max_triangles_per_node() > 0);
        assert(geometry->type() == RTXObjectTypeStandardGeometry);
        std::shared_ptr<StandardGeometry> standard = std::static_pointer_cast<StandardGeometry>(geometry);
        std::shared_ptr<BVH> bvh = std::make_shared<BVH>(standard);
        int bvh_index = _map_object_bvh[object_index];
        _bvh_array[bvh_index] = bvh;
    }

    for (auto& bvh : _bvh_array) {
        total_nodes += bvh->num_nodes();
    }

    _cpu_threaded_bvh_array = rtx::array<RTXThreadedBVH>(total_nodes);
    _cpu_threaded_bvh_node_array = rtx::array<RTXThreadedBVHNode>(total_nodes);

    int node_index_offset = 0;
    for (int bvh_index = 0; bvh_index < (int)_bvh_array.size(); bvh_index++) {
        auto& bvh = _bvh_array[bvh_index];

        RTXThreadedBVH cuda_bvh;
        cuda_bvh.node_index_offset = node_index_offset;
        cuda_bvh.num_nodes = bvh->num_nodes();
        _cpu_threaded_bvh_array[bvh_index] = cuda_bvh;

        bvh->serialize(_cpu_threaded_bvh_node_array, node_index_offset);
        node_index_offset += bvh->num_nodes();
    }
}
void Renderer::serialize_rays(int height, int width)
{
    int num_rays_per_pixel = _rt_args->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;
    _cpu_ray_array = rtx::array<RTXRay>(num_rays);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);
    glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
    if (_camera->type() == RTXCameraTypePerspective) {
        PerspectiveCamera* perspective = static_cast<PerspectiveCamera*>(_camera.get());
        origin.z = 1.0f / tanf(perspective->_fov_rad / 2.0f);
    }
    float aspect_ratio = float(width) / float(height);
    if (_prev_height != height || _prev_width != width) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int m = 0; m < num_rays_per_pixel; m++) {
                    int index = y * width * num_rays_per_pixel + x * num_rays_per_pixel + m;
                    RTXRay ray;
                    ray.direction.x = 2.0f * float(x + supersampling_noise(generator)) / float(width) - 1.0f;
                    ray.direction.y = -(2.0f * float(y + supersampling_noise(generator)) / float(height) - 1.0f) / aspect_ratio;
                    ray.direction.z = -origin.z;
                    ray.origin.x = origin.x;
                    ray.origin.y = origin.y;
                    ray.origin.z = origin.z;
                    _cpu_ray_array[index] = ray;
                }
            }
        }
    }
}
void Renderer::render_objects(int height, int width)
{
    auto start = std::chrono::system_clock::now();
    bool should_transform_geometry = false;
    bool should_serialize_bvh = false;
    bool should_serialize_geometry = false;
    bool should_reallocate_gpu_memory = false;
    bool should_transfer_to_gpu = false;
    bool should_update_ray = false;
    if (_scene->updated()) {
        should_transform_geometry = true;
        should_serialize_geometry = true;
        should_serialize_bvh = true;
        should_reallocate_gpu_memory = true;
        should_transfer_to_gpu = true;
    } else {
        if (_camera->updated()) {
            should_transform_geometry = true;
            should_serialize_geometry = true;
            should_serialize_bvh = true;
            should_transfer_to_gpu = true;
        }
    }
    if (_prev_height != height || _prev_width != width) {
        should_update_ray = true;
    }

    if (should_transform_geometry) {
        transform_geometries_to_view_space();
    }

    // 現在のカメラ座標系でのBVHを構築
    // Construct BVH in current camera coordinate system
    if (should_serialize_bvh) {
        construct_bvh();
        rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
        rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_array, sizeof(RTXThreadedBVH) * _cpu_threaded_bvh_array.size());
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_node_array, sizeof(RTXThreadedBVHNode) * _cpu_threaded_bvh_node_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_threaded_bvh_array, (void*)_cpu_threaded_bvh_array.data(), sizeof(RTXThreadedBVH) * _cpu_threaded_bvh_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_threaded_bvh_node_array, (void*)_cpu_threaded_bvh_node_array.data(), sizeof(RTXThreadedBVHNode) * _cpu_threaded_bvh_node_array.size());
    }

    // オブジェクトをシリアライズ
    // Serialize objects
    if (should_serialize_geometry) {
        serialize_objects();
    }
    if (should_reallocate_gpu_memory) {
        rtx_cuda_free((void**)&_gpu_face_vertex_indices_array);
        rtx_cuda_free((void**)&_gpu_vertex_array);
        rtx_cuda_free((void**)&_gpu_object_array);
        rtx_cuda_malloc((void**)&_gpu_face_vertex_indices_array, sizeof(RTXFace) * _cpu_face_vertex_indices_array.size());
        rtx_cuda_malloc((void**)&_gpu_vertex_array, sizeof(RTXVertex) * _cpu_vertex_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_array, sizeof(RTXObject) * _cpu_object_array.size());
    }
    if (should_transfer_to_gpu) {
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_vertex_indices_array, (void*)_cpu_face_vertex_indices_array.data(), sizeof(RTXFace) * _cpu_face_vertex_indices_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_cpu_vertex_array.data(), sizeof(RTXVertex) * _cpu_vertex_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_array, (void*)_cpu_object_array.data(), sizeof(RTXObject) * _cpu_object_array.size());
    }

    int num_rays_per_pixel = _rt_args->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;

    // レイ
    // Ray
    if (should_update_ray) {
        serialize_rays(height, width);
        rtx_cuda_free((void**)&_gpu_ray_array);
        rtx_cuda_free((void**)&_gpu_render_array);
        rtx_cuda_malloc((void**)&_gpu_ray_array, sizeof(RTXRay) * num_rays);
        rtx_cuda_malloc((void**)&_gpu_render_array, sizeof(RTXPixel) * num_rays);
        rtx_cuda_memcpy_host_to_device((void*)_gpu_ray_array, (void*)_cpu_ray_array.data(), sizeof(RTXRay) * _cpu_ray_array.size());
        _cpu_render_array = rtx::array<RTXPixel>(num_rays);
        _prev_height = height;
        _prev_width = width;
    }

    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("preprocessing: %lf msec\n", elapsed);

    start = std::chrono::system_clock::now();
    rtx_cuda_render(
        _gpu_ray_array, _cpu_ray_array.size(),
        _gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.size(),
        _gpu_vertex_array, _cpu_vertex_array.size(),
        _gpu_object_array, _cpu_object_array.size(),
        _gpu_threaded_bvh_array, _cpu_threaded_bvh_array.size(),
        _gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.size(),
        _gpu_render_array, _cpu_render_array.size(),
        _cuda_args->num_threads(),
        _cuda_args->num_blocks(),
        _rt_args->num_rays_per_pixel(),
        _rt_args->max_bounce());
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("kernel: %lf msec\n", elapsed);

    rtx_cuda_memcpy_device_to_host((void*)_cpu_render_array.data(), (void*)_gpu_render_array, sizeof(RTXPixel) * num_rays);

    _scene->set_updated(true);
    _camera->set_updated(true);
}
void Renderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingArguments> rt_args,
    std::shared_ptr<CUDAKernelLaunchArguments> cuda_args,
    py::array_t<float, py::array::c_style> render_buffer)
{
    _scene = scene;
    _camera = camera;
    _rt_args = rt_args;
    _cuda_args = cuda_args;

    int height = render_buffer.shape(0);
    int width = render_buffer.shape(1);
    auto pixel = render_buffer.mutable_unchecked<3>();

    render_objects(height, width);

    int num_rays_per_pixel = _rt_args->num_rays_per_pixel();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RTXPixel sum = { 0.0f, 0.0f, 0.0f };
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel + x * num_rays_per_pixel + m;
                RTXPixel pixel = _cpu_render_array[index];
                sum.r += pixel.r;
                sum.g += pixel.g;
                sum.b += pixel.b;
            }
            pixel(y, x, 0) = sum.r / float(num_rays_per_pixel);
            pixel(y, x, 1) = sum.g / float(num_rays_per_pixel);
            pixel(y, x, 2) = sum.b / float(num_rays_per_pixel);
        }
    }
}
void Renderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingArguments> rt_args,
    std::shared_ptr<CUDAKernelLaunchArguments> cuda_args,
    unsigned char* render_buffer,
    int height,
    int width,
    int channels,
    int num_blocks,
    int num_threads)
{
    _scene = scene;
    _camera = camera;
    _rt_args = rt_args;
    _cuda_args = cuda_args;

    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }

    render_objects(height, width);

    int num_rays_per_pixel = _rt_args->num_rays_per_pixel();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RTXPixel sum = { 0.0f, 0.0f, 0.0f };
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel + x * num_rays_per_pixel + m;
                RTXPixel pixel = _cpu_render_array[index];
                sum.r += pixel.r;
                sum.g += pixel.g;
                sum.b += pixel.b;
            }
            int index = y * width * channels + x * channels;
            render_buffer[index + 0] = std::min(std::max((int)(sum.r / float(num_rays_per_pixel) * 255.0f), 0), 255);
            render_buffer[index + 1] = std::min(std::max((int)(sum.g / float(num_rays_per_pixel) * 255.0f), 0), 255);
            render_buffer[index + 2] = std::min(std::max((int)(sum.b / float(num_rays_per_pixel) * 255.0f), 0), 255);
        }
    }
}
}