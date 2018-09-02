#include "renderer.h"
#include "../../../camera/perspective.h"
#include "../../../geometry/box.h"
#include "../../../geometry/sphere.h"
#include "../../../geometry/standard.h"
#include "../../../header/enum.h"
#include "../header/ray_tracing.h"
#include <iostream>
#include <memory>
#include <vector>

namespace rtx {

namespace py = pybind11;

RayTracingCUDARenderer::RayTracingCUDARenderer()
{
    _gpu_ray_array = NULL;
    _gpu_face_vertex_index_array = NULL;
    _gpu_vertex_array = NULL;
    _gpu_object_face_offset_array = NULL;
    _gpu_object_face_count_array = NULL;
    _gpu_object_vertex_offset_array = NULL;
    _gpu_object_vertex_count_array = NULL;
    _gpu_scene_threaded_bvh_node_array = NULL;
    _gpu_scene_threaded_bvh_aabb_array = NULL;
    _gpu_render_array = NULL;
}
RayTracingCUDARenderer::~RayTracingCUDARenderer()
{
    rtx_cuda_free((void**)&_gpu_ray_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_index_array);
    rtx_cuda_free((void**)&_gpu_vertex_array);
    rtx_cuda_free((void**)&_gpu_object_face_offset_array);
    rtx_cuda_free((void**)&_gpu_object_face_count_array);
    rtx_cuda_free((void**)&_gpu_object_vertex_offset_array);
    rtx_cuda_free((void**)&_gpu_object_vertex_count_array);
    rtx_cuda_free((void**)&_gpu_scene_threaded_bvh_node_array);
    rtx_cuda_free((void**)&_gpu_render_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_index_array);
    rtx_cuda_free((void**)&_gpu_ray_array);
}
void RayTracingCUDARenderer::transform_geometries_to_view_space()
{
    int num_objects = _scene->_mesh_array.size();
    std::vector<std::shared_ptr<Geometry>> transformed_geometry_array;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& mesh = _scene->_mesh_array[object_index];
        auto& geometry = mesh->_geometry;
        glm::mat4 transformation_matrix = _camera->_view_matrix * mesh->_model_matrix;
        // std::cout << "transform: " << std::endl;
        // std::cout << transformation_matrix[0][0] << ", " << transformation_matrix[0][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[0][3] << std::endl;
        // std::cout << transformation_matrix[1][0] << ", " << transformation_matrix[1][1] << ", " << transformation_matrix[1][2] << ", " << transformation_matrix[1][3] << std::endl;
        // std::cout << transformation_matrix[2][0] << ", " << transformation_matrix[2][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[2][3] << std::endl;
        // std::cout << transformation_matrix[3][0] << ", " << transformation_matrix[3][1] << ", " << transformation_matrix[0][2] << ", " << transformation_matrix[3][3] << std::endl;
        // Transform vertices from model space to view space
        auto transformed_geometry = geometry->transoform(transformation_matrix);
        transformed_geometry_array.push_back(transformed_geometry);
    }
    _transformed_geometry_array = transformed_geometry_array;
}
void RayTracingCUDARenderer::serialize_geometries()
{
    int num_faces = 0;
    int num_vertices = 0;
    for (auto& mesh : _scene->_mesh_array) {
        auto& geometry = mesh->_geometry;
        num_faces += geometry->num_faces();
        num_vertices += geometry->num_vertices();
    }
    int num_objects = _scene->_mesh_array.size();
    int stride = 4;
    _face_vertex_index_array = rtx::array<int>(num_faces * stride);
    _vertex_array = rtx::array<float>(num_vertices * stride);
    _object_face_offset_array = rtx::array<int>(num_objects);
    _object_face_count_array = rtx::array<int>(num_objects);
    _object_vertex_offset_array = rtx::array<int>(num_objects);
    _object_vertex_count_array = rtx::array<int>(num_objects);
    int array_index = 0;

    int vertex_offset = 0;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        int next_array_index = geometry->serialize_vertices(_vertex_array, array_index);
        // std::cout << "vertex: ";
        // for(int i = 0;i < geometry->num_vertices();i++){
        //     std::cout << "(" << _vertex_array[(vertex_offset + i) * 4 + 0] << ", " << _vertex_array[(vertex_offset + i) * 4 + 1] << ", " << _vertex_array[(vertex_offset + i) * 4 + 2] << ") ";
        // }
        // std::cout << std::endl;
        assert(next_array_index == array_index + geometry->num_vertices() * stride);
        _object_vertex_offset_array[object_index] = vertex_offset;
        _object_vertex_count_array[object_index] = geometry->num_vertices();
        vertex_offset += geometry->num_vertices();
        array_index = next_array_index;
    }
    assert(array_index == _vertex_array.size());

    array_index = 0;
    int face_offset = 0;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        // std::cout << "object: " << object_index << std::endl;
        auto& geometry = _transformed_geometry_array.at(object_index);
        int vertex_offset = _object_vertex_offset_array[object_index];
        int next_array_index = geometry->serialize_faces(_face_vertex_index_array, array_index, vertex_offset);
        // std::cout << "face: ";
        // for(int i = array_index;i < next_array_index;i++){
        //     std::cout << _face_vertex_index_array[i] << " ";
        // }
        // std::cout << std::endl;
        assert(next_array_index == array_index + geometry->num_faces() * stride);
        _object_face_offset_array[object_index] = face_offset;
        _object_face_count_array[object_index] = geometry->num_faces();
        face_offset += geometry->num_faces();
        array_index = next_array_index;
    }

    // for (int object_index = 0; object_index < num_objects; object_index++) {
    //     std::cout << "vertex_offset: " << _object_vertex_offset_array[object_index] << " face_offset: " << _face_offset_array[object_index] << std::endl;
    // }
    assert(array_index == _face_vertex_index_array.size());
}
void RayTracingCUDARenderer::construct_bvh()
{
    _bvh = std::make_unique<bvh::scene::SceneBVH>(_transformed_geometry_array);
    int num_nodes = _bvh->num_nodes();
    _scene_threaded_bvh_node_array = rtx::array<unsigned int>(num_nodes);
    _scene_threaded_bvh_aabb_array = rtx::array<float>(num_nodes * 2 * 4);
    _bvh->serialize(_scene_threaded_bvh_node_array, _scene_threaded_bvh_aabb_array);
}

void RayTracingCUDARenderer::serialize_rays(int height, int width)
{
    int num_rays_per_pixel = _options->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;
    int stride = 8;
    _ray_array = rtx::array<float>(num_rays * stride);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);
    glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
    if (_camera->type() == RTX_CAMERA_TYPE_PERSPECTIVE) {
        PerspectiveCamera* perspective = static_cast<PerspectiveCamera*>(_camera.get());
        origin.z = 1.0f / tanf(perspective->_fov_rad / 2.0f);
    }
    float aspect_ratio = float(width) / float(height);
    if (_prev_height != height || _prev_width != width) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                for (int m = 0; m < num_rays_per_pixel; m++) {
                    int index = y * width * num_rays_per_pixel * stride + x * num_rays_per_pixel * stride + m * stride;

                    // direction
                    _ray_array[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(width) - 1.0f;
                    _ray_array[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(height) - 1.0f) / aspect_ratio;
                    _ray_array[index + 2] = -origin.z;
                    _ray_array[index + 3] = 1.0f;

                    // origin
                    _ray_array[index + 4] = origin.x;
                    _ray_array[index + 5] = origin.y;
                    _ray_array[index + 6] = origin.z;
                    _ray_array[index + 7] = 1.0f;
                }
            }
        }
    }
}
void RayTracingCUDARenderer::render_objects(int height, int width)
{
    // GPUの線形メモリへ転送するデータを準備する
    // Construct arrays to transfer to the Linear Memory of GPU
    if (_scene->updated()) {
        transform_geometries_to_view_space();
        serialize_geometries();
        rtx_cuda_free((void**)&_gpu_vertex_array);
        rtx_cuda_free((void**)&_gpu_face_vertex_index_array);
        rtx_cuda_free((void**)&_gpu_object_face_count_array);
        rtx_cuda_free((void**)&_gpu_object_face_offset_array);
        rtx_cuda_free((void**)&_gpu_object_vertex_count_array);
        rtx_cuda_free((void**)&_gpu_object_vertex_offset_array);
        rtx_cuda_malloc((void**)&_gpu_vertex_array, sizeof(float) * _vertex_array.size());
        rtx_cuda_malloc((void**)&_gpu_face_vertex_index_array, sizeof(int) * _face_vertex_index_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_face_count_array, sizeof(int) * _object_face_count_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_face_offset_array, sizeof(int) * _object_face_offset_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_vertex_count_array, sizeof(float) * _object_vertex_count_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_vertex_offset_array, sizeof(float) * _object_vertex_offset_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_vertex_array.data(), sizeof(float) * _vertex_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_vertex_index_array, (void*)_face_vertex_index_array.data(), sizeof(int) * _face_vertex_index_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_face_count_array, (void*)_object_face_count_array.data(), sizeof(int) * _object_face_count_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_face_offset_array, (void*)_object_face_offset_array.data(), sizeof(int) * _object_face_offset_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_vertex_count_array, (void*)_object_vertex_count_array.data(), sizeof(float) * _object_vertex_count_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_vertex_offset_array, (void*)_object_vertex_offset_array.data(), sizeof(float) * _object_vertex_offset_array.size());
    } else {
        if (_camera->updated()) {
            transform_geometries_to_view_space();
            serialize_geometries();
            rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_vertex_array.data(), sizeof(float) * _vertex_array.size());
        }
    }

    // 現在のカメラ座標系でのBVHを構築
    // Construct BVH in current camera coordinate system
    if (_scene->updated()) {
        construct_bvh();
        rtx_cuda_free((void**)&_gpu_scene_threaded_bvh_node_array);
        rtx_cuda_free((void**)&_gpu_scene_threaded_bvh_aabb_array);
        rtx_cuda_malloc((void**)&_gpu_scene_threaded_bvh_node_array, sizeof(unsigned int) * _scene_threaded_bvh_node_array.size());
        rtx_cuda_malloc((void**)&_gpu_scene_threaded_bvh_aabb_array, sizeof(float) * _scene_threaded_bvh_aabb_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_scene_threaded_bvh_node_array, (void*)_scene_threaded_bvh_node_array.data(), sizeof(unsigned int) * _scene_threaded_bvh_node_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_scene_threaded_bvh_aabb_array, (void*)_scene_threaded_bvh_aabb_array.data(), sizeof(unsigned int) * _scene_threaded_bvh_aabb_array.size());
    } else {
        if (_camera->updated()) {
            construct_bvh();
            rtx_cuda_memcpy_host_to_device((void*)_gpu_scene_threaded_bvh_node_array, (void*)_scene_threaded_bvh_node_array.data(), sizeof(unsigned int) * _scene_threaded_bvh_node_array.size());
            rtx_cuda_memcpy_host_to_device((void*)_gpu_scene_threaded_bvh_aabb_array, (void*)_scene_threaded_bvh_aabb_array.data(), sizeof(float) * _scene_threaded_bvh_aabb_array.size());
        }
    }

    // レイ
    // Ray
    if (_prev_height != height || _prev_width != width) {
        int render_array_size = height * width * 3 * _options->num_rays_per_pixel();
        serialize_rays(height, width);
        rtx_cuda_free((void**)&_gpu_ray_array);
        rtx_cuda_free((void**)&_gpu_render_array);
        rtx_cuda_malloc((void**)&_gpu_ray_array, sizeof(float) * _ray_array.size());
        rtx_cuda_malloc((void**)&_gpu_render_array, sizeof(float) * render_array_size);
        rtx_cuda_memcpy_host_to_device((void*)_gpu_ray_array, (void*)_ray_array.data(), sizeof(unsigned int) * _ray_array.size());
        _render_array = rtx::array<float>(render_array_size);
        _prev_height = height;
        _prev_width = width;
    }

    int num_rays_per_pixel = _options->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;

    rtx_cuda_ray_tracing_render(
        _gpu_ray_array, _ray_array.size(),
        _gpu_face_vertex_index_array, _face_vertex_index_array.size(),
        _gpu_vertex_array, _vertex_array.size(),
        _gpu_object_face_count_array, _object_face_count_array.size(),
        _gpu_object_face_offset_array, _object_face_offset_array.size(),
        _gpu_object_vertex_count_array, _object_vertex_count_array.size(),
        _gpu_object_vertex_offset_array, _object_vertex_offset_array.size(),
        _gpu_scene_threaded_bvh_node_array, _scene_threaded_bvh_node_array.size(),
        _gpu_scene_threaded_bvh_aabb_array, _scene_threaded_bvh_aabb_array.size(),
        _gpu_render_array, _render_array.size(),
        num_rays,
        _options->num_rays_per_pixel(),
        _options->path_depth());

    int render_array_size = height * width * 3 * _options->num_rays_per_pixel();
    assert(_render_array.size() == render_array_size);
    rtx_cuda_memcpy_device_to_host((void*)_render_array.data(), (void*)_gpu_render_array, sizeof(float) * render_array_size);

    _scene->set_updated(true);
    _camera->set_updated(true);
}
void RayTracingCUDARenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<float, py::array::c_style> array)
{
    _scene = scene;
    _camera = camera;
    _options = options;

    int height = array.shape(0);
    int width = array.shape(1);
    auto pixel = array.mutable_unchecked<3>();

    render_objects(height, width);

    int num_rays_per_pixel = _options->num_rays_per_pixel();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel * 3 + x * num_rays_per_pixel * 3 + m * 3;
                sum_r += _render_array[index + 0];
                sum_g += _render_array[index + 1];
                sum_b += _render_array[index + 2];
            }
            pixel(y, x, 0) = sum_r / float(num_rays_per_pixel);
            pixel(y, x, 1) = sum_g / float(num_rays_per_pixel);
            pixel(y, x, 2) = sum_b / float(num_rays_per_pixel);
        }
    }
}
void RayTracingCUDARenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    unsigned char* array,
    int height,
    int width,
    int channels)
{
    _scene = scene;
    _camera = camera;
    _options = options;

    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }

    render_objects(height, width);

    int num_rays_per_pixel = _options->num_rays_per_pixel();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel * 3 + x * num_rays_per_pixel * 3 + m * 3;
                sum_r += _render_array[index + 0];
                sum_g += _render_array[index + 1];
                sum_b += _render_array[index + 2];
            }
            int index = y * width * channels + x * channels;
            array[index + 0] = std::min(std::max((int)(sum_r / float(num_rays_per_pixel) * 255.0f), 0), 255);
            array[index + 1] = std::min(std::max((int)(sum_g / float(num_rays_per_pixel) * 255.0f), 0), 255);
            array[index + 2] = std::min(std::max((int)(sum_b / float(num_rays_per_pixel) * 255.0f), 0), 255);
        }
    }
}
}