#include "renderer.h"
#include "../../../camera/perspective.h"
#include "../../../geometry/box.h"
#include "../../../geometry/sphere.h"
#include "../../../geometry/standard.h"
#include "../../../header/enum.h"
#include <iostream>
#include <memory>
#include <vector>

namespace rtx {

namespace py = pybind11;

RayTracingCUDARenderer::RayTracingCUDARenderer()
{
    _gpu_ray_array = NULL;
    _gpu_face_vertex_indices_array = NULL;
    _gpu_vertex_array = NULL;
    _gpu_object_array = NULL;
    _gpu_threaded_bvh_array = NULL;
    _gpu_threaded_bvh_node_array = NULL;
    _gpu_render_array = NULL;
}
RayTracingCUDARenderer::~RayTracingCUDARenderer()
{
    rtx_cuda_free((void**)&_gpu_ray_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_indices_array);
    rtx_cuda_free((void**)&_gpu_vertex_array);
    rtx_cuda_free((void**)&_gpu_object_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
    rtx_cuda_free((void**)&_gpu_render_array);
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
void RayTracingCUDARenderer::serialize_objects()
{
    assert(_transformed_geometry_array.size() > 0);
    int num_faces = 0;
    int num_vertices = 0;
    for (auto& geometry : _transformed_geometry_array) {
        num_faces += geometry->num_faces();
        num_vertices += geometry->num_vertices();
    }
    int num_objects = _transformed_geometry_array.size();
    _cpu_face_vertex_indices_array = rtx::array<RTXGeometryFace>(num_faces);
    _cpu_vertex_array = rtx::array<RTXGeometryVertex>(num_vertices);
    _cpu_object_array = rtx::array<RTXObject>(num_objects);

    int array_offset = 0;
    int vertex_offset = 0;
    std::vector<int> vertex_index_offset_array;
    std::vector<int> vertex_count_array;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        geometry->serialize_vertices(_cpu_vertex_array, array_offset);
        vertex_index_offset_array.push_back(array_offset);
        vertex_count_array.push_back(geometry->num_vertices());
        array_offset += geometry->num_vertices();
    }

    array_offset = 0;
    std::vector<int> face_index_offset_array;
    std::vector<int> face_count_array;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        // std::cout << "object: " << object_index << std::endl;
        auto& geometry = _transformed_geometry_array.at(object_index);
        int vertex_offset = vertex_index_offset_array[object_index];
        geometry->serialize_faces(_cpu_face_vertex_indices_array, array_offset, vertex_offset);
        // std::cout << "face: ";array_offset
        // for(int i = array_index;i < next_array_index;i++){
        //     std::cout << _cpu_face_vertex_indices_array[i] << " ";
        // }
        // std::cout << std::endl;
        face_index_offset_array.push_back(array_offset);
        face_count_array.push_back(geometry->num_faces());
        array_offset += geometry->num_faces();
    }

    assert(vertex_index_offset_array.size() == vertex_count_array.size());
    assert(face_index_offset_array.size() == face_count_array.size());
    assert(vertex_index_offset_array.size() == face_index_offset_array.size());

    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
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
void RayTracingCUDARenderer::construct_bvh()
{
    assert(_transformed_geometry_array.size() > 0);
    _map_object_bvh = std::unordered_map<int, int>();
    _bvh_array = std::vector<std::shared_ptr<BVH>>();
    int bvh_index = 0;
    int total_nodes = 0;
    for (int object_index = 0; object_index < (int)_transformed_geometry_array.size(); object_index++) {
        auto& geometry = _transformed_geometry_array[object_index];
        if (geometry->bvh_enabled() == false) {
            continue;
        }
        assert(geometry->bvh_max_triangles_per_node() > 0);
        assert(geometry->type() == RTX_GEOMETRY_TYPE_STANDARD);
        std::shared_ptr<StandardGeometry> standard = std::static_pointer_cast<StandardGeometry>(geometry);
        std::shared_ptr<BVH> bvh = std::make_shared<BVH>(standard);
        _bvh_array.emplace_back(bvh);
        total_nodes += bvh->num_nodes();
        _map_object_bvh[object_index] = bvh_index;
        bvh_index++;
    }
    if (_bvh_array.size() > 128) {
        throw std::runtime_error("We only support up to 128 BVH enabled objects.");
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
void RayTracingCUDARenderer::serialize_rays(int height, int width)
{
    int num_rays_per_pixel = _options->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;
    _cpu_ray_array = rtx::array<RTXRay>(num_rays);
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
void RayTracingCUDARenderer::render_objects(int height, int width)
{
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
    }
    if (should_reallocate_gpu_memory) {
        rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
        rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_array, sizeof(RTXThreadedBVH) * _cpu_threaded_bvh_array.size());
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_node_array, sizeof(RTXThreadedBVHNode) * _cpu_threaded_bvh_node_array.size());
    }
    if (should_transfer_to_gpu) {
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
        rtx_cuda_malloc((void**)&_gpu_face_vertex_indices_array, sizeof(RTXGeometryFace) * _cpu_face_vertex_indices_array.size());
        rtx_cuda_malloc((void**)&_gpu_vertex_array, sizeof(RTXGeometryVertex) * _cpu_vertex_array.size());
        rtx_cuda_malloc((void**)&_gpu_object_array, sizeof(RTXObject) * _cpu_object_array.size());
    }
    if (should_transfer_to_gpu) {
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_vertex_indices_array, (void*)_cpu_face_vertex_indices_array.data(), sizeof(RTXGeometryFace) * _cpu_face_vertex_indices_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_cpu_vertex_array.data(), sizeof(RTXGeometryVertex) * _cpu_vertex_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_array, (void*)_cpu_object_array.data(), sizeof(RTXObject) * _cpu_object_array.size());
    }

    int num_rays_per_pixel = _options->num_rays_per_pixel();
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

    rtx_cuda_ray_tracing_render(
        _gpu_ray_array, _cpu_ray_array.size(),
        _gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.size(),
        _gpu_vertex_array, _cpu_vertex_array.size(),
        _gpu_object_array, _cpu_object_array.size(),
        _gpu_threaded_bvh_array, _cpu_threaded_bvh_array.size(),
        _gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.size(),
        _gpu_render_array, _cpu_render_array.size(),
        _options->num_rays_per_pixel(),
        _options->max_bounce());

    rtx_cuda_memcpy_device_to_host((void*)_cpu_render_array.data(), (void*)_gpu_render_array, sizeof(RTXPixel) * num_rays);

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
            RTXPixel sum = { 0.0f, 0.0f, 0.0f };
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel + x * num_rays_per_pixel + m;
                RTXPixel pixel = _cpu_render_array[index];
                sum.r += pixel.r;
                sum.g += pixel.g;
                sum.b += pixel.b;
            }
            int index = y * width * channels + x * channels;
            array[index * 3 + 0] = std::min(std::max((int)(sum.r / float(num_rays_per_pixel) * 255.0f), 0), 255);
            array[index * 3 + 1] = std::min(std::max((int)(sum.g / float(num_rays_per_pixel) * 255.0f), 0), 255);
            array[index * 3 + 2] = std::min(std::max((int)(sum.b / float(num_rays_per_pixel) * 255.0f), 0), 255);
        }
    }
}
}