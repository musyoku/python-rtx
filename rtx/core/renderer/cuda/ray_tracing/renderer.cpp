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
    _gpu_face_offset_array = NULL;
    _gpu_face_count_array = NULL;
    _gpu_vertex_offset_array = NULL;
    _gpu_vertex_count_array = NULL;
    _gpu_scene_threaded_bvh_node_array = NULL;
    _gpu_scene_threaded_bvh_aabb_array = NULL;
    _gpu_render_array = NULL;
}
RayTracingCUDARenderer::~RayTracingCUDARenderer()
{
    rtx_cuda_free((void**)&_gpu_ray_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_index_array);
    rtx_cuda_free((void**)&_gpu_vertex_array);
    rtx_cuda_free((void**)&_gpu_face_offset_array);
    rtx_cuda_free((void**)&_gpu_face_count_array);
    rtx_cuda_free((void**)&_gpu_vertex_offset_array);
    rtx_cuda_free((void**)&_gpu_vertex_count_array);
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
    _face_offset_array = rtx::array<int>(num_objects);
    _face_count_array = rtx::array<int>(num_objects);
    _vertex_offset_array = rtx::array<int>(num_objects);
    _vertex_count_array = rtx::array<int>(num_objects);
    int array_index = 0;

    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        int next_array_index = geometry->serialize_vertices(_vertex_array, array_index);
        assert(next_array_index == array_index + geometry->num_vertices() * stride);
        _vertex_offset_array[object_index] = array_index;
        _vertex_count_array[object_index] = geometry->num_vertices();
        array_index = next_array_index;
    }
    assert(array_index == _vertex_array.size());

    array_index = 0;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        int vertex_offset = _vertex_offset_array[object_index];
        int next_array_index = geometry->serialize_faces(_face_vertex_index_array, array_index, vertex_offset);
        assert(next_array_index == array_index + geometry->num_faces() * stride);
        _face_offset_array[object_index] = array_index;
        _face_count_array[object_index] = geometry->num_faces();
        array_index = next_array_index;
    }

    // for (int object_index = 0; object_index < num_objects; object_index++) {
    //     std::cout << "vertex_offset: " << _vertex_offset_array[object_index] << " face_offset: " << _face_offset_array[object_index] << std::endl;
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
    float aspect_ratio = float(_width) / float(_height);
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
        rtx_cuda_free((void**)&_gpu_vertex_count_array);
        rtx_cuda_free((void**)&_gpu_face_vertex_index_array);
        rtx_cuda_free((void**)&_gpu_face_count_array);
        rtx_cuda_malloc((void**)&_gpu_vertex_array, sizeof(float) * _vertex_array.size());
        rtx_cuda_malloc((void**)&_gpu_vertex_count_array, sizeof(float) * _vertex_count_array.size());
        rtx_cuda_malloc((void**)&_gpu_face_vertex_index_array, sizeof(int) * _face_vertex_index_array.size());
        rtx_cuda_malloc((void**)&_gpu_face_count_array, sizeof(int) * _face_count_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_vertex_array.data(), sizeof(float) * _vertex_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_count_array, (void*)_vertex_count_array.data(), sizeof(float) * _vertex_count_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_vertex_index_array, (void*)_face_vertex_index_array.data(), sizeof(int) * _face_vertex_index_array.size());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_count_array, (void*)_face_count_array.data(), sizeof(int) * _face_count_array.size());
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
        _gpu_face_count_array, _face_count_array.size(),
        _gpu_vertex_array, _vertex_array.size(),
        _gpu_vertex_count_array, _vertex_count_array.size(),
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

    // int channels = array.shape(2);
    // if (channels != 3) {
    //     throw std::runtime_error("channels != 3");
    // }
    // auto pixel = array.mutable_unchecked<3>();

    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);

    // int num_faces = 0;

    // for (auto& mesh : scene->_mesh_array) {
    //     auto& geometry = mesh->_geometry;
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //         num_faces += 1;
    //         continue;
    //     }
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //         StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //         num_faces += standard_geometry->_face_vertex_indices_array.size();
    //         continue;
    //     }
    // }

    // int faces_stride = 4 * 3;
    // int color_stride = 3;
    // if (_initialized == false) {
    //     std::cout << num_faces << " * " << faces_stride << std::endl;
    //     _vertex_array = new float[num_faces * faces_stride];
    //     _face_colors = new float[num_faces * color_stride];
    //     _object_types = new int[num_faces];
    //     _material_types = new int[num_faces];
    // }
    // std::vector<std::shared_ptr<Mesh>> mesh_array;
    // for (auto& mesh : scene->_mesh_array) {
    //     auto& geometry = mesh->_geometry;
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //         SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
    //         std::shared_ptr<SphereGeometry> geometry_in_view_space = std::make_shared<SphereGeometry>(sphere->_radius);
    //         glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;
    //         glm::vec4 homogeneous_center = glm::vec4(sphere->_center, 1.0f);
    //         glm::vec4 homogeneous_center_in_view_space = mv_matrix * homogeneous_center;
    //         geometry_in_view_space->_center = glm::vec3(homogeneous_center_in_view_space.x, homogeneous_center_in_view_space.y, homogeneous_center_in_view_space.z);

    //         std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(geometry_in_view_space, mesh->_material);
    //         mesh_array.emplace_back(mesh_in_view_space);
    //     }
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //         StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //         std::shared_ptr<StandardGeometry> standard_geometry_in_view_space = std::make_shared<StandardGeometry>();
    //         standard_geometry_in_view_space->_face_vertex_indices_array = standard_geometry->_face_vertex_indices_array;

    //         glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;

    //         for (auto& vertex : standard_geometry->_vertex_array) {
    //             glm::vec4 homogeneous_vertex = glm::vec4(vertex, 1.0f);
    //             glm::vec4 homogeneous_vertex_in_view_space = mv_matrix * homogeneous_vertex;
    //             standard_geometry_in_view_space->_vertex_array.emplace_back(glm::vec3(homogeneous_vertex_in_view_space.x, homogeneous_vertex_in_view_space.y, homogeneous_vertex_in_view_space.z));
    //         }

    //         for (auto& face_normal : standard_geometry->_face_normal_array) {
    //             glm::vec4 homogeneous_face_normal = glm::vec4(face_normal, 1.0f);
    //             glm::vec4 homogeneous_face_normal_in_view_space = mv_matrix * homogeneous_face_normal;
    //             glm::vec3 face_normal_in_view_space = glm::normalize(glm::vec3(homogeneous_face_normal_in_view_space.x, homogeneous_face_normal_in_view_space.y, homogeneous_face_normal_in_view_space.z));
    //             standard_geometry_in_view_space->_face_normal_array.emplace_back(face_normal_in_view_space);
    //         }

    //         std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(standard_geometry_in_view_space, mesh->_material);
    //         mesh_array.emplace_back(mesh_in_view_space);
    //     }
    // }

    // int face_index = 0;
    // for (auto& mesh : mesh_array) {
    //     auto& geometry = mesh->_geometry;
    //     auto& material = mesh->_material;
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //         SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
    //         int index = face_index * faces_stride;
    //         _vertex_array[index + 0] = sphere->_center.x;
    //         _vertex_array[index + 1] = sphere->_center.y;
    //         _vertex_array[index + 2] = sphere->_center.z;
    //         _vertex_array[index + 3] = 1.0f;
    //         _vertex_array[index + 4] = sphere->_radius;

    //         index = face_index * color_stride;
    //         glm::vec3 color = material->color();
    //         _face_colors[index + 0] = color.r;
    //         _face_colors[index + 1] = color.g;
    //         _face_colors[index + 2] = color.b;

    //         _object_types[face_index] = RTX_GEOMETRY_TYPE_SPHERE;
    //         _material_types[face_index] = material->type();
    //         face_index += 1;
    //         continue;
    //     }
    //     if (geometry->type() == RTX_GEOMETRY_TYPE) {
    //         BoxGeometry* box = static_cast<BoxGeometry*>(geometry.get());
    //         int index = face_index * faces_stride;
    //         _vertex_array[index + 0] = box->_min.x;
    //         _vertex_array[index + 1] = box->_min.y;
    //         _vertex_array[index + 2] = box->_min.z;
    //         _vertex_array[index + 3] = 1.0f;
    //         _vertex_array[index + 4] = box->_max.x;
    //         _vertex_array[index + 5] = box->_max.y;
    //         _vertex_array[index + 6] = box->_max.z;
    //         _vertex_array[index + 7] = 1.0f;

    //         index = face_index * color_stride;
    //         glm::vec3 color = material->color();
    //         _face_colors[index + 0] = color.r;
    //         _face_colors[index + 1] = color.g;
    //         _face_colors[index + 2] = color.b;

    //         _object_types[face_index] = RTX_GEOMETRY_TYPE;
    //         _material_types[face_index] = material->type();
    //         face_index += 1;
    //         continue;
    //     }
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //         StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //         for (auto& face : standard_geometry->_face_vertex_indices_array) {
    //             glm::vec3& va = standard_geometry->_vertex_array[face[0]];
    //             glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
    //             glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

    //             int index = face_index * faces_stride;
    //             _vertex_array[index + 0] = va.x;
    //             _vertex_array[index + 1] = va.y;
    //             _vertex_array[index + 2] = va.z;
    //             _vertex_array[index + 3] = 1.0f;

    //             _vertex_array[index + 4] = vb.x;
    //             _vertex_array[index + 5] = vb.y;
    //             _vertex_array[index + 6] = vb.z;
    //             _vertex_array[index + 7] = 1.0f;

    //             _vertex_array[index + 8] = vc.x;
    //             _vertex_array[index + 9] = vc.y;
    //             _vertex_array[index + 10] = vc.z;
    //             _vertex_array[index + 11] = 1.0f;

    //             index = face_index * color_stride;
    //             glm::vec3 color = material->color();
    //             _face_colors[index + 0] = color.r;
    //             _face_colors[index + 1] = color.g;
    //             _face_colors[index + 2] = color.b;

    //             _object_types[face_index] = RTX_GEOMETRY_TYPE_STANDARD;
    //             _material_types[face_index] = material->type();
    //             face_index += 1;
    //         }
    //         continue;
    //     }
    // }

    // int num_rays_per_pixel = options->num_rays_per_pixel();
    // int num_rays = _height * _width * num_rays_per_pixel;
    // int rays_stride = 7;

    // if (_initialized == false) {
    //     _ray_array = new float[num_rays * rays_stride];
    // }
    // glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
    // if (camera->type() == RTX_CAMERA_TYPE_PERSPECTIVE) {
    //     PerspectiveCamera* perspective = static_cast<PerspectiveCamera*>(camera.get());
    //     origin.z = 1.0f / tanf(perspective->_fov_rad / 2.0f);
    // }
    // float aspect_ratio = float(_width) / float(_height);

    // for (int y = 0; y < _height; y++) {
    //     for (int x = 0; x < _width; x++) {

    //         for (int m = 0; m < num_rays_per_pixel; m++) {
    //             int index = y * _width * num_rays_per_pixel * rays_stride + x * num_rays_per_pixel * rays_stride + m * rays_stride;

    //             // direction
    //             _ray_array[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
    //             _ray_array[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f) / aspect_ratio;
    //             _ray_array[index + 2] = -origin.z;

    //             // origin
    //             _ray_array[index + 3] = origin.x;
    //             _ray_array[index + 4] = origin.y;
    //             _ray_array[index + 5] = origin.z;

    //             // length
    //             _ray_array[index + 6] = 1.0f;
    //         }
    //     }
    // }

    // int num_pixels = _height * _width;
    // if (_initialized == false) {
    //     _color_per_ray = new float[num_pixels * 3 * num_rays_per_pixel];
    // }

    // if (_initialized == false) {
    //     _camera_inv_matrix = new float[16];
    //     rtx_cuda_alloc(
    //         _gpu_ray_array,
    //         _gpu_face_vertices,
    //         _gpu_face_colors,
    //         _gpu_object_types,
    //         _gpu_material_types,
    //         _gpu_color_per_ray,
    //         _gpu_camera_inv_matrix,
    //         _ray_array,
    //         _vertex_array,
    //         _face_colors,
    //         _object_types,
    //         _material_types,
    //         _camera_inv_matrix,
    //         num_rays,
    //         rays_stride,
    //         num_faces,
    //         faces_stride,
    //         color_stride,
    //         num_pixels,
    //         num_rays_per_pixel);
    // }

    // glm::mat4 inv_camera = glm::inverse(camera->_view_matrix);
    // std::cout << inv_camera[0][0] << ", " << inv_camera[0][1] << ", " << inv_camera[0][2] << ", " << inv_camera[0][3] << std::endl;
    // std::cout << inv_camera[1][0] << ", " << inv_camera[1][1] << ", " << inv_camera[1][2] << ", " << inv_camera[1][3] << std::endl;
    // std::cout << inv_camera[2][0] << ", " << inv_camera[2][1] << ", " << inv_camera[2][2] << ", " << inv_camera[2][3] << std::endl;
    // std::cout << inv_camera[3][0] << ", " << inv_camera[3][1] << ", " << inv_camera[3][2] << ", " << inv_camera[3][3] << std::endl;
    // _camera_inv_matrix[0] = inv_camera[0][0];
    // _camera_inv_matrix[1] = inv_camera[0][1];
    // _camera_inv_matrix[2] = inv_camera[0][2];
    // _camera_inv_matrix[3] = inv_camera[0][3];
    // _camera_inv_matrix[4] = inv_camera[1][0];
    // _camera_inv_matrix[5] = inv_camera[1][1];
    // _camera_inv_matrix[6] = inv_camera[1][2];
    // _camera_inv_matrix[7] = inv_camera[1][3];
    // _camera_inv_matrix[8] = inv_camera[2][0];
    // _camera_inv_matrix[9] = inv_camera[2][1];
    // _camera_inv_matrix[10] = inv_camera[2][2];
    // _camera_inv_matrix[11] = inv_camera[2][3];
    // _camera_inv_matrix[12] = inv_camera[3][0];
    // _camera_inv_matrix[13] = inv_camera[3][1];
    // _camera_inv_matrix[14] = inv_camera[3][2];
    // _camera_inv_matrix[15] = inv_camera[3][3];
    // rtx_cuda_copy(_gpu_ray_array,
    //     _gpu_face_vertices,
    //     _gpu_camera_inv_matrix,
    //     _ray_array,
    //     _vertex_array,
    //     _camera_inv_matrix,
    //     num_rays,
    //     rays_stride,
    //     num_faces,
    //     faces_stride);

    // rtx_cuda_ray_tracing_render(
    //     _gpu_ray_array,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray,
    //     _color_per_ray,
    //     _gpu_camera_inv_matrix,
    //     num_rays,
    //     num_faces,
    //     faces_stride,
    //     color_stride,
    //     options->path_depth(),
    //     num_pixels,
    //     num_rays_per_pixel);

    // // _initialized = true;

    // for (int y = 0; y < _height; y++) {
    //     for (int x = 0; x < _width; x++) {
    //         float sum_r = 0.0f;
    //         float sum_g = 0.0f;
    //         float sum_b = 0.0f;
    //         for (int m = 0; m < num_rays_per_pixel; m++) {
    //             int index = y * _width * num_rays_per_pixel * 3 + x * num_rays_per_pixel * 3 + m * 3;
    //             sum_r += _color_per_ray[index + 0];
    //             sum_g += _color_per_ray[index + 1];
    //             sum_b += _color_per_ray[index + 2];
    //         }
    //         pixel(y, x, 0) = sum_r / float(num_rays_per_pixel);
    //         pixel(y, x, 1) = sum_g / float(num_rays_per_pixel);
    //         pixel(y, x, 2) = sum_b / float(num_rays_per_pixel);
    //     }
    // }

    // _initialized = true;
    // rtx_cuda_delete(_gpu_ray_array,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _vertex_array;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _ray_array;
    // delete[] _color_per_ray;
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

    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);

    // int num_faces = 0;

    // for (auto& mesh : scene->_mesh_array) {
    //     auto& geometry = mesh->_geometry;
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //         num_faces += 1;
    //         continue;
    //     }
    //     if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //         StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //         num_faces += standard_geometry->_face_vertex_indices_array.size();
    //         continue;
    //     }
    // }

    // int faces_stride = 4 * 3;
    // int color_stride = 3;
    // if (_initialized == false) {
    //     _vertex_array = new float[num_faces * faces_stride];
    //     _face_colors = new float[num_faces * color_stride];
    //     _object_types = new int[num_faces];
    //     _material_types = new int[num_faces];

    //     std::vector<std::shared_ptr<Mesh>> mesh_array;
    //     for (auto& mesh : scene->_mesh_array) {
    //         auto& geometry = mesh->_geometry;
    //         if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //             SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
    //             std::shared_ptr<SphereGeometry> geometry_in_view_space = std::make_shared<SphereGeometry>(sphere->_radius);
    //             glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;
    //             glm::vec4 homogeneous_center = glm::vec4(sphere->_center, 1.0f);
    //             glm::vec4 homogeneous_center_in_view_space = mv_matrix * homogeneous_center;
    //             geometry_in_view_space->_center = glm::vec3(homogeneous_center_in_view_space.x, homogeneous_center_in_view_space.y, homogeneous_center_in_view_space.z);

    //             std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(geometry_in_view_space, mesh->_material);
    //             mesh_array.emplace_back(mesh_in_view_space);
    //         }
    //         if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //             StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //             std::shared_ptr<StandardGeometry> standard_geometry_in_view_space = std::make_shared<StandardGeometry>();
    //             standard_geometry_in_view_space->_face_vertex_indices_array = standard_geometry->_face_vertex_indices_array;

    //             glm::mat4 mv_matrix = camera->_view_matrix * mesh->_model_matrix;

    //             for (auto& vertex : standard_geometry->_vertex_array) {
    //                 glm::vec4 homogeneous_vertex = glm::vec4(vertex, 1.0f);
    //                 glm::vec4 homogeneous_vertex_in_view_space = mv_matrix * homogeneous_vertex;
    //                 standard_geometry_in_view_space->_vertex_array.emplace_back(glm::vec3(homogeneous_vertex_in_view_space.x, homogeneous_vertex_in_view_space.y, homogeneous_vertex_in_view_space.z));
    //             }

    //             for (auto& face_normal : standard_geometry->_face_normal_array) {
    //                 glm::vec4 homogeneous_face_normal = glm::vec4(face_normal, 1.0f);
    //                 glm::vec4 homogeneous_face_normal_in_view_space = mv_matrix * homogeneous_face_normal;
    //                 glm::vec3 face_normal_in_view_space = glm::normalize(glm::vec3(homogeneous_face_normal_in_view_space.x, homogeneous_face_normal_in_view_space.y, homogeneous_face_normal_in_view_space.z));
    //                 standard_geometry_in_view_space->_face_normal_array.emplace_back(face_normal_in_view_space);
    //             }

    //             std::shared_ptr<Mesh> mesh_in_view_space = std::make_shared<Mesh>(standard_geometry_in_view_space, mesh->_material);
    //             mesh_array.emplace_back(mesh_in_view_space);
    //         }
    //     }

    //     int face_index = 0;
    //     for (auto& mesh : mesh_array) {
    //         auto& geometry = mesh->_geometry;
    //         auto& material = mesh->_material;
    //         if (geometry->type() == RTX_GEOMETRY_TYPE_SPHERE) {
    //             SphereGeometry* sphere = static_cast<SphereGeometry*>(geometry.get());
    //             int index = face_index * faces_stride;
    //             _vertex_array[index + 0] = sphere->_center.x;
    //             _vertex_array[index + 1] = sphere->_center.y;
    //             _vertex_array[index + 2] = sphere->_center.z;
    //             _vertex_array[index + 3] = 1.0f;
    //             _vertex_array[index + 4] = sphere->_radius;

    //             index = face_index * color_stride;
    //             glm::vec3 color = material->color();
    //             _face_colors[index + 0] = color.r;
    //             _face_colors[index + 1] = color.g;
    //             _face_colors[index + 2] = color.b;

    //             _object_types[face_index] = RTX_GEOMETRY_TYPE_SPHERE;
    //             _material_types[face_index] = material->type();
    //             face_index += 1;
    //         }
    //         if (geometry->type() == RTX_GEOMETRY_TYPE) {
    //             BoxGeometry* box = static_cast<BoxGeometry*>(geometry.get());
    //             int index = face_index * faces_stride;
    //             _vertex_array[index + 0] = box->_min.x;
    //             _vertex_array[index + 1] = box->_min.y;
    //             _vertex_array[index + 2] = box->_min.z;
    //             _vertex_array[index + 3] = 1.0f;
    //             _vertex_array[index + 4] = box->_max.x;
    //             _vertex_array[index + 5] = box->_max.y;
    //             _vertex_array[index + 6] = box->_max.z;
    //             _vertex_array[index + 7] = 1.0f;

    //             index = face_index * color_stride;
    //             glm::vec3 color = material->color();
    //             _face_colors[index + 0] = color.r;
    //             _face_colors[index + 1] = color.g;
    //             _face_colors[index + 2] = color.b;

    //             _object_types[face_index] = RTX_GEOMETRY_TYPE;
    //             _material_types[face_index] = material->type();
    //             face_index += 1;
    //         }
    //         if (geometry->type() == RTX_GEOMETRY_TYPE_STANDARD) {
    //             StandardGeometry* standard_geometry = static_cast<StandardGeometry*>(geometry.get());
    //             for (auto& face : standard_geometry->_face_vertex_indices_array) {
    //                 glm::vec3& va = standard_geometry->_vertex_array[face[0]];
    //                 glm::vec3& vb = standard_geometry->_vertex_array[face[1]];
    //                 glm::vec3& vc = standard_geometry->_vertex_array[face[2]];

    //                 int index = face_index * faces_stride;
    //                 _vertex_array[index + 0] = va.x;
    //                 _vertex_array[index + 1] = va.y;
    //                 _vertex_array[index + 2] = va.z;
    //                 _vertex_array[index + 3] = 1.0f;

    //                 _vertex_array[index + 4] = vb.x;
    //                 _vertex_array[index + 5] = vb.y;
    //                 _vertex_array[index + 6] = vb.z;
    //                 _vertex_array[index + 7] = 1.0f;

    //                 _vertex_array[index + 8] = vc.x;
    //                 _vertex_array[index + 9] = vc.y;
    //                 _vertex_array[index + 10] = vc.z;
    //                 _vertex_array[index + 11] = 1.0f;

    //                 index = face_index * color_stride;
    //                 glm::vec3 color = material->color();
    //                 _face_colors[index + 0] = color.r;
    //                 _face_colors[index + 1] = color.g;
    //                 _face_colors[index + 2] = color.b;

    //                 _object_types[face_index] = RTX_GEOMETRY_TYPE_STANDARD;
    //                 _material_types[face_index] = material->type();
    //                 face_index += 1;
    //             }
    //         }
    //     }
    // }

    // int num_rays_per_pixel = options->num_rays_per_pixel();
    // int num_rays = _height * _width * num_rays_per_pixel;
    // int rays_stride = 7;

    // if (_initialized == false) {
    //     _ray_array = new float[num_rays * rays_stride];

    //     for (int y = 0; y < _height; y++) {
    //         for (int x = 0; x < _width; x++) {
    //             glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);

    //             for (int m = 0; m < num_rays_per_pixel; m++) {
    //                 int index = y * _width * num_rays_per_pixel * rays_stride + x * num_rays_per_pixel * rays_stride + m * rays_stride;

    //                 // direction
    //                 _ray_array[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
    //                 _ray_array[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
    //                 _ray_array[index + 2] = -1.0f;

    //                 // origin
    //                 _ray_array[index + 3] = 0.0f;
    //                 _ray_array[index + 4] = 0.0f;
    //                 _ray_array[index + 5] = 1.0f;

    //                 // length
    //                 _ray_array[index + 6] = 1.0f;
    //             }
    //         }
    //     }
    // }
    // int num_pixels = _height * _width;
    // if (_initialized == false) {
    //     _color_per_ray = new float[num_pixels * 3 * num_rays_per_pixel];
    // }

    // if (_initialized == false) {
    //     _camera_inv_matrix = new float[16];
    //     rtx_cuda_alloc(
    //         _gpu_ray_array,
    //         _gpu_face_vertices,
    //         _gpu_face_colors,
    //         _gpu_object_types,
    //         _gpu_material_types,
    //         _gpu_color_per_ray,
    //         _gpu_camera_inv_matrix,
    //         _ray_array,
    //         _vertex_array,
    //         _face_colors,
    //         _object_types,
    //         _material_types,
    //         _camera_inv_matrix,
    //         num_rays,
    //         rays_stride,
    //         num_faces,
    //         faces_stride,
    //         color_stride,
    //         num_pixels,
    //         num_rays_per_pixel);
    // }

    // glm::mat4 inv_camera = glm::inverse(camera->_view_matrix);
    // // std::cout << inv_camera[0][0] << ", " << inv_camera[0][1] << ", " << inv_camera[0][2] << ", " << inv_camera[0][3] << std::endl;
    // // std::cout << inv_camera[1][0] << ", " << inv_camera[1][1] << ", " << inv_camera[1][2] << ", " << inv_camera[1][3] << std::endl;
    // // std::cout << inv_camera[2][0] << ", " << inv_camera[2][1] << ", " << inv_camera[2][2] << ", " << inv_camera[2][3] << std::endl;
    // // std::cout << inv_camera[3][0] << ", " << inv_camera[3][1] << ", " << inv_camera[3][2] << ", " << inv_camera[3][3] << std::endl;
    // rtx_cuda_copy(_gpu_ray_array, _gpu_face_vertices, _gpu_camera_inv_matrix, _ray_array, _vertex_array, _camera_inv_matrix, num_rays, rays_stride, num_faces, faces_stride);

    // rtx_cuda_ray_tracing_render(
    //     _gpu_ray_array,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray,
    //     _color_per_ray,
    //     _gpu_camera_inv_matrix,
    //     num_rays,
    //     num_faces,
    //     faces_stride,
    //     color_stride,
    //     options->path_depth(),
    //     num_pixels,
    //     num_rays_per_pixel);

    // for (int y = 0; y < _height; y++) {
    //     for (int x = 0; x < _width; x++) {
    //         float sum_r = 0.0f;
    //         float sum_g = 0.0f;
    //         float sum_b = 0.0f;
    //         for (int m = 0; m < num_rays_per_pixel; m++) {
    //             int index = y * _width * num_rays_per_pixel * 3 + x * num_rays_per_pixel * 3 + m * 3;
    //             sum_r += _color_per_ray[index + 0];
    //             sum_g += _color_per_ray[index + 1];
    //             sum_b += _color_per_ray[index + 2];
    //         }
    //         int index = y * width * channels + x * channels;
    //         array[index + 0] = std::min(std::max((int)(sum_r / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //         array[index + 1] = std::min(std::max((int)(sum_g / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //         array[index + 2] = std::min(std::max((int)(sum_b / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //     }
    // }

    // _initialized = true;
    // rtx_cuda_delete(_gpu_ray_array,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _vertex_array;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _ray_array;
    // delete[] _color_per_ray;
}
}