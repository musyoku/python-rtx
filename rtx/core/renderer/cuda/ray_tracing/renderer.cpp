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
    _gpu_rays = nullptr;
    _gpu_faces = nullptr;
    _gpu_vertices = nullptr;
    _gpu_object_colors = nullptr;
    _gpu_geometry_types = nullptr;
    _gpu_material_types = nullptr;
    _gpu_color_buffer = nullptr;
    _gpu_bvh_hit_path = nullptr;
    _gpu_bvh_miss_path = nullptr;
    _gpu_bvh_is_leaf = nullptr;
    _gpu_bvh_object_index = nullptr;
    _gpu_bvh_face_start_index = nullptr;
    _gpu_bvh_face_end_index = nullptr;
}
RayTracingCUDARenderer::~RayTracingCUDARenderer()
{
    rtx_cuda_free((void*)_gpu_faces);
    rtx_cuda_free((void*)_gpu_vertices);
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
    _faces = rtx::array<int>(num_faces * stride);
    _vertices = rtx::array<float>(num_vertices * stride);
    _vertex_serialization_offsets = rtx::array<int>(num_objects);
    _face_serialization_offsets = rtx::array<int>(num_objects);
    int buffer_index = 0;

    // 各頂点をモデル座標系からカメラ座標系に変換
    // Transform vertices from model space to camera space
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        int next_buffer_index = geometry->serialize_vertices(_vertices, buffer_index);
        assert(next_buffer_index == buffer_index + geometry->num_vertices() * stride);
        _vertex_serialization_offsets[object_index] = buffer_index;
        buffer_index = next_buffer_index;
    }
    assert(buffer_index == _vertices.size());

    buffer_index = 0;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& geometry = _transformed_geometry_array.at(object_index);
        int vertex_offset = _vertex_serialization_offsets[object_index];
        int next_buffer_index = geometry->serialize_faces(_faces, buffer_index, vertex_offset);
        assert(next_buffer_index == buffer_index + geometry->num_faces() * stride);
        _face_serialization_offsets[object_index] = buffer_index;
        buffer_index = next_buffer_index;
    }

    for (int object_index = 0; object_index < num_objects; object_index++) {
        std::cout << "vertex_offset: " << _vertex_serialization_offsets[object_index] << " face_offset: " << _face_serialization_offsets[object_index] << std::endl;
    }
    assert(buffer_index == _faces.size());
}
void RayTracingCUDARenderer::construct_bvh()
{
    _bvh = std::make_unique<bvh::scene::SceneBVH>(_transformed_geometry_array);
    int num_nodes = _bvh->num_nodes();
    _scene_bvh_nodes = rtx::array<unsigned int>(num_nodes);
    _bvh->serialize(_scene_bvh_nodes);
}

void RayTracingCUDARenderer::serialize_rays(int height, int width)
{
    int num_rays_per_pixel = _options->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;
    int stride = 6;
    _rays = rtx::array<float>(num_rays * stride);
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
                    _rays[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(width) - 1.0f;
                    _rays[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(height) - 1.0f) / aspect_ratio;
                    _rays[index + 2] = -origin.z;

                    // origin
                    _rays[index + 3] = origin.x;
                    _rays[index + 4] = origin.y;
                    _rays[index + 5] = origin.z;
                }
            }
        }
    }
}
void RayTracingCUDARenderer::render_objects(int height, int width)
{
    // Transform
    if (_scene->updated()) {
        transform_geometries_to_view_space();
    }

    // GPUの線形メモリへ転送するデータを準備する
    // Construct arrays to transfer to the Linear Memory of GPU
    if (_scene->updated()) {
        rtx_cuda_free(_gpu_vertices);
        rtx_cuda_free(_gpu_faces);
        serialize_geometries();
        rtx_cuda_malloc_float(_gpu_vertices, _vertices.size());
        rtx_cuda_malloc_integer(_gpu_faces, _faces.size());
    }

    // 現在のカメラ座標系でのBVHを構築
    // Construct BVH in current camera coordinate system
    if (_camera->updated() || _scene->updated()) {
        rtx_cuda_free(_gpu_scene_bvh_nodes);
        construct_bvh();
        rtx_cuda_malloc_float(_gpu_scene_bvh_nodes, _scene_bvh_nodes.size());
    }

    // レイ
    // Ray
    if (_prev_height != height || _prev_width != width) {
        rtx_cuda_free(_gpu_rays);
        serialize_rays(height, width);
        rtx_cuda_malloc_float(_gpu_rays, _rays.size());
        _prev_height = height;
        _prev_width = width;
    }
}
void RayTracingCUDARenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<float, py::array::c_style> buffer)
{
    _scene = scene;
    _camera = camera;
    _options = options;

    int height = buffer.shape(0);
    int width = buffer.shape(1);

    render_objects(height, width);

    // int channels = buffer.shape(2);
    // if (channels != 3) {
    //     throw std::runtime_error("channels != 3");
    // }
    // auto pixel = buffer.mutable_unchecked<3>();

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
    //     _vertices = new float[num_faces * faces_stride];
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
    //         _vertices[index + 0] = sphere->_center.x;
    //         _vertices[index + 1] = sphere->_center.y;
    //         _vertices[index + 2] = sphere->_center.z;
    //         _vertices[index + 3] = 1.0f;
    //         _vertices[index + 4] = sphere->_radius;

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
    //         _vertices[index + 0] = box->_min.x;
    //         _vertices[index + 1] = box->_min.y;
    //         _vertices[index + 2] = box->_min.z;
    //         _vertices[index + 3] = 1.0f;
    //         _vertices[index + 4] = box->_max.x;
    //         _vertices[index + 5] = box->_max.y;
    //         _vertices[index + 6] = box->_max.z;
    //         _vertices[index + 7] = 1.0f;

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
    //             _vertices[index + 0] = va.x;
    //             _vertices[index + 1] = va.y;
    //             _vertices[index + 2] = va.z;
    //             _vertices[index + 3] = 1.0f;

    //             _vertices[index + 4] = vb.x;
    //             _vertices[index + 5] = vb.y;
    //             _vertices[index + 6] = vb.z;
    //             _vertices[index + 7] = 1.0f;

    //             _vertices[index + 8] = vc.x;
    //             _vertices[index + 9] = vc.y;
    //             _vertices[index + 10] = vc.z;
    //             _vertices[index + 11] = 1.0f;

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
    //     _rays = new float[num_rays * rays_stride];
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
    //             _rays[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
    //             _rays[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f) / aspect_ratio;
    //             _rays[index + 2] = -origin.z;

    //             // origin
    //             _rays[index + 3] = origin.x;
    //             _rays[index + 4] = origin.y;
    //             _rays[index + 5] = origin.z;

    //             // length
    //             _rays[index + 6] = 1.0f;
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
    //         _gpu_rays,
    //         _gpu_face_vertices,
    //         _gpu_face_colors,
    //         _gpu_object_types,
    //         _gpu_material_types,
    //         _gpu_color_per_ray,
    //         _gpu_camera_inv_matrix,
    //         _rays,
    //         _vertices,
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
    // rtx_cuda_copy(_gpu_rays,
    //     _gpu_face_vertices,
    //     _gpu_camera_inv_matrix,
    //     _rays,
    //     _vertices,
    //     _camera_inv_matrix,
    //     num_rays,
    //     rays_stride,
    //     num_faces,
    //     faces_stride);

    // rtx_cuda_ray_tracing_render(
    //     _gpu_rays,
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
    // rtx_cuda_delete(_gpu_rays,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _vertices;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _rays;
    // delete[] _color_per_ray;
}
void RayTracingCUDARenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    unsigned char* buffer,
    int height,
    int width,
    int channels)
{
    _scene = scene;
    _camera = camera;
    _options = options;

    render_objects();

    // _height = height;
    // _width = width;
    // if (channels != 3) {
    //     throw std::runtime_error("channels != 3");
    // }

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
    //     _vertices = new float[num_faces * faces_stride];
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
    //             _vertices[index + 0] = sphere->_center.x;
    //             _vertices[index + 1] = sphere->_center.y;
    //             _vertices[index + 2] = sphere->_center.z;
    //             _vertices[index + 3] = 1.0f;
    //             _vertices[index + 4] = sphere->_radius;

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
    //             _vertices[index + 0] = box->_min.x;
    //             _vertices[index + 1] = box->_min.y;
    //             _vertices[index + 2] = box->_min.z;
    //             _vertices[index + 3] = 1.0f;
    //             _vertices[index + 4] = box->_max.x;
    //             _vertices[index + 5] = box->_max.y;
    //             _vertices[index + 6] = box->_max.z;
    //             _vertices[index + 7] = 1.0f;

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
    //                 _vertices[index + 0] = va.x;
    //                 _vertices[index + 1] = va.y;
    //                 _vertices[index + 2] = va.z;
    //                 _vertices[index + 3] = 1.0f;

    //                 _vertices[index + 4] = vb.x;
    //                 _vertices[index + 5] = vb.y;
    //                 _vertices[index + 6] = vb.z;
    //                 _vertices[index + 7] = 1.0f;

    //                 _vertices[index + 8] = vc.x;
    //                 _vertices[index + 9] = vc.y;
    //                 _vertices[index + 10] = vc.z;
    //                 _vertices[index + 11] = 1.0f;

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
    //     _rays = new float[num_rays * rays_stride];

    //     for (int y = 0; y < _height; y++) {
    //         for (int x = 0; x < _width; x++) {
    //             glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);

    //             for (int m = 0; m < num_rays_per_pixel; m++) {
    //                 int index = y * _width * num_rays_per_pixel * rays_stride + x * num_rays_per_pixel * rays_stride + m * rays_stride;

    //                 // direction
    //                 _rays[index + 0] = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
    //                 _rays[index + 1] = -(2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f);
    //                 _rays[index + 2] = -1.0f;

    //                 // origin
    //                 _rays[index + 3] = 0.0f;
    //                 _rays[index + 4] = 0.0f;
    //                 _rays[index + 5] = 1.0f;

    //                 // length
    //                 _rays[index + 6] = 1.0f;
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
    //         _gpu_rays,
    //         _gpu_face_vertices,
    //         _gpu_face_colors,
    //         _gpu_object_types,
    //         _gpu_material_types,
    //         _gpu_color_per_ray,
    //         _gpu_camera_inv_matrix,
    //         _rays,
    //         _vertices,
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
    // rtx_cuda_copy(_gpu_rays, _gpu_face_vertices, _gpu_camera_inv_matrix, _rays, _vertices, _camera_inv_matrix, num_rays, rays_stride, num_faces, faces_stride);

    // rtx_cuda_ray_tracing_render(
    //     _gpu_rays,
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
    //         buffer[index + 0] = std::min(std::max((int)(sum_r / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //         buffer[index + 1] = std::min(std::max((int)(sum_g / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //         buffer[index + 2] = std::min(std::max((int)(sum_b / float(num_rays_per_pixel) * 255.0f), 0), 255);
    //     }
    // }

    // _initialized = true;
    // rtx_cuda_delete(_gpu_rays,
    //     _gpu_face_vertices,
    //     _gpu_face_colors,
    //     _gpu_object_types,
    //     _gpu_material_types,
    //     _gpu_color_per_ray);
    // delete[] _vertices;
    // delete[] _face_colors;
    // delete[] _object_types;
    // delete[] _material_types;
    // delete[] _rays;
    // delete[] _color_per_ray;
}
}