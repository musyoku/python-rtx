#include "renderer.h"
#include "../camera/perspective.h"
#include "../geometry/box.h"
#include "../geometry/sphere.h"
#include "../geometry/standard.h"
#include "../header/enum.h"
#include "../mapping/solid_color.h"
#include "../material/emissive.h"
#include "../material/lambert.h"
#include "header/bridge.h"
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
    _gpu_material_attribute_byte_array = NULL;
    _gpu_threaded_bvh_array = NULL;
    _gpu_threaded_bvh_node_array = NULL;
    _gpu_light_sampling_table = NULL;
    _gpu_color_mapping_array = NULL;
    _gpu_render_array = NULL;
    _total_frames = 0;

    rtx_cuda_allocate_linear_memory_texture_objects();
}
Renderer::~Renderer()
{
    rtx_cuda_free((void**)&_gpu_ray_array);
    rtx_cuda_free((void**)&_gpu_face_vertex_indices_array);
    rtx_cuda_free((void**)&_gpu_vertex_array);
    rtx_cuda_free((void**)&_gpu_object_array);
    rtx_cuda_free((void**)&_gpu_material_attribute_byte_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
    rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
    rtx_cuda_free((void**)&_gpu_color_mapping_array);
    rtx_cuda_free((void**)&_gpu_render_array);

    rtx_cuda_delete_linear_memory_texture_objects();
}
void Renderer::transform_objects_to_view_space()
{
    int num_objects = _scene->_object_array.size();
    if (num_objects == 0) {
        return;
    }
    _transformed_object_array = std::vector<std::shared_ptr<Object>>();
    _transformed_object_array.reserve(num_objects);
    for (auto& object : _scene->_object_array) {
        auto& geometry = object->geometry();
        glm::mat4 transformation_matrix = _camera->view_matrix() * geometry->model_matrix();
        auto transformed_geometry = geometry->transoform(transformation_matrix);
        _transformed_object_array.emplace_back(std::make_shared<Object>(transformed_geometry, object->material(), object->mapping()));
    }
}
void Renderer::serialize_objects()
{
    int num_objects = _transformed_object_array.size();
    assert(num_objects > 0);

    int total_faces = 0;
    int total_vertices = 0;
    for (auto& object : _transformed_object_array) {
        auto& geometry = object->geometry();
        total_faces += geometry->num_faces();
        total_vertices += geometry->num_vertices();
    }

    _cpu_face_vertex_indices_array = rtx::array<RTXFace>(total_faces);
    _cpu_vertex_array = rtx::array<RTXVertex>(total_vertices);
    _cpu_object_array = rtx::array<RTXObject>(num_objects);
    _cpu_light_sampling_table = std::vector<int>();

    // serialize vertices
    int vertex_index_offset = 0;
    int vertex_offset = 0;
    std::vector<int> vertex_index_offset_array(num_objects);
    std::vector<int> vertex_count_array(num_objects);
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& object = _transformed_object_array.at(object_index);
        auto& geometry = object->geometry();
        geometry->serialize_vertices(_cpu_vertex_array, vertex_index_offset);
        vertex_index_offset_array[object_index] = vertex_index_offset;
        vertex_count_array[object_index] = geometry->num_vertices();
        vertex_index_offset += geometry->num_vertices();
    }

    // serialize faces
    int face_index_offset = 0;
    std::vector<int> face_index_offset_array(num_objects);
    std::vector<int> face_count_array(num_objects);
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& object = _transformed_object_array.at(object_index);
        auto& geometry = object->geometry();
        auto& bvh = _geometry_bvh_array.at(object_index);
        bvh->serialize_faces(_cpu_face_vertex_indices_array, face_index_offset);
        face_index_offset_array[object_index] = face_index_offset;
        face_count_array[object_index] = geometry->num_faces();
        face_index_offset += geometry->num_faces();
    }

    assert(vertex_index_offset_array.size() == num_objects);
    assert(face_index_offset_array.size() == num_objects);

    int total_material_bytes = 0;
    int num_geometries = 0;
    _cpu_light_sampling_table = std::vector<int>();
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& object = _transformed_object_array.at(object_index);
        auto& material = object->material();
        if (material->is_emissive()) {
            _cpu_light_sampling_table.push_back(object_index);
        }
        total_material_bytes += material->attribute_bytes();
    }
    _cpu_material_attribute_byte_array = rtx::array<RTXMaterialAttributeByte>(total_material_bytes / sizeof(RTXMaterialAttributeByte));

    int material_attribute_byte_array_offset = 0;
    _cpu_color_mapping_array = std::vector<RTXColor>();
    _texture_mapping_array = std::vector<TextureMapping*>();
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& object = _transformed_object_array.at(object_index);
        auto& geometry = object->geometry();
        auto& material = object->material();
        auto& mapping = object->mapping();

        RTXObject cuda_object;
        cuda_object.num_faces = face_count_array[object_index];
        cuda_object.face_index_offset = face_index_offset_array[object_index];
        cuda_object.num_vertices = vertex_count_array[object_index];
        cuda_object.vertex_index_offset = vertex_index_offset_array[object_index];
        cuda_object.geometry_type = geometry->type();
        cuda_object.num_material_layers = material->num_layers();
        cuda_object.layerd_material_types = material->types();
        cuda_object.material_attribute_byte_array_offset = material_attribute_byte_array_offset;
        cuda_object.mapping_type = mapping->type();
        cuda_object.mapping_index = -1;

        if (mapping->type() == RTXMappingTypeSolidColor) {
            SolidColorMapping* m = static_cast<SolidColorMapping*>(mapping.get());
            auto color = m->color();
            _cpu_color_mapping_array.emplace_back(RTXColor({ color.r, color.g, color.b, color.a }));
            cuda_object.mapping_index = _cpu_color_mapping_array.size() - 1;
        } else if (mapping->type() == RTXMappingTypeTexture) {
            TextureMapping* m = static_cast<TextureMapping*>(mapping.get());
            _texture_mapping_array.push_back(m);
            cuda_object.mapping_index = _texture_mapping_array.size() - 1;
        }

        _cpu_object_array[object_index] = cuda_object;

        material->serialize_attributes(_cpu_material_attribute_byte_array, material_attribute_byte_array_offset);
        material_attribute_byte_array_offset += material->attribute_bytes() / sizeof(RTXMaterialAttributeByte);
    }
}
void Renderer::construct_bvh()
{
    int num_objects = _transformed_object_array.size();
    assert(num_objects > 0);
    _geometry_bvh_array = std::vector<std::shared_ptr<BVH>>(num_objects);
    int total_nodes = 0;

#pragma omp parallel for
    for (int object_index = 0; object_index < (int)_transformed_object_array.size(); object_index++) {
        auto& object = _transformed_object_array[object_index];
        auto& geometry = object->geometry();
        assert(geometry->bvh_max_triangles_per_node() > 0);
        std::shared_ptr<BVH> bvh = std::make_shared<BVH>(geometry);
        _geometry_bvh_array[object_index] = bvh;
    }

    for (auto& bvh : _geometry_bvh_array) {
        total_nodes += bvh->num_nodes();
    }

    _cpu_threaded_bvh_array = rtx::array<RTXThreadedBVH>(num_objects);
    _cpu_threaded_bvh_node_array = rtx::array<RTXThreadedBVHNode>(total_nodes);

    int node_index_offset = 0;
    for (int object_index = 0; object_index < num_objects; object_index++) {
        auto& bvh = _geometry_bvh_array[object_index];

        RTXThreadedBVH cuda_bvh;
        cuda_bvh.node_index_offset = node_index_offset;
        cuda_bvh.num_nodes = bvh->num_nodes();
        _cpu_threaded_bvh_array[object_index] = cuda_bvh;

        bvh->serialize_nodes(_cpu_threaded_bvh_node_array, node_index_offset);
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
void Renderer::launch_kernel()
{
    size_t available_shared_memory_bytes = rtx_cuda_get_available_shared_memory_bytes();

    int num_rays = _cpu_ray_array.size();
    int num_rays_per_thread = num_rays / (_cuda_args->num_threads() * _cuda_args->num_blocks()) + 1;

    size_t required_shared_memory_bytes;
    required_shared_memory_bytes += _cpu_face_vertex_indices_array.bytes();
    required_shared_memory_bytes += _cpu_vertex_array.bytes();
    required_shared_memory_bytes += _cpu_object_array.bytes();
    required_shared_memory_bytes += _cpu_material_attribute_byte_array.bytes();
    required_shared_memory_bytes += _cpu_threaded_bvh_array.bytes();
    required_shared_memory_bytes += _cpu_threaded_bvh_node_array.bytes();
    required_shared_memory_bytes += sizeof(RTXColor) * _cpu_color_mapping_array.size();

    int curand_seed = _total_frames;

    if (required_shared_memory_bytes <= available_shared_memory_bytes) {
        rtx_cuda_launch_standard_shared_memory_kernel(
            _gpu_ray_array, _cpu_ray_array.size(),
            _gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.size(),
            _gpu_vertex_array, _cpu_vertex_array.size(),
            _gpu_object_array, _cpu_object_array.size(),
            _gpu_material_attribute_byte_array, _cpu_material_attribute_byte_array.size(),
            _gpu_threaded_bvh_array, _cpu_threaded_bvh_array.size(),
            _gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.size(),
            _gpu_color_mapping_array, _cpu_color_mapping_array.size(),
            _gpu_render_array, _cpu_render_array.size(),
            _cuda_args->num_threads(),
            _cuda_args->num_blocks(),
            num_rays_per_thread,
            required_shared_memory_bytes,
            _rt_args->max_bounce(),
            curand_seed);
        return;
    }
    required_shared_memory_bytes = 0;
    required_shared_memory_bytes += _cpu_object_array.bytes();
    required_shared_memory_bytes += _cpu_material_attribute_byte_array.bytes();
    required_shared_memory_bytes += _cpu_threaded_bvh_array.bytes();
    required_shared_memory_bytes += sizeof(RTXColor) * _cpu_color_mapping_array.size();

    if (required_shared_memory_bytes <= available_shared_memory_bytes) {
        // テクスチャメモリに直列データを入れる場合
        // こちらの方が若干早い
        rtx_cuda_launch_standard_texture_memory_kernel(
            _gpu_ray_array, _cpu_ray_array.size(),
            _gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.size(),
            _gpu_vertex_array, _cpu_vertex_array.size(),
            _gpu_object_array, _cpu_object_array.size(),
            _gpu_material_attribute_byte_array, _cpu_material_attribute_byte_array.size(),
            _gpu_threaded_bvh_array, _cpu_threaded_bvh_array.size(),
            _gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.size(),
            _gpu_color_mapping_array, _cpu_color_mapping_array.size(),
            _gpu_render_array, _cpu_render_array.size(),
            _cuda_args->num_threads(),
            _cuda_args->num_blocks(),
            num_rays_per_thread,
            required_shared_memory_bytes,
            _rt_args->max_bounce(),
            curand_seed);

        // グローバルメモリに直列データを入れる場合
        // rtx_cuda_launch_standard_global_memory_kernel(
        //     _gpu_ray_array, _cpu_ray_array.size(),
        //     _gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.size(),
        //     _gpu_vertex_array, _cpu_vertex_array.size(),
        //     _gpu_object_array, _cpu_object_array.size(),
        //     _gpu_material_attribute_byte_array, _cpu_material_attribute_byte_array.size(),
        //     _gpu_threaded_bvh_array, _cpu_threaded_bvh_array.size(),
        //     _gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.size(),
        //     _gpu_color_mapping_array, _cpu_color_mapping_array.size(),
        //     _gpu_render_array, _cpu_render_array.size(),
        //     _cuda_args->num_threads(),
        //     _cuda_args->num_blocks(),
        //     num_rays_per_thread,
        //     required_shared_memory_bytes,
        //     _rt_args->max_bounce(),
        //     curand_seed);
        return;
    }

    throw std::runtime_error("Error: Not implemented");
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
    bool should_reset_total_frames = false;
    if (_scene->updated()) {
        should_transform_geometry = true;
        should_serialize_geometry = true;
        should_serialize_bvh = true;
        should_reallocate_gpu_memory = true;
        should_transfer_to_gpu = true;
        should_reset_total_frames = true;
    } else {
        if (_camera->updated()) {
            should_transform_geometry = true;
            should_serialize_geometry = true;
            should_serialize_bvh = true;
            should_transfer_to_gpu = true;
            should_reset_total_frames = true;
        }
    }
    if (_prev_height != height || _prev_width != width) {
        should_update_ray = true;
    }

    if (should_transform_geometry) {
        transform_objects_to_view_space();
    }

    // 現在のカメラ座標系でのBVHを構築
    // Construct BVH in current camera coordinate system
    if (should_serialize_bvh) {
        construct_bvh();
        rtx_cuda_free((void**)&_gpu_threaded_bvh_array);
        rtx_cuda_free((void**)&_gpu_threaded_bvh_node_array);
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_array, _cpu_threaded_bvh_array.bytes());
        rtx_cuda_malloc((void**)&_gpu_threaded_bvh_node_array, _cpu_threaded_bvh_node_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_threaded_bvh_array, (void*)_cpu_threaded_bvh_array.data(), _cpu_threaded_bvh_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_threaded_bvh_node_array, (void*)_cpu_threaded_bvh_node_array.data(), _cpu_threaded_bvh_node_array.bytes());
    }

    // オブジェクトをシリアライズ
    // Serialize objects
    if (should_serialize_geometry) {
        serialize_objects();
    }
    if (should_reallocate_gpu_memory) {
        assert(_cpu_face_vertex_indices_array.size() > 0);
        assert(_cpu_vertex_array.size() > 0);
        assert(_cpu_object_array.size() > 0);
        assert(_cpu_material_attribute_byte_array.size() > 0);
        rtx_cuda_free((void**)&_gpu_face_vertex_indices_array);
        rtx_cuda_free((void**)&_gpu_vertex_array);
        rtx_cuda_free((void**)&_gpu_object_array);
        rtx_cuda_free((void**)&_gpu_material_attribute_byte_array);
        rtx_cuda_malloc((void**)&_gpu_face_vertex_indices_array, _cpu_face_vertex_indices_array.bytes());
        rtx_cuda_malloc((void**)&_gpu_vertex_array, _cpu_vertex_array.bytes());
        rtx_cuda_malloc((void**)&_gpu_object_array, _cpu_object_array.bytes());
        rtx_cuda_malloc((void**)&_gpu_material_attribute_byte_array, _cpu_material_attribute_byte_array.bytes());
        if (_cpu_light_sampling_table.size() > 0) {
            rtx_cuda_free((void**)&_gpu_light_sampling_table);
            rtx_cuda_malloc((void**)&_gpu_light_sampling_table, sizeof(int) * _cpu_light_sampling_table.size());
        }
        if (_cpu_color_mapping_array.size() > 0) {
            rtx_cuda_free((void**)&_gpu_color_mapping_array);
            rtx_cuda_malloc((void**)&_gpu_color_mapping_array, sizeof(RTXColor) * _cpu_color_mapping_array.size());
        }
        if (_texture_mapping_array.size() > 0) {
            for (int mapping_index = 0; mapping_index < (int)_texture_mapping_array.size(); mapping_index++) {
                TextureMapping* mapping = _texture_mapping_array[mapping_index];
                rtx_cuda_malloc_texture(mapping_index, mapping->width(), mapping->height());
                rtx_cuda_memcpy_to_texture(mapping_index, 0, mapping->width(), mapping->pointer(), mapping->bytes());
                // rtx_cuda_memcpy_to_texture(mapping_index, mapping->width(), mapping->height(), mapping->pointer(), mapping->bytes());
                rtx_cuda_bind_texture(mapping_index);
            }
        }
    }
    if (should_transfer_to_gpu) {
        rtx_cuda_memcpy_host_to_device((void*)_gpu_face_vertex_indices_array, (void*)_cpu_face_vertex_indices_array.data(), _cpu_face_vertex_indices_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_vertex_array, (void*)_cpu_vertex_array.data(), _cpu_vertex_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_object_array, (void*)_cpu_object_array.data(), _cpu_object_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_material_attribute_byte_array, (void*)_cpu_material_attribute_byte_array.data(), _cpu_material_attribute_byte_array.bytes());
        if (_cpu_light_sampling_table.size() > 0) {
            rtx_cuda_memcpy_host_to_device((void*)_gpu_light_sampling_table, (void*)_cpu_light_sampling_table.data(), sizeof(int) * _cpu_light_sampling_table.size());
        }
        if (_cpu_color_mapping_array.size() > 0) {
            rtx_cuda_memcpy_host_to_device((void*)_gpu_color_mapping_array, (void*)_cpu_color_mapping_array.data(), sizeof(RTXColor) * _cpu_color_mapping_array.size());
        }
    }

    int num_rays_per_pixel = _rt_args->num_rays_per_pixel();
    int num_rays = height * width * num_rays_per_pixel;

    // レイ
    // Ray
    if (should_update_ray) {
        _cpu_render_array = rtx::array<RTXPixel>(num_rays);
        for (int n = 0; n < num_rays; n++) {
            _cpu_render_array[n] = { -1.0f, -1.0f, -1.0f, -1.0f };
        }
        _cpu_render_buffer_array = rtx::array<RTXPixel>(height * width * 3);
        serialize_rays(height, width);
        rtx_cuda_free((void**)&_gpu_ray_array);
        rtx_cuda_free((void**)&_gpu_render_array);
        rtx_cuda_malloc((void**)&_gpu_ray_array, _cpu_ray_array.bytes());
        rtx_cuda_malloc((void**)&_gpu_render_array, _cpu_render_array.bytes());
        rtx_cuda_memcpy_host_to_device((void*)_gpu_ray_array, (void*)_cpu_ray_array.data(), _cpu_ray_array.bytes());
        _prev_height = height;
        _prev_width = width;
    }

    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("preprocessing: %lf msec\n", elapsed);

    start = std::chrono::system_clock::now();
    launch_kernel();
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("kernel: %lf msec\n", elapsed);

    rtx_cuda_memcpy_device_to_host((void*)_cpu_render_array.data(), (void*)_gpu_render_array, _cpu_render_array.bytes());

    _scene->set_updated(false);
    _camera->set_updated(false);

    if (should_reset_total_frames) {
        _total_frames = 0;
    }
    _total_frames++;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RTXPixel pixel_buffer = _cpu_render_buffer_array[y * width * 3 + x * 3];
            if (_total_frames == 1) {
                pixel_buffer.r = 0.0;
                pixel_buffer.g = 0.0;
                pixel_buffer.b = 0.0;
            }
            for (int m = 0; m < num_rays_per_pixel; m++) {
                int index = y * width * num_rays_per_pixel + x * num_rays_per_pixel + m;
                RTXPixel pixel = _cpu_render_array[index];
                pixel_buffer.r += pixel.r / float(num_rays_per_pixel);
                pixel_buffer.g += pixel.g / float(num_rays_per_pixel);
                pixel_buffer.b += pixel.b / float(num_rays_per_pixel);
            }
            _cpu_render_buffer_array[y * width * 3 + x * 3] = pixel_buffer;
        }
    }
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

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RTXPixel pixel_buffer = _cpu_render_buffer_array[y * width * 3 + x * 3];
            pixel(y, x, 0) = pixel_buffer.r / _total_frames;
            pixel(y, x, 1) = pixel_buffer.g / _total_frames;
            pixel(y, x, 2) = pixel_buffer.b / _total_frames;
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