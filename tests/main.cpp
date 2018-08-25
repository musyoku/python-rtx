#include "../rtx/core/camera/perspective.h"
#include "../rtx/core/class/scene.h"
#include "../rtx/core/geometry/sphere.h"
#include "../rtx/core/material/mesh/lambert.h"
#include "../rtx/core/renderer/cpu/ray_tracing/renderer.h"
#include "../rtx/core/renderer/cuda/header/ray_tracing.h"
#include "../rtx/core/renderer/cuda/ray_tracing/renderer.h"
#include "../rtx/core/renderer/options/ray_tracing.h"
#include <chrono>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace rtx;
namespace py = pybind11;

int main()
{
    std::shared_ptr<Scene> scene = std::make_shared<Scene>();
    float colors[3][3] = { { 1.0f, 1.0f, 0.25f }, { 1.0f, 0.25f, 1.0f }, { 0.25f, 1.0f, 1.0f } };
    float shift[3] = { -1.125, 0, 1.125 };

    for (int n = 0; n < 27; n++) {
        std::shared_ptr<SphereGeometry> geometry = std::make_shared<SphereGeometry>(0.5);

        float color[3] = { colors[n % 3][0], colors[n % 3][1], colors[n % 3][2] };
        std::shared_ptr<MeshLambertMaterial> material = std::make_shared<MeshLambertMaterial>(color, 0.8f);

        std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(geometry, material);
        float position[3] = { shift[n % 3], shift[(n / 3) % 3], shift[n / 9] };
        mesh->set_position(position);

        scene->add(mesh);
    }

    float eye[] = { 0.0f, 0.0f, 2.0f };
    float center[] = { 0.0f, 0.0f, 0.0f };
    float up[] = { 0.0f, 1.0f, 0.0f };
    std::shared_ptr<PerspectiveCamera> camera = std::make_shared<PerspectiveCamera>(eye, center, up, 1.0f, 1.0f, 1.0f, 1.0f);

    std::shared_ptr<RayTracingOptions> options = std::make_shared<RayTracingOptions>();
    options->set_num_rays_per_pixel(512);
    options->set_path_depth(5);
    std::shared_ptr<RayTracingCUDARenderer> render = std::make_shared<RayTracingCUDARenderer>();

    int width = 64;
    int height = 64;
    int channels = 3;
    unsigned char* pixels = new unsigned char[height * width * channels];
    render->render(scene, camera, options, pixels, height, width, channels);
    auto start = std::chrono::system_clock::now();
    int repeat = 50;
    for (int i = 0; i < repeat; i++) {
        render->render(scene, camera, options, pixels, height, width, channels);
    }
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "fps: " << (double(repeat) / elapsed * 1000) << std::endl;

    stbi_write_bmp("render.bmp", width, height, 3, pixels);
    delete[] pixels;

    cuda_device_reset();
    return 0;
}