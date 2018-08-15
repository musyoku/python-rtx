#include "../three/core/camera/perspective.h"
#include "../three/core/class/scene.h"
#include "../three/core/geometry/sphere.h"
#include "../three/core/material/mesh/lambert.h"
#include "../three/core/renderer/cpu/ray_tracing/renderer.h"
#include "../three/core/renderer/options/ray_tracing.h"

using namespace three;
namespace py = pybind11;

int main()
{
    std::shared_ptr<Scene> scene = std::make_shared<Scene>();

    std::shared_ptr<SphereGeometry> geometry = std::make_shared<SphereGeometry>(1.0f);

    py::tuple color = py::make_tuple(1.0f, 1.0f, 1.0f);
    std::shared_ptr<MeshLambertMaterial> material = std::make_shared<MeshLambertMaterial>(color, 0.8f);

    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(geometry, material);

    scene->add(mesh);

    py::tuple eye = py::make_tuple(0.0f, 0.0f, 2.0f);
    py::tuple center = py::make_tuple(0.0f, 0.0f, 0.0f);
    py::tuple up = py::make_tuple(0.0f, 1.0f, 0.0f);
    std::shared_ptr<PerspectiveCamera> camera = std::make_shared<PerspectiveCamera>(eye, center, up, 1.0f, 1.0f, 1.0f, 1.0f);

    std::shared_ptr<RayTracingOptions> options = std::make_shared<RayTracingOptions>();
    options->set_num_rays_per_pixel(64);
    options->set_path_depth(4);
    std::shared_ptr<RayTracingCPURenderer> render = std::make_shared<RayTracingCPURenderer>();

    int width = 128;
    int height = 128;
    int channels = 3;
    float*** buffer = new float**[height];
    for (int h = 0; h < height; h++) {
        buffer[h] = new float*[width];
        for (int w = 0; w < width; w++) {
            buffer[h][w] = new float[channels];
            for (int c = 0; c < channels; c++) {
                buffer[h][w][c] = 0.0f;
            }
        }
    }
    render->render(scene, camera, options, buffer, height, width, channels);

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            delete[] buffer[h][w];
        }
        delete[] buffer[h];
    }
    delete[] buffer;

    return 0;
}