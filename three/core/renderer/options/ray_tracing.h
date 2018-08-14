#pragma once

namespace three {
class RayTracingOptions {
private:
    int _num_rays_per_pixel; // for supersampling
    int _path_depth;
public:
    RayTracingOptions();
    int num_rays_per_pixel();
    void set_num_rays_per_pixel(int num);
    int path_depth();
    void set_path_depth(int depth);
};
}