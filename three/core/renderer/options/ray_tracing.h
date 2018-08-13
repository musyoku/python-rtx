#pragma once

namespace three {
class RayTracingOptions {
private:
    int _num_rays_per_pixel; // for supersampling
public:
    RayTracingOptions();
    int num_rays_per_pixel();
    void set_num_rays_per_pixel(int num);
};
}