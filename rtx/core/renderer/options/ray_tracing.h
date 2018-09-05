#pragma once

namespace rtx {
class RayTracingOptions {
private:
    int _num_rays_per_pixel; // for supersampling
    int _max_bounce;
public:
    RayTracingOptions();
    int num_rays_per_pixel();
    void set_num_rays_per_pixel(int num);
    int max_bounce();
    void set_max_bounce(int bounce);
};
}