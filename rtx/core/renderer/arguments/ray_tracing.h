#pragma once

namespace rtx {
class RayTracingArguments {
private:
    int _num_rays_per_pixel;
    int _max_bounce;
    bool _next_event_estimation_enabled;

public:
    RayTracingArguments();
    int num_rays_per_pixel();
    void set_num_rays_per_pixel(int num);
    int max_bounce();
    void set_max_bounce(int bounce);
    bool next_event_estimation_enabled();
    void set_next_event_estimation_enabled(bool enabled);
};
}