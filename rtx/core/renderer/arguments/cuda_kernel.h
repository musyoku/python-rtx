#pragma once

namespace rtx {
class CUDAKernelLaunchArguments {
private:
    int _num_threads;
    int _num_rays_per_thread;
    int _num_blocks;

public:
    CUDAKernelLaunchArguments();
    int num_threads();
    void set_num_threads(int num);
    int num_rays_per_thread();
    void set_num_rays_per_thread(int num);
    int num_blocks();
    void set_num_blocks(int num);
};
}